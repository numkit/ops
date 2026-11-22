// toolboxes/builtin/src/lang/arrays/nd_manip.cpp
//
// Phase 6: N-D array manipulation. nk's Value is capped at
// 3D so the perm vector must have length 2 or 3. The general
// approach: compute output dims from input dims via the permutation,
// then iterate over output indices and compute the corresponding
// input index. Pure scalar gather; no SIMD opportunity in the
// general case (input strides differ per axis after permutation).

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <numkit/builtin/elmat.hpp>

#include <numkit/builtin/elmat.hpp>  // reshape, horzcat, vertcat
#include <numkit/value/value.hpp>
#include <numkit/value/shape_ops.hpp>      // computeStridesColMajor, incrementCoords
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>

#include <algorithm>
#include <numeric>

namespace numkit::builtin {

namespace {

// Check the perm list is a valid 1-based permutation of [1..N].
// Throws on bad input. Stack-mounted scratch (perm length is bounded
// by Dims::kMaxRank); no heap traffic.
void validatePerm(const keel::i32 *perm, keel::usize N, const char *fn)
{
    if (N == 0)
        throw Error(keel::format("{}: perm vector must not be empty", fn),
                     0, 0, fn, "", keel::format("numkit:{}:emptyPerm", fn));
    if (N > Dims::kMaxRank)
        throw Error(keel::format("{}: perm length exceeds 32", fn),
                     0, 0, fn, "", keel::format("numkit:{}:tooManyDims", fn));
    keel::i32 sorted[Dims::kMaxRank];
    for (keel::usize i = 0; i < N; ++i) sorted[i] = perm[i];
    std::sort(sorted, sorted + N);
    for (keel::usize i = 0; i < N; ++i) {
        if (sorted[i] != keel::as<keel::i32>(i + 1))
            throw Error(keel::format("{}: perm must be a permutation of 1..N", fn),
                         0, 0, fn, "", keel::format("numkit:{}:badPerm", fn));
    }
}

} // namespace

namespace {

// Phase P6: cache-blocked transpose for the per-page [2 1] perm. The
// straight strided gather (read down columns of input, write across
// rows of output) thrashes L1 — at R=512 doubles per column, every
// load misses cache for the first column-tile of each output row. A
// 32×32 block fits in L1 (8 KB), so an inner block transpose copies
// each input element exactly once with locality on both sides.
constexpr keel::usize TRANSPOSE_BLOCK = 32;

void transposePage(const keel::f64 *src, keel::f64 *dst, keel::usize inR, keel::usize inC)
{
    const keel::usize outR = inC; // output rows = input cols
    // Block over output coordinates so that the inner write loop is
    // CONTIGUOUS in the destination (stride 1 along dst column). Reads
    // from src are then strided by inR but stay within a small block,
    // so they're cached after the first access in each block. This
    // minimises write traffic — strided writes evict the write-combiner
    // and require read-for-ownership per cache line, which is the
    // dominant cost in the unblocked transpose.
    for (keel::usize rBlk = 0; rBlk < inR; rBlk += TRANSPOSE_BLOCK) {
        const keel::usize rEnd = std::min(rBlk + TRANSPOSE_BLOCK, inR);
        for (keel::usize cBlk = 0; cBlk < inC; cBlk += TRANSPOSE_BLOCK) {
            const keel::usize cEnd = std::min(cBlk + TRANSPOSE_BLOCK, inC);
            // Inner block: dst column r holds input row r.
            // For fixed r, vary c: dst write is stride 1 in c.
            for (keel::usize r = rBlk; r < rEnd; ++r) {
                keel::f64 *dstCol = dst + r * outR;
                for (keel::usize c = cBlk; c < cEnd; ++c) {
                    dstCol[c] = src[c * inR + r];
                }
            }
        }
    }
}

} // namespace

// ────────────────────────────────────────────────────────────────────
// permute / ipermute
// ────────────────────────────────────────────────────────────────────
//
// Semantics (MATLAB):
//   B(i_1, i_2, ..., i_N) = A(i_{p_1}, i_{p_2}, ..., i_{p_N})
// i.e. output axis k corresponds to input axis perm[k]. So the size
// of output along axis k equals the size of input along perm[k].
Value permute(const Value &x, keel::ArrayRef<keel::i32> perm, keel::Allocator *mr)
{
    const keel::usize n = perm.size();
    validatePerm(perm.data(), n, "permute");

    const auto &dd = x.dims();
    const keel::i32 inNd = std::max<keel::i32>(dd.ndim(), keel::as<keel::i32>(n));
    if (inNd > Dims::kMaxRank)
        throw Error("permute: rank exceeds 32",
                     0, 0, "permute", "", "numkit:permute:tooManyDims");

    // All shape arrays on the stack — no per-call heap traffic. Avoids
    // 4 std::vector allocs that dominated cost at small sizes (post-ND
    // generalisation regressed BM_Permute3D /16 by 1.77× before this).
    keel::i32 p[Dims::kMaxRank];
    keel::usize inDims[Dims::kMaxRank];
    keel::usize outDimsArr[Dims::kMaxRank];
    for (keel::i32 i = 0; i < inNd; ++i)
        p[i] = (i < keel::as<keel::i32>(n)) ? perm[i] : (i + 1);
    for (keel::i32 i = 0; i < inNd; ++i) inDims[i] = dd.dim(i);
    for (keel::i32 k = 0; k < inNd; ++k) outDimsArr[k] = inDims[p[k] - 1];

    // OBJECT: permute dimensions via the strided gather map → objectGather.
    if (x.isObject()) {
        Dims outDimsObj(outDimsArr, inNd);
        if (x.numel() == 0)
            return x.objectGather(nullptr, outDimsObj, mr);
        keel::usize inStrides[Dims::kMaxRank];
        computeStridesColMajor(Dims(inDims, inNd), inStrides);
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<keel::usize> map(x.numel(), scratch.allocator());
        keel::usize outCoords[Dims::kMaxRank] = {0};
        keel::usize dstOff = 0;
        do {
            keel::usize srcOff = 0;
            for (keel::i32 k = 0; k < inNd; ++k)
                srcOff += outCoords[k] * inStrides[p[k] - 1];
            map[dstOff++] = srcOff;
        } while (incrementCoords(outCoords, outDimsObj));
        return x.objectGather(map.data(), outDimsObj, mr);
    }

    // Non-DOUBLE (char / logical / complex / single / int / cell): permute is
    // a pure rearrangement -> type-preserving. Strided gather, copying raw
    // bytes (POD) or Value elements (CELL). The DOUBLE fast paths below are
    // left untouched.
    if (x.type() != ValueType::DOUBLE) {
        const ValueType t = x.type();
        const bool isCell = (t == ValueType::CELL);
        if (t == ValueType::STRING || t == ValueType::STRUCT || t == ValueType::FUNC_HANDLE)
            throw Error(keel::format("permute: does not support type '{}'", mtypeName(t)),
                         0, 0, "permute", "", "numkit:permute:type");
        if (isCell && inNd > 2)
            throw Error("permute: cell array rank > 2 not supported",
                         0, 0, "permute", "", "numkit:permute:cellND");
        Value r = isCell ? Value::cell(outDimsArr[0], inNd >= 2 ? outDimsArr[1] : 1, mr)
                         : Value::matrixND(outDimsArr, inNd, t, mr);
        if (x.numel() == 0) return r;
        keel::usize inStrides[Dims::kMaxRank];
        computeStridesColMajor(Dims(inDims, inNd), inStrides);
        const keel::usize es = isCell ? 0 : elementSize(t);
        const char *src = isCell ? nullptr : static_cast<const char *>(x.rawData());
        char *dst = isCell ? nullptr : static_cast<char *>(r.rawDataMut());
        keel::usize outCoords[Dims::kMaxRank] = {0};
        Dims outDimsObj(outDimsArr, inNd);
        keel::usize dstOff = 0;
        do {
            keel::usize srcOff = 0;
            for (keel::i32 k = 0; k < inNd; ++k)
                srcOff += outCoords[k] * inStrides[p[k] - 1];
            if (isCell) r.cellAt(dstOff) = x.cellAt(srcOff);
            else        keel::mem_copy(dst + dstOff * es, src + srcOff * es, es);
            ++dstOff;
        } while (incrementCoords(outCoords, outDimsObj));
        return r;
    }

    // 2D / 3D fast path uses createMatrix / createMatrix3d via matrixND;
    // ≥ 4D goes through Value::matrixND. Trailing 1s are kept.
    auto r = Value::matrixND(outDimsArr, inNd, ValueType::DOUBLE, mr);
    if (x.numel() == 0) return r;

    const keel::f64 *src = x.doubleData();
    keel::f64 *dst       = r.doubleDataMut();

    // Phase P6 fast path: per-page transpose ([2, 1] or [2, 1, 3] for
    // ≤ 3D inputs). 2.3× win on 512×512 (cache-blocked vs strided gather).
    // For ND inputs, iterate over the trailing pages of the first 2 axes.
    if (inNd >= 2 && p[0] == 2 && p[1] == 1) {
        bool tailIsIdentity = true;
        for (keel::i32 k = 2; k < inNd; ++k)
            if (p[k] != k + 1) { tailIsIdentity = false; break; }
        if (tailIsIdentity) {
            const keel::usize inR = inDims[0], inC = inDims[1];
            keel::usize P = 1;
            for (keel::i32 k = 2; k < inNd; ++k) P *= inDims[k];
            for (keel::usize pp = 0; pp < P; ++pp)
                transposePage(src + pp * inR * inC,
                              dst + pp * inC * inR,
                              inR, inC);
            return r;
        }
    }

    // 3D fast path — explicit nested loops with constant strides. Avoids
    // the per-element coord-walk overhead that dominated pre-fix at
    // small sizes (BM_Permute3D /16: 7.5us → expected ≤ 5us).
    if (inNd == 3) {
        const keel::usize s[3] = {1, inDims[0], inDims[0] * inDims[1]};
        const keel::usize sa = s[p[0] - 1];
        const keel::usize sb = s[p[1] - 1];
        const keel::usize sc = s[p[2] - 1];
        const keel::usize outR = outDimsArr[0], outC = outDimsArr[1], outP = outDimsArr[2];
        keel::usize dstIdx = 0;
        for (keel::usize k = 0; k < outP; ++k) {
            const keel::usize baseK = k * sc;
            for (keel::usize j = 0; j < outC; ++j) {
                const keel::usize baseJK = baseK + j * sb;
                for (keel::usize i = 0; i < outR; ++i)
                    dst[dstIdx++] = src[baseJK + i * sa];
            }
        }
        return r;
    }

    // 2D fast path — same idea, two nested loops.
    if (inNd == 2) {
        const keel::usize s[2] = {1, inDims[0]};
        const keel::usize sa = s[p[0] - 1];
        const keel::usize sb = s[p[1] - 1];
        const keel::usize outR = outDimsArr[0], outC = outDimsArr[1];
        keel::usize dstIdx = 0;
        for (keel::usize j = 0; j < outC; ++j) {
            const keel::usize baseJ = j * sb;
            for (keel::usize i = 0; i < outR; ++i)
                dst[dstIdx++] = src[baseJ + i * sa];
        }
        return r;
    }

    // General ND permute (4D+) — strided gather via incrementCoords.
    keel::usize inStrides[Dims::kMaxRank];
    computeStridesColMajor(Dims(inDims, inNd), inStrides);

    keel::usize outCoords[Dims::kMaxRank] = {0};
    Dims outDimsObj(outDimsArr, inNd);
    keel::usize dstOff = 0;
    do {
        keel::usize srcOff = 0;
        for (keel::i32 k = 0; k < inNd; ++k) {
            const keel::i32 axIn = p[k] - 1;
            srcOff += outCoords[k] * inStrides[axIn];
        }
        dst[dstOff] = src[srcOff];
        ++dstOff;
    } while (incrementCoords(outCoords, outDimsObj));
    return r;
}

Value ipermute(const Value &x, keel::ArrayRef<keel::i32> perm, keel::Allocator *mr)
{
    const keel::usize n = perm.size();
    validatePerm(perm.data(), n, "ipermute");
    // Compute inverse permutation: invPerm[perm[i] - 1] = i + 1.
    // Stack-mounted: perm length is bounded by Dims::kMaxRank.
    keel::i32 inv[Dims::kMaxRank];
    for (keel::usize i = 0; i < n; ++i)
        inv[perm[i] - 1] = keel::as<keel::i32>(i + 1);
    return permute(x, keel::ArrayRef<keel::i32>(inv, n), mr);
}

// ────────────────────────────────────────────────────────────────────
// squeeze
// ────────────────────────────────────────────────────────────────────
//
// MATLAB squeeze removes singleton dimensions but leaves scalars and
// 2D matrices alone (it never reduces below 2D). So:
//   1×3×4  → 3×4
//   3×1×4  → 3×4
//   3×4×1  → 3×4 (already effectively 2D)
//   3×4    → 3×4 (no-op)
//   1×3    → 1×3 (no-op, MATLAB convention)
//
// Since the underlying data is the same (column-major + page stride),
// squeeze is essentially reshape-with-new-dims. For 1×3×4 this means
// the data layout still matches: the singleton dim collapses cleanly
// because a stride of 1 doesn't introduce gaps.
Value squeeze(const Value &x, keel::Allocator *mr)
{
    const auto &dd = x.dims();
    keel::i32 nd = dd.ndim();
    while (nd > 2 && dd.dim(nd - 1) == 1)
        --nd;

    // 2D and below: shape preserved (MATLAB never collapses below 2D).
    if (nd <= 2)
        return reshape(x, dd.rows(), dd.cols(), 0, mr);

    // ND (≥ 3): drop every singleton dim, preserve the rest in order.
    // Pad to at least 2 dims with trailing 1s so a fully-singleton input
    // (1×1×1, 1×1×1×1, etc.) collapses to scalar shape (1×1) rather than
    // an invalid 0D shape.
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto kept = keel::Array<keel::usize>(scratch.allocator());
    kept.reserve(nd);
    for (keel::i32 i = 0; i < nd; ++i) {
        const keel::usize d = dd.dim(i);
        if (d != 1) kept.push_back(d);
    }
    while (kept.size() < 2) kept.push_back(1);

    return reshape(x, keel::ArrayRef<keel::usize>(kept.data(), kept.size()), mr);
}

// ────────────────────────────────────────────────────────────────────
// cat(dim, ...)
// ────────────────────────────────────────────────────────────────────
//
// dim=1 → vertcat (delegate); dim=2 → horzcat (delegate); dim=3 →
// stack 2D pages or extend 3D page count. Other dims throw.
namespace {

Value catDim3(const Value *values, keel::usize count, keel::Allocator *mr)
{
    if (count == 0) return Value();

    // First non-empty input fixes (R, C); empties are tolerated and
    // skipped (matches MATLAB).
    keel::usize R = 0, C = 0;
    bool anchored = false;
    keel::usize totalPages = 0;
    ValueType outType = ValueType::DOUBLE;
    for (keel::usize i = 0; i < count; ++i) {
        const auto &v = values[i];
        if (v.isEmpty() || v.numel() == 0) continue;
        const ValueType t = v.type();
        // STRING / STRUCT / FUNC_HANDLE along dim 3 deferred; same-type only
        // (mixed-type promotion not implemented), matching ND cat.
        if (t == ValueType::STRING || t == ValueType::STRUCT
            || t == ValueType::FUNC_HANDLE)
            throw Error(keel::format("cat: dim 3 does not support type '{}'", mtypeName(t)),
                         0, 0, "cat", "", "numkit:cat:typeDim3");
        const auto &dd = v.dims();
        if (!anchored) {
            R = dd.rows();
            C = dd.cols();
            outType = t;
            anchored = true;
        } else {
            if (dd.rows() != R || dd.cols() != C)
                throw Error("cat: dim 3 inputs must agree on rows and cols",
                             0, 0, "cat", "", "numkit:cat:badDims");
            if (t != outType)
                throw Error("cat: dim 3 requires all inputs to share a type",
                             0, 0, "cat", "", "numkit:cat:typeMismatchDim3");
        }
        totalPages += dd.is3D() ? dd.pages() : 1;
    }
    if (!anchored) return Value();

    // OBJECT arrays: dim-3 cat is an ordered page-stack, i.e. appending each
    // input's column-major states in turn → exactly objectConcatN.
    if (outType == ValueType::OBJECT)
        return Value::objectConcatN(values, count, Dims(R, C, totalPages), mr);

    // CELL permutes element-wise (Value copy); POD types copy raw bytes.
    // cat is a pure rearrangement, so it is type-preserving.
    const bool isCell = (outType == ValueType::CELL);
    Value r = isCell ? Value::cell3D(R, C, totalPages, mr)
                     : Value::matrix3d(R, C, totalPages, outType, mr);
    if (isCell) {
        keel::usize pageOff = 0;
        for (keel::usize i = 0; i < count; ++i) {
            const auto &v = values[i];
            if (v.isEmpty() || v.numel() == 0) continue;
            const keel::usize P = v.dims().is3D() ? v.dims().pages() : 1;
            const keel::usize base = pageOff * R * C, n = R * C * P;
            for (keel::usize e = 0; e < n; ++e) r.cellAt(base + e) = v.cellAt(e);
            pageOff += P;
        }
        return r;
    }
    const keel::usize es = elementSize(outType);
    char *dst = static_cast<char *>(r.rawDataMut());
    keel::usize pageOff = 0;
    for (keel::usize i = 0; i < count; ++i) {
        const auto &v = values[i];
        if (v.isEmpty() || v.numel() == 0) continue;
        const keel::usize P = v.dims().is3D() ? v.dims().pages() : 1;
        keel::mem_copy(dst + pageOff * R * C * es,
                    static_cast<const char *>(v.rawData()), R * C * P * es);
        pageOff += P;
    }
    return r;
}

// ND cat for dim >= 4. All non-cat axes must agree across inputs (treating
// ranks past an input's actual ndim as trailing 1s). Result rank =
// max(dim, max input ndim). All numeric types supported via byte-copy
// (elementSize-based). All inputs must share a type (no implicit
// promotion). CELL / STRUCT / STRING / FUNC_HANDLE rejected.
Value catND(keel::i32 dim, const Value *values, keel::usize count, keel::Allocator *mr)
{
    if (count == 0) return Value();
    const keel::i32 k = dim - 1;

    // Determine output rank: at least `dim`, plus any input may force higher.
    keel::i32 outNdim = dim;
    for (keel::usize i = 0; i < count; ++i) {
        const auto &v = values[i];
        if (v.isEmpty() || v.numel() == 0) continue;
        outNdim = std::max(outNdim, v.dims().ndim());
    }
    constexpr keel::i32 kMaxNd = Dims::kMaxRank;
    if (outNdim > kMaxNd)
        throw Error("cat: rank exceeds 32",
                     0, 0, "cat", "", "numkit:cat:tooManyDims");

    keel::usize outDim[kMaxNd];
    for (keel::i32 j = 0; j < outNdim; ++j) outDim[j] = 0;
    bool anchored = false;
    ValueType outType = ValueType::DOUBLE;

    for (keel::usize i = 0; i < count; ++i) {
        const auto &v = values[i];
        if (v.isEmpty() || v.numel() == 0) continue;
        const ValueType t = v.type();
        if (t == ValueType::CELL || t == ValueType::STRUCT || t == ValueType::STRING
            || t == ValueType::FUNC_HANDLE)
            throw Error(keel::format("cat: ND cat does not support type '{}'", mtypeName(t)),
                         0, 0, "cat", "", "numkit:cat:typeND");
        const auto &d = v.dims();
        if (!anchored) {
            for (keel::i32 j = 0; j < outNdim; ++j)
                outDim[j] = (j < d.ndim()) ? d.dim(j) : 1;
            outType = t;
            anchored = true;
        } else {
            if (t != outType)
                throw Error("cat: ND cat requires all inputs to share a type",
                             0, 0, "cat", "", "numkit:cat:typeMismatchND");
            for (keel::i32 j = 0; j < outNdim; ++j) {
                const keel::usize vd = (j < d.ndim()) ? d.dim(j) : 1;
                if (j == k) {
                    outDim[j] += vd;
                } else if (vd != outDim[j]) {
                    throw Error(keel::format("cat: dim {} inputs must agree on all axes except dim {}", dim, dim),
                                 0, 0, "cat", "", "numkit:cat:badDims");
                }
            }
        }
    }
    if (!anchored) return Value();

    // Inner block size B = prod(outDim[0..k-1]); outer count O = prod(outDim[k+1..]).
    keel::usize B = 1;
    for (keel::i32 j = 0; j < k; ++j) B *= outDim[j];
    keel::usize O = 1;
    for (keel::i32 j = k + 1; j < outNdim; ++j) O *= outDim[j];

    auto result = Value::matrixND(outDim, outNdim, outType, mr);
    if (B == 0 || O == 0 || outDim[k] == 0) return result;

    const keel::usize es = elementSize(outType);
    char *dst = static_cast<char *>(result.rawDataMut());
    const keel::usize resultOuterStride = outDim[k] * B;
    keel::usize accumK = 0;
    for (keel::usize i = 0; i < count; ++i) {
        const auto &v = values[i];
        if (v.isEmpty() || v.numel() == 0) continue;
        const auto &d = v.dims();
        const keel::usize inputDimK = (k < d.ndim()) ? d.dim(k) : 1;
        if (inputDimK == 0) continue;
        const char *src = static_cast<const char *>(v.rawData());
        const keel::usize inputOuterStride = inputDimK * B;
        const keel::usize blockBytes = inputDimK * B * es;
        for (keel::usize o = 0; o < O; ++o) {
            keel::mem_copy(dst + (o * resultOuterStride + accumK * B) * es,
                        src + o * inputOuterStride * es,
                        blockBytes);
        }
        accumK += inputDimK;
    }
    return result;
}

} // namespace

Value cat(keel::i32 dim, keel::ArrayRef<Value> values, keel::Allocator *mr)
{
    if (dim < 1)
        throw Error("cat: dim must be a positive integer",
                     0, 0, "cat", "", "numkit:cat:badDim");
    switch (dim) {
        case 1: return vertcat(values, mr);
        case 2: return horzcat(values, mr);
        case 3: return catDim3(values.data(), values.size(), mr);
        default: return catND(dim, values.data(), values.size(), mr);
    }
}

// ────────────────────────────────────────────────────────────────────
// blkdiag
// ────────────────────────────────────────────────────────────────────
//
// Block-diagonal matrix: diagonal blocks are the inputs (in order),
// off-diagonal regions are zero. 2D inputs only.
Value blkdiag(keel::ArrayRef<Value> values, keel::Allocator *mr)
{
    const keel::usize count = values.size();
    if (count == 0) return Value();

    // Type-preserving: anchor the output type on the first NON-EMPTY block
    // (empties contribute neither size on the diagonal mismatch nor a type
    // vote). CHAR / LOGICAL / SINGLE / int / COMPLEX preserved via raw-byte
    // copy into a zero-filled output. CELL / STRING / STRUCT rejected; mixed
    // input types deferred (MATLAB would promote — out of scope here).
    keel::usize totalRows = 0, totalCols = 0;
    ValueType t = ValueType::DOUBLE;
    bool typeSet = false;
    for (keel::usize i = 0; i < count; ++i) {
        if (values[i].dims().is3D())
            throw Error("blkdiag: 3D inputs are not supported",
                         0, 0, "blkdiag", "", "numkit:blkdiag:3D");
        totalRows += values[i].dims().rows();
        totalCols += values[i].dims().cols();
        if (values[i].numel() == 0) continue;
        const ValueType vt = values[i].type();
        if (vt == ValueType::CELL || vt == ValueType::STRING ||
            vt == ValueType::STRUCT || vt == ValueType::FUNC_HANDLE)
            throw Error("blkdiag: inputs must be numeric, char, or logical",
                         0, 0, "blkdiag", "", "numkit:blkdiag:badType");
        if (!typeSet) { t = vt; typeSet = true; }
        else if (vt != t)
            throw Error("blkdiag: mixed input types are not supported",
                         0, 0, "blkdiag", "", "numkit:blkdiag:mixedType");
    }

    auto r = Value::matrix(totalRows, totalCols, t, mr);
    const keel::usize es = elementSize(t);
    char *dst = static_cast<char *>(r.rawDataMut());
    // Zero-init (matrix() may return an uninitialised buffer in some builds);
    // the zero byte pattern is the canonical zero for every supported type.
    keel::mem_zero(dst, totalRows * totalCols * es);

    keel::usize rowOff = 0, colOff = 0;
    for (keel::usize k = 0; k < count; ++k) {
        const auto &v = values[k];
        const keel::usize R = v.dims().rows(), C = v.dims().cols();
        if (v.numel() > 0) {
            const char *src = static_cast<const char *>(v.rawData());
            for (keel::usize c = 0; c < C; ++c) {
                const keel::usize dstColStart = ((colOff + c) * totalRows + rowOff) * es;
                keel::mem_copy(dst + dstColStart, src + c * R * es, R * es);
            }
        }
        rowOff += R;
        colOff += C;
    }
    return r;
}

// ────────────────────────────────────────────────────────────────────
// shiftdim
// ────────────────────────────────────────────────────────────────────
//
// shiftdim(A, n) cyclically shifts the dim ordering. n > 0 promotes
// permute([n+1..N, 1..n]); n < 0 prepends |n| singleton dims via
// reshape; n == 0 is the identity. The auto form `[B, k] =
// shiftdim(A)` drops leading singletons and reports the count.
//
// Cyclic semantics: n is taken mod N for n > 0 (so shiftdim(A, N) is
// the identity, matching MATLAB).

Value shiftdim(const Value &x, keel::i32 n, keel::Allocator *mr)
{
    if (n == 0) return x;

    const auto &d = x.dims();
    const keel::i32 N = d.ndim();

    if (n > 0) {
        // Reduce n mod N — shifting by N is the identity.
        const keel::i32 eff = (N > 0) ? (n % N) : 0;
        if (eff == 0) return x;
        constexpr keel::i32 kMaxNd = Dims::kMaxRank;
        keel::i32 perm[kMaxNd];
        for (keel::i32 i = 0; i < N; ++i)
            perm[i] = ((i + eff) % N) + 1;  // 1-based
        return permute(x, keel::ArrayRef<keel::i32>(perm, keel::as<keel::usize>(N)), mr);
    }

    // n < 0: prepend |n| singleton dims via reshape.
    const keel::i32 k = -n;
    const keel::i32 newN = N + k;
    constexpr keel::i32 kMaxNd = Dims::kMaxRank;
    if (newN > kMaxNd)
        throw Error("shiftdim: rank exceeds 32",
                     0, 0, "shiftdim", "", "numkit:shiftdim:tooManyDims");
    keel::usize newDims[kMaxNd];
    for (keel::i32 i = 0; i < k; ++i) newDims[i] = 1;
    for (keel::i32 i = 0; i < N; ++i) newDims[k + i] = d.dim(i);
    return reshape(x, keel::ArrayRef<keel::usize>(newDims, keel::as<keel::usize>(newN)), mr);
}

ShiftDimAuto shiftdimAuto(const Value &x, keel::Allocator *mr)
{
    const auto &d = x.dims();
    const keel::i32 N = d.ndim();
    keel::i32 k = 0;
    while (k < N && d.dim(k) == 1) ++k;
    if (k == 0) return { x, 0 };
    // Avoid collapsing all dims to "scalar with rank 0" — keep at least
    // 1D worth of structure (matches MATLAB: shiftdim(ones(1,1,3))
    // returns a 1×3 row, not a 0-D scalar).
    if (k == N) k = N - 1;
    if (k <= 0) return { x, 0 };
    Value y = shiftdim(x, k, mr);
    // MATLAB-spec: the auto form ALSO strips trailing singletons (so
    // shiftdim(ones(1,1000,1000)) returns a 2-D 1000x1000, not a
    // 3-D 1000x1000x1). The explicit shiftdim(A, n) form does not.
    // See BUGS.md #20.
    const auto &yd = y.dims();
    const keel::i32 Ny = yd.ndim();
    keel::i32 newN = Ny;
    while (newN > 2 && yd.dim(newN - 1) == 1) --newN;
    if (newN < Ny) {
        constexpr keel::i32 kMaxNd = Dims::kMaxRank;
        keel::usize trimmed[kMaxNd];
        for (keel::i32 i = 0; i < newN; ++i) trimmed[i] = yd.dim(i);
        y = reshape(y, keel::ArrayRef<keel::usize>(trimmed, keel::as<keel::usize>(newN)), mr);
    }
    return { std::move(y), k };
}

} // namespace numkit::builtin
