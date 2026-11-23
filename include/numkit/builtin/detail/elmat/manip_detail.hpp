// toolboxes/.../manip_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by manip.cpp + manip_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/format.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)



namespace numkit::builtin {

namespace {

// CELL / STRING store their contents as a vector<Value> (cellData), not a
// contiguous byte buffer, so the byte-copy paths below would dereference a
// null rawData(). These helpers walk elements and copy Values directly,
// mirroring the repmat cell/string handling above. MATLAB's flip family
// (flip/fliplr/flipud/rot90) is type-agnostic — it just permutes elements
// — so cell and string arrays must reorder exactly like numeric ones.

inline bool isCellOrString(ValueType t)
{
    return t == ValueType::CELL || t == ValueType::STRING;
}

// Allocate a CELL or STRING result of shape rows×cols.
inline Value makeCellOrString(ValueType t, keel::usize rows, keel::usize cols,
                              keel::Allocator *mr)
{
    return (t == ValueType::STRING) ? Value::stringArray(rows, cols, mr)
                                    : Value::cell(rows, cols, mr);
}

// Copy element at column-major index srcIdx of src into dstIdx of dst.
inline void copyCellElem(Value &dst, keel::usize dstIdx, const Value &src, keel::usize srcIdx)
{
    if (src.type() == ValueType::STRING)
        dst.stringElemSet(dstIdx, src.stringElem(srcIdx));
    else
        dst.cellAt(dstIdx) = src.cellAt(srcIdx);
}

// Reverse a CELL/STRING array along `axis` (0=rows, 1=cols). Supports up to
// 2-D (the only ranks Value::cell can represent); higher-rank cell/string
// flips are rare and deferred with a clear error.
Value flipCellStr(const Value &x, keel::i32 axis, const char *fn, keel::Allocator *mr)
{
    const auto &d = x.dims();
    if (d.ndim() > 2)
        throw Error(keel::format("{}: cell/string arrays support up to 2-D", fn),
                     0, 0, fn, "", keel::format("numkit:{}:cellRank", fn));
    const keel::usize R = d.rows(), C = d.cols();
    Value r = makeCellOrString(x.type(), R, C, mr);
    if (x.numel() == 0) return r;
    const keel::usize flipDim = (axis == 0) ? R : (axis == 1) ? C : 1;
    if (axis > 1 || flipDim <= 1) {                  // identity copy
        for (keel::usize i = 0; i < x.numel(); ++i) copyCellElem(r, i, x, i);
        return r;
    }
    for (keel::usize c = 0; c < C; ++c)
        for (keel::usize rr = 0; rr < R; ++rr) {
            const keel::usize outIdx = c * R + rr;
            const keel::usize srcIdx = (axis == 0) ? (c * R + (R - 1 - rr))
                                              : ((C - 1 - c) * R + rr);
            copyCellElem(r, outIdx, x, srcIdx);
        }
    return r;
}

// ND flip helper: reverses the order of slabs along `axis` (0-based).
// The slab stride is B = prod(dims[0..axis-1]) elements; outer count
// O = prod(dims[axis+1..N-1]). Used by the ND fallback for fliplr
// (axis=1) and flipud (axis=0). Type-preserving via byte-copy.
[[maybe_unused]] Value flipNDAlongAxis(const Value &x, keel::i32 axis, const char *fn, keel::Allocator *mr)
{
    const ValueType t = x.type();
    if (isCellOrString(t)) return flipCellStr(x, axis, fn, mr);
    if (x.isObject()) {
        // Reverse the `axis` dimension via an index map → objectGather.
        const auto &d = x.dims();
        const keel::i32 nd = d.ndim();
        const keel::usize flipDim = (axis < nd) ? d.dim(axis) : 1;
        const keel::usize total = d.numel();
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<keel::usize> map(total, scratch.allocator());
        if (flipDim <= 1) {
            for (keel::usize i = 0; i < total; ++i) map[i] = i;
        } else {
            keel::usize B = 1;
            for (keel::i32 i = 0; i < axis; ++i) B *= d.dim(i);
            keel::usize O = 1;
            for (keel::i32 i = axis + 1; i < nd; ++i) O *= d.dim(i);
            for (keel::usize o = 0; o < O; ++o) {
                const keel::usize outerBase = o * flipDim * B;
                for (keel::usize i = 0; i < flipDim; ++i)
                    for (keel::usize b = 0; b < B; ++b)
                        map[outerBase + i * B + b] = outerBase + (flipDim - 1 - i) * B + b;
            }
        }
        return x.objectGather(map.data(), d, mr);
    }
    if (t == ValueType::STRUCT || t == ValueType::FUNC_HANDLE)
        throw Error(keel::format("{}: ND fallback does not support type '{}'", fn, mtypeName(t)),
                     0, 0, fn, "", keel::format("numkit:{}:typeND", fn));
    const auto &d = x.dims();
    const keel::i32 nd = d.ndim();
    constexpr keel::i32 kMaxNd = Dims::kMaxRank;
    if (nd > kMaxNd)
        throw Error(keel::format("{}: rank exceeds 32", fn),
                     0, 0, fn, "", keel::format("numkit:{}:tooManyDims", fn));

    keel::usize outDimArr[kMaxNd];
    for (keel::i32 i = 0; i < nd; ++i) outDimArr[i] = d.dim(i);
    auto r = Value::matrixND(outDimArr, nd, t, mr);
    if (x.numel() == 0) return r;

    const keel::usize es = elementSize(t);
    const char *src = static_cast<const char *>(x.rawData());
    char *dst = static_cast<char *>(r.rawDataMut());

    // axis past the actual rank, or singleton dim → identity copy.
    const keel::usize flipDim = (axis < nd) ? d.dim(axis) : 1;
    if (flipDim <= 1) {
        keel::mem_copy(dst, src, x.numel() * es);
        return r;
    }

    keel::usize B = 1;
    for (keel::i32 i = 0; i < axis; ++i) B *= d.dim(i);
    keel::usize O = 1;
    for (keel::i32 i = axis + 1; i < nd; ++i) O *= d.dim(i);

    for (keel::usize o = 0; o < O; ++o) {
        const keel::usize outerBase = o * flipDim * B;
        for (keel::usize i = 0; i < flipDim; ++i) {
            keel::mem_copy(dst + (outerBase + i * B) * es,
                        src + (outerBase + (flipDim - 1 - i) * B) * es,
                        B * es);
        }
    }
    return r;
}

} // namespace
namespace {

// Per-page rotation kernels. Each takes a single page (R*C contiguous
// elements, column-major) of the input and writes the rotated page to
// the output buffer. Output strides depend on the rotation: rot90/270
// swap (R, C); rot180 keeps (R, C). 3D dispatch in rot90() loops these
// over all pages.

inline void rot90OncePage(const keel::f64 *src, keel::f64 *dst, keel::usize R, keel::usize C)
{
    // Output shape (C, R): out[(C-1-c)*1 + r*C] = src[c*R + r]
    for (keel::usize c = 0; c < C; ++c)
        for (keel::usize rr = 0; rr < R; ++rr)
            dst[rr * C + (C - 1 - c)] = src[c * R + rr];
}

inline void rot180Page(const keel::f64 *src, keel::f64 *dst, keel::usize R, keel::usize C)
{
    for (keel::usize c = 0; c < C; ++c)
        for (keel::usize rr = 0; rr < R; ++rr)
            dst[(C - 1 - c) * R + (R - 1 - rr)] = src[c * R + rr];
}

inline void rot270Page(const keel::f64 *src, keel::f64 *dst, keel::usize R, keel::usize C)
{
    // Output shape (C, R): out[c + (R-1-r)*C] = src[c*R + r]
    for (keel::usize c = 0; c < C; ++c)
        for (keel::usize rr = 0; rr < R; ++rr)
            dst[(R - 1 - rr) * C + c] = src[c * R + rr];
}

// Type-agnostic byte-copy variants for ND fallback. Same index math as
// the DOUBLE kernels above, but each cell is es bytes via memcpy.
inline void rot90OncePageBytes(const char *src, char *dst,
                               keel::usize R, keel::usize C, keel::usize es)
{
    for (keel::usize c = 0; c < C; ++c)
        for (keel::usize rr = 0; rr < R; ++rr)
            keel::mem_copy(dst + (rr * C + (C - 1 - c)) * es,
                        src + (c * R + rr) * es, es);
}

inline void rot180PageBytes(const char *src, char *dst,
                            keel::usize R, keel::usize C, keel::usize es)
{
    for (keel::usize c = 0; c < C; ++c)
        for (keel::usize rr = 0; rr < R; ++rr)
            keel::mem_copy(dst + ((C - 1 - c) * R + (R - 1 - rr)) * es,
                        src + (c * R + rr) * es, es);
}

inline void rot270PageBytes(const char *src, char *dst,
                            keel::usize R, keel::usize C, keel::usize es)
{
    for (keel::usize c = 0; c < C; ++c)
        for (keel::usize rr = 0; rr < R; ++rr)
            keel::mem_copy(dst + ((R - 1 - rr) * C + c) * es,
                        src + (c * R + rr) * es, es);
}

// rot90 for CELL / STRING arrays (element-wise Value copy). kMod selects
// the rotation (0/1/2/3 → 0/90/180/270° CCW). Supports up to 2-D.
[[maybe_unused]] Value rot90CellStr(const Value &x, keel::i32 kMod, keel::Allocator *mr)
{
    const auto &d = x.dims();
    if (d.ndim() > 2)
        throw Error("rot90: cell/string arrays support up to 2-D",
                     0, 0, "rot90", "", "numkit:rot90:cellRank");
    const keel::usize R = d.rows(), C = d.cols();
    const ValueType t = x.type();
    if (kMod == 0 || kMod == 2) {                    // shape stays R×C
        Value r = makeCellOrString(t, R, C, mr);
        if (x.numel() == 0) return r;
        for (keel::usize c = 0; c < C; ++c)
            for (keel::usize rr = 0; rr < R; ++rr) {
                const keel::usize outIdx = c * R + rr;
                const keel::usize srcIdx = (kMod == 0)
                    ? (c * R + rr)
                    : ((C - 1 - c) * R + (R - 1 - rr));
                copyCellElem(r, outIdx, x, srcIdx);
            }
        return r;
    }
    // kMod 1 (90°) / 3 (270°): output shape is C×R (rows/cols swap).
    Value r = makeCellOrString(t, C, R, mr);
    if (x.numel() == 0) return r;
    for (keel::usize c = 0; c < C; ++c)
        for (keel::usize rr = 0; rr < R; ++rr) {
            const keel::usize srcIdx = c * R + rr;            // src(rr, c)
            const keel::usize outIdx = (kMod == 1)
                ? ((C - 1 - c) + rr * C)                 // out(C-1-c, rr)
                : (c + (R - 1 - rr) * C);                // out(c, R-1-rr)
            copyCellElem(r, outIdx, x, srcIdx);
        }
    return r;
}

} // namespace
namespace {

inline keel::usize wrap(keel::i64 k, keel::usize n)
{
    if (n == 0) return 0;
    keel::i64 m = k % keel::as<keel::i64>(n);
    if (m < 0) m += keel::as<keel::i64>(n);
    return keel::as<keel::usize>(m);
}

[[maybe_unused]] void shift2D(const keel::f64 *src, keel::f64 *dst, keel::usize R, keel::usize C,
             keel::i64 kRow, keel::i64 kCol)
{
    const keel::usize shR = wrap(kRow, R);
    const keel::usize shC = wrap(kCol, C);
    for (keel::usize c = 0; c < C; ++c) {
        const keel::usize srcC = (c + C - shC) % C;
        for (keel::usize rr = 0; rr < R; ++rr) {
            const keel::usize srcR = (rr + R - shR) % R;
            dst[c * R + rr] = src[srcC * R + srcR];
        }
    }
}

} // namespace
namespace {

// Type-agnostic per-page tril/triu via byte-copy + memset(0). All numeric
// types (DOUBLE, SINGLE, integer, LOGICAL, COMPLEX) zero correctly via
// memset since their zero element is all-zero bits.
inline void trilPageBytes(const char *src, char *dst, keel::usize R, keel::usize C,
                          keel::i32 k, keel::usize es)
{
    for (keel::usize c = 0; c < C; ++c)
        for (keel::usize rr = 0; rr < R; ++rr) {
            const keel::i32 diff = keel::as<keel::i32>(c) - keel::as<keel::i32>(rr);
            const keel::usize off = (c * R + rr) * es;
            if (diff <= k) keel::mem_copy(dst + off, src + off, es);
            else           keel::mem_zero(dst + off, es);
        }
}

inline void triuPageBytes(const char *src, char *dst, keel::usize R, keel::usize C,
                          keel::i32 k, keel::usize es)
{
    for (keel::usize c = 0; c < C; ++c)
        for (keel::usize rr = 0; rr < R; ++rr) {
            const keel::i32 diff = keel::as<keel::i32>(c) - keel::as<keel::i32>(rr);
            const keel::usize off = (c * R + rr) * es;
            if (diff >= k) keel::mem_copy(dst + off, src + off, es);
            else           keel::mem_zero(dst + off, es);
        }
}

} // namespace
namespace {

template <typename PageBytesFn>
Value trilTriuND(const Value &x, keel::i32 k, PageBytesFn pageFn, const char *fn, keel::Allocator *mr)
{
    const auto &dd = x.dims();
    constexpr keel::i32 kMaxNd = Dims::kMaxRank;
    const keel::i32 nd = dd.ndim();
    if (nd > kMaxNd)
        throw Error(keel::format("{}: rank exceeds 32", fn),
                     0, 0, fn, "", keel::format("numkit:{}:tooManyDims", fn));
    const ValueType t = x.type();
    if (t == ValueType::CELL || t == ValueType::STRUCT || t == ValueType::STRING
        || t == ValueType::FUNC_HANDLE)
        throw Error(keel::format("{}: ND fallback does not support type '{}'", fn, mtypeName(t)),
                     0, 0, fn, "", keel::format("numkit:{}:typeND", fn));

    const keel::usize R = dd.rows(), C = dd.cols();
    keel::usize outDimArr[kMaxNd];
    for (keel::i32 i = 0; i < nd; ++i) outDimArr[i] = dd.dim(i);
    auto r = Value::matrixND(outDimArr, nd, t, mr);
    if (x.numel() == 0) return r;

    const keel::usize es = elementSize(t);
    const char *src = static_cast<const char *>(x.rawData());
    char *dst = static_cast<char *>(r.rawDataMut());
    const keel::usize pageBytes = R * C * es;
    forEachOuterPage(dd, [&](keel::usize pp, const keel::usize *) {
        pageFn(src + pp * pageBytes, dst + pp * pageBytes, R, C, k, es);
    });
    return r;
}

} // namespace
namespace {

// Build the output→source index map for one dimension given a replication
// count spec. `counts` is a scalar (broadcast to all `dimLen` entries) or a
// DOUBLE vector of length `dimLen`. map[k] = source index for output slot k.
[[maybe_unused]] keel::Array<keel::usize> repelemMap(const Value &counts, keel::usize dimLen, const keel::ArenaTemp &arena)
{
    keel::Array<keel::usize> map(arena.allocator());
    auto checkCount = [](keel::f64 c) -> keel::usize {
        if (!(c >= 0.0) || keel::floor(c) != c)
            throw Error("repelem: replication counts must be nonnegative integers",
                         0, 0, "repelem", "", "numkit:repelem:badCount");
        return keel::as<keel::usize>(c);
    };
    if (counts.isScalar()) {
        const keel::usize rep = checkCount(counts.toScalar());
        map.reserve(dimLen * rep);
        for (keel::usize i = 0; i < dimLen; ++i)
            for (keel::usize k = 0; k < rep; ++k) map.push_back(i);
        return map;
    }
    if (counts.type() != ValueType::DOUBLE)
        throw Error("repelem: count vector must be DOUBLE",
                     0, 0, "repelem", "", "numkit:repelem:type");
    if (counts.numel() != dimLen)
        throw Error("repelem: count vector length must equal the dimension size",
                     0, 0, "repelem", "", "numkit:repelem:countLen");
    const keel::f64 *cd = counts.doubleData();
    for (keel::usize i = 0; i < dimLen; ++i) {
        const keel::usize rep = checkCount(cd[i]);
        for (keel::usize k = 0; k < rep; ++k) map.push_back(i);
    }
    return map;
}

} // namespace
namespace {
inline bool isVectorLike(const Value &v)
{
    const auto &d = v.dims();
    // Accept scalars, 1×N / N×1 vectors, and the genuinely-empty 0×0
    // (so paddata([], n) doesn't throw — it just expands the empty
    // store with zeros).
    return d.ndim() <= 2 && (d.rows() <= 1 || d.cols() <= 1 || v.isScalar());
}
} // anon

} // namespace numkit::builtin
