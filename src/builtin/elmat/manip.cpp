// toolboxes/builtin/src/lang/arrays/manip.cpp
//
// Phase-5 array manipulation kernels. All input/output is column-major
// double; 3D layout stored as page-stride R*C. Most ops are pure data
// movement (memcpy where possible) — no SIMD opportunity beyond what
// the compiler auto-vectorises in the inner copy loops.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <numkit/builtin/elmat.hpp>
#include <numkit/builtin/ops.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/shape_ops.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "manip_detail.hpp"

namespace numkit::builtin {

// ────────────────────────────────────────────────────────────────────
// repmat
// ────────────────────────────────────────────────────────────────────
//
// Tile the input matrix m × n times (and optionally p times along
// pages for the 3D form). Output dims: (R*m) × (C*n) × (P*p).
Value repmat(const Value &x, keel::usize m, keel::usize n, keel::usize p, keel::Allocator *mr)
{
    // STRING / CELL / non-DOUBLE types delegate to the ND path which
    // handles them. The fast 2-D DOUBLE memcpy loop below assumes
    // contiguous double storage. See BUGS.md #6.
    const ValueType t = x.type();
    if (t != ValueType::DOUBLE) {
        const keel::usize tiles[3] = { m, n, p };
        const keel::usize nd = (p > 1 || x.dims().is3D()) ? 3 : 2;
        return repmatND(x, keel::ArrayRef<keel::usize>(tiles, nd), mr);
    }
    const auto &dd = x.dims();
    const keel::usize R = dd.rows(), C = dd.cols();
    const keel::usize P = dd.is3D() ? dd.pages() : 1;
    const keel::usize outR = R * m, outC = C * n, outP = P * p;
    const bool out3D = (outP > 1) || dd.is3D();

    auto r = out3D ? Value::matrix3d(outR, outC, outP, ValueType::DOUBLE, mr)
                   : Value::matrix(outR, outC, ValueType::DOUBLE, mr);
    if (R == 0 || C == 0 || P == 0 || m == 0 || n == 0 || p == 0)
        return r;

    const double *src = x.doubleData();
    double *dst = r.doubleDataMut();

    // Phase P6: build the first output page once, then replicate via
    // page memcpy for the remaining p-1 page tiles. The within-page
    // copy is the same m*n*C tile-of-memcpy structure as before — the
    // total bytes written floor at outR*outC*outP, no algorithmic
    // shortcut on the per-page cost. The win here is ONLY for p > 1
    // (page replication) where one big memcpy per page beats redoing
    // the per-column tile loop.
    for (keel::usize srcPage = 0; srcPage < P; ++srcPage) {
        double *firstPage = dst + (0 * P + srcPage) * outR * outC;
        for (keel::usize cTile = 0; cTile < n; ++cTile) {
            for (keel::usize c = 0; c < C; ++c) {
                const double *colSrc = src + srcPage * R * C + c * R;
                for (keel::usize rTile = 0; rTile < m; ++rTile) {
                    double *colDst = firstPage + (cTile * C + c) * outR
                                                 + rTile * R;
                    std::memcpy(colDst, colSrc, R * sizeof(double));
                }
            }
        }
        for (keel::usize pTile = 1; pTile < p; ++pTile) {
            double *targetPage = dst + (pTile * P + srcPage) * outR * outC;
            std::memcpy(targetPage, firstPage, outR * outC * sizeof(double));
        }
    }
    return r;
}

// ND repmat — tile vector of arbitrary length. Outer-coord walk maps
// each output column-of-axis-0 back to its source column via per-axis
// modulo, then memcpys axis-0-bytes tilesPadded[0] times to fill the
// output column. Type-preserving via byte-copy (elementSize-based).
Value repmatND(const Value &x, keel::ArrayRef<keel::usize> tiles, keel::Allocator *mr)
{
    const ValueType t = x.type();
    if (t == ValueType::FUNC_HANDLE)
        throw Error(std::string("repmat: ND repmat does not support type '")
                     + mtypeName(t) + "'",
                     0, 0, "repmat", "", "numkit:repmat:typeND");

    const auto &inDims = x.dims();
    constexpr int kMaxNd = Dims::kMaxRank;
    const int ntiles = static_cast<int>(tiles.size());
    int outNdim = std::max(inDims.ndim(), ntiles);
    if (outNdim > kMaxNd)
        throw Error("repmat: rank exceeds 32",
                     0, 0, "repmat", "", "numkit:repmat:tooManyDims");
    if (outNdim < 1) outNdim = 1;

    // OBJECT arrays store per-element state — build the tile→source index
    // map (per-axis modulo, same as the CELL path) and gather.
    if (x.isObject()) {
        keel::usize inDP[kMaxNd], tilesP[kMaxNd], outD[kMaxNd];
        for (int i = 0; i < outNdim; ++i) {
            inDP[i] = (i < inDims.ndim()) ? inDims.dim(i) : 1;
            tilesP[i] = (i < ntiles) ? tiles[i] : 1;
            outD[i] = inDP[i] * tilesP[i];
        }
        Dims outDims(outD, outNdim);
        const keel::usize total = outDims.numel();
        std::vector<keel::usize> srcMap(total, 0);
        if (x.numel() > 0 && total > 0) {
            keel::usize outStrides[kMaxNd], srcStrides[kMaxNd];
            computeStridesColMajor(outDims, outStrides);
            computeStridesColMajor(inDims, srcStrides);
            for (int i = inDims.ndim(); i < outNdim; ++i) srcStrides[i] = 0;
            for (keel::usize outIdx = 0; outIdx < total; ++outIdx) {
                keel::usize rem = outIdx, srcIdx = 0;
                for (int d = outNdim - 1; d >= 0; --d) {
                    const keel::usize coord = rem / outStrides[d];
                    rem -= coord * outStrides[d];
                    if (d < inDims.ndim())
                        srcIdx += (coord % inDP[d]) * srcStrides[d];
                }
                srcMap[outIdx] = srcIdx;
            }
        }
        return x.objectGather(srcMap.data(), outDims, mr);
    }

    // STRING / CELL / STRUCT store contents as non-POD containers, not a
    // contiguous byte buffer — the memcpy paths below would dereference
    // a null rawData(). Walk by element and copy Values directly.
    // See BUGS.md #6 and #7-related cell-array handling.
    if (t == ValueType::STRING || t == ValueType::CELL || t == ValueType::STRUCT) {
        keel::usize inDP[kMaxNd];
        keel::usize tilesP[kMaxNd];
        keel::usize outD[kMaxNd];
        for (int i = 0; i < outNdim; ++i) {
            inDP[i]   = (i < inDims.ndim()) ? inDims.dim(i) : 1;
            tilesP[i] = (i < ntiles)        ? tiles[i]      : 1;
            outD[i]   = inDP[i] * tilesP[i];
        }
        int effNdim = outNdim;
        while (effNdim > 2 && outD[effNdim - 1] == 1) --effNdim;
        // Build output of the same value-type (STRING, CELL, or STRUCT) with
        // the new shape.
        Value r;
        if (t == ValueType::STRING) {
            if (effNdim <= 2)
                r = Value::stringArray(outD[0], effNdim >= 2 ? outD[1] : 1, mr);
            else if (effNdim == 3)
                r = Value::stringArray3D(outD[0], outD[1], outD[2], mr);
            else
                throw Error("repmat: ND > 3 not supported for string",
                             0, 0, "repmat", "", "numkit:repmat:tooManyDimsStr");
        } else if (t == ValueType::CELL) {
            if (effNdim <= 2)
                r = Value::cell(outD[0], effNdim >= 2 ? outD[1] : 1, mr);
            else
                throw Error("repmat: ND > 2 not supported for cell",
                             0, 0, "repmat", "", "numkit:repmat:tooManyDimsCell");
        } else { // STRUCT
            if (effNdim <= 2) {
                r = Value::structArray(outD[0], effNdim >= 2 ? outD[1] : 1, mr);
            } else {
                r = Value::structArray(0, 0, mr);
                keel::usize maxCoords[kMaxNd];
                bool anyZero = false;
                for (int i = 0; i < effNdim; ++i) {
                    if (outD[i] == 0) { anyZero = true; break; }
                    maxCoords[i] = outD[i] - 1;
                }
                if (!anyZero)
                    r.growStructArrayND(maxCoords, effNdim, mr);
            }
        }
        if (x.numel() == 0 || r.numel() == 0) return r;

        keel::usize outStrides[kMaxNd];
        computeStridesColMajor(r.dims(), outStrides);
        keel::usize srcStrides[kMaxNd];
        computeStridesColMajor(inDims, srcStrides);
        for (int i = inDims.ndim(); i < outNdim; ++i) srcStrides[i] = 0;

        const keel::Array<keel::String> structFieldsOrder =
            (t == ValueType::STRUCT) ? x.fieldNamesInOrder() : keel::Array<keel::String>(mr);

        const keel::usize total = r.numel();
        for (keel::usize outIdx = 0; outIdx < total; ++outIdx) {
            // Recover output coord, map back to source coord by per-axis modulo.
            keel::usize rem = outIdx;
            keel::usize srcIdx = 0;
            for (int d = outNdim - 1; d >= 0; --d) {
                const keel::usize coord = rem / outStrides[d];
                rem -= coord * outStrides[d];
                if (d < inDims.ndim())
                    srcIdx += (coord % inDP[d]) * srcStrides[d];
            }
            if (t == ValueType::STRING) {
                r.stringElemSet(outIdx, x.stringElem(srcIdx));
            } else if (t == ValueType::CELL) {
                r.cellAt(outIdx) = x.cellAt(srcIdx);
            } else {
                const auto &srcFields = x.isStructArray() ? x.structArrayElem(srcIdx) : x.structFields();
                for (const auto &fname : structFieldsOrder) {
                    const Value *val = srcFields.get(fname);
                    if (val)
                        r.setField(outIdx, fname, *val);
                }
            }
        }
        return r;
    }

    keel::usize inDimPadded[kMaxNd];
    keel::usize tilesPadded[kMaxNd];
    keel::usize outDim[kMaxNd];
    for (int i = 0; i < outNdim; ++i) {
        inDimPadded[i] = (i < inDims.ndim()) ? inDims.dim(i) : 1;
        tilesPadded[i] = (i < ntiles) ? tiles[i] : 1;
        outDim[i] = inDimPadded[i] * tilesPadded[i];
    }

    auto r = Value::matrixND(outDim, outNdim, t, mr);
    if (r.numel() == 0 || x.numel() == 0) return r;

    const keel::usize es = elementSize(t);
    const char *src = static_cast<const char *>(x.rawData());
    char *dst = static_cast<char *>(r.rawDataMut());

    // 1D special case: input is a row/col, output is `tilesPadded[0]` copies.
    if (outNdim == 1) {
        for (keel::usize k = 0; k < tilesPadded[0]; ++k)
            std::memcpy(dst + k * inDimPadded[0] * es, src,
                        inDimPadded[0] * es);
        return r;
    }

    keel::usize outStrides[kMaxNd];
    computeStridesColMajor(r.dims(), outStrides);
    keel::usize srcStrides[kMaxNd];
    computeStridesColMajor(inDims, srcStrides);
    for (int i = inDims.ndim(); i < outNdim; ++i) srcStrides[i] = 0;

    keel::usize outerDimsArr[kMaxNd];
    for (int i = 1; i < outNdim; ++i) outerDimsArr[i - 1] = outDim[i];
    Dims outerIter(outerDimsArr, outNdim - 1);

    keel::usize outerCoords[kMaxNd] = {0};
    do {
        keel::usize srcOff = 0;
        keel::usize dstOff = 0;
        for (int i = 1; i < outNdim; ++i) {
            const keel::usize oc = outerCoords[i - 1];
            const keel::usize inDimI = inDimPadded[i];
            srcOff += (oc % inDimI) * srcStrides[i];
            dstOff += oc * outStrides[i];
        }
        for (keel::usize k = 0; k < tilesPadded[0]; ++k)
            std::memcpy(dst + (dstOff + k * inDimPadded[0]) * es,
                        src + srcOff * es,
                        inDimPadded[0] * es);
    } while (incrementCoords(outerCoords, outerIter));

    return r;
}

// ────────────────────────────────────────────────────────────────────
// fliplr / flipud
// ────────────────────────────────────────────────────────────────────


Value fliplr(const Value &x, keel::Allocator *mr)
{
    const auto &dd = x.dims();
    if (isCellOrString(x.type())) return flipCellStr(x, 1, "fliplr", mr);
    if (x.isObject()) return flipNDAlongAxis(x, 1, "fliplr", mr); // flip dim 2
    if (dd.ndim() >= 4) return flipNDAlongAxis(x, 1, "fliplr", mr);

    // POD types (DOUBLE/CHAR/LOGICAL/COMPLEX/single/int) copy raw bytes — flip
    // is a pure rearrangement, so it is type-preserving (cell/string handled
    // above via flipCellStr).
    const ValueType t = x.type();
    const keel::usize R = dd.rows(), C = dd.cols();
    const keel::usize P = dd.is3D() ? dd.pages() : 1;
    auto r = dd.is3D() ? Value::matrix3d(R, C, P, t, mr)
                       : Value::matrix(R, C, t, mr);
    if (x.numel() == 0) return r;

    const keel::usize es = elementSize(t);
    const char *src = static_cast<const char *>(x.rawData());
    char *dst = static_cast<char *>(r.rawDataMut());
    for (keel::usize pp = 0; pp < P; ++pp)
        for (keel::usize c = 0; c < C; ++c) {
            const char *colSrc = src + (pp * R * C + (C - 1 - c) * R) * es;
            char *colDst = dst + (pp * R * C + c * R) * es;
            std::memcpy(colDst, colSrc, R * es);
        }
    return r;
}

Value flipud(const Value &x, keel::Allocator *mr)
{
    const auto &dd = x.dims();
    if (isCellOrString(x.type())) return flipCellStr(x, 0, "flipud", mr);
    if (x.isObject()) return flipNDAlongAxis(x, 0, "flipud", mr); // flip dim 1
    if (dd.ndim() >= 4) return flipNDAlongAxis(x, 0, "flipud", mr);

    // POD types copy raw bytes — type-preserving (cell/string handled above).
    const ValueType t = x.type();
    const keel::usize R = dd.rows(), C = dd.cols();
    const keel::usize P = dd.is3D() ? dd.pages() : 1;
    auto r = dd.is3D() ? Value::matrix3d(R, C, P, t, mr)
                       : Value::matrix(R, C, t, mr);
    if (x.numel() == 0) return r;

    const keel::usize es = elementSize(t);
    const char *src = static_cast<const char *>(x.rawData());
    char *dst = static_cast<char *>(r.rawDataMut());
    for (keel::usize pp = 0; pp < P; ++pp)
        for (keel::usize c = 0; c < C; ++c) {
            const char *colSrc = src + (pp * R * C + c * R) * es;
            char *colDst = dst + (pp * R * C + c * R) * es;
            for (keel::usize rr = 0; rr < R; ++rr)
                std::memcpy(colDst + rr * es, colSrc + (R - 1 - rr) * es, es);
        }
    return r;
}

// ────────────────────────────────────────────────────────────────────
// rot90
// ────────────────────────────────────────────────────────────────────
//
// rot90(A) rotates 90° counter-clockwise. For 2D matrix R×C, output
// is C×R with element (rNew, cNew) = A(cNew, C-1-rNew). The k-arg
// generalisation: k mod 4 selects 0/90/180/270° rotation. Negative k
// is clockwise.

Value rot90(const Value &x, int k, keel::Allocator *mr)
{
    int kMod = k % 4;
    if (kMod < 0) kMod += 4;

    if (isCellOrString(x.type())) return rot90CellStr(x, kMod, mr);

    const auto &dd = x.dims();
    const int nd = dd.ndim();
    const keel::usize R = dd.rows(), C = dd.cols();

    // OBJECT (2-D): rotate via the same index map as rot90CellStr (each
    // assignment is result[outIdx] = src[srcIdx]) → objectGather.
    if (x.isObject() && nd < 4) {
        std::vector<keel::usize> map(R * C);
        Dims outD = (kMod == 1 || kMod == 3) ? Dims(C, R) : Dims(R, C);
        for (keel::usize c = 0; c < C; ++c)
            for (keel::usize rr = 0; rr < R; ++rr) {
                if (kMod == 0)
                    map[c * R + rr] = c * R + rr;
                else if (kMod == 2)
                    map[c * R + rr] = (C - 1 - c) * R + (R - 1 - rr);
                else if (kMod == 1)
                    map[(C - 1 - c) + rr * C] = c * R + rr;
                else
                    map[c + (R - 1 - rr) * C] = c * R + rr;
            }
        return x.objectGather(map.data(), outD, mr);
    }

    // ND fallback (rank ≥ 4): rotate every (R×C) slice indexed by axes
    // 2..N-1. Output rank matches input; axes 0 and 1 swap for kMod 1/3.
    // Type-agnostic via byte-copy.
    if (nd >= 4) {
        const ValueType t = x.type();
        if (t == ValueType::CELL || t == ValueType::STRUCT || t == ValueType::STRING
            || t == ValueType::FUNC_HANDLE)
            throw Error(std::string("rot90: ND fallback does not support type '")
                         + mtypeName(t) + "'",
                         0, 0, "rot90", "", "numkit:rot90:typeND");
        constexpr int kMaxNd = Dims::kMaxRank;
        if (nd > kMaxNd)
            throw Error("rot90: rank exceeds 32",
                         0, 0, "rot90", "", "numkit:rot90:tooManyDims");
        keel::usize outDims[kMaxNd];
        outDims[0] = (kMod == 1 || kMod == 3) ? C : R;
        outDims[1] = (kMod == 1 || kMod == 3) ? R : C;
        for (int i = 2; i < nd; ++i) outDims[i] = dd.dim(i);
        auto r = Value::matrixND(outDims, nd, t, mr);
        if (x.numel() == 0) return r;
        const keel::usize es = elementSize(t);
        const char *src = static_cast<const char *>(x.rawData());
        char *dst = static_cast<char *>(r.rawDataMut());
        const keel::usize pageElems = R * C;
        if (kMod == 0) {
            std::memcpy(dst, src, x.numel() * es);
        } else {
            auto pageFn = (kMod == 1) ? rot90OncePageBytes
                        : (kMod == 2) ? rot180PageBytes
                                      : rot270PageBytes;
            forEachOuterPage(dd, [&](keel::usize pp, const keel::usize *) {
                pageFn(src + pp * pageElems * es,
                       dst + pp * pageElems * es, R, C, es);
            });
        }
        return r;
    }

    const keel::usize P = dd.is3D() ? dd.pages() : 1;
    const bool is3D = dd.is3D();

    // Non-DOUBLE 2-D/3-D POD types (char/logical/complex/single/int): rot90 is
    // a pure rearrangement, so reuse the byte-copy kernels (cell/string were
    // handled above by rot90CellStr). Output preserves x's type.
    if (x.type() != ValueType::DOUBLE) {
        const ValueType t = x.type();
        const keel::usize es = elementSize(t);
        const char *src = static_cast<const char *>(x.rawData());
        if (kMod == 0) {
            auto r = is3D ? Value::matrix3d(R, C, P, t, mr) : Value::matrix(R, C, t, mr);
            if (x.numel() > 0) std::memcpy(r.rawDataMut(), src, x.numel() * es);
            return r;
        }
        if (kMod == 2) {
            auto r = is3D ? Value::matrix3d(R, C, P, t, mr) : Value::matrix(R, C, t, mr);
            if (x.numel() == 0) return r;
            char *dst = static_cast<char *>(r.rawDataMut());
            for (keel::usize pp = 0; pp < P; ++pp)
                rot180PageBytes(src + pp * R * C * es, dst + pp * R * C * es, R, C, es);
            return r;
        }
        // kMod 1 (90° CCW) / 3 (270°): output shape (C, R, P).
        auto r = is3D ? Value::matrix3d(C, R, P, t, mr) : Value::matrix(C, R, t, mr);
        if (x.numel() == 0) return r;
        char *dst = static_cast<char *>(r.rawDataMut());
        const auto kern = (kMod == 1) ? rot90OncePageBytes : rot270PageBytes;
        for (keel::usize pp = 0; pp < P; ++pp)
            kern(src + pp * R * C * es, dst + pp * C * R * es, R, C, es);
        return r;
    }

    // k mod 4 == 0 → identity (just copy). Same shape as input.
    if (kMod == 0) {
        auto r = is3D ? Value::matrix3d(R, C, P, ValueType::DOUBLE, mr)
                      : Value::matrix(R, C, ValueType::DOUBLE, mr);
        if (x.numel() > 0)
            std::memcpy(r.doubleDataMut(), x.doubleData(),
                        x.numel() * sizeof(double));
        return r;
    }

    // k mod 4 == 2 → same shape (R, C) but elements reflected.
    if (kMod == 2) {
        auto r = is3D ? Value::matrix3d(R, C, P, ValueType::DOUBLE, mr)
                      : Value::matrix(R, C, ValueType::DOUBLE, mr);
        if (x.numel() == 0) return r;
        const double *src = x.doubleData();
        double *dst = r.doubleDataMut();
        for (keel::usize pp = 0; pp < P; ++pp)
            rot180Page(src + pp * R * C, dst + pp * R * C, R, C);
        return r;
    }

    // k mod 4 == 1 (90° CCW) or == 3 (90° CW = 270°): output shape is
    // (C, R, P) — the per-page rows and cols swap.
    auto r = is3D ? Value::matrix3d(C, R, P, ValueType::DOUBLE, mr)
                  : Value::matrix(C, R, ValueType::DOUBLE, mr);
    if (x.numel() == 0) return r;
    const double *src = x.doubleData();
    double *dst = r.doubleDataMut();
    if (kMod == 1) {
        for (keel::usize pp = 0; pp < P; ++pp)
            rot90OncePage(src + pp * R * C, dst + pp * C * R, R, C);
    } else { // kMod == 3
        for (keel::usize pp = 0; pp < P; ++pp)
            rot270Page(src + pp * R * C, dst + pp * C * R, R, C);
    }
    return r;
}

// ────────────────────────────────────────────────────────────────────
// circshift
// ────────────────────────────────────────────────────────────────────
//
// MATLAB:
//   circshift(V, k)        — vector: rotate by k (positive = right/down).
//   circshift(M, k)        — matrix: shift along first non-singleton dim.
//   circshift(M, [r c])    — shift rows by r, cols by c.
// Modulo arithmetic ensures shifts >= dim length wrap correctly.
// Negative shifts work via the same modulo (k%n then add n if negative).

Value circshift(const Value &x, keel::i64 k, keel::Allocator *mr)
{
    const auto &dd = x.dims();
    // Non-DOUBLE (char / logical / complex / cell / string): circshift is a
    // pure rearrangement — route through the type-agnostic ND permute. Scalar
    // k shifts along the first non-singleton dimension.
    if (x.type() != ValueType::DOUBLE) {
        if (x.isScalar() || x.numel() == 0) return x;
        int fnsd = 0;
        for (int i = 0; i < dd.ndim(); ++i)
            if (dd.dim(i) > 1) { fnsd = i; break; }
        keel::i64 sh[Dims::kMaxRank] = {0};
        sh[fnsd] = k;
        return circshiftND(x, keel::ArrayRef<keel::i64>(sh, fnsd + 1), mr);
    }
    if (x.isScalar()) return Value::scalar(x.toScalar(), mr);
    if (dd.isVector()) {
        const keel::usize n = x.numel();
        auto r = Value::matrix(dd.rows(), dd.cols(), ValueType::DOUBLE, mr);
        const keel::usize sh = wrap(k, n);
        for (keel::usize i = 0; i < n; ++i)
            r.doubleDataMut()[i] = x.doubleData()[(i + n - sh) % n];
        return r;
    }
    // 2D / 3D matrix: scalar k shifts along first non-singleton dim.
    // For 2D matrices that's dim=1 (rows). For 3D, dim 1 too (since
    // rows is typically > 1).
    return circshift(x, k, 0, mr);
}

Value circshiftND(const Value &x, keel::ArrayRef<keel::i64> shifts, keel::Allocator *mr)
{
    const ValueType t = x.type();
    // STRING arrays + struct arrays + function-handle arrays are deferred.
    if (t == ValueType::STRUCT || t == ValueType::FUNC_HANDLE
        || t == ValueType::STRING)
        throw Error(std::string("circshift: does not support type '")
                     + mtypeName(t) + "'",
                     0, 0, "circshift", "", "numkit:circshift:typeND");
    const auto &d = x.dims();
    const int nd = d.ndim();
    constexpr int kMaxNd = Dims::kMaxRank;
    if (nd > kMaxNd)
        throw Error("circshift: rank exceeds 32",
                     0, 0, "circshift", "", "numkit:circshift:tooManyDims");

    const bool isCell = (t == ValueType::CELL);
    if (isCell && nd > 2)
        throw Error("circshift: cell array rank > 2 not supported",
                     0, 0, "circshift", "", "numkit:circshift:cellND");

    keel::usize outDims[kMaxNd];
    for (int i = 0; i < nd; ++i) outDims[i] = d.dim(i);
    // CELL needs proper cell storage (Value::matrixND does not allocate it).
    Value r = isCell ? Value::cell(outDims[0], nd >= 2 ? outDims[1] : 1, mr)
                     : Value::matrixND(outDims, nd, t, mr);
    if (x.numel() == 0) return r;

    const int nshifts = static_cast<int>(shifts.size());
    keel::usize shiftMod[kMaxNd] = {0};
    for (int i = 0; i < nd; ++i) {
        const keel::i64 s = (i < nshifts) ? shifts[i] : 0;
        shiftMod[i] = wrap(s, d.dim(i));
    }

    // OBJECT: build the per-element shifted source map → objectGather.
    if (x.isObject()) {
        keel::usize strides[kMaxNd];
        computeStridesColMajor(d, strides);
        const keel::usize total = d.numel();
        std::vector<keel::usize> map(total);
        for (keel::usize dstLin = 0; dstLin < total; ++dstLin) {
            keel::usize rem = dstLin, srcLin = 0;
            for (int i = nd - 1; i >= 0; --i) {
                const keel::usize coord = rem / strides[i];
                rem -= coord * strides[i];
                const keel::usize srcCoord = (coord + d.dim(i) - shiftMod[i]) % d.dim(i);
                srcLin += srcCoord * strides[i];
            }
            map[dstLin] = srcLin;
        }
        return x.objectGather(map.data(), d, mr);
    }

    // CELL permutes element-by-element (Value copy); POD types copy raw
    // bytes. circshift is a pure rearrangement, so it is type-agnostic.
    const keel::usize es = isCell ? 0 : elementSize(t);
    const char *src = isCell ? nullptr : static_cast<const char *>(x.rawData());
    char *dst = isCell ? nullptr : static_cast<char *>(r.rawDataMut());
    auto copyElem = [&](keel::usize dIdx, keel::usize sIdx) {
        if (isCell) r.cellAt(dIdx) = x.cellAt(sIdx);
        else        std::memcpy(dst + dIdx * es, src + sIdx * es, es);
    };
    const keel::usize R = d.dim(0);
    const keel::usize shift0 = shiftMod[0];

    if (nd == 1) {
        for (keel::usize i = 0; i < R; ++i)
            copyElem(i, (i + R - shift0) % R);
        return r;
    }

    keel::usize srcStrides[kMaxNd];
    computeStridesColMajor(d, srcStrides);

    keel::usize outerDimsArr[kMaxNd];
    for (int i = 1; i < nd; ++i) outerDimsArr[i - 1] = d.dim(i);
    Dims outerIter(outerDimsArr, nd - 1);

    keel::usize outerCoords[kMaxNd] = {0};
    do {
        keel::usize srcOuterOff = 0, dstOuterOff = 0;
        for (int i = 1; i < nd; ++i) {
            const keel::usize dimI = d.dim(i);
            const keel::usize srcCoord = (outerCoords[i - 1] + dimI - shiftMod[i]) % dimI;
            srcOuterOff += srcCoord * srcStrides[i];
            dstOuterOff += outerCoords[i - 1] * srcStrides[i];
        }
        if (shift0 == 0) {
            if (isCell)
                for (keel::usize i = 0; i < R; ++i) copyElem(dstOuterOff + i, srcOuterOff + i);
            else
                std::memcpy(dst + dstOuterOff * es, src + srcOuterOff * es, R * es);
        } else {
            for (keel::usize i = 0; i < R; ++i)
                copyElem(dstOuterOff + i, srcOuterOff + (i + R - shift0) % R);
        }
    } while (incrementCoords(outerCoords, outerIter));

    return r;
}

Value circshift(const Value &x, keel::i64 kRow, keel::i64 kCol, keel::Allocator *mr)
{
    const auto &dd = x.dims();
    // Non-DOUBLE: route through the type-agnostic ND permute (shift dim1 by
    // kRow, dim2 by kCol; higher dims unshifted).
    if (x.type() != ValueType::DOUBLE) {
        if (x.isScalar() || x.numel() == 0) return x;
        const keel::i64 shifts[2] = {kRow, kCol};
        return circshiftND(x, keel::ArrayRef<keel::i64>(shifts, 2), mr);
    }
    if (x.isScalar()) return Value::scalar(x.toScalar(), mr);
    if (dd.is3D()) {
        const keel::usize R = dd.rows(), C = dd.cols(), P = dd.pages();
        auto r = Value::matrix3d(R, C, P, ValueType::DOUBLE, mr);
        for (keel::usize pp = 0; pp < P; ++pp)
            shift2D(x.doubleData() + pp * R * C,
                    r.doubleDataMut() + pp * R * C,
                    R, C, kRow, kCol);
        return r;
    }
    if (dd.ndim() >= 4) {
        const keel::i64 shifts[2] = {kRow, kCol};
        return circshiftND(x, keel::ArrayRef<keel::i64>(shifts, 2), mr);
    }
    const keel::usize R = dd.rows(), C = dd.cols();
    auto r = Value::matrix(R, C, ValueType::DOUBLE, mr);
    shift2D(x.doubleData(), r.doubleDataMut(), R, C, kRow, kCol);
    return r;
}

// ────────────────────────────────────────────────────────────────────
// tril / triu
// ────────────────────────────────────────────────────────────────────
//
// k is the diagonal offset:
//   k =  0  → main diagonal
//   k =  1  → first super-diagonal (above main)
//   k = -1  → first sub-diagonal (below main)
// tril keeps elements where col - row <= k. triu keeps col - row >= k.


// ND tril/triu: apply the 2D byte-level mask to every outer-axis slice.
// The first two axes form the matrix; axes 2..N-1 are "outer pages".
// All numeric types supported via byte-copy + memset(0) (zero bit pattern
// is the canonical zero for DOUBLE/SINGLE/integer/LOGICAL/COMPLEX).

Value tril(const Value &x, int k, keel::Allocator *mr)
{
    const auto &dd = x.dims();
    if (dd.ndim() >= 4)
        return trilTriuND(x, k, trilPageBytes, "tril", mr);

    // Type-preserving: keep the lower triangle, zero-fill the rest. The
    // zeroed bit pattern is the canonical zero for every supported type
    // (DOUBLE / SINGLE / int / LOGICAL / CHAR / COMPLEX). CELL / STRING /
    // STRUCT rejected, matching MATLAB ("must be numeric, char, or logical").
    const ValueType t = x.type();
    if (t == ValueType::CELL || t == ValueType::STRUCT || t == ValueType::STRING
        || t == ValueType::FUNC_HANDLE)
        throw Error("tril: inputs must be numeric, char, or logical",
                     0, 0, "tril", "", "numkit:tril:badType");

    const keel::usize R = dd.rows(), C = dd.cols();
    const keel::usize P = dd.is3D() ? dd.pages() : 1;
    auto r = dd.is3D() ? Value::matrix3d(R, C, P, t, mr)
                       : Value::matrix(R, C, t, mr);
    if (x.numel() == 0) return r;
    const keel::usize es = elementSize(t);
    const char *src = static_cast<const char *>(x.rawData());
    char *dst = static_cast<char *>(r.rawDataMut());
    for (keel::usize pp = 0; pp < P; ++pp)
        trilPageBytes(src + pp * R * C * es, dst + pp * R * C * es, R, C, k, es);
    return r;
}

Value triu(const Value &x, int k, keel::Allocator *mr)
{
    const auto &dd = x.dims();
    if (dd.ndim() >= 4)
        return trilTriuND(x, k, triuPageBytes, "triu", mr);

    // Type-preserving (see tril).
    const ValueType t = x.type();
    if (t == ValueType::CELL || t == ValueType::STRUCT || t == ValueType::STRING
        || t == ValueType::FUNC_HANDLE)
        throw Error("triu: inputs must be numeric, char, or logical",
                     0, 0, "triu", "", "numkit:triu:badType");

    const keel::usize R = dd.rows(), C = dd.cols();
    const keel::usize P = dd.is3D() ? dd.pages() : 1;
    auto r = dd.is3D() ? Value::matrix3d(R, C, P, t, mr)
                       : Value::matrix(R, C, t, mr);
    if (x.numel() == 0) return r;
    const keel::usize es = elementSize(t);
    const char *src = static_cast<const char *>(x.rawData());
    char *dst = static_cast<char *>(r.rawDataMut());
    for (keel::usize pp = 0; pp < P; ++pp)
        triuPageBytes(src + pp * R * C * es, dst + pp * R * C * es, R, C, k, es);
    return r;
}

// ────────────────────────────────────────────────────────────────────
// flip — generalized N-D flip
// ────────────────────────────────────────────────────────────────────

Value flip(const Value &x, int dim1Based, keel::Allocator *mr)
{
    int axis;
    if (dim1Based <= 0) {
        // First non-singleton dim, 0-based.
        axis = 0;
        const auto &d = x.dims();
        for (int i = 0; i < d.ndim(); ++i) {
            if (d.dim(i) > 1) { axis = i; break; }
        }
    } else {
        axis = dim1Based - 1;
    }
    return flipNDAlongAxis(x, axis, "flip", mr);
}

// ────────────────────────────────────────────────────────────────────
// repelem — element-wise replication
// ────────────────────────────────────────────────────────────────────
//
// 1-D form: repelem(v, n) returns a vector where v(i) is repeated n
// times for each i. Output length = numel(v) * n. Result shape mirrors
// the input vector's row/column orientation (row → row, column →
// column, scalar → 1×n row).
//
// 2-D form: repelem(A, m, n) replaces each element a(i,j) with an
// m × n constant block of a(i,j) values. Output is (R*m) × (C*n).
//
// DOUBLE inputs only for now (covers ~all script use). Other types
// would need the typed-byte-copy treatment used by repmat / flip.

Value repelem(const Value &x, keel::usize n, keel::Allocator *mr)
{
    const auto &d = x.dims();
    if (d.ndim() > 2 || (d.rows() != 1 && d.cols() != 1 && !x.isScalar()))
        throw Error("repelem: 1-arg form requires a vector input",
                     0, 0, "repelem", "", "numkit:repelem:notVector");
    if (x.type() != ValueType::DOUBLE)
        throw Error("repelem: only DOUBLE inputs are supported",
                     0, 0, "repelem", "", "numkit:repelem:type");

    const keel::usize inN = x.numel();
    const keel::usize outN = inN * n;
    // Preserve column-vector orientation; scalar → row vector of n.
    const bool isCol = (d.rows() > 1 && d.cols() == 1);
    auto r = isCol ? Value::matrix(outN, 1, ValueType::DOUBLE, mr)
                   : Value::matrix(1, outN, ValueType::DOUBLE, mr);
    if (outN == 0) return r;

    const double *src = x.doubleData();
    double *dst = r.doubleDataMut();
    for (keel::usize i = 0; i < inN; ++i) {
        const double v = src[i];
        for (keel::usize k = 0; k < n; ++k)
            dst[i * n + k] = v;
    }
    return r;
}

Value repelem(const Value &x, keel::usize m, keel::usize n, keel::Allocator *mr)
{
    const auto &d = x.dims();
    if (d.ndim() > 2)
        throw Error("repelem: 3-arg form is 2-D only",
                     0, 0, "repelem", "", "numkit:repelem:rank");
    if (x.type() != ValueType::DOUBLE)
        throw Error("repelem: only DOUBLE inputs are supported",
                     0, 0, "repelem", "", "numkit:repelem:type");

    const keel::usize R = d.rows(), C = d.cols();
    const keel::usize outR = R * m, outC = C * n;
    auto r = Value::matrix(outR, outC, ValueType::DOUBLE, mr);
    if (outR == 0 || outC == 0) return r;

    const double *src = x.doubleData();
    double *dst = r.doubleDataMut();
    // Column-major: walk output by (outRow, outCol) and pull from
    // src(outRow / m, outCol / n).
    for (keel::usize oc = 0; oc < outC; ++oc) {
        const keel::usize srcCol = oc / n;
        for (keel::usize orow = 0; orow < outR; ++orow) {
            const keel::usize srcRow = orow / m;
            dst[oc * outR + orow] = src[srcCol * R + srcRow];
        }
    }
    return r;
}


// repelem(v, counts): per-element counts (scalar or vector the length of v).
Value repelem(const Value &x, const Value &counts, keel::Allocator *mr)
{
    if (counts.isScalar())   // fast path / preserves scalar-form semantics
        return repelem(x, static_cast<keel::usize>(counts.toScalar()), mr);

    const auto &d = x.dims();
    if (d.ndim() > 2 || (d.rows() != 1 && d.cols() != 1 && !x.isScalar()))
        throw Error("repelem: vector-count form requires a vector input",
                     0, 0, "repelem", "", "numkit:repelem:notVector");
    if (x.type() != ValueType::DOUBLE)
        throw Error("repelem: only DOUBLE inputs are supported",
                     0, 0, "repelem", "", "numkit:repelem:type");

    keel::ArenaTemp arena = keel::get_scratch(mr);
    auto map = repelemMap(counts, x.numel(), arena);
    const keel::usize outN = map.size();
    const bool isCol = (d.rows() > 1 && d.cols() == 1);
    auto r = isCol ? Value::matrix(outN, 1, ValueType::DOUBLE, mr)
                   : Value::matrix(1, outN, ValueType::DOUBLE, mr);
    if (outN == 0) return r;

    const double *src = x.doubleData();
    double *dst = r.doubleDataMut();
    for (keel::usize k = 0; k < outN; ++k) dst[k] = src[map[k]];
    return r;
}

// repelem(A, r, c): per-row / per-column counts (each scalar or vector).
Value repelem(const Value &x, const Value &rCounts, const Value &cCounts,
              keel::Allocator *mr)
{
    if (rCounts.isScalar() && cCounts.isScalar())   // fast path
        return repelem(x, static_cast<keel::usize>(rCounts.toScalar()),
                          static_cast<keel::usize>(cCounts.toScalar()), mr);

    const auto &d = x.dims();
    if (d.ndim() > 2)
        throw Error("repelem: 3-arg form is 2-D only",
                     0, 0, "repelem", "", "numkit:repelem:rank");
    if (x.type() != ValueType::DOUBLE)
        throw Error("repelem: only DOUBLE inputs are supported",
                     0, 0, "repelem", "", "numkit:repelem:type");

    const keel::usize R = d.rows(), C = d.cols();
    keel::ArenaTemp arena = keel::get_scratch(mr);
    auto rowMap = repelemMap(rCounts, R, arena);
    auto colMap = repelemMap(cCounts, C, arena);
    const keel::usize outR = rowMap.size(), outC = colMap.size();
    auto r = Value::matrix(outR, outC, ValueType::DOUBLE, mr);
    if (outR == 0 || outC == 0) return r;

    const double *src = x.doubleData();
    double *dst = r.doubleDataMut();
    for (keel::usize oc = 0; oc < outC; ++oc) {
        const keel::usize srcCol = colMap[oc];
        for (keel::usize orow = 0; orow < outR; ++orow)
            dst[oc * outR + orow] = src[srcCol * R + rowMap[orow]];
    }
    return r;
}

// ────────────────────────────────────────────────────────────────────
// Pack 32: paddata / trimdata / resize
// ────────────────────────────────────────────────────────────────────
//
// Vector-only for now; row/column orientation is preserved. Numeric
// inputs only — the pad cell is `0` (matches MATLAB R2024 default).


Value paddata(const Value &v, keel::usize n, int dim, keel::StringRef side, double fillValue, keel::Allocator *mr)
{
    if (!v.isEmpty() && v.type() != ValueType::DOUBLE)
        throw Error("paddata: only DOUBLE inputs supported",
                     0, 0, "paddata", "", "numkit:paddata:type");
    const auto &d = v.dims();
    if (d.ndim() > 2)
        throw Error("paddata: only 1-D or 2-D inputs supported",
                     0, 0, "paddata", "", "numkit:paddata:rank");

    std::string sd(side.data(), side.size());
    for (char &c : sd) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (sd != "right" && sd != "trailing" && sd != "left" && sd != "leading" && sd != "both")
        throw Error("paddata: side must be 'trailing', 'leading', or 'both'",
                     0, 0, "paddata", "", "numkit:paddata:badSide");

    // Determine operating dimension: 1 = rows, 2 = cols.
    int opDim = dim;
    if (opDim <= 0) {
        if (d.rows() == 1 && d.cols() >= 1) opDim = 2;
        else if (d.rows() == 0 && d.cols() == 0) opDim = 2;
        else opDim = 1;
    }

    const keel::usize R = d.rows(), C = d.cols();
    const keel::usize cur = (opDim == 1) ? R : C;
    if (cur >= n) return v;

    const keel::usize padTotal = n - cur;
    keel::usize padBefore = 0;
    if (sd == "left" || sd == "leading") padBefore = padTotal;
    else if (sd == "both")               padBefore = padTotal / 2;

    if (opDim == 1) {
        const keel::usize outC = (R == 0 && C == 0) ? 1 : C;
        auto out = Value::matrix(n, outC, ValueType::DOUBLE, mr);
        double *dst = out.doubleDataMut();
        for (keel::usize i = 0; i < n * outC; ++i) dst[i] = fillValue;
        if (R > 0 && C > 0) {
            const double *src = v.doubleData();
            for (keel::usize c = 0; c < C; ++c)
                std::memcpy(dst + c * n + padBefore, src + c * R, R * sizeof(double));
        }
        return out;
    } else {
        const keel::usize outR = (R == 0 && C == 0) ? 1 : R;
        auto out = Value::matrix(outR, n, ValueType::DOUBLE, mr);
        double *dst = out.doubleDataMut();
        for (keel::usize i = 0; i < outR * n; ++i) dst[i] = fillValue;
        if (R > 0 && C > 0) {
            const double *src = v.doubleData();
            std::memcpy(dst + padBefore * outR, src, R * C * sizeof(double));
        }
        return out;
    }
}

Value paddata(const Value &v, keel::usize n, keel::Allocator *mr)
{
    if (!isVectorLike(v))
        throw Error("paddata: vector input required",
                     0, 0, "paddata", "", "numkit:paddata:notVector");
    return paddata(v, n, 0, "trailing", 0.0, mr);
}

Value paddata(const Value &x, keel::usize len, int dim, keel::StringRef side, keel::Allocator *mr)
{
    return paddata(x, len, dim, side, 0.0, mr);
}

Value trimdata(const Value &v, keel::usize n, int dim, keel::StringRef side, keel::Allocator *mr)
{
    if (!v.isEmpty() && v.type() != ValueType::DOUBLE)
        throw Error("trimdata: only DOUBLE inputs supported",
                     0, 0, "trimdata", "", "numkit:trimdata:type");
    const auto &d = v.dims();
    if (d.ndim() > 2)
        throw Error("trimdata: only 1-D or 2-D inputs supported",
                     0, 0, "trimdata", "", "numkit:trimdata:rank");

    std::string sd(side.data(), side.size());
    for (char &c : sd) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (sd != "right" && sd != "trailing" && sd != "left" && sd != "leading" && sd != "both")
        throw Error("trimdata: side must be 'trailing', 'leading', or 'both'",
                     0, 0, "trimdata", "", "numkit:trimdata:badSide");

    int opDim = dim;
    if (opDim <= 0) {
        if (d.rows() == 1 && d.cols() >= 1) opDim = 2;
        else opDim = 1;
    }

    const keel::usize R = d.rows(), C = d.cols();
    const keel::usize cur = (opDim == 1) ? R : C;
    if (cur <= n) return v;

    const keel::usize trimTotal = cur - n;
    keel::usize keepStart = 0;
    if (sd == "left" || sd == "leading") keepStart = trimTotal;
    else if (sd == "both")               keepStart = trimTotal / 2;

    if (opDim == 1) {
        auto out = Value::matrix(n, C, ValueType::DOUBLE, mr);
        if (n > 0 && C > 0) {
            const double *src = v.doubleData();
            double *dst = out.doubleDataMut();
            for (keel::usize c = 0; c < C; ++c)
                std::memcpy(dst + c * n, src + c * R + keepStart, n * sizeof(double));
        }
        return out;
    } else {
        auto out = Value::matrix(R, n, ValueType::DOUBLE, mr);
        if (R > 0 && n > 0) {
            const double *src = v.doubleData();
            double *dst = out.doubleDataMut();
            std::memcpy(dst, src + keepStart * R, R * n * sizeof(double));
        }
        return out;
    }
}

Value trimdata(const Value &v, keel::usize n, keel::Allocator *mr)
{
    if (!isVectorLike(v))
        throw Error("trimdata: vector input required",
                     0, 0, "trimdata", "", "numkit:trimdata:notVector");
    return trimdata(v, n, 0, "trailing", mr);
}

Value resize(const Value &v, keel::usize n, int dim, keel::StringRef side, double fillValue, keel::Allocator *mr)
{
    const auto &d = v.dims();
    int opDim = dim;
    if (opDim <= 0) {
        if (d.rows() == 1 && d.cols() >= 1) opDim = 2;
        else if (d.rows() == 0 && d.cols() == 0) opDim = 2;
        else opDim = 1;
    }
    const keel::usize cur = (opDim == 1) ? d.rows() : d.cols();
    return (cur < n) ? paddata(v, n, dim, side, fillValue, mr)
                     : trimdata(v, n, dim, side, mr);
}

Value resize(const Value &v, keel::usize n, keel::Allocator *mr)
{
    return (v.numel() < n) ? paddata(v, n, mr) : trimdata(v, n, mr);
}

Value head(const Value &x, keel::usize k, keel::Allocator *mr)
{
    if (x.isEmpty()) return x;
    const auto &d = x.dims();
    const keel::usize R = d.rows();
    const keel::usize C = d.cols();
    const keel::usize r = std::min(k, R);
    if (r == R) return x;
    const keel::usize totalElems = x.numel();
    const keel::usize elemSize = totalElems > 0 ? (x.rawBytes() / totalElems) : sizeof(double);
    auto out = Value::matrix(r, C, x.type(), mr);
    if (elemSize > 0 && x.rawData() != nullptr) {
        for (keel::usize c = 0; c < C; ++c) {
            const char *srcCol = static_cast<const char *>(x.rawData()) + c * R * elemSize;
            char *dstCol = static_cast<char *>(out.rawDataMut()) + c * r * elemSize;
            std::memcpy(dstCol, srcCol, r * elemSize);
        }
    }
    return out;
}

Value tail(const Value &x, keel::usize k, keel::Allocator *mr)
{
    if (x.isEmpty()) return x;
    const auto &d = x.dims();
    const keel::usize R = d.rows();
    const keel::usize C = d.cols();
    const keel::usize r = std::min(k, R);
    if (r == R) return x;
    const keel::usize startRow = R - r;
    const keel::usize totalElems = x.numel();
    const keel::usize elemSize = totalElems > 0 ? (x.rawBytes() / totalElems) : sizeof(double);
    auto out = Value::matrix(r, C, x.type(), mr);
    if (elemSize > 0 && x.rawData() != nullptr) {
        for (keel::usize c = 0; c < C; ++c) {
            const char *srcCol = static_cast<const char *>(x.rawData()) + (c * R + startRow) * elemSize;
            char *dstCol = static_cast<char *>(out.rawDataMut()) + c * r * elemSize;
            std::memcpy(dstCol, srcCol, r * elemSize);
        }
    }
    return out;
}

Value bsxfun(keel::StringRef op, const Value &a, const Value &b, keel::Allocator *mr)
{
    if (op == "plus" || op == "@plus") return plus(a, b, mr);
    if (op == "minus" || op == "@minus") return minus(a, b, mr);
    if (op == "times" || op == "@times") return times(a, b, mr);
    if (op == "rdivide" || op == "@rdivide") return rdivide(a, b, mr);
    if (op == "ldivide" || op == "@ldivide") return ldivide(a, b, mr);
    if (op == "power" || op == "@power") return power(a, b, mr);
    if (op == "eq" || op == "@eq") return eq(a, b, mr);
    if (op == "ne" || op == "@ne") return ne(a, b, mr);
    if (op == "lt" || op == "@lt") return lt(a, b, mr);
    if (op == "le" || op == "@le") return le(a, b, mr);
    if (op == "gt" || op == "@gt") return gt(a, b, mr);
    if (op == "ge" || op == "@ge") return ge(a, b, mr);
    if (op == "and" || op == "@and") return logical_and(a, b, mr);
    if (op == "or" || op == "@or") return logical_or(a, b, mr);
    throw Error(keel::format("bsxfun: unsupported function '{}'", op), 0, 0, "bsxfun", "", "numkit:bsxfun:unsupported");
}

// ── sub2ind / ind2sub (see manip.hpp) ─────────────────────────────────

Value sub2ind(const Value &siz, keel::ArrayRef<Value> subs,
              keel::Allocator *mr)
{
    const keel::usize nDims = siz.numel();
    if (nDims == 0)
        throw Error("sub2ind: siz must not be empty",
                     0, 0, "sub2ind", "", "numkit:sub2ind:badSiz");
    if (subs.empty())
        throw Error("sub2ind: requires at least 1 subscript",
                     0, 0, "sub2ind", "", "numkit:sub2ind:nargin");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto dims = keel::Array<keel::usize>(nDims, scratch.allocator());
    for (keel::usize i = 0; i < nDims; ++i)
        dims[i] = static_cast<keel::usize>(siz.doubleData()[i]);

    const keel::usize nSubs = subs.size();
    if (nSubs > nDims)
        throw Error("sub2ind: too many subscript arrays for given siz",
                     0, 0, "sub2ind", "", "numkit:sub2ind:tooManySubs");

    // All sub arrays must agree on shape; result inherits that shape.
    const Value &shapeRef = subs[0];
    const keel::usize outN = shapeRef.numel();
    for (keel::usize a = 0; a < nSubs; ++a)
        if (subs[a].numel() != outN)
            throw Error("sub2ind: subscript arrays must be the same size",
                         0, 0, "sub2ind", "", "numkit:sub2ind:shape");

    auto r = (shapeRef.isScalar())
                ? Value::scalar(0.0, mr)
                : Value::matrix(shapeRef.dims().rows(), shapeRef.dims().cols(),
                                ValueType::DOUBLE, mr);
    double *dst = r.doubleDataMut();
    // Strides for column-major: stride[0]=1, stride[1]=dim0, stride[2]=dim0*dim1.
    auto strideOfDim = [&](keel::usize d) {
        keel::usize s = 1;
        for (keel::usize i = 0; i < d && i < nDims; ++i) s *= dims[i];
        return s;
    };
    for (keel::usize k = 0; k < outN; ++k) {
        keel::usize lin = 0;
        for (keel::usize d = 0; d < nSubs; ++d) {
            const double sd = subs[d].isScalar() ? subs[d].toScalar()
                                                 : subs[d].doubleData()[k];
            const keel::usize idx = static_cast<keel::usize>(sd) - 1;  // 1-based → 0-based
            lin += idx * strideOfDim(d);
        }
        dst[k] = static_cast<double>(lin + 1);  // back to 1-based
    }
    return r;
}

std::vector<Value> ind2sub(const Value &siz, const Value &ind, keel::usize nout,
                           keel::Allocator *mr)
{
    const keel::usize nDims = siz.numel();
    if (nDims == 0)
        throw Error("ind2sub: siz must not be empty",
                     0, 0, "ind2sub", "", "numkit:ind2sub:badSiz");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto dims = keel::Array<keel::usize>(nDims, scratch.allocator());
    for (keel::usize i = 0; i < nDims; ++i)
        dims[i] = static_cast<keel::usize>(siz.doubleData()[i]);

    const keel::usize outDims = (nout == 0) ? nDims : nout;
    const keel::usize outN = ind.numel();

    // Build effective dim list: outDims entries. The first outDims-1
    // match siz; the last absorbs the product of remaining dims.
    auto effDim = [&](keel::usize d) -> keel::usize {
        if (d + 1 < outDims) return d < nDims ? dims[d] : 1;
        keel::usize r = 1;
        for (keel::usize i = d; i < nDims; ++i) r *= dims[i];
        return r ? r : 1;
    };
    keel::Array<keel::usize> stride(outDims, scratch.allocator());
    {
        keel::usize s = 1;
        for (keel::usize d = 0; d < outDims; ++d) {
            stride[d] = s;
            s *= effDim(d);
        }
    }

    auto makeLike = [&]() {
        return ind.isScalar()
                   ? Value::scalar(0.0, mr)
                   : Value::matrix(ind.dims().rows(), ind.dims().cols(),
                                   ValueType::DOUBLE, mr);
    };
    std::vector<Value> rs;
    rs.reserve(outDims);
    for (keel::usize i = 0; i < outDims; ++i) rs.emplace_back(makeLike());

    for (keel::usize k = 0; k < outN; ++k) {
        const double iv = ind.isScalar() ? ind.toScalar() : ind.doubleData()[k];
        keel::usize lin = static_cast<keel::usize>(iv) - 1;
        for (keel::usize d = 0; d < outDims; ++d) {
            const keel::usize v = (d + 1 < outDims)
                                 ? (lin / stride[d]) % effDim(d)
                                 : (lin / stride[d]);
            rs[d].doubleDataMut()[k] = static_cast<double>(v + 1);
        }
    }
    return rs;
}

} // namespace numkit::builtin
