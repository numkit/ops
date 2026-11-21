// toolboxes/.../matrix_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by matrix.cpp + matrix_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/format.h>
#include <keel/math.h>
#include <numkit/value/value.hpp>
#include <numkit/ops/helpers.hpp>
#include <numkit/ops/reductions.hpp>
#include <numkit/ops/rows_helpers.hpp>

#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::builtin {

namespace {

// Siamese / de la Loubère method for odd N >= 3.
// Fills positions [0..N²-1] starting at (0, N/2) and stepping
// (-1, +1) mod N; on collision step (+1, 0) instead.
void magicOdd(keel::f64 *p, keel::usize N)
{
    keel::usize r = 0;
    keel::usize c = N / 2;
    for (keel::usize k = 1; k <= N * N; ++k) {
        p[r * N + c] = keel::as<keel::f64>(k);
        const keel::usize nr = (r == 0) ? (N - 1) : (r - 1);
        const keel::usize nc = (c + 1) % N;
        if (p[nr * N + nc] != 0.0) {
            r = (r + 1) % N;          // collision: drop down
        } else {
            r = nr;
            c = nc;
        }
    }
}

// Doubly-even (N ≡ 0 mod 4): start with the natural 1..N² fill and
// swap each cell whose (i mod 4, j mod 4) is on either of the two
// 4×4-block diagonals.
[[maybe_unused]] void magicDoublyEven(keel::f64 *p, keel::usize N)
{
    const keel::usize total = N * N;
    for (keel::usize i = 0; i < N; ++i)
        for (keel::usize j = 0; j < N; ++j) {
            const keel::usize k = i * N + j + 1;       // 1-based natural fill
            const keel::usize mi = i % 4;
            const keel::usize mj = j % 4;
            const bool diag = (mi == mj) || (mi + mj == 3);
            p[i * N + j] = keel::as<keel::f64>(diag ? (total + 1 - k) : k);
        }
}

// Singly-even (N ≡ 2 mod 4, N >= 6) via Strachey's method.
// This mirrors MATLAB R2025b's magic.m verbatim:
//   p = N/2;  K = (N-2)/4;
//   M = [Q  Q+2p²; Q+3p²  Q+p²]   where Q = magic(p)
//   For columns j in {0..K-1, N-K+1..N-1}:
//     swap rows r ∈ {0..p-1} with rows r+p (column-by-column).
//   Then for row mid = K (0-indexed = (p-1)/2):
//     swap (mid, 0) ↔ (mid+p, 0)   -- undo the previous swap on this cell
//     swap (mid, K) ↔ (mid+p, K)   -- and apply the strached swap instead
[[maybe_unused]] void magicSinglyEven(keel::f64 *p, keel::usize N)
{
    const keel::usize P = N / 2;          // odd
    const keel::usize S = P * P;
    const keel::usize K = (N - 2) / 4;    // num "full" left/right cols to swap

    // Build one (P×P) odd-magic and tile into four quadrants.
    keel::Array<keel::f64> sq(P * P, 0.0);
    magicOdd(sq.data(), P);

    for (keel::usize i = 0; i < P; ++i) {
        for (keel::usize j = 0; j < P; ++j) {
            const keel::f64 s = sq[i * P + j];
            p[(i)     * N + (j)]     = s;                    // A (top-left)
            p[(i)     * N + (j + P)] = s + 2.0 * S;          // C (top-right)
            p[(i + P) * N + (j)]     = s + 3.0 * S;          // D (bottom-left)
            p[(i + P) * N + (j + P)] = s + 1.0 * S;          // B (bottom-right)
        }
    }

    // Bulk column swaps: leftmost K and rightmost K-1 columns get
    // top-half / bottom-half rows swapped. (For N=6, K=1 → swap col 0
    // only; right side has K-1=0 cols, none.)
    auto swapRowsAtCol = [&](keel::usize col) {
        for (keel::usize r = 0; r < P; ++r)
            std::swap(p[r * N + col], p[(r + P) * N + col]);
    };
    for (keel::usize c = 0; c < K; ++c)
        swapRowsAtCol(c);
    for (keel::usize c = N - K + 1; c < N && K > 0; ++c)
        swapRowsAtCol(c);

    // Middle-row fix: for row mid = K (0-indexed), undo column-0 swap
    // and apply column-K swap instead. (MATLAB: i=k+1, j=[1, i].)
    const keel::usize mid = K;
    if (mid < P) {
        std::swap(p[mid * N + 0], p[(mid + P) * N + 0]);  // undo
        std::swap(p[mid * N + K], p[(mid + P) * N + K]);  // apply
    }
}

} // anonymous namespace
namespace {

// Extract Value as a flat double buffer in linear element order via
// elemAsDouble (handles every numeric/logical type). Returns the
// buffer; live for the lifetime of `scratch`.
[[maybe_unused]] keel::Array<keel::f64> valueToScratchDoubles(const Value &v, const keel::ArenaTemp &scratch)
{
    const keel::usize n = v.numel();
    keel::Array<keel::f64> out(n, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    return out;
}

} // namespace
namespace {

// Per-page transpose helper. `conjugate` flips the sign of imaginary
// parts when input is COMPLEX. For DOUBLE / SINGLE inputs the flag is
// ignored at the element level (no-op).
template <typename T>
Value pageTransposeT(const Value &x, ValueType ty, bool conjugate, keel::Allocator *mr)
{
    const auto &d = x.dims();
    const keel::usize M = d.rows(), N = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1u;

    auto out = (P == 1u)
        ? Value::matrix(N, M, ty, mr)
        : Value::matrix3d(N, M, P, ty, mr);

    const T *src = static_cast<const T *>(x.rawData());
    T *dst       = static_cast<T *>(out.rawDataMut());
    const keel::usize pageInElems  = M * N;
    const keel::usize pageOutElems = N * M;

    for (keel::usize p = 0; p < P; ++p) {
        const T *sp = src + p * pageInElems;
        T *dp       = dst + p * pageOutElems;
        for (keel::usize j = 0; j < N; ++j) {
            for (keel::usize i = 0; i < M; ++i) {
                if constexpr (std::is_same_v<T, keel::c64>) {
                    keel::c64 v = sp[j * M + i];
                    dp[i * N + j] = conjugate ? keel::conj(v) : v;
                } else {
                    (void)conjugate;
                    dp[i * N + j] = sp[j * M + i];
                }
            }
        }
    }
    return out;
}

[[maybe_unused]] Value pageTransposeAny(const Value &x, bool conjugate, keel::Allocator *mr)
{
    switch (x.type()) {
    case ValueType::DOUBLE:  return pageTransposeT<keel::f64>(x, ValueType::DOUBLE, conjugate, mr);
    case ValueType::SINGLE:  return pageTransposeT<keel::f32>(x, ValueType::SINGLE, conjugate, mr);
    case ValueType::COMPLEX: return pageTransposeT<keel::c64>(x, ValueType::COMPLEX, conjugate, mr);
    case ValueType::INT8:    return pageTransposeT<keel::i8>(x, ValueType::INT8, conjugate, mr);
    case ValueType::INT16:   return pageTransposeT<keel::i16>(x, ValueType::INT16, conjugate, mr);
    case ValueType::INT32:   return pageTransposeT<keel::i32>(x, ValueType::INT32, conjugate, mr);
    case ValueType::INT64:   return pageTransposeT<keel::i64>(x, ValueType::INT64, conjugate, mr);
    case ValueType::UINT8:   return pageTransposeT<keel::u8>(x, ValueType::UINT8, conjugate, mr);
    case ValueType::UINT16:  return pageTransposeT<keel::u16>(x, ValueType::UINT16, conjugate, mr);
    case ValueType::UINT32:  return pageTransposeT<keel::u32>(x, ValueType::UINT32, conjugate, mr);
    case ValueType::UINT64:  return pageTransposeT<keel::u64>(x, ValueType::UINT64, conjugate, mr);
    case ValueType::LOGICAL: return pageTransposeT<keel::u8>(x, ValueType::LOGICAL, conjugate, mr);
    default:
        throw Error("pagetranspose: unsupported input type",
                     0, 0, "pagetranspose", "", "numkit:pagetranspose:badType");
    }
}

} // namespace
namespace {

// Per-page matmul kernel, parameterised by element type. The DOUBLE
// specialisation hands off to the SIMD-aware matmulDoubleLoop in the
// backend; the SINGLE one uses the same (j, k, i) ordering as a
// portable inline loop.
template <typename T>
inline void runPageMatmul(const T *, const T *, T *,
                          keel::usize, keel::usize, keel::usize);

template <>
inline void runPageMatmul<keel::f64>(const keel::f64 *a, const keel::f64 *b, keel::f64 *c,
                                  keel::usize M, keel::usize N, keel::usize K)
{
    ops::detail::matmulDoubleLoop(a, b, c, M, N, K);
}

template <>
inline void runPageMatmul<keel::f32>(const keel::f32 *a, const keel::f32 *b, keel::f32 *c,
                                 keel::usize M, keel::usize N, keel::usize K)
{
    for (keel::usize j = 0; j < N; ++j) {
        keel::f32 *cj = c + j * M;
        for (keel::usize i = 0; i < M; ++i) cj[i] = 0.0f;
        for (keel::usize k = 0; k < K; ++k) {
            const keel::f32 bkj = b[j * K + k];
            const keel::f32 *ak = a + k * M;
            for (keel::usize i = 0; i < M; ++i)
                cj[i] += ak[i] * bkj;
        }
    }
}

template <>
inline void runPageMatmul<keel::c64>(const keel::c64 *a, const keel::c64 *b, keel::c64 *c,
                                   keel::usize M, keel::usize N, keel::usize K)
{
    for (keel::usize j = 0; j < N; ++j) {
        keel::c64 *cj = c + j * M;
        for (keel::usize i = 0; i < M; ++i) cj[i] = keel::c64(0.0, 0.0);
        for (keel::usize k = 0; k < K; ++k) {
            const keel::c64 bkj = b[j * K + k];
            const keel::c64 *ak = a + k * M;
            for (keel::usize i = 0; i < M; ++i)
                cj[i] += ak[i] * bkj;
        }
    }
}

template <typename T> constexpr ValueType pagemtimesElemMType();
template <> constexpr ValueType pagemtimesElemMType<keel::f64 >() { return ValueType::DOUBLE;  }
template <> constexpr ValueType pagemtimesElemMType<keel::f32  >() { return ValueType::SINGLE;  }
template <> constexpr ValueType pagemtimesElemMType<keel::c64>() { return ValueType::COMPLEX; }

// Read element i of `src` as T. For T = Complex, real-typed sources
// upgrade to (real, 0); for T ∈ {double, float}, complex sources are
// rejected upstream so we never reach the if-branch with COMPLEX input.
template <typename T>
inline T readElemAsT(const Value &src, keel::usize i, bool typeMatches)
{
    if constexpr (std::is_same_v<T, keel::c64>) {
        if (typeMatches) return src.complexData()[i];
        return keel::c64(src.elemAsDouble(i), 0.0);
    } else {
        if (typeMatches) return static_cast<const T *>(src.rawData())[i];
        return static_cast<T>(src.elemAsDouble(i));
    }
}

// Conjugate a value if T is Complex; identity for real T.
template <typename T>
inline T conjIfComplex(T v)
{
    if constexpr (std::is_same_v<T, keel::c64>) return keel::conj(v);
    else return v;
}

// Materialise one page from `src` into typed scratch `dst`, optionally
// transposing (and conjugating, for ctranspose on Complex). Direct copy
// (no per-element conversion) when src already holds the target type
// AND no transpose is needed.
template <typename T>
void materialisePage(T *dst, const Value &src, keel::usize pageOff,
                     keel::usize rowDim, keel::usize colDim, TranspOp tr)
{
    const keel::usize pageElems = rowDim * colDim;
    const keel::usize base = pageOff * pageElems;
    const bool typeMatches = (src.type() == pagemtimesElemMType<T>());

    if (tr == TranspOp::None) {
        if (typeMatches) {
            std::memcpy(dst, static_cast<const T *>(src.rawData()) + base,
                        pageElems * sizeof(T));
        } else {
            for (keel::usize i = 0; i < pageElems; ++i)
                dst[i] = readElemAsT<T>(src, base + i, false);
        }
        return;
    }
    // Transpose: dst is colDim × rowDim col-major;
    // dst[r * colDim + c] = src[c * rowDim + r] (then conjugate if ctranspose+Complex).
    const bool needsConj = (tr == TranspOp::CTranspose);
    for (keel::usize r = 0; r < rowDim; ++r) {
        for (keel::usize c = 0; c < colDim; ++c) {
            const keel::usize srcOff = base + c * rowDim + r;
            T v = readElemAsT<T>(src, srcOff, typeMatches);
            if (needsConj) v = conjIfComplex<T>(v);
            dst[r * colDim + c] = v;
        }
    }
}

template <typename T>
Value pagemtimesImpl(const Value &x, TranspOp tx, const Value &y, TranspOp ty, keel::Allocator *mr)
{
    const auto &xd = x.dims();
    const auto &yd = y.dims();
    const int xnd = xd.ndim();
    const int ynd = yd.ndim();
    if (xnd < 2 || ynd < 2)
        throw Error("pagemtimes: each input must have at least 2 dimensions",
                     0, 0, "pagemtimes", "", "numkit:pagemtimes:rank");

    const keel::usize xRowDim = xd.dim(0), xColDim = xd.dim(1);
    const keel::usize yRowDim = yd.dim(0), yColDim = yd.dim(1);

    const keel::usize M  = (tx == TranspOp::None) ? xRowDim : xColDim;
    const keel::usize Kx = (tx == TranspOp::None) ? xColDim : xRowDim;
    const keel::usize Ky = (ty == TranspOp::None) ? yRowDim : yColDim;
    const keel::usize N  = (ty == TranspOp::None) ? yColDim : yRowDim;
    if (Kx != Ky)
        throw Error("pagemtimes: inner matrix dimensions must agree",
                     0, 0, "pagemtimes", "", "numkit:pagemtimes:innerdim");
    const keel::usize K = Kx;

    constexpr int kMaxNd = Dims::kMaxRank;
    const int xb = std::max(0, xnd - 2);
    const int yb = std::max(0, ynd - 2);
    const int outBatchNd = std::max(xb, yb);
    keel::usize xBatch[kMaxNd], yBatch[kMaxNd], outBatch[kMaxNd];
    for (int i = 0; i < outBatchNd; ++i) {
        xBatch[i] = (i < xb) ? xd.dim(2 + i) : 1;
        yBatch[i] = (i < yb) ? yd.dim(2 + i) : 1;
        if (xBatch[i] != yBatch[i] && xBatch[i] != 1 && yBatch[i] != 1)
            throw Error("pagemtimes: batch dimensions must broadcast "
                         "(each axis must match or be 1)",
                         0, 0, "pagemtimes", "", "numkit:pagemtimes:dimagree");
        outBatch[i] = std::max(xBatch[i], yBatch[i]);
    }

    keel::usize batchN = 1;
    for (int i = 0; i < outBatchNd; ++i) batchN *= outBatch[i];

    const int outNd = 2 + outBatchNd;
    keel::usize outDimArr[kMaxNd];
    outDimArr[0] = M;
    outDimArr[1] = N;
    for (int i = 0; i < outBatchNd; ++i) outDimArr[2 + i] = outBatch[i];
    auto z = createForDims(Dims(outDimArr, outNd), pagemtimesElemMType<T>(), mr);
    if (M == 0 || N == 0 || batchN == 0)
        return z;

    T *zData = static_cast<T *>(z.rawDataMut());
    const keel::usize xPageStride = xRowDim * xColDim;
    const keel::usize yPageStride = yRowDim * yColDim;
    const keel::usize zPageStride = M * N;

    // Direct-pass when source already matches T and no transpose is
    // needed; otherwise materialise into typed scratch (one per call,
    // reused across all batch pages).
    const bool xDirect = (x.type() == pagemtimesElemMType<T>()) && (tx == TranspOp::None);
    const bool yDirect = (y.type() == pagemtimesElemMType<T>()) && (ty == TranspOp::None);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<T> scratchX(scratch.allocator()), scratchY(scratch.allocator());
    if (!xDirect) scratchX.resize(xPageStride);
    if (!yDirect) scratchY.resize(yPageStride);

    auto getXPage = [&](keel::usize pageOff) -> const T * {
        if (xDirect)
            return static_cast<const T *>(x.rawData()) + pageOff * xPageStride;
        materialisePage(scratchX.data(), x, pageOff, xRowDim, xColDim, tx);
        return scratchX.data();
    };
    auto getYPage = [&](keel::usize pageOff) -> const T * {
        if (yDirect)
            return static_cast<const T *>(y.rawData()) + pageOff * yPageStride;
        materialisePage(scratchY.data(), y, pageOff, yRowDim, yColDim, ty);
        return scratchY.data();
    };

    if (outBatchNd == 0) {
        runPageMatmul<T>(getXPage(0), getYPage(0), zData, M, N, K);
        return z;
    }

    keel::usize xBatchStride[kMaxNd], yBatchStride[kMaxNd];
    {
        keel::usize sx = 1, sy = 1;
        for (int i = 0; i < outBatchNd; ++i) {
            xBatchStride[i] = sx;
            yBatchStride[i] = sy;
            sx *= xBatch[i];
            sy *= yBatch[i];
        }
    }

    keel::usize coords[kMaxNd] = {0};
    Dims outBatchDims(outBatch, outBatchNd);
    keel::usize pageIdx = 0;
    do {
        keel::usize xOff = 0, yOff = 0;
        for (int i = 0; i < outBatchNd; ++i) {
            const keel::usize xc = (xBatch[i] == 1) ? 0 : coords[i];
            const keel::usize yc = (yBatch[i] == 1) ? 0 : coords[i];
            xOff += xc * xBatchStride[i];
            yOff += yc * yBatchStride[i];
        }
        runPageMatmul<T>(getXPage(xOff), getYPage(yOff),
                         zData + pageIdx * zPageStride,
                         M, N, K);
        ++pageIdx;
    } while (incrementCoords(coords, outBatchDims));

    return z;
}

} // namespace
namespace {

// Promote to a 2D DOUBLE matrix for row-tuple ops. Returns a copy if the
// type or shape differs; for already-2D-DOUBLE input returns by value
// (cheap COW in the engine).
Value toDoubleMatrix2D(const Value &x, const char *fn, keel::Allocator *mr)
{
    if (x.dims().is3D() || x.dims().ndim() > 2)
        throw Error(keel::format("{}: input must be 2D", fn),
                     0, 0, fn, "", keel::format("numkit:{}:bad2D", fn));
    const keel::usize R = x.dims().rows();
    const keel::usize C = x.dims().cols();
    if (x.type() == ValueType::DOUBLE) {
        // Return a fresh DOUBLE matrix identical to x — cheap, avoids
        // touching the input through a shared buffer later.
        auto r = Value::matrix(R, C, ValueType::DOUBLE, mr);
        if (x.numel() > 0)
            std::memcpy(r.doubleDataMut(), x.doubleData(),
                        x.numel() * sizeof(keel::f64));
        return r;
    }
    auto r = Value::matrix(R, C, ValueType::DOUBLE, mr);
    keel::f64 *dst = r.doubleDataMut();
    for (keel::usize i = 0; i < x.numel(); ++i)
        dst[i] = x.elemAsDouble(i);
    return r;
}

// Complex sortrows (MATLAB rule): rows are ordered lexicographically by the
// requested columns; each column compares complex entries by magnitude |z|
// then phase angle arg(z) ascending (a negative column index sorts that
// column descending). A NaN component sorts last. Must NOT drop the
// imaginary part (toDoubleMatrix2D does) — that silently returns wrong rows.
inline bool cxRowLess(keel::c64 a, keel::c64 b)
{
    const keel::f64 am = keel::abs(a), bm = keel::abs(b);
    const bool an = std::isnan(am), bn = std::isnan(bm);
    if (an || bn) { if (an && bn) return false; return bn; }   // non-NaN < NaN
    if (am != bm) return am < bm;
    return keel::arg(a) < keel::arg(b);
}

std::tuple<Value, Value>
sortRowsComplex(const Value &x, const int *cols, keel::usize nCols,
                keel::Allocator *mr)
{
    if (x.dims().is3D() || x.dims().ndim() > 2)
        throw Error("sortrows: input must be 2D",
                     0, 0, "sortrows", "", "numkit:sortrows:bad2D");
    const keel::usize R = x.dims().rows();
    const keel::usize C = x.dims().cols();
    if (R == 0)
        return std::make_tuple(x, Value::matrix(0, 1, ValueType::DOUBLE, mr));

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<int> sortKeys(scratch.allocator());
    if (nCols == 0) {
        sortKeys.reserve(C);
        for (keel::usize c = 1; c <= C; ++c) sortKeys.push_back(keel::as<int>(c));
    } else {
        sortKeys.assign(keel::ArrayRef<int>(cols, nCols));
        for (int rawCol : sortKeys) {
            const int absC = (rawCol < 0) ? -rawCol : rawCol;
            if (rawCol == 0 || keel::as<keel::usize>(absC) > C)
                throw Error("sortrows: column index out of range",
                             0, 0, "sortrows", "", "numkit:sortrows:badCol");
        }
    }

    const keel::c64 *src = x.complexData();
    auto perm = keel::Array<keel::usize>(R, scratch.allocator());
    for (keel::usize i = 0; i < R; ++i) perm[i] = i;
    std::stable_sort(perm.begin(), perm.end(),
        [&](keel::usize a, keel::usize b) {
            for (int key : sortKeys) {
                const bool desc = key < 0;
                const keel::usize c = keel::as<keel::usize>((desc ? -key : key) - 1);
                const keel::c64 va = src[c * R + a], vb = src[c * R + b];
                if (cxRowLess(va, vb)) return !desc;
                if (cxRowLess(vb, va)) return desc;
            }
            return false;   // all keys equal → stable
        });

    auto out = Value::matrix(R, C, ValueType::COMPLEX, mr);
    keel::c64 *dst = out.complexDataMut();
    for (keel::usize i = 0; i < R; ++i)
        for (keel::usize c = 0; c < C; ++c)
            dst[c * R + i] = src[c * R + perm[i]];
    auto idx = Value::matrix(R, 1, ValueType::DOUBLE, mr);
    keel::f64 *idxP = idx.doubleDataMut();
    for (keel::usize i = 0; i < R; ++i)
        idxP[i] = keel::as<keel::f64>(perm[i] + 1);
    return std::make_tuple(std::move(out), std::move(idx));
}

[[maybe_unused]] std::tuple<Value, Value>
sortRowsImpl(const Value &x, const int *cols, keel::usize nCols, keel::Allocator *mr)
{
    if (x.type() == ValueType::COMPLEX)
        return sortRowsComplex(x, cols, nCols, mr);
    auto m = toDoubleMatrix2D(x, "sortrows", mr);
    const keel::usize R = m.dims().rows();
    const keel::usize C = m.dims().cols();

    if (R == 0) {
        // Empty rows — return as-is and an empty 0×1 idx column.
        auto idx = Value::matrix(0, 1, ValueType::DOUBLE, mr);
        return std::make_tuple(std::move(m), std::move(idx));
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);

    // Validate cols list. nCols==0 ⇒ all columns ascending in order.
    keel::Array<int> sortKeys(scratch.allocator());
    if (nCols == 0) {
        sortKeys.reserve(C);
        for (keel::usize c = 1; c <= C; ++c)
            sortKeys.push_back(keel::as<int>(c));
    } else {
        sortKeys.assign(keel::ArrayRef<int>(cols, nCols));
        for (int rawCol : sortKeys) {
            const int absC = (rawCol < 0) ? -rawCol : rawCol;
            if (rawCol == 0 || keel::as<keel::usize>(absC) > C)
                throw Error("sortrows: column index out of range",
                             0, 0, "sortrows", "", "numkit:sortrows:badCol");
        }
    }

    auto perm = keel::Array<keel::usize>(R, scratch.allocator());
    for (keel::usize i = 0; i < R; ++i) perm[i] = i;

    const keel::f64 *src = m.doubleData();
    std::stable_sort(perm.begin(), perm.end(),
        [&](keel::usize a, keel::usize b) {
            return numkit::ops::rowLexCmpByCols(src, C, R, a, b,
                                            sortKeys.data(), sortKeys.size()) < 0;
        });

    auto sorted = numkit::ops::collectRowsByIndex(m, perm.data(), perm.size(), mr);
    auto idx = Value::matrix(R, 1, ValueType::DOUBLE, mr);
    keel::f64 *idxP = idx.doubleDataMut();
    for (keel::usize i = 0; i < R; ++i)
        idxP[i] = keel::as<keel::f64>(perm[i] + 1);
    return std::make_tuple(std::move(sorted), std::move(idx));
}

} // namespace
namespace {

// Type-aware predicate: element at linear index i non-zero?
// NaN counts as non-zero (NaN != 0). For COMPLEX both parts checked.
template <typename T>
inline bool isNonzeroElemT(const T *p, keel::usize i) { return p[i] != T{0}; }

inline bool isNonzeroComplex(const keel::c64 *p, keel::usize i)
{
    return p[i].real() != 0.0 || p[i].imag() != 0.0;
}

template <typename Visit>
void forEachNonzero(const Value &x, Visit visit)
{
    const keel::usize n = x.numel();
    switch (x.type()) {
    case ValueType::LOGICAL: {
        const keel::u8 *p = x.logicalData();
        for (keel::usize i = 0; i < n; ++i) if (p[i]) visit(i);
        break;
    }
    case ValueType::DOUBLE: {
        const keel::f64 *p = x.doubleData();
        for (keel::usize i = 0; i < n; ++i) if (isNonzeroElemT(p, i)) visit(i);
        break;
    }
    case ValueType::SINGLE: {
        const keel::f32 *p = x.singleData();
        for (keel::usize i = 0; i < n; ++i) if (isNonzeroElemT(p, i)) visit(i);
        break;
    }
    case ValueType::COMPLEX: {
        const keel::c64 *p = x.complexData();
        for (keel::usize i = 0; i < n; ++i) if (isNonzeroComplex(p, i)) visit(i);
        break;
    }
    case ValueType::INT8: {
        const keel::i8 *p = x.int8Data();
        for (keel::usize i = 0; i < n; ++i) if (isNonzeroElemT(p, i)) visit(i);
        break;
    }
    case ValueType::INT16: {
        const keel::i16 *p = x.int16Data();
        for (keel::usize i = 0; i < n; ++i) if (isNonzeroElemT(p, i)) visit(i);
        break;
    }
    case ValueType::INT32: {
        const keel::i32 *p = x.int32Data();
        for (keel::usize i = 0; i < n; ++i) if (isNonzeroElemT(p, i)) visit(i);
        break;
    }
    case ValueType::INT64: {
        const keel::i64 *p = x.int64Data();
        for (keel::usize i = 0; i < n; ++i) if (isNonzeroElemT(p, i)) visit(i);
        break;
    }
    case ValueType::UINT8: {
        const keel::u8 *p = x.uint8Data();
        for (keel::usize i = 0; i < n; ++i) if (isNonzeroElemT(p, i)) visit(i);
        break;
    }
    case ValueType::UINT16: {
        const keel::u16 *p = x.uint16Data();
        for (keel::usize i = 0; i < n; ++i) if (isNonzeroElemT(p, i)) visit(i);
        break;
    }
    case ValueType::UINT32: {
        const keel::u32 *p = x.uint32Data();
        for (keel::usize i = 0; i < n; ++i) if (isNonzeroElemT(p, i)) visit(i);
        break;
    }
    case ValueType::UINT64: {
        const keel::u64 *p = x.uint64Data();
        for (keel::usize i = 0; i < n; ++i) if (isNonzeroElemT(p, i)) visit(i);
        break;
    }
    default:
        throw Error("nnz/nonzeros: unsupported element type",
                     0, 0, "nnz", "", "numkit:nnz:badType");
    }
}

template <typename T>
T *typedDstFor(Value &r, ValueType outType)
{
    switch (outType) {
    case ValueType::LOGICAL: return reinterpret_cast<T *>(r.logicalDataMut());
    case ValueType::DOUBLE:  return reinterpret_cast<T *>(r.doubleDataMut());
    case ValueType::SINGLE:  return reinterpret_cast<T *>(r.singleDataMut());
    case ValueType::COMPLEX: return reinterpret_cast<T *>(r.complexDataMut());
    case ValueType::INT8:    return reinterpret_cast<T *>(r.int8DataMut());
    case ValueType::INT16:   return reinterpret_cast<T *>(r.int16DataMut());
    case ValueType::INT32:   return reinterpret_cast<T *>(r.int32DataMut());
    case ValueType::INT64:   return reinterpret_cast<T *>(r.int64DataMut());
    case ValueType::UINT8:   return reinterpret_cast<T *>(r.uint8DataMut());
    case ValueType::UINT16:  return reinterpret_cast<T *>(r.uint16DataMut());
    case ValueType::UINT32:  return reinterpret_cast<T *>(r.uint32DataMut());
    case ValueType::UINT64:  return reinterpret_cast<T *>(r.uint64DataMut());
    default: return nullptr;
    }
}

template <typename T, typename Reader>
Value collectTypedNonzeros(const Value &x, ValueType outType, Reader read, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<T> vals(scratch.allocator());
    forEachNonzero(x, [&](keel::usize i) { vals.push_back(read(i)); });
    auto r = Value::matrix(vals.size(), 1, outType, mr);
    if (!vals.empty()) {
        T *dst = typedDstFor<T>(r, outType);
        std::memcpy(dst, vals.data(), vals.size() * sizeof(T));
    }
    return r;
}

} // namespace
namespace {

// Integer cumsum / cumprod. MATLAB keeps the integer class and accumulates
// NATIVELY with saturation at each step — the saturated running value is
// carried forward, so cumsum(int8([100 100 -100]))=[100 127 27] int8 and
// cumprod(int8([5 10 10]))=[5 50 127] int8. Generic strided scan over the
// chosen dimension d (column-major: stride = prod(dims[0..d-2]),
// len = dims[d-1]). Linear iteration is valid for any dim because element i
// depends only on i-stride (< i). NOTE: int64/uint64 above 2^53 lose
// precision (accumulated through double) — the same limitation as the rest
// of nk's numeric core; int8/16/32 and uint8/16/32 are exact.
template <typename T>
void cumIntegerScanInto(const Value &x, T *dst, keel::usize strideD, keel::usize lenD,
                        bool isProd)
{
    const keel::f64 lo = keel::as<keel::f64>(std::numeric_limits<T>::min());
    const keel::f64 hi = keel::as<keel::f64>(std::numeric_limits<T>::max());
    const keel::usize n = x.numel();
    for (keel::usize i = 0; i < n; ++i) {
        const keel::usize coord = (i / strideD) % lenD;
        const keel::f64 cur = x.elemAsDouble(i);
        keel::f64 v = cur;
        if (coord != 0) {
            const keel::f64 prev = keel::as<keel::f64>(dst[i - strideD]);
            v = isProd ? prev * cur : prev + cur;
        }
        if (v < lo) v = lo;
        else if (v > hi) v = hi;        // saturate to the class range
        dst[i] = static_cast<T>(v);
    }
}

[[maybe_unused]] Value cumIntegerNative(const Value &x, int dim, bool isProd,
                       keel::Allocator *mr)
{
    const auto &dd = x.dims();
    const int nd = dd.ndim();
    int d;
    if (dim > 0) {
        d = numkit::ops::resolveDim(x, dim, isProd ? "cumprod" : "cumsum");
    } else {
        d = 1;                          // first non-singleton dim (MATLAB default)
        for (int k = 0; k < nd; ++k)
            if (dd.dim(k) > 1) { d = k + 1; break; }
    }
    keel::usize strideD = 1;
    for (int k = 0; k < d - 1; ++k) strideD *= dd.dim(k);
    const keel::usize lenD = (d - 1 < nd) ? dd.dim(d - 1) : 1;

    keel::usize outDims[Dims::kMaxRank];
    for (int k = 0; k < nd; ++k) outDims[k] = dd.dim(k);
    Value r = Value::matrixND(outDims, nd, x.type(), mr);
    if (x.numel() == 0 || lenD == 0 || strideD == 0) return r;

    switch (x.type()) {
    case ValueType::INT8:   cumIntegerScanInto<keel::i8>  (x, r.int8DataMut(),   strideD, lenD, isProd); break;
    case ValueType::INT16:  cumIntegerScanInto<keel::i16> (x, r.int16DataMut(),  strideD, lenD, isProd); break;
    case ValueType::INT32:  cumIntegerScanInto<keel::i32> (x, r.int32DataMut(),  strideD, lenD, isProd); break;
    case ValueType::INT64:  cumIntegerScanInto<keel::i64> (x, r.int64DataMut(),  strideD, lenD, isProd); break;
    case ValueType::UINT8:  cumIntegerScanInto<keel::u8> (x, r.uint8DataMut(),  strideD, lenD, isProd); break;
    case ValueType::UINT16: cumIntegerScanInto<keel::u16>(x, r.uint16DataMut(), strideD, lenD, isProd); break;
    case ValueType::UINT32: cumIntegerScanInto<keel::u32>(x, r.uint32DataMut(), strideD, lenD, isProd); break;
    case ValueType::UINT64: cumIntegerScanInto<keel::u64>(x, r.uint64DataMut(), strideD, lenD, isProd); break;
    default: break;
    }
    return r;
}

// Inclusive cumulative scan of a COMPLEX array along the 1-based dim d (sum
// when isProd=false, product when isProd=true). Output shape == input shape.
// Uniform inner-block (B) / outer-count (O) strides cover vector/2D/3D/ND.
// (Complex isn't perf-critical here — correctness over a SIMD prefix scan.)
[[maybe_unused]] Value cumComplexAlongDim(const Value &x, int d, bool isProd,
                         keel::Allocator *mr)
{
    const auto &dd = x.dims();
    const int nd = dd.ndim();
    constexpr int kMaxNd = Dims::kMaxRank;
    if (nd > kMaxNd)
        throw Error("cumsum: rank exceeds 32",
                     0, 0, "cumsum", "", "numkit:cumsum:tooManyDims");
    keel::usize outDims[kMaxNd];
    for (int i = 0; i < nd; ++i) outDims[i] = dd.dim(i);
    auto r = Value::matrixND(outDims, nd, ValueType::COMPLEX, mr);
    if (x.numel() == 0) return r;
    const keel::usize sliceLen = (d >= 1 && d <= nd) ? dd.dim(d - 1) : 1;
    if (sliceLen == 0) return r;
    keel::usize B = 1; for (int i = 0; i < d - 1 && i < nd; ++i) B *= dd.dim(i);
    keel::usize O = 1; for (int i = d; i < nd; ++i) O *= dd.dim(i);
    const keel::c64 *src = x.complexData();
    keel::c64 *dst = r.complexDataMut();
    for (keel::usize o = 0; o < O; ++o)
        for (keel::usize b = 0; b < B; ++b) {
            const keel::usize base = o * sliceLen * B + b;
            keel::c64 acc = src[base];
            dst[base] = acc;
            for (keel::usize k = 1; k < sliceLen; ++k) {
                const keel::c64 v = src[base + k * B];
                acc = isProd ? acc * v : acc + v;
                dst[base + k * B] = acc;
            }
        }
    return r;
}

// Running complex min/max along the 1-based dim d (MATLAB R2025b). Order rule:
// compare by |z| (modulus), ties by angle(z) — the BINARY max/min comparator
// with NO all-real fallback, so cummax(complex([-1 -3 -2])) is [-1 -3 -3] (it
// differs from the real-input cummax). A NaN-component element is skipped
// (omitnan, the cummax/cummin default); a leading run of NaNs is carried until
// the first finite element. Mirrors complexMinMaxPick (omitNan=true) in
// math/.../reductions_detail.hpp — keep the order rule in sync. (Complex isn't
// perf-critical here — correctness over a SIMD scan.)
[[maybe_unused]] Value cumComplexMinMaxAlongDim(const Value &x, int d, bool isMax,
                               keel::Allocator *mr)
{
    const auto &dd = x.dims();
    const int nd = dd.ndim();
    constexpr int kMaxNd = Dims::kMaxRank;
    if (nd > kMaxNd)
        throw Error("cummax: rank exceeds 32",
                     0, 0, "cummax", "", "numkit:cummax:tooManyDims");
    keel::usize outDims[kMaxNd];
    for (int i = 0; i < nd; ++i) outDims[i] = dd.dim(i);
    auto r = Value::matrixND(outDims, nd, ValueType::COMPLEX, mr);
    if (x.numel() == 0) return r;
    const keel::usize sliceLen = (d >= 1 && d <= nd) ? dd.dim(d - 1) : 1;
    if (sliceLen == 0) return r;
    keel::usize B = 1; for (int i = 0; i < d - 1 && i < nd; ++i) B *= dd.dim(i);
    keel::usize O = 1; for (int i = d; i < nd; ++i) O *= dd.dim(i);
    const keel::c64 *src = x.complexData();
    keel::c64 *dst = r.complexDataMut();
    auto isMissing = [](keel::c64 z) {
        return std::isnan(z.real()) || std::isnan(z.imag());
    };
    auto vBetter = [&](keel::c64 v, keel::c64 best) {
        const keel::f64 av = keel::abs(v), ab = keel::abs(best);
        if (av != ab) return isMax ? (av > ab) : (av < ab);
        const keel::f64 gv = keel::arg(v), gb = keel::arg(best);
        return isMax ? (gv > gb) : (gv < gb);
    };
    for (keel::usize o = 0; o < O; ++o)
        for (keel::usize b = 0; b < B; ++b) {
            const keel::usize base = o * sliceLen * B + b;
            keel::c64 acc = src[base];
            dst[base] = acc;
            for (keel::usize k = 1; k < sliceLen; ++k) {
                const keel::c64 v = src[base + k * B];
                if (isMissing(v)) {
                    // omitnan: skip the missing operand, carry acc forward.
                } else if (isMissing(acc) || vBetter(v, acc)) {
                    acc = v;  // first finite replaces a carried NaN, or v wins
                }
                dst[base + k * B] = acc;
            }
        }
    return r;
}

// First non-singleton dimension (1-based), default 1 — MATLAB's default op dim.
[[maybe_unused]] int firstNonSingletonDim(const Value &x)
{
    const auto &dd = x.dims();
    for (int k = 0; k < dd.ndim(); ++k)
        if (dd.dim(k) > 1) return k + 1;
    return 1;
}

} // namespace
namespace {

template <typename Op>
void cumKernel(const Value &x, int d, Op op, keel::f64 *dst)
{
    const auto &dd = x.dims();
    const keel::usize R = dd.rows(), C = dd.cols();
    const keel::usize P = dd.is3D() ? dd.pages() : 1;
    const keel::f64 *src = x.doubleData();

    if (d == 1) {
        for (keel::usize pp = 0; pp < P; ++pp)
            for (keel::usize c = 0; c < C; ++c) {
                const keel::usize base = pp * R * C + c * R;
                if (R == 0) continue;
                keel::f64 acc = src[base];
                dst[base] = acc;
                for (keel::usize rr = 1; rr < R; ++rr) {
                    acc = op(acc, src[base + rr]);
                    dst[base + rr] = acc;
                }
            }
    } else if (d == 2) {
        for (keel::usize pp = 0; pp < P; ++pp)
            for (keel::usize rr = 0; rr < R; ++rr) {
                const keel::usize pageBase = pp * R * C;
                if (C == 0) continue;
                keel::f64 acc = src[pageBase + rr];
                dst[pageBase + rr] = acc;
                for (keel::usize c = 1; c < C; ++c) {
                    acc = op(acc, src[pageBase + c * R + rr]);
                    dst[pageBase + c * R + rr] = acc;
                }
            }
    } else if (d == 3) {
        for (keel::usize c = 0; c < C; ++c)
            for (keel::usize rr = 0; rr < R; ++rr) {
                if (P == 0) continue;
                keel::f64 acc = src[c * R + rr];
                dst[c * R + rr] = acc;
                for (keel::usize pp = 1; pp < P; ++pp) {
                    acc = op(acc, src[pp * R * C + c * R + rr]);
                    dst[pp * R * C + c * R + rr] = acc;
                }
            }
    }
}

template <typename Op>
Value cumImpl(const Value &x, int dim, Op op, const char *fn, keel::Allocator *mr)
{
    if (x.isEmpty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);

    if (x.dims().isVector() || x.isScalar()) {
        auto r = Value::matrix(x.dims().rows(), x.dims().cols(),
                                ValueType::DOUBLE, mr);
        if (x.numel() == 0) return r;
        keel::f64 acc = x.doubleData()[0];
        r.doubleDataMut()[0] = acc;
        for (keel::usize i = 1; i < x.numel(); ++i) {
            acc = op(acc, x.doubleData()[i]);
            r.doubleDataMut()[i] = acc;
        }
        return r;
    }

    const int d = numkit::ops::resolveDim(x, dim, fn);
    const auto &dd = x.dims();
    auto r = dd.is3D() ? Value::matrix3d(dd.rows(), dd.cols(), dd.pages(),
                                          ValueType::DOUBLE, mr)
                       : Value::matrix(dd.rows(), dd.cols(),
                                        ValueType::DOUBLE, mr);
    cumKernel(x, d, op, r.doubleDataMut());
    return r;
}

} // namespace
namespace {

using ScanFn = void (*)(const keel::f64 *, keel::f64 *, keel::usize);

template <typename Op>
Value cumScanDispatch(const Value &x, int dim, ScanFn scan, Op scalarOp, const char *fn, keel::Allocator *mr)
{
    if (x.isEmpty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if (x.isScalar()) {
        auto r = Value::matrix(x.dims().rows(), x.dims().cols(), ValueType::DOUBLE, mr);
        r.doubleDataMut()[0] = x.toScalar();
        return r;
    }
    if (x.dims().isVector()) {
        auto r = Value::matrix(x.dims().rows(), x.dims().cols(), ValueType::DOUBLE, mr);
        scan(x.doubleData(), r.doubleDataMut(), x.numel());
        return r;
    }

    const int d = numkit::ops::resolveDim(x, dim, fn);
    const auto &dd = x.dims();

    // ND fallback (rank ≥ 4): per-slice scan along axis d-1.
    if (dd.ndim() >= 4) {
        constexpr int kMaxNd = Dims::kMaxRank;
        if (dd.ndim() > kMaxNd)
            throw Error(keel::format("{}: rank exceeds 32", fn),
                         0, 0, fn, "", keel::format("numkit:{}:tooManyDims", fn));
        keel::usize outDims[kMaxNd];
        for (int i = 0; i < dd.ndim(); ++i) outDims[i] = dd.dim(i);
        auto r = Value::matrixND(outDims, dd.ndim(), ValueType::DOUBLE, mr);
        const keel::usize sliceLen = dd.dim(d - 1);
        keel::usize B = 1;
        for (int i = 0; i < d - 1; ++i) B *= dd.dim(i);
        keel::usize O = 1;
        for (int i = d; i < dd.ndim(); ++i) O *= dd.dim(i);
        const keel::f64 *src = x.doubleData();
        keel::f64 *dst = r.doubleDataMut();
        if (B == 1) {
            for (keel::usize o = 0; o < O; ++o) {
                const keel::usize base = o * sliceLen;
                scan(src + base, dst + base, sliceLen);
            }
        } else {
            for (keel::usize o = 0; o < O; ++o)
                for (keel::usize b = 0; b < B; ++b) {
                    const keel::usize base = o * sliceLen * B + b;
                    if (sliceLen == 0) continue;
                    keel::f64 acc = src[base];
                    dst[base] = acc;
                    for (keel::usize k = 1; k < sliceLen; ++k) {
                        acc = scalarOp(acc, src[base + k * B]);
                        dst[base + k * B] = acc;
                    }
                }
        }
        return r;
    }

    const keel::usize R = dd.rows(), C = dd.cols();
    const keel::usize P = dd.is3D() ? dd.pages() : 1;
    auto r = dd.is3D() ? Value::matrix3d(R, C, P, ValueType::DOUBLE, mr)
                       : Value::matrix(R, C, ValueType::DOUBLE, mr);
    const keel::f64 *src = x.doubleData();
    keel::f64 *dst = r.doubleDataMut();

    if (d == 1) {
        // Per-column scan — column data is contiguous, route through SIMD.
        for (keel::usize pp = 0; pp < P; ++pp)
            for (keel::usize c = 0; c < C; ++c) {
                const keel::usize base = pp * R * C + c * R;
                scan(src + base, dst + base, R);
            }
    } else {
        // dim=2/3: strided access; reuse the existing scalar cumKernel.
        cumKernel(x, d, scalarOp, dst);
    }
    return r;
}

} // namespace
namespace {
// cummax / cummin PRESERVE the logical class (unlike cumsum / cumprod, which
// promote logical → double). Run the double kernel on a promoted copy, then
// narrow the 0/1 result back to LOGICAL — mirrors xorOf's double→logical step.
Value logicalizeCumResult(const Value &d, keel::Allocator *mr)
{
    if (d.isScalar()) return Value::logicalScalar(d.toScalar() != 0.0, mr);
    Value r = createLike(d, ValueType::LOGICAL, mr);
    keel::u8 *dst = r.logicalDataMut();
    const keel::f64 *src = d.doubleData();
    const keel::usize n = d.numel();
    for (keel::usize i = 0; i < n; ++i) dst[i] = (src[i] != 0.0) ? 1 : 0;
    return r;
}

// Narrow a sorted DOUBLE result (exact char code points) back to a CHAR Value
// of the SAME shape — used by the char branch of sort_reg (MATLAB sorts char
// by code point, preserving the char class). Cannot reuse toChar(): it routes
// through fromString and FLATTENS a matrix to a row. createLike preserves dims
// (2-D / N-D); the column-major layout matches the double buffer 1:1.
[[maybe_unused]] Value charizeSortResult(const Value &d, keel::Allocator *mr)
{
    Value r = createLike(d, ValueType::CHAR, mr);
    char *dst = r.charDataMut();
    const keel::f64 *src = d.doubleData();
    const keel::usize n = d.numel();
    for (keel::usize i = 0; i < n; ++i)
        dst[i] = keel::as<char>(keel::as<int>(src[i]));
    return r;
}
} // namespace
namespace {

// One pass of forward differences along axis `d` (1-based). Source has
// dim[d-1] = sliceLen; output has dim[d-1] = sliceLen - 1. Column-major
// strides (innerStride = prod(dim[0..d-2])).
[[maybe_unused]] void diffOnceDouble(const keel::f64 *src, keel::f64 *dst,
                    const Dims &srcDims, int d)
{
    const int nd = srcDims.ndim();
    const keel::usize sliceLen = srcDims.dim(d - 1);
    if (sliceLen < 2) return;  // out has zero elements

    keel::usize innerStride = 1;
    for (int i = 0; i < d - 1; ++i) innerStride *= srcDims.dim(i);
    keel::usize outerCount = 1;
    for (int i = d; i < nd; ++i) outerCount *= srcDims.dim(i);
    const keel::usize outSliceLen = sliceLen - 1;

    if (innerStride == 1) {
        // Contiguous along the diff axis — simple linear pass per outer block.
        for (keel::usize o = 0; o < outerCount; ++o) {
            const keel::f64 *s = src + o * sliceLen;
            keel::f64 *t = dst + o * outSliceLen;
            for (keel::usize k = 0; k < outSliceLen; ++k)
                t[k] = s[k + 1] - s[k];
        }
    } else {
        for (keel::usize o = 0; o < outerCount; ++o)
            for (keel::usize b = 0; b < innerStride; ++b) {
                const keel::usize srcBase = o * innerStride * sliceLen + b;
                const keel::usize dstBase = o * innerStride * outSliceLen + b;
                for (keel::usize k = 0; k < outSliceLen; ++k)
                    dst[dstBase + k * innerStride] =
                        src[srcBase + (k + 1) * innerStride] -
                        src[srcBase + k * innerStride];
            }
    }
}

[[maybe_unused]] Value makeDiffOutput(const Dims &srcDims, int d, keel::usize step, ValueType vt,
                     keel::Allocator *mr)
{
    const int nd = srcDims.ndim();
    constexpr int kMaxNd = Dims::kMaxRank;
    if (nd > kMaxNd)
        throw Error("diff: rank exceeds 32",
                     0, 0, "diff", "", "numkit:diff:tooManyDims");
    keel::usize outDims[kMaxNd];
    for (int i = 0; i < nd; ++i) outDims[i] = srcDims.dim(i);
    outDims[d - 1] = (outDims[d - 1] >= step) ? outDims[d - 1] - step : 0;
    return Value::matrixND(outDims, nd, vt, mr);
}

// Complex counterpart of diffOnceDouble — successive differences over Complex
// storage (real and imaginary differenced together), same strided layout.
[[maybe_unused]] void diffOnceComplex(const keel::c64 *src, keel::c64 *dst, const Dims &srcDims, int d)
{
    const int nd = srcDims.ndim();
    const keel::usize sliceLen = srcDims.dim(d - 1);
    if (sliceLen < 2) return;
    keel::usize innerStride = 1;
    for (int i = 0; i < d - 1; ++i) innerStride *= srcDims.dim(i);
    keel::usize outerCount = 1;
    for (int i = d; i < nd; ++i) outerCount *= srcDims.dim(i);
    const keel::usize outSliceLen = sliceLen - 1;
    if (innerStride == 1) {
        for (keel::usize o = 0; o < outerCount; ++o) {
            const keel::c64 *s = src + o * sliceLen;
            keel::c64 *t = dst + o * outSliceLen;
            for (keel::usize k = 0; k < outSliceLen; ++k) t[k] = s[k + 1] - s[k];
        }
    } else {
        for (keel::usize o = 0; o < outerCount; ++o)
            for (keel::usize b = 0; b < innerStride; ++b) {
                const keel::usize srcBase = o * innerStride * sliceLen + b;
                const keel::usize dstBase = o * innerStride * outSliceLen + b;
                for (keel::usize k = 0; k < outSliceLen; ++k)
                    dst[dstBase + k * innerStride] =
                        src[srcBase + (k + 1) * innerStride] -
                        src[srcBase + k * innerStride];
            }
    }
}

// Identity copy of a COMPLEX value (preserves shape + both parts).
[[maybe_unused]] Value copyComplexSameShape(const Value &x, keel::Allocator *mr)
{
    const auto &dd = x.dims();
    const int nd = dd.ndim();
    keel::usize dims[Dims::kMaxRank];
    for (int i = 0; i < nd; ++i) dims[i] = dd.dim(i);
    auto r = Value::matrixND(dims, nd, ValueType::COMPLEX, mr);
    if (x.numel() > 0)
        std::memcpy(r.complexDataMut(), x.complexData(), x.numel() * sizeof(keel::c64));
    return r;
}

Value copyToDouble(const Value &x, keel::Allocator *mr)
{
    const auto &dd = x.dims();
    const int nd = dd.ndim();
    constexpr int kMaxNd = Dims::kMaxRank;
    keel::usize dims[kMaxNd];
    for (int i = 0; i < nd; ++i) dims[i] = dd.dim(i);
    auto r = Value::matrixND(dims, nd, ValueType::DOUBLE, mr);
    if (x.type() == ValueType::DOUBLE) {
        std::memcpy(r.doubleDataMut(), x.doubleData(),
                    x.numel() * sizeof(keel::f64));
    } else {
        keel::f64 *dst = r.doubleDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            dst[i] = x.elemAsDouble(i);
    }
    return r;
}

// One pass of an integer-typed difference along dim d, with NATIVE
// saturation (matches MATLAB: diff(int8([-100 100]))=127 int8). Same strided
// layout as diffOnceDouble.
template <typename T>
void diffOnceIntegerT(const T *src, T *dst, const Dims &srcDims, int d)
{
    const int nd = srcDims.ndim();
    const keel::usize sliceLen = srcDims.dim(d - 1);
    keel::usize innerStride = 1;
    for (int i = 0; i < d - 1; ++i) innerStride *= srcDims.dim(i);
    keel::usize outerCount = 1;
    for (int i = d; i < nd; ++i) outerCount *= srcDims.dim(i);
    const keel::usize outSliceLen = sliceLen - 1;
    const keel::f64 lo = keel::as<keel::f64>(std::numeric_limits<T>::min());
    const keel::f64 hi = keel::as<keel::f64>(std::numeric_limits<T>::max());
    auto sub = [&](T a, T b) -> T {
        keel::f64 v = keel::as<keel::f64>(a) - keel::as<keel::f64>(b);
        if (v < lo) v = lo;
        else if (v > hi) v = hi;
        return static_cast<T>(v);
    };
    if (innerStride == 1) {
        for (keel::usize o = 0; o < outerCount; ++o) {
            const T *s = src + o * sliceLen;
            T *t = dst + o * outSliceLen;
            for (keel::usize k = 0; k < outSliceLen; ++k) t[k] = sub(s[k + 1], s[k]);
        }
    } else {
        for (keel::usize o = 0; o < outerCount; ++o)
            for (keel::usize b = 0; b < innerStride; ++b) {
                const keel::usize srcBase = o * innerStride * sliceLen + b;
                const keel::usize dstBase = o * innerStride * outSliceLen + b;
                for (keel::usize k = 0; k < outSliceLen; ++k)
                    dst[dstBase + k * innerStride] =
                        sub(src[srcBase + (k + 1) * innerStride],
                            src[srcBase + k * innerStride]);
            }
    }
}

// n-th order diff of an integer-typed Value: keeps the class and saturates at
// each pass (the saturated pass-k result feeds pass k+1). Caller guarantees
// sliceLen > n along d.
[[maybe_unused]] Value diffInteger(const Value &x, int n, int d, keel::Allocator *mr)
{
    const ValueType vt = x.type();
    Value cur = copyIntegerSameClass(x, mr);
    for (int pass = 0; pass < n; ++pass) {
        const auto &curDims = cur.dims();
        const int nd = curDims.ndim();
        keel::usize outDims[Dims::kMaxRank];
        for (int i = 0; i < nd; ++i) outDims[i] = curDims.dim(i);
        outDims[d - 1] = (outDims[d - 1] >= 1) ? outDims[d - 1] - 1 : 0;
        Value out = Value::matrixND(outDims, nd, vt, mr);
        switch (vt) {
        case ValueType::INT8:   diffOnceIntegerT<keel::i8>  (cur.int8Data(),   out.int8DataMut(),   curDims, d); break;
        case ValueType::INT16:  diffOnceIntegerT<keel::i16> (cur.int16Data(),  out.int16DataMut(),  curDims, d); break;
        case ValueType::INT32:  diffOnceIntegerT<keel::i32> (cur.int32Data(),  out.int32DataMut(),  curDims, d); break;
        case ValueType::INT64:  diffOnceIntegerT<keel::i64> (cur.int64Data(),  out.int64DataMut(),  curDims, d); break;
        case ValueType::UINT8:  diffOnceIntegerT<keel::u8> (cur.uint8Data(),  out.uint8DataMut(),  curDims, d); break;
        case ValueType::UINT16: diffOnceIntegerT<keel::u16>(cur.uint16Data(), out.uint16DataMut(), curDims, d); break;
        case ValueType::UINT32: diffOnceIntegerT<keel::u32>(cur.uint32Data(), out.uint32DataMut(), curDims, d); break;
        case ValueType::UINT64: diffOnceIntegerT<keel::u64>(cur.uint64Data(), out.uint64DataMut(), curDims, d); break;
        default: break;
        }
        cur = std::move(out);
    }
    return cur;
}

} // namespace
namespace {

// Used by xor below — small inputs, no need for a SIMD path.
[[maybe_unused]] Value promoteToDouble(const Value &x, keel::Allocator *mr)
{
    if (x.type() == ValueType::DOUBLE) return x;
    auto r = createLike(x, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < x.numel(); ++i)
        r.doubleDataMut()[i] = x.elemAsDouble(i);
    return r;
}

} // namespace

// topkrows worker (def in matrix.cpp, external).
Value topkrows_full(const Value &A, keel::usize k,
                    keel::ArrayRef<keel::usize> colsIn,
                    keel::ArrayRef<keel::u8> descIn,
                    keel::Array<keel::usize> *out_idx,
                    keel::Allocator *mr);

} // namespace numkit::builtin
