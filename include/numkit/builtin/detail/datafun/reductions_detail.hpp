// toolboxes/.../reductions_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by reductions.cpp + reductions_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/format.h>
#include <keel/math.h>
#include <numkit/value/value.hpp>
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

// Generic column-/dim-wise reducer: applies op(acc, x) and initializes
// acc with init. For 2D: reduces across rows → row vector of cols. For 3D:
// reduces along first non-singleton dimension. For vectors/scalars: scalar.
template<typename Op>
Value reduce(const Value &x, Op op, keel::f64 init, keel::Allocator *mr, bool meanMode = false)
{
    // Reads each element as double — supports DOUBLE / SINGLE / integer
    // / LOGICAL transparently. DOUBLE keeps the contiguous fast path.
    const bool fastDouble = (x.type() == ValueType::DOUBLE);
    auto readD = [&](keel::usize i) -> keel::f64 {
        return fastDouble ? x.doubleData()[i] : x.elemAsDouble(i);
    };

    // MATLAB: a default reduction of the 0x0 empty [] returns the scalar
    // identity, NOT a 1x0 empty: sum([])=0, prod([])=1, mean([])=NaN.
    // (Partial empties like zeros(0,3) keep their per-column shape and are
    // handled by the matrix path below.)
    if (x.dims().ndim() == 2 && x.dims().rows() == 0 && x.dims().cols() == 0)
        return Value::scalar(meanMode ? std::nan("") : init, mr);

    if (x.dims().isVector() || x.isScalar()) {
        keel::f64 acc = init;
        for (keel::usize i = 0; i < x.numel(); ++i)
            acc = op(acc, readD(i));
        if (meanMode)
            acc /= keel::as<keel::f64>(x.numel());
        return Value::scalar(acc, mr);
    }

    const keel::usize R = x.dims().rows(), C = x.dims().cols();

    if (x.dims().is3D()) {
        const keel::usize P = x.dims().pages();
        const int redDim = (R > 1) ? 0 : (C > 1) ? 1 : 2;
        const keel::usize outR = (redDim == 0) ? 1 : R;
        const keel::usize outC = (redDim == 1) ? 1 : C;
        const keel::usize outP = (redDim == 2) ? 1 : P;
        const keel::usize N = (redDim == 0) ? R : (redDim == 1) ? C : P;
        auto r = Value::matrix3d(outR, outC, outP, ValueType::DOUBLE, mr);
        for (keel::usize pp = 0; pp < outP; ++pp)
            for (keel::usize c = 0; c < outC; ++c)
                for (keel::usize rr = 0; rr < outR; ++rr) {
                    keel::f64 acc = init;
                    for (keel::usize k = 0; k < N; ++k) {
                        const keel::usize rIdx = (redDim == 0) ? k : rr;
                        const keel::usize cIdx = (redDim == 1) ? k : c;
                        const keel::usize pIdx = (redDim == 2) ? k : pp;
                        acc = op(acc, readD(pIdx * R * C + cIdx * R + rIdx));
                    }
                    if (meanMode)
                        acc /= keel::as<keel::f64>(N);
                    r.doubleDataMut()[pp * outR * outC + c * outR + rr] = acc;
                }
        return r;
    }

    auto r = Value::matrix(1, C, ValueType::DOUBLE, mr);
    for (keel::usize c = 0; c < C; ++c) {
        keel::f64 acc = init;
        for (keel::usize rr = 0; rr < R; ++rr)
            acc = op(acc, readD(c * R + rr));
        if (meanMode)
            acc /= keel::as<keel::f64>(R);
        r.doubleDataMut()[c] = acc;
    }
    return r;
}

} // anonymous namespace
namespace {

// Read x[i] as T. Direct buffer access when source storage matches T;
// otherwise convert via elemAsDouble (with saturating cast for
// integers) — this branch only fires for the (rare) typeMatch=false
// dispatch error case, but kept for safety.
template <typename T>
inline T readSrcAsT(const Value &x, keel::usize i, bool typeMatch)
{
    if (typeMatch)
        return static_cast<const T *>(x.rawData())[i];
    if constexpr (std::is_floating_point_v<T>) {
        return static_cast<T>(x.elemAsDouble(i));
    } else {
        const keel::f64 d = x.elemAsDouble(i);
        return static_cast<T>(std::clamp(std::round(d),
            keel::as<keel::f64>(std::numeric_limits<T>::lowest()),
            keel::as<keel::f64>(std::numeric_limits<T>::max())));
    }
}

template <typename T>
inline Value makeScalarT(T v, ValueType outType, keel::Allocator *mr)
{
    if (outType == ValueType::DOUBLE)
        return Value::scalar(keel::as<keel::f64>(v), mr);
    if (outType == ValueType::LOGICAL)
        return Value::logicalScalar(v != 0, mr);
    auto r = Value::matrix(1, 1, outType, mr);
    static_cast<T *>(r.rawDataMut())[0] = v;
    return r;
}

// Walk every output cell along dim `redDim` (1-based). For each cell,
// gather the slice into `scratch`, find best via cmp, write best to
// dst[outIdx] and (1-based) source position to dstI[outIdx]. Handles
// 2D / 3D / ND uniformly via stride arithmetic.
template <typename T, typename Cmp>
void minMaxAlongDim(const Value &x, int redDim, T *dst, keel::f64 *dstI, Cmp cmp, bool typeMatch, keel::Allocator *mr)
{
    const auto &d = x.dims();
    const int redAxis = redDim - 1;
    const keel::usize sliceLen = d.dim(redAxis);
    keel::usize B = 1;
    for (int i = 0; i < redAxis; ++i) B *= d.dim(i);
    keel::usize O = 1;
    for (int i = redAxis + 1; i < d.ndim(); ++i) O *= d.dim(i);

    if (sliceLen == 0) return;  // empty slice — caller has set output to defaults

    keel::Array<T> scratch(sliceLen, mr);
    auto runSlice = [&](keel::usize outIdx, keel::usize baseOff, keel::usize stride) {
        for (keel::usize k = 0; k < sliceLen; ++k)
            scratch[k] = readSrcAsT<T>(x, baseOff + k * stride, typeMatch);
        T best = scratch[0];
        keel::usize bi = 0;
        for (keel::usize k = 1; k < sliceLen; ++k)
            if (cmp(scratch[k], best)) { best = scratch[k]; bi = k; }
        dst[outIdx] = best;
        dstI[outIdx] = keel::as<keel::f64>(bi + 1);
    };

    if (B == 1) {
        // Reducing axis 0 → contiguous gather (stride = 1).
        for (keel::usize o = 0; o < O; ++o)
            runSlice(o, o * sliceLen, 1);
        return;
    }
    for (keel::usize o = 0; o < O; ++o) {
        for (keel::usize b = 0; b < B; ++b) {
            const keel::usize base = o * sliceLen * B + b;
            const keel::usize outIdx = o * B + b;
            runSlice(outIdx, base, B);
        }
    }
}

// Construct the right-shaped (value, idx) output pair for reduction
// along `redDim` of x: value array has `outType`, idx array has DOUBLE.
inline std::pair<Value, Value>
allocMinMaxOutputs(const Value &x, int redDim, ValueType outType, keel::Allocator *mr)
{
    if (x.dims().ndim() >= 4 && redDim >= 1 && redDim <= x.dims().ndim()) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        auto shape = numkit::ops::outShapeForDimND(x, redDim, scratch.allocator());
        return {Value::matrixND(shape.data(), (int) shape.size(), outType, mr),
                Value::matrixND(shape.data(), (int) shape.size(), ValueType::DOUBLE, mr)};
    }
    auto outShape = numkit::ops::outShapeForDim(x, redDim);
    return {createMatrix(outShape, outType, mr),
            createMatrix(outShape, ValueType::DOUBLE, mr)};
}

// MATLAB: max/min of an EMPTY array returns empty and never errors. The
// result has the input's size with the operating dimension clamped to
// min(size,1):  max([])=0x0, max(zeros(0,3))=0x3, max(zeros(3,0))=1x0.
// The operating dim is the explicit dim when given (dimArg>=1), otherwise
// the first dimension whose size != 1 (MATLAB treats a size-0 dim as
// non-singleton here), defaulting to 1. Value keeps the input class; the
// index output is an empty DOUBLE of the same shape.
inline std::tuple<Value, Value>
emptyMinMaxResult(const Value &x, int dimArg, ValueType outType, keel::Allocator *mr)
{
    const auto &d = x.dims();
    int opDim = dimArg;
    if (opDim < 1) {
        opDim = 1;
        const int nd = d.ndim();
        for (int i = 0; i < nd; ++i)
            if (d.dim(i) != 1) { opDim = i + 1; break; }
    }
    auto clamp1 = [](keel::usize s) -> keel::usize { return s < 1 ? s : 1; };
    DimsArg o{d.rows(), d.cols(), d.is3D() ? d.pages() : 0};
    switch (opDim) {
        case 1: o.rows  = clamp1(o.rows); break;
        case 2: o.cols  = clamp1(o.cols); break;
        case 3: o.pages = (o.pages == 0) ? 0 : 1; break;
        default: break;
    }
    return {createMatrix(o, outType, mr),
            createMatrix(o, ValueType::DOUBLE, mr)};
}

template <typename T, typename Cmp>
std::tuple<Value, Value>
reduceMinMaxAllT(const Value &x, Cmp cmp, ValueType outType, keel::Allocator *mr)
{
    const bool typeMatch = (x.type() == outType);
    if (x.numel() == 0)
        throw std::runtime_error("min/max of empty array is not supported");
    if (x.isScalar() || x.dims().isVector()) {
        T best = readSrcAsT<T>(x, 0, typeMatch);
        keel::usize bi = 0;
        for (keel::usize i = 1; i < x.numel(); ++i) {
            const T v = readSrcAsT<T>(x, i, typeMatch);
            if (cmp(v, best)) { best = v; bi = i; }
        }
        return std::make_tuple(makeScalarT<T>(best, outType, mr),
                               Value::scalar(keel::as<keel::f64>(bi + 1), mr));
    }
    // Multi-dim: reduce along first non-singleton dim (MATLAB rule).
    const int redDim = numkit::ops::firstNonSingletonDim(x);
    auto [out, outIdx] = allocMinMaxOutputs(x, redDim, outType, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    minMaxAlongDim<T>(x, redDim, static_cast<T *>(out.rawDataMut()), outIdx.doubleDataMut(), cmp, typeMatch, scratch.allocator());
    return std::make_tuple(std::move(out), std::move(outIdx));
}

template <typename T, typename Cmp>
std::tuple<Value, Value>
reduceMinMaxAlongDimT(const Value &x, int dim, Cmp cmp, ValueType outType, keel::Allocator *mr)
{
    const bool typeMatch = (x.type() == outType);
    if (x.isScalar() || x.dims().isVector()) {
        // For vectors, MATLAB ignores explicit dim and reduces all elements
        // when dim == firstNonSingleton; otherwise (dim past ndim or singleton
        // dim) it returns the input unchanged with idx = ones.
        if (dim != numkit::ops::firstNonSingletonDim(x)) {
            // Identity reduction: copy x as outType (cast where needed) and
            // return ones as idx.
            const keel::usize n = x.numel();
            Value out, outIdx;
            if (x.dims().isVector()) {
                out    = createMatrix({x.dims().rows(), x.dims().cols(), 0}, outType, mr);
                outIdx = createMatrix({x.dims().rows(), x.dims().cols(), 0}, ValueType::DOUBLE, mr);
            } else {
                out    = makeScalarT<T>(readSrcAsT<T>(x, 0, typeMatch), outType, mr);
                outIdx = Value::scalar(1.0, mr);
                return std::make_tuple(std::move(out), std::move(outIdx));
            }
            T *dst = static_cast<T *>(out.rawDataMut());
            keel::f64 *dstI = outIdx.doubleDataMut();
            for (keel::usize i = 0; i < n; ++i) {
                dst[i]  = readSrcAsT<T>(x, i, typeMatch);
                dstI[i] = 1.0;
            }
            return std::make_tuple(std::move(out), std::move(outIdx));
        }
        // Reduce-all on a vector → scalar (matches the no-dim form).
        return reduceMinMaxAllT<T>(x, cmp, outType, mr);
    }
    auto [out, outIdx] = allocMinMaxOutputs(x, dim, outType, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    minMaxAlongDim<T>(x, dim, static_cast<T *>(out.rawDataMut()), outIdx.doubleDataMut(), cmp, typeMatch, scratch.allocator());
    return std::make_tuple(std::move(out), std::move(outIdx));
}

// COMPLEX min/max — MATLAB rule: compare by |z| (modulus) primary key,
// angle(z) (argument) secondary key for ties. Special case: when the
// entire input has all-zero imaginary parts, MATLAB falls back to
// comparing the real parts directly (so min([1 -3 2]) = -3, not the
// element with smallest |z|).
//
// Output value array is COMPLEX; index array is always DOUBLE (matches
// the round-3 typed reducers). The all-real check is one O(n) scan over
// the input — fast and amortised across the per-slice work.
inline bool allImagZero(const keel::c64 *data, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i)
        if (data[i].imag() != 0.0) return false;
    return true;
}

template <bool IsMax>
inline bool complexBetter(keel::c64 v, keel::c64 best, bool allReal)
{
    if (allReal) {
        if constexpr (IsMax) return v.real() > best.real();
        else                 return v.real() < best.real();
    }
    const keel::f64 absV = keel::abs(v), absB = keel::abs(best);
    if (absV != absB) {
        if constexpr (IsMax) return absV > absB;
        else                 return absV < absB;
    }
    const keel::f64 angV = keel::arg(v), angB = keel::arg(best);
    if constexpr (IsMax) return angV > angB;
    else                 return angV < angB;
}

// Binary (elementwise) complex min/max picker — MATLAB R2025b semantics
// (validated, bugs/math/maxmin-complex.md): compare by |z|, ties broken by
// angle(z). NO all-real fallback here (unlike the reduction form): MATLAB's
// binary max(complex(-3,0),1) is -3 (|−3|=3>1), so allReal is fixed false.
// NaN: a complex value with EITHER component NaN is "missing". omitnan (the
// max/min default) skips a missing operand (the other wins); both missing →
// the FIRST operand (MATLAB: max(complex(NaN,2),complex(1,NaN)) = NaN+2i).
// includenan: a missing operand propagates and wins, first operand first.
template <bool IsMax>
inline keel::c64 complexMinMaxPick(keel::c64 a, keel::c64 b, bool omitNan)
{
    const bool aNan = std::isnan(a.real()) || std::isnan(a.imag());
    const bool bNan = std::isnan(b.real()) || std::isnan(b.imag());
    if (omitNan) {
        if (aNan && bNan) return a;
        if (aNan)         return b;
        if (bNan)         return a;
    } else {
        if (aNan) return a;
        if (bNan) return b;
    }
    return complexBetter<IsMax>(a, b, /*allReal=*/false) ? a : b;
}

inline std::pair<Value, Value>
allocComplexMinMaxOutputs(const Value &x, int redDim, keel::Allocator *mr)
{
    if (x.dims().ndim() >= 4 && redDim >= 1 && redDim <= x.dims().ndim()) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        auto shape = numkit::ops::outShapeForDimND(x, redDim, scratch.allocator());
        return {Value::matrixND(shape.data(), (int) shape.size(), ValueType::COMPLEX, mr),
                Value::matrixND(shape.data(), (int) shape.size(), ValueType::DOUBLE,  mr)};
    }
    auto outShape = numkit::ops::outShapeForDim(x, redDim);
    return {createMatrix(outShape, ValueType::COMPLEX, mr),
            createMatrix(outShape, ValueType::DOUBLE,  mr)};
}

template <bool IsMax>
std::tuple<Value, Value>
reduceMinMaxComplexAll(const Value &x, keel::Allocator *mr, const char *fn)
{
    if (x.numel() == 0)
        throw Error(keel::format("{} of empty array is not supported", fn),
                     0, 0, fn, "", keel::format("numkit:{}:empty", fn));
    const keel::c64 *data = x.complexData();
    const bool allReal = allImagZero(data, x.numel());
    if (x.isScalar() || x.dims().isVector()) {
        keel::c64 best = data[0];
        keel::usize bi = 0;
        for (keel::usize i = 1; i < x.numel(); ++i)
            if (complexBetter<IsMax>(data[i], best, allReal)) { best = data[i]; bi = i; }
        return std::make_tuple(Value::complexScalar(best, mr),
                               Value::scalar(keel::as<keel::f64>(bi + 1), mr));
    }
    const int redDim = numkit::ops::firstNonSingletonDim(x);
    auto [out, outIdx] = allocComplexMinMaxOutputs(x, redDim, mr);
    keel::c64 *dst  = out.complexDataMut();
    keel::f64  *dstI = outIdx.doubleDataMut();

    const auto &d = x.dims();
    const int redAxis = redDim - 1;
    const keel::usize sliceLen = d.dim(redAxis);
    keel::usize B = 1;
    for (int i = 0; i < redAxis; ++i) B *= d.dim(i);
    keel::usize O = 1;
    for (int i = redAxis + 1; i < d.ndim(); ++i) O *= d.dim(i);

    auto runSlice = [&](keel::usize outOff, keel::usize baseOff, keel::usize stride) {
        keel::c64 best = data[baseOff];
        keel::usize bi = 0;
        for (keel::usize k = 1; k < sliceLen; ++k) {
            const keel::c64 v = data[baseOff + k * stride];
            if (complexBetter<IsMax>(v, best, allReal)) { best = v; bi = k; }
        }
        dst[outOff] = best;
        dstI[outOff] = keel::as<keel::f64>(bi + 1);
    };
    if (B == 1) {
        for (keel::usize o = 0; o < O; ++o) runSlice(o, o * sliceLen, 1);
    } else {
        for (keel::usize o = 0; o < O; ++o)
            for (keel::usize b = 0; b < B; ++b)
                runSlice(o * B + b, o * sliceLen * B + b, B);
    }
    return std::make_tuple(std::move(out), std::move(outIdx));
}

template <bool IsMax>
std::tuple<Value, Value>
reduceMinMaxComplexAlongDim(const Value &x, int dim, keel::Allocator *mr, const char *fn)
{
    if (x.isScalar() || x.dims().isVector()) {
        if (dim != numkit::ops::firstNonSingletonDim(x)) {
            // Identity reduction: copy x as COMPLEX, idx = ones.
            const keel::usize n = x.numel();
            Value out, outIdx;
            if (x.dims().isVector()) {
                out    = createMatrix({x.dims().rows(), x.dims().cols(), 0}, ValueType::COMPLEX, mr);
                outIdx = createMatrix({x.dims().rows(), x.dims().cols(), 0}, ValueType::DOUBLE,  mr);
            } else {
                out    = Value::complexScalar(x.complexData()[0], mr);
                outIdx = Value::scalar(1.0, mr);
                return std::make_tuple(std::move(out), std::move(outIdx));
            }
            keel::c64 *dst = out.complexDataMut();
            keel::f64  *dstI = outIdx.doubleDataMut();
            const keel::c64 *src = x.complexData();
            for (keel::usize i = 0; i < n; ++i) { dst[i] = src[i]; dstI[i] = 1.0; }
            return std::make_tuple(std::move(out), std::move(outIdx));
        }
        return reduceMinMaxComplexAll<IsMax>(x, mr, fn);
    }
    const keel::c64 *data = x.complexData();
    const bool allReal = allImagZero(data, x.numel());
    auto [out, outIdx] = allocComplexMinMaxOutputs(x, dim, mr);
    keel::c64 *dst  = out.complexDataMut();
    keel::f64  *dstI = outIdx.doubleDataMut();

    const auto &d = x.dims();
    const int redAxis = dim - 1;
    const keel::usize sliceLen = d.dim(redAxis);
    keel::usize B = 1;
    for (int i = 0; i < redAxis; ++i) B *= d.dim(i);
    keel::usize O = 1;
    for (int i = redAxis + 1; i < d.ndim(); ++i) O *= d.dim(i);

    auto runSlice = [&](keel::usize outOff, keel::usize baseOff, keel::usize stride) {
        keel::c64 best = data[baseOff];
        keel::usize bi = 0;
        for (keel::usize k = 1; k < sliceLen; ++k) {
            const keel::c64 v = data[baseOff + k * stride];
            if (complexBetter<IsMax>(v, best, allReal)) { best = v; bi = k; }
        }
        dst[outOff] = best;
        dstI[outOff] = keel::as<keel::f64>(bi + 1);
    };
    if (B == 1) {
        for (keel::usize o = 0; o < O; ++o) runSlice(o, o * sliceLen, 1);
    } else {
        for (keel::usize o = 0; o < O; ++o)
            for (keel::usize b = 0; b < B; ++b)
                runSlice(o * B + b, o * sliceLen * B + b, B);
    }
    return std::make_tuple(std::move(out), std::move(outIdx));
}

// Forward declarations for the regular dispatchers (defined further
// down). Needed because the nan-aware dispatchers fall through to
// these for integer types — clang's strict two-phase lookup requires
// the declaration to be visible at the point of (template) reference.
template <bool IsMax, typename Cmp>
std::tuple<Value, Value>
dispatchMinMaxAll(const Value &x, Cmp cmp, keel::Allocator *mr, const char *fn);

template <bool IsMax, typename Cmp>
std::tuple<Value, Value>
dispatchMinMaxAlongDim(const Value &x, int dim, Cmp cmp, keel::Allocator *mr, const char *fn);

// ── NaN-aware min/max (omitnan flag) ─────────────────────────────────
//
// Per-ValueType dispatch identical to round-3 minMaxAlongDim, but skips
// NaN values during comparison. For floating types (DOUBLE/SINGLE) and
// COMPLEX (NaN if real or imag is NaN), filter out NaNs. For integer
// types, NaN can't occur so the regular kernel is used.
//
// All-NaN slice handling: the value array is set to NaN (for float/
// complex) and the index array to 1 (matching MATLAB convention).

template <typename T>
inline bool isElemNan(T v)
{
    if constexpr (std::is_floating_point_v<T>) return std::isnan(v);
    else                                       return false;
}

template <typename T, typename Cmp>
void minMaxNanAlongDim(const Value &x, int redDim, T *dst, keel::f64 *dstI,
                       Cmp cmp, bool typeMatch)
{
    const auto &d = x.dims();
    const int redAxis = redDim - 1;
    const keel::usize sliceLen = d.dim(redAxis);
    keel::usize B = 1;
    for (int i = 0; i < redAxis; ++i) B *= d.dim(i);
    keel::usize O = 1;
    for (int i = redAxis + 1; i < d.ndim(); ++i) O *= d.dim(i);

    if (sliceLen == 0) return;

    auto runSlice = [&](keel::usize outIdx, keel::usize baseOff, keel::usize stride) {
        keel::usize firstValid = SIZE_MAX;
        T best{};
        for (keel::usize k = 0; k < sliceLen; ++k) {
            const T v = readSrcAsT<T>(x, baseOff + k * stride, typeMatch);
            if (isElemNan(v)) continue;
            if (firstValid == SIZE_MAX) {
                best = v; firstValid = k;
            } else if (cmp(v, best)) {
                best = v; firstValid = k;
            }
        }
        if (firstValid == SIZE_MAX) {
            // All-NaN slice: NaN value, idx = 1 (MATLAB convention).
            if constexpr (std::is_floating_point_v<T>)
                dst[outIdx] = std::numeric_limits<T>::quiet_NaN();
            else
                dst[outIdx] = T{};
            dstI[outIdx] = 1.0;
        } else {
            dst[outIdx] = best;
            dstI[outIdx] = keel::as<keel::f64>(firstValid + 1);
        }
    };
    if (B == 1) {
        for (keel::usize o = 0; o < O; ++o) runSlice(o, o * sliceLen, 1);
        return;
    }
    for (keel::usize o = 0; o < O; ++o)
        for (keel::usize b = 0; b < B; ++b)
            runSlice(o * B + b, o * sliceLen * B + b, B);
}

template <typename T, typename Cmp>
std::tuple<Value, Value>
reduceMinMaxNanAllT(const Value &x, Cmp cmp, ValueType outType, keel::Allocator *mr)
{
    if (x.numel() == 0)
        throw std::runtime_error("min/max of empty array is not supported");
    const bool typeMatch = (x.type() == outType);
    if (x.isScalar() || x.dims().isVector()) {
        keel::usize firstValid = SIZE_MAX;
        T best{};
        for (keel::usize i = 0; i < x.numel(); ++i) {
            const T v = readSrcAsT<T>(x, i, typeMatch);
            if (isElemNan(v)) continue;
            if (firstValid == SIZE_MAX) { best = v; firstValid = i; }
            else if (cmp(v, best))      { best = v; firstValid = i; }
        }
        if (firstValid == SIZE_MAX) {
            if constexpr (std::is_floating_point_v<T>)
                return std::make_tuple(makeScalarT<T>(std::numeric_limits<T>::quiet_NaN(), outType, mr),
                                       Value::scalar(1.0, mr));
            else
                return std::make_tuple(makeScalarT<T>(T{}, outType, mr),
                                       Value::scalar(1.0, mr));
        }
        return std::make_tuple(makeScalarT<T>(best, outType, mr),
                               Value::scalar(keel::as<keel::f64>(firstValid + 1), mr));
    }
    const int redDim = numkit::ops::firstNonSingletonDim(x);
    auto [out, outIdx] = allocMinMaxOutputs(x, redDim, outType, mr);
    minMaxNanAlongDim<T>(x, redDim,
                         static_cast<T *>(out.rawDataMut()),
                         outIdx.doubleDataMut(),
                         cmp, typeMatch);
    return std::make_tuple(std::move(out), std::move(outIdx));
}

template <typename T, typename Cmp>
std::tuple<Value, Value>
reduceMinMaxNanAlongDimT(const Value &x, int dim, Cmp cmp, ValueType outType, keel::Allocator *mr)
{
    const bool typeMatch = (x.type() == outType);
    if (x.isScalar() || x.dims().isVector()) {
        if (dim != numkit::ops::firstNonSingletonDim(x))
            return reduceMinMaxAlongDimT<T>(x, dim, cmp, outType, mr);
        return reduceMinMaxNanAllT<T>(x, cmp, outType, mr);
    }
    auto [out, outIdx] = allocMinMaxOutputs(x, dim, outType, mr);
    minMaxNanAlongDim<T>(x, dim,
                         static_cast<T *>(out.rawDataMut()),
                         outIdx.doubleDataMut(),
                         cmp, typeMatch);
    return std::make_tuple(std::move(out), std::move(outIdx));
}

// COMPLEX nan-aware min/max — same |z|+angle comparator, but skips
// elements where either real or imag part is NaN.
template <bool IsMax>
std::tuple<Value, Value>
reduceMinMaxComplexNanAll(const Value &x, keel::Allocator *mr, const char *fn)
{
    if (x.numel() == 0)
        throw Error(keel::format("{} of empty array is not supported", fn),
                     0, 0, fn, "", keel::format("numkit:{}:empty", fn));
    const keel::c64 *data = x.complexData();
    auto isNan = [](keel::c64 c) { return std::isnan(c.real()) || std::isnan(c.imag()); };
    // For all-real check, only consider non-NaN elements.
    bool allReal = true;
    for (keel::usize i = 0; i < x.numel(); ++i) {
        if (isNan(data[i])) continue;
        if (data[i].imag() != 0.0) { allReal = false; break; }
    }
    auto findBest = [&](keel::usize baseOff, keel::usize stride, keel::usize n,
                        keel::c64 &best, keel::usize &bi) {
        keel::usize firstValid = SIZE_MAX;
        for (keel::usize k = 0; k < n; ++k) {
            const keel::c64 v = data[baseOff + k * stride];
            if (isNan(v)) continue;
            if (firstValid == SIZE_MAX) {
                best = v; bi = k; firstValid = k;
            } else if (complexBetter<IsMax>(v, best, allReal)) {
                best = v; bi = k; firstValid = k;
            }
        }
        return firstValid != SIZE_MAX;
    };

    if (x.isScalar() || x.dims().isVector()) {
        keel::c64 best;
        keel::usize bi = 0;
        if (!findBest(0, 1, x.numel(), best, bi))
            return std::make_tuple(Value::complexScalar(keel::c64(std::nan(""), 0.0), mr),
                                   Value::scalar(1.0, mr));
        return std::make_tuple(Value::complexScalar(best, mr),
                               Value::scalar(keel::as<keel::f64>(bi + 1), mr));
    }
    const int redDim = numkit::ops::firstNonSingletonDim(x);
    auto [out, outIdx] = allocComplexMinMaxOutputs(x, redDim, mr);
    keel::c64 *dst = out.complexDataMut();
    keel::f64 *dstI = outIdx.doubleDataMut();

    const auto &d = x.dims();
    const int redAxis = redDim - 1;
    const keel::usize sliceLen = d.dim(redAxis);
    keel::usize B = 1;
    for (int i = 0; i < redAxis; ++i) B *= d.dim(i);
    keel::usize O = 1;
    for (int i = redAxis + 1; i < d.ndim(); ++i) O *= d.dim(i);

    auto runSlice = [&](keel::usize outOff, keel::usize baseOff, keel::usize stride) {
        keel::c64 best;
        keel::usize bi = 0;
        if (!findBest(baseOff, stride, sliceLen, best, bi)) {
            dst[outOff] = keel::c64(std::nan(""), 0.0);
            dstI[outOff] = 1.0;
        } else {
            dst[outOff] = best;
            dstI[outOff] = keel::as<keel::f64>(bi + 1);
        }
    };
    if (B == 1) {
        for (keel::usize o = 0; o < O; ++o) runSlice(o, o * sliceLen, 1);
    } else {
        for (keel::usize o = 0; o < O; ++o)
            for (keel::usize b = 0; b < B; ++b)
                runSlice(o * B + b, o * sliceLen * B + b, B);
    }
    return std::make_tuple(std::move(out), std::move(outIdx));
}

template <bool IsMax>
std::tuple<Value, Value>
reduceMinMaxComplexNanAlongDim(const Value &x, int dim, keel::Allocator *mr, const char *fn)
{
    if (x.isScalar() || x.dims().isVector()) {
        if (dim != numkit::ops::firstNonSingletonDim(x))
            return reduceMinMaxComplexAlongDim<IsMax>(x, dim, mr, fn);
        return reduceMinMaxComplexNanAll<IsMax>(x, mr, fn);
    }
    const keel::c64 *data = x.complexData();
    auto isNan = [](keel::c64 c) { return std::isnan(c.real()) || std::isnan(c.imag()); };
    bool allReal = true;
    for (keel::usize i = 0; i < x.numel(); ++i) {
        if (isNan(data[i])) continue;
        if (data[i].imag() != 0.0) { allReal = false; break; }
    }
    auto [out, outIdx] = allocComplexMinMaxOutputs(x, dim, mr);
    keel::c64 *dst = out.complexDataMut();
    keel::f64 *dstI = outIdx.doubleDataMut();

    const auto &d = x.dims();
    const int redAxis = dim - 1;
    const keel::usize sliceLen = d.dim(redAxis);
    keel::usize B = 1;
    for (int i = 0; i < redAxis; ++i) B *= d.dim(i);
    keel::usize O = 1;
    for (int i = redAxis + 1; i < d.ndim(); ++i) O *= d.dim(i);

    auto runSlice = [&](keel::usize outOff, keel::usize baseOff, keel::usize stride) {
        keel::usize firstValid = SIZE_MAX;
        keel::c64 best;
        keel::usize bi = 0;
        for (keel::usize k = 0; k < sliceLen; ++k) {
            const keel::c64 v = data[baseOff + k * stride];
            if (isNan(v)) continue;
            if (firstValid == SIZE_MAX) { best = v; bi = k; firstValid = k; }
            else if (complexBetter<IsMax>(v, best, allReal)) { best = v; bi = k; firstValid = k; }
        }
        if (firstValid == SIZE_MAX) {
            dst[outOff] = keel::c64(std::nan(""), 0.0);
            dstI[outOff] = 1.0;
        } else {
            dst[outOff] = best;
            dstI[outOff] = keel::as<keel::f64>(bi + 1);
        }
    };
    if (B == 1) {
        for (keel::usize o = 0; o < O; ++o) runSlice(o, o * sliceLen, 1);
    } else {
        for (keel::usize o = 0; o < O; ++o)
            for (keel::usize b = 0; b < B; ++b)
                runSlice(o * B + b, o * sliceLen * B + b, B);
    }
    return std::make_tuple(std::move(out), std::move(outIdx));
}

template <bool IsMax, typename Cmp>
std::tuple<Value, Value>
dispatchMinMaxNanAll(const Value &x, Cmp cmp, keel::Allocator *mr, const char *fn)
{
    if (x.numel() == 0) return emptyMinMaxResult(x, -1, x.type(), mr);
    switch (x.type()) {
    case ValueType::DOUBLE:  return reduceMinMaxNanAllT<keel::f64>(x, cmp, ValueType::DOUBLE, mr);
    case ValueType::SINGLE:  return reduceMinMaxNanAllT<keel::f32 >(x, cmp, ValueType::SINGLE, mr);
    case ValueType::COMPLEX: return reduceMinMaxComplexNanAll<IsMax>(x, mr, fn);
    // Integer/logical/char have no NaN — fall through to the regular path.
    default: return dispatchMinMaxAll<IsMax>(x, cmp, mr, fn);
    }
}

template <bool IsMax, typename Cmp>
std::tuple<Value, Value>
dispatchMinMaxNanAlongDim(const Value &x, int dim, Cmp cmp,
                          keel::Allocator *mr, const char *fn)
{
    if (x.numel() == 0) return emptyMinMaxResult(x, dim, x.type(), mr);
    switch (x.type()) {
    case ValueType::DOUBLE:  return reduceMinMaxNanAlongDimT<keel::f64>(x, dim, cmp, ValueType::DOUBLE, mr);
    case ValueType::SINGLE:  return reduceMinMaxNanAlongDimT<keel::f32 >(x, dim, cmp, ValueType::SINGLE, mr);
    case ValueType::COMPLEX: return reduceMinMaxComplexNanAlongDim<IsMax>(x, dim, mr, fn);
    default: return dispatchMinMaxAlongDim<IsMax>(x, dim, cmp, mr, fn);
    }
}

// Dispatch on x.type(), instantiate reducer with right T/outType pair.
// LOGICAL maps to T=uint8_t (storage type). CHAR maps to T=char.
// COMPLEX uses the |z|-then-angle comparator (MATLAB rule).
template <bool IsMax, typename Cmp>
std::tuple<Value, Value>
dispatchMinMaxAll(const Value &x, Cmp cmp, keel::Allocator *mr, const char *fn)
{
    if (x.numel() == 0) return emptyMinMaxResult(x, -1, x.type(), mr);
    switch (x.type()) {
    case ValueType::DOUBLE:  return reduceMinMaxAllT<keel::f64  >(x, cmp, ValueType::DOUBLE,  mr);
    case ValueType::SINGLE:  return reduceMinMaxAllT<keel::f32   >(x, cmp, ValueType::SINGLE,  mr);
    case ValueType::INT8:    return reduceMinMaxAllT<keel::i8  >(x, cmp, ValueType::INT8,    mr);
    case ValueType::INT16:   return reduceMinMaxAllT<keel::i16 >(x, cmp, ValueType::INT16,   mr);
    case ValueType::INT32:   return reduceMinMaxAllT<keel::i32 >(x, cmp, ValueType::INT32,   mr);
    case ValueType::INT64:   return reduceMinMaxAllT<keel::i64 >(x, cmp, ValueType::INT64,   mr);
    case ValueType::UINT8:   return reduceMinMaxAllT<keel::u8 >(x, cmp, ValueType::UINT8,   mr);
    case ValueType::UINT16:  return reduceMinMaxAllT<keel::u16>(x, cmp, ValueType::UINT16,  mr);
    case ValueType::UINT32:  return reduceMinMaxAllT<keel::u32>(x, cmp, ValueType::UINT32,  mr);
    case ValueType::UINT64:  return reduceMinMaxAllT<keel::u64>(x, cmp, ValueType::UINT64,  mr);
    case ValueType::LOGICAL: return reduceMinMaxAllT<keel::u8 >(x, cmp, ValueType::LOGICAL, mr);
    case ValueType::CHAR:    return reduceMinMaxAllT<char    >(x, cmp, ValueType::CHAR,    mr);
    case ValueType::COMPLEX: return reduceMinMaxComplexAll<IsMax>(x, mr, fn);
    default:
        throw Error(keel::format("{}: unsupported input type", fn),
                     0, 0, fn, "", keel::format("numkit:{}:type", fn));
    }
}

template <bool IsMax, typename Cmp>
std::tuple<Value, Value>
dispatchMinMaxAlongDim(const Value &x, int dim, Cmp cmp, keel::Allocator *mr, const char *fn)
{
    if (x.numel() == 0) return emptyMinMaxResult(x, dim, x.type(), mr);
    switch (x.type()) {
    case ValueType::DOUBLE:  return reduceMinMaxAlongDimT<keel::f64  >(x, dim, cmp, ValueType::DOUBLE,  mr);
    case ValueType::SINGLE:  return reduceMinMaxAlongDimT<keel::f32   >(x, dim, cmp, ValueType::SINGLE,  mr);
    case ValueType::INT8:    return reduceMinMaxAlongDimT<keel::i8  >(x, dim, cmp, ValueType::INT8,    mr);
    case ValueType::INT16:   return reduceMinMaxAlongDimT<keel::i16 >(x, dim, cmp, ValueType::INT16,   mr);
    case ValueType::INT32:   return reduceMinMaxAlongDimT<keel::i32 >(x, dim, cmp, ValueType::INT32,   mr);
    case ValueType::INT64:   return reduceMinMaxAlongDimT<keel::i64 >(x, dim, cmp, ValueType::INT64,   mr);
    case ValueType::UINT8:   return reduceMinMaxAlongDimT<keel::u8 >(x, dim, cmp, ValueType::UINT8,   mr);
    case ValueType::UINT16:  return reduceMinMaxAlongDimT<keel::u16>(x, dim, cmp, ValueType::UINT16,  mr);
    case ValueType::UINT32:  return reduceMinMaxAlongDimT<keel::u32>(x, dim, cmp, ValueType::UINT32,  mr);
    case ValueType::UINT64:  return reduceMinMaxAlongDimT<keel::u64>(x, dim, cmp, ValueType::UINT64,  mr);
    case ValueType::LOGICAL: return reduceMinMaxAlongDimT<keel::u8 >(x, dim, cmp, ValueType::LOGICAL, mr);
    case ValueType::CHAR:    return reduceMinMaxAlongDimT<char    >(x, dim, cmp, ValueType::CHAR,    mr);
    case ValueType::COMPLEX: return reduceMinMaxComplexAlongDim<IsMax>(x, dim, mr, fn);
    default:
        throw Error(keel::format("{}: unsupported input type", fn),
                     0, 0, fn, "", keel::format("numkit:{}:type", fn));
    }
}

} // anonymous namespace

// nan-omitting max/min workers (defs in reductions.cpp, external).
std::tuple<Value, Value> maxOmitNan(const Value &x, int dim, keel::Allocator *mr);
std::tuple<Value, Value> minOmitNan(const Value &x, int dim, keel::Allocator *mr);
Value maxOmitNanBinary(const Value &a, const Value &b, keel::Allocator *mr);
Value minOmitNanBinary(const Value &a, const Value &b, keel::Allocator *mr);

} // namespace numkit::builtin
