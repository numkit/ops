// toolboxes/builtin/src/reduction_helpers.hpp
//
// Shared infrastructure for "reduce-along-dim" style operations.
// Used by Phase-1 stats (var/std/median/quantile/prctile/mode) and
// any future reductions that want explicit `dim` support.
//
// Two layouts handled:
//   * 2D matrix (R×C, column-major): dim=1 reduces rows, dim=2 reduces cols
//   * 3D array  (R×C×P, page = R*C stride): dim=1/2/3 reduces that axis
//
// Vectors and scalars: ignore the explicit dim and reduce all elements
// (matching MATLAB's "first non-singleton" behavior for vectors).
//
// API: applyAlongDim(x, dim, F) collects each slice into a contiguous
// scratch buffer and calls F(scratch.data(), n, out_index) → double.
// `out_index` is the linear index into the output buffer, useful when
// F needs to write multiple outputs per slice (e.g. mode wants both
// value and count — but we expose that via a separate API).

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <numkit/ops/value_factory.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace numkit::ops {

// Return MATLAB's "first non-singleton dim" (1-based). For an all-
// singleton input (scalar) returns 1. Used when the user omits `dim`.
inline int firstNonSingletonDim(const Value &x)
{
    const auto &d = x.dims();
    const int nd = d.ndim();
    for (int i = 0; i < nd; ++i)
        if (d.dim(i) > 1) return i + 1;
    return 1;
}

// Validate user-supplied dim. dim must be a positive integer; dim
// past the input's actual ndim is treated as a trailing-singleton
// (returns sentinel = ndim+1 so callers know to do an identity copy).
inline int validateDim(const Value &x, int dim, const char *fn)
{
    if (dim < 1)
        throw Error(std::string(fn) + ": dim must be a positive integer",
                     0, 0, fn, "", std::string("numkit:") + fn + ":badDim");
    const int nd = x.dims().ndim();
    if (dim > nd) {
        // Trailing-singleton semantics — caller produces an identity copy.
        return nd + 1;
    }
    return dim;
}

// Compute output shape for a reduce-along-dim operation. The reduced
// dim becomes 1; other dims preserved.
inline DimsArg outShapeForDim(const Value &x, int dim)
{
    const auto &d = x.dims();
    DimsArg o{d.rows(), d.cols(), d.is3D() ? d.pages() : 0};
    switch (dim) {
        case 1: o.rows = 1; break;
        case 2: o.cols = 1; break;
        case 3: o.pages = (o.pages == 0) ? 0 : 1; break;
        default: break; // sentinel: no reduction (handled by caller)
    }
    return o;
}

// Shape of one slice (count of elements traversed when collapsing dim).
inline keel::usize sliceLenForDim(const Value &x, int dim)
{
    const auto &d = x.dims();
    switch (dim) {
        case 1: return d.rows();
        case 2: return d.cols();
        case 3: return d.is3D() ? d.pages() : 1;
        default: return 1;
    }
}

// Number of independent slices we'll produce.
inline keel::usize sliceCountForDim(const Value &x, int dim)
{
    return x.numel() / sliceLenForDim(x, dim);
}

// Iterate every (output_index, slice_data) pair. F receives a pointer
// to a contiguous buffer of length sliceLenForDim and writes outputs
// via `outBuf[outIdx] = …`. Non-DOUBLE inputs are read via
// elemAsDouble (DOUBLE keeps the contiguous-memcpy fast path).
//
// `mr` backs the per-call scratch buffer reused across slices. Pass
// the same memory_resource you used for any sibling scratches in the
// caller (typically `arena.resource()` from a per-call ScratchArena).
//
// NOTE: F is called once per output element. For multi-output ops
// (like mode returning value+count) the caller must allocate two
// MValues and capture both buffers in F's closure.
template <typename F>
void forEachSlice(const Value &x, int dim, F &&f,
                  keel::Allocator *mr = nullptr)
{
    const auto &d = x.dims();
    const keel::usize R = d.rows(), C = d.cols(), P = d.is3D() ? d.pages() : 1;
    const keel::usize N = sliceLenForDim(x, dim);
    const bool fastDouble = (x.type() == ValueType::DOUBLE);
    const keel::f64 *src = fastDouble ? x.doubleData() : nullptr;

    keel::Array<keel::f64> scratch(N, mr);

    if (dim == 1) {
        // Slice = a column (or column within a page). Stride = 1.
        keel::usize outIdx = 0;
        for (keel::usize pp = 0; pp < P; ++pp) {
            for (keel::usize c = 0; c < C; ++c) {
                const keel::usize base = pp * R * C + c * R;
                if (fastDouble) {
                    std::copy(src + base, src + base + N, scratch.data());
                } else {
                    for (keel::usize k = 0; k < N; ++k)
                        scratch[k] = x.elemAsDouble(base + k);
                }
                f(outIdx, scratch.data(), N);
                ++outIdx;
            }
        }
    } else if (dim == 2) {
        // Slice = a row (within a page). Stride = R.
        keel::usize outIdx = 0;
        for (keel::usize pp = 0; pp < P; ++pp) {
            for (keel::usize r = 0; r < R; ++r) {
                if (fastDouble) {
                    for (keel::usize c = 0; c < C; ++c)
                        scratch[c] = src[pp * R * C + c * R + r];
                } else {
                    for (keel::usize c = 0; c < C; ++c)
                        scratch[c] = x.elemAsDouble(pp * R * C + c * R + r);
                }
                f(outIdx, scratch.data(), N);
                ++outIdx;
            }
        }
    } else if (dim == 3) {
        // Slice = pages at fixed (r,c). Stride = R*C. Only meaningful for 3D.
        keel::usize outIdx = 0;
        for (keel::usize c = 0; c < C; ++c) {
            for (keel::usize r = 0; r < R; ++r) {
                if (fastDouble) {
                    for (keel::usize pp = 0; pp < P; ++pp)
                        scratch[pp] = src[pp * R * C + c * R + r];
                } else {
                    for (keel::usize pp = 0; pp < P; ++pp)
                        scratch[pp] = x.elemAsDouble(pp * R * C + c * R + r);
                }
                f(outIdx, scratch.data(), N);
                ++outIdx;
            }
        }
    }
}

// ND output-shape helper (any rank): copies all input dims, sets the
// reduced dim to 1. dim is 1-based; dim > ndim means trailing
// singleton (identity reduction) — caller handles separately. The
// caller threads in `mr` (typically the same arena that backs the
// reduction's per-slice scratch) so the shape buffer never hits the
// global heap.
inline keel::Array<keel::usize> outShapeForDimND(const Value &x, int dim,
                                            keel::Allocator *mr = nullptr)
{
    const auto &d = x.dims();
    keel::Array<keel::usize> shape(d.ndim(), mr);
    for (int i = 0; i < d.ndim(); ++i)
        shape[i] = (i + 1 == dim) ? 1 : d.dim(i);
    return shape;
}

// ND apply core: walks each axis-`redAxis` slice via stride arithmetic
// and feeds it to F. Output is dense column-major matching the
// reduced-dim shape. Output is always DOUBLE (matches MATLAB convention
// for sum/mean/etc.); non-DOUBLE input is read via elemAsDouble so
// integer / single / logical inputs reduce correctly.
//
// `mr` backs the per-call scratch buffer (see forEachSlice).
template <typename F>
void forEachSliceND(const Value &x, int dim, keel::f64 *dst, F &&f,
                    keel::Allocator *mr = nullptr)
{
    const auto &d = x.dims();
    const int redAxis = dim - 1;
    const keel::usize sliceLen = d.dim(redAxis);
    keel::usize B = 1;
    for (int i = 0; i < redAxis; ++i) B *= d.dim(i);
    keel::usize O = 1;
    for (int i = redAxis + 1; i < d.ndim(); ++i) O *= d.dim(i);

    const bool fastDouble = (x.type() == ValueType::DOUBLE);
    const keel::f64 *src = fastDouble ? x.doubleData() : nullptr;
    keel::Array<keel::f64> scratch(sliceLen, mr);
    if (B == 1) {
        // Reducing axis 0 → contiguous gather.
        if (fastDouble) {
            for (keel::usize o = 0; o < O; ++o) {
                const keel::f64 *base = src + o * sliceLen;
                std::copy(base, base + sliceLen, scratch.data());
                dst[o] = f(o, scratch.data(), sliceLen);
            }
        } else {
            for (keel::usize o = 0; o < O; ++o) {
                const keel::usize base = o * sliceLen;
                for (keel::usize k = 0; k < sliceLen; ++k)
                    scratch[k] = x.elemAsDouble(base + k);
                dst[o] = f(o, scratch.data(), sliceLen);
            }
        }
        return;
    }
    for (keel::usize o = 0; o < O; ++o) {
        for (keel::usize b = 0; b < B; ++b) {
            const keel::usize base = o * sliceLen * B + b;
            if (fastDouble) {
                for (keel::usize k = 0; k < sliceLen; ++k)
                    scratch[k] = src[base + k * B];
            } else {
                for (keel::usize k = 0; k < sliceLen; ++k)
                    scratch[k] = x.elemAsDouble(base + k * B);
            }
            const keel::usize outIdx = o * B + b;
            dst[outIdx] = f(outIdx, scratch.data(), sliceLen);
        }
    }
}

// applyAlongDim: most common case — F maps a slice to a single double.
// Returns an Value with the dim collapsed to 1. dim==-1 means "reduce
// all elements to a single scalar" (vector/scalar case).
template <typename F>
Value applyAlongDim(const Value &x, int dim, F &&f, keel::Allocator *mr = nullptr)
{
    // 2D/3D empty: collapse to 0×0 (legacy behaviour; MATLAB's
    // shape-preservation for empty 2D inputs was never wired up here).
    // ND empty (rank ≥ 4): fall through to the ND path, which produces
    // a correctly-shaped output (e.g. sum(zeros([2,0,3,2]), 2) → 2×1×3×2).
    if (x.isEmpty() && x.dims().ndim() < 4) {
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    }
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (x.dims().isVector() || x.isScalar()) {
        auto buf = keel::Array<keel::f64>(x.numel(), scratch.allocator());
        if (x.type() == ValueType::DOUBLE)
            std::copy(x.doubleData(), x.doubleData() + x.numel(), buf.data());
        else
            for (keel::usize i = 0; i < x.numel(); ++i)
                buf[i] = x.elemAsDouble(i);
        return Value::scalar(f(0, buf.data(), x.numel()), mr);
    }
    // ND fallback for rank ≥ 4. dim out-of-range was already mapped by
    // validateDim → ndim+1 sentinel, but we may receive dim ∈ [1, ndim].
    if (x.dims().ndim() >= 4 && dim >= 1 && dim <= x.dims().ndim()) {
        auto shape = outShapeForDimND(x, dim, scratch.allocator());
        Value out = Value::matrixND(shape.data(),
                                      static_cast<int>(shape.size()),
                                      ValueType::DOUBLE, mr);
        keel::f64 *dst = out.doubleDataMut();
        forEachSliceND(x, dim, dst, std::forward<F>(f), scratch.allocator());
        return out;
    }
    auto outShape = outShapeForDim(x, dim);
    Value out = createMatrix(outShape, ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();
    forEachSlice(x, dim, [&](keel::usize outIdx, keel::f64 *slice, keel::usize n) {
        dst[outIdx] = f(outIdx, slice, n);
    }, scratch.allocator());
    return out;
}

// Index-aware variant: F writes (value, 1-based index of source element)
// per slice. Used by min/max with dim. Returns (values, indices).
template <typename F>
std::pair<Value, Value>
applyAlongDimWithIndex(const Value &x, int dim, F &&f, keel::Allocator *mr = nullptr)
{
    if (x.isEmpty() && x.dims().ndim() < 4) {
        return {Value::matrix(0, 0, ValueType::DOUBLE, mr),
                Value::matrix(0, 0, ValueType::DOUBLE, mr)};
    }
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (x.dims().isVector() || x.isScalar()) {
        auto buf = keel::Array<keel::f64>(x.numel(), scratch.allocator());
        if (x.type() == ValueType::DOUBLE)
            std::copy(x.doubleData(), x.doubleData() + x.numel(), buf.data());
        else
            for (keel::usize i = 0; i < x.numel(); ++i)
                buf[i] = x.elemAsDouble(i);
        keel::f64 v = 0;
        keel::usize idx = 0;
        f(buf.data(), x.numel(), v, idx);
        return {Value::scalar(v, mr),
                Value::scalar(static_cast<keel::f64>(idx + 1), mr)};
    }
    // ND fallback (rank ≥ 4)
    if (x.dims().ndim() >= 4 && dim >= 1 && dim <= x.dims().ndim()) {
        auto shape = outShapeForDimND(x, dim, scratch.allocator());
        Value out    = Value::matrixND(shape.data(),
                                         static_cast<int>(shape.size()),
                                         ValueType::DOUBLE, mr);
        Value outIdx = Value::matrixND(shape.data(),
                                         static_cast<int>(shape.size()),
                                         ValueType::DOUBLE, mr);
        keel::f64 *dst  = out.doubleDataMut();
        keel::f64 *dstI = outIdx.doubleDataMut();
        forEachSliceND(x, dim, dst,
            [&](keel::usize outOff, keel::f64 *slice, keel::usize n) {
                keel::f64 v = 0; keel::usize idx = 0;
                f(slice, n, v, idx);
                dstI[outOff] = static_cast<keel::f64>(idx + 1);
                return v;
            }, scratch.allocator());
        return {std::move(out), std::move(outIdx)};
    }
    auto outShape = outShapeForDim(x, dim);
    Value out = createMatrix(outShape, ValueType::DOUBLE, mr);
    Value outIdx = createMatrix(outShape, ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();
    keel::f64 *dstI = outIdx.doubleDataMut();
    forEachSlice(x, dim, [&](keel::usize outOff, keel::f64 *slice, keel::usize n) {
        keel::f64 v = 0;
        keel::usize idx = 0;
        f(slice, n, v, idx);
        dst[outOff] = v;
        dstI[outOff] = static_cast<keel::f64>(idx + 1);
    }, scratch.allocator());
    return {std::move(out), std::move(outIdx)};
}

// Resolve effective dim: explicit user-supplied if provided, else the
// "first non-singleton" rule. Returns 1-based dim or 0 for "scalar".
inline int resolveDim(const Value &x, int explicitDim, const char *fn)
{
    if (explicitDim > 0) return validateDim(x, explicitDim, fn);
    return firstNonSingletonDim(x);
}

// Compact NaN-free elements to the front of [data, data+n). Returns the
// new count k (0 <= k <= n). Used by nan* reductions: after compacting
// they call the regular reduction kernel on [data, data+k) and decide
// what to return when k == 0 (which depends on the function — nansum
// returns 0, nanmean/nanvar/nanstd/nanmedian/nanmax/nanmin return NaN).
inline keel::usize compactNonNan(keel::f64 *data, keel::usize n)
{
    keel::usize w = 0;
    for (keel::usize i = 0; i < n; ++i) {
        if (!std::isnan(data[i])) {
            if (w != i) data[w] = data[i];
            ++w;
        }
    }
    return w;
}

} // namespace numkit::ops
