// toolboxes/builtin/src/math/elementary/backends/logical_reductions_portable.cpp
//
// Scalar reference any() / all(). Compiled when NUMKIT_WITH_SIMD=OFF.
// The Highway-dispatched variant lives in logical_reductions_highway.cpp;
// behaviour is identical — same dim-dispatch, same LOGICAL output type,
// same MATLAB-empty semantics (any([])=false, all([])=true). Only the
// inner unit-stride scan loops differ (no Highway types here so the
// portable build doesn't need the Highway include path).

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <numkit/builtin/elmat.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/ops/reductions.hpp>

#include <cstddef>
#include <cstdint>

namespace numkit::builtin {
namespace {

template <bool IsAny>
keel::u8 scanGeneric(const Value &x, keel::usize base, keel::usize n, keel::usize stride)
{
    if (x.isComplex()) {
        const auto *p = x.complexData();
        for (keel::usize i = 0; i < n; ++i) {
            const auto z = p[base + i * stride];
            const bool nz = z.real() != 0.0 || z.imag() != 0.0;
            if constexpr (IsAny) { if (nz)  return 1; }
            else                 { if (!nz) return 0; }
        }
    } else {
        for (keel::usize i = 0; i < n; ++i) {
            const bool nz = x.elemAsDouble(base + i * stride) != 0.0;
            if constexpr (IsAny) { if (nz)  return 1; }
            else                 { if (!nz) return 0; }
        }
    }
    return IsAny ? 0 : 1;
}

template <bool IsAny>
keel::u8 scanSlice(const Value &x, keel::usize base, keel::usize n, keel::usize stride)
{
    if (n == 0) return IsAny ? 0 : 1;
    if (x.type() == ValueType::DOUBLE) {
        const double *p = x.doubleData() + base;
        for (keel::usize i = 0; i < n; ++i) {
            const bool nz = p[i * stride] != 0.0;
            if constexpr (IsAny) { if (nz)  return 1; }
            else                 { if (!nz) return 0; }
        }
        return IsAny ? 0 : 1;
    }
    if (x.type() == ValueType::LOGICAL) {
        const keel::u8 *p = x.logicalData() + base;
        for (keel::usize i = 0; i < n; ++i) {
            const bool nz = p[i * stride] != 0;
            if constexpr (IsAny) { if (nz)  return 1; }
            else                 { if (!nz) return 0; }
        }
        return IsAny ? 0 : 1;
    }
    return scanGeneric<IsAny>(x, base, n, stride);
}

template <bool IsAny>
Value logicalReduceImpl(const Value &x, int dim, keel::Allocator *mr)
{
    if (x.isEmpty())
        return Value::logicalScalar(IsAny ? false : true, mr);
    if (x.isScalar()) {
        const bool nz = x.isComplex()
            ? (x.toComplex().real() != 0.0 || x.toComplex().imag() != 0.0)
            : (x.elemAsDouble(0) != 0.0);
        return Value::logicalScalar(nz, mr);
    }
    if (x.dims().isVector()) {
        const keel::u8 v = scanSlice<IsAny>(x, 0, x.numel(), 1);
        return Value::logicalScalar(v != 0, mr);
    }

    const int d = numkit::ops::resolveDim(x, dim, IsAny ? "any" : "all");
    const auto &dd = x.dims();

    // ND fallback: rank ≥ 4 — stride arithmetic via scanSlice.
    if (dd.ndim() >= 4 && d >= 1 && d <= dd.ndim()) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        auto shape = numkit::ops::outShapeForDimND(x, d, scratch.allocator());
        Value out = Value::matrixND(shape.data(),
                                      keel::as<int>(shape.size()),
                                      ValueType::LOGICAL, mr);
        keel::u8 *dst = out.logicalDataMut();
        const keel::usize sliceLen = dd.dim(d - 1);
        keel::usize B = 1;
        for (int i = 0; i < d - 1; ++i) B *= dd.dim(i);
        keel::usize O = 1;
        for (int i = d; i < dd.ndim(); ++i) O *= dd.dim(i);
        for (keel::usize o = 0; o < O; ++o)
            for (keel::usize b = 0; b < B; ++b) {
                const keel::usize base = o * sliceLen * B + b;
                dst[o * B + b] = scanSlice<IsAny>(x, base, sliceLen, B);
            }
        return out;
    }

    const auto outShape = numkit::ops::outShapeForDim(x, d);
    Value out = createMatrix(outShape, ValueType::LOGICAL, mr);
    keel::u8 *dst = out.logicalDataMut();

    const keel::usize R = dd.rows(), C = dd.cols(), P = dd.is3D() ? dd.pages() : 1;

    if (d == 1) {
        keel::usize outIdx = 0;
        for (keel::usize pp = 0; pp < P; ++pp)
            for (keel::usize c = 0; c < C; ++c) {
                const keel::usize base = pp * R * C + c * R;
                dst[outIdx++] = scanSlice<IsAny>(x, base, R, 1);
            }
    } else if (d == 2) {
        keel::usize outIdx = 0;
        for (keel::usize pp = 0; pp < P; ++pp)
            for (keel::usize r = 0; r < R; ++r) {
                const keel::usize base = pp * R * C + r;
                dst[outIdx++] = scanSlice<IsAny>(x, base, C, R);
            }
    } else if (d == 3) {
        keel::usize outIdx = 0;
        for (keel::usize c = 0; c < C; ++c)
            for (keel::usize r = 0; r < R; ++r) {
                const keel::usize base = c * R + r;
                dst[outIdx++] = scanSlice<IsAny>(x, base, P, R * C);
            }
    }
    return out;
}

} // namespace

Value anyOf(const Value &x, int dim, keel::Allocator *mr)
{
    return logicalReduceImpl<true>(x, dim, mr);
}

Value allOf(const Value &x, int dim, keel::Allocator *mr)
{
    return logicalReduceImpl<false>(x, dim, mr);
}

} // namespace numkit::builtin
