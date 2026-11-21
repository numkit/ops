// toolboxes/.../polynomials_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by polynomials.cpp + polynomials_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
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

keel::Array<keel::f64> readPolyAsDouble(const Value &p, const char *fn, keel::Allocator *mr)
{
    if (p.type() == ValueType::COMPLEX)
        throw Error(keel::format("{}: complex coefficient input is not supported", fn),
                     0, 0, fn, "", keel::format("numkit:{}:complex", fn));
    if (!p.dims().isVector() && !p.isScalar() && !p.isEmpty())
        throw Error(keel::format("{}: argument must be a vector", fn),
                     0, 0, fn, "", keel::format("numkit:{}:notVector", fn));
    const keel::usize n = p.numel();
    keel::Array<keel::f64> v(n, mr);
    for (keel::usize i = 0; i < n; ++i) v[i] = p.elemAsDouble(i);
    return v;
}

// Convolve two real polynomial coefficient vectors (length-N + length-M
// → length-N+M-1). Pointer + size for inputs so the same helper composes
// with std::vector and std::pmr::vector backings.
[[maybe_unused]] keel::Array<keel::f64> polyConv(const keel::f64 *a, keel::usize na, const keel::f64 *b, keel::usize nb, keel::Allocator *mr)
{
    if (na == 0 || nb == 0) return keel::Array<keel::f64>(mr);
    keel::Array<keel::f64> r(na + nb - 1, mr);
    for (keel::usize i = 0; i < na; ++i)
        for (keel::usize j = 0; j < nb; ++j)
            r[i + j] += a[i] * b[j];
    return r;
}

// d/dx of a coefficient vector in MATLAB order.
[[maybe_unused]] keel::Array<keel::f64> polyderRaw(const keel::f64 *p, keel::usize pn, keel::Allocator *mr)
{
    if (pn <= 1) {
        keel::Array<keel::f64> r(1, mr);
        r[0] = 0.0;  // constant → derivative is [0].
        return r;
    }
    const keel::usize n = pn - 1;  // degree
    keel::Array<keel::f64> r(n, mr);
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 exponent = keel::as<keel::f64>(n - i);
        r[i] = p[i] * exponent;
    }
    return r;
}

[[maybe_unused]] Value rowFromVec(const keel::f64 *v, keel::usize n, keel::Allocator *mr)
{
    auto out = Value::matrix(1, n, ValueType::DOUBLE, mr);
    if (n > 0)
        std::memcpy(out.doubleDataMut(), v, n * sizeof(keel::f64));
    return out;
}

// Trim trailing zeros that arise from a-b cancellation in polyder(b,a).
[[maybe_unused]] void trimLeadingZeros(keel::Array<keel::f64> &v)
{
    keel::usize lo = 0;
    while (lo + 1 < v.size() && v[lo] == 0.0) ++lo;
    if (lo > 0) v.assign(v.slice(lo));
}

} // namespace
namespace {

[[maybe_unused]] keel::Array<keel::c64> readVecAsComplex(const Value &v, const char *fn, keel::Allocator *mr)
{
    if (!v.dims().isVector() && !v.isScalar() && !v.isEmpty())
        throw Error(keel::format("{}: argument must be a vector", fn),
                     0, 0, fn, "", keel::format("numkit:{}:notVector", fn));
    const keel::usize n = v.numel();
    keel::Array<keel::c64> r(n, mr);
    if (v.type() == ValueType::COMPLEX) {
        const auto *p = v.complexData();
        for (keel::usize i = 0; i < n; ++i) r[i] = p[i];
    } else {
        for (keel::usize i = 0; i < n; ++i)
            r[i] = keel::c64(v.elemAsDouble(i), 0.0);
    }
    return r;
}

Value complexColFromVec(const keel::c64 *v, keel::usize n, keel::Allocator *mr)
{
    auto out = Value::complexMatrix(n, 1, mr);
    for (keel::usize i = 0; i < n; ++i)
        out.complexDataMut()[i] = v[i];
    return out;
}

[[maybe_unused]] Value realColIfFlat(const keel::c64 *v, keel::usize n, keel::Allocator *mr)
{
    bool anyComplex = false;
    for (keel::usize i = 0; i < n; ++i)
        if (keel::abs(v[i].imag()) > 1e-12 * (keel::abs(v[i].real()) + 1.0)) {
            anyComplex = true;
            break;
        }
    if (!anyComplex) {
        auto out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < n; ++i)
            out.doubleDataMut()[i] = v[i].real();
        return out;
    }
    return complexColFromVec(v, n, mr);
}

} // namespace

} // namespace numkit::builtin
