// src/builtin/src/elfun/exponents.cpp
//
// Scalar exponentials, power scaling, and roots for numkit::builtin.

#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/builtin/elfun.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/helpers.hpp>

#include <cmath>
#include "_unary_hint.hpp"

namespace numkit::builtin {

Value exp(const Value &x, keel::Allocator *mr) { return exp(x, nullptr, mr); }
Value log(const Value &x, keel::Allocator *mr) { return log(x, nullptr, mr); }

Value pow2(const Value &f, const Value &e, keel::Allocator *mr)
{
    return elementwiseDouble(f, e,
        [](keel::f64 ff, keel::f64 ee) {
            return std::ldexp(ff, keel::as<int>(std::floor(ee)));
        }, mr);
}

Value nextpow2(const Value &n, keel::Allocator *mr)
{
    auto nextPow2Scalar = [](keel::f64 v) -> keel::f64 {
        if (std::isnan(v)) return v;
        keel::f64 a = std::fabs(v);
        if (a <= 1.0) return 0.0;
        int exp_val;
        keel::f64 frac = std::frexp(a, &exp_val);
        if (frac == 0.5) return keel::as<keel::f64>(exp_val - 1);
        return keel::as<keel::f64>(exp_val);
    };
    return unaryDouble(n, nextPow2Scalar, mr);
}

Value cbrt(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::pow(c, 1.0 / 3.0); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return std::cbrt(v); }, mr);
}

Value nthroot(const Value &x, const Value &n, keel::Allocator *mr)
{
    return elementwiseDouble(x, n, [](keel::f64 xv, keel::f64 nv) {
        if (nv == 0.0) return std::nan("");
        if (xv >= 0.0) return std::pow(xv, 1.0 / nv);
        const keel::f64 rounded = std::round(nv);
        if (rounded != nv) return std::nan("");
        const long long ni = keel::as<long long>(rounded);
        if (ni % 2 == 0) return std::nan("");
        return -std::pow(-xv, 1.0 / nv);
    }, mr);
}

Value pow(const Value &x, const Value &y, keel::Allocator *mr)
{
    return elementwiseDouble(x, y, [](keel::f64 xx, keel::f64 yy) { return std::pow(xx, yy); }, mr);
}

Value realpow(const Value &x, const Value &y, keel::Allocator *mr)
{
    auto checkPair = [](keel::f64 xx, keel::f64 yy) {
        if (xx < 0.0 && yy != std::floor(yy)) {
            throw std::runtime_error(
                "realpow produced complex result — use power(.^) instead");
        }
        return std::pow(xx, yy);
    };
    return elementwiseDouble(x, y, checkPair, mr);
}

} // namespace numkit::builtin
