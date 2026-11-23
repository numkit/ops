// toolboxes/builtin/src/math/elementary/rounding.cpp
//
// Rounding and sign builtins. abs lives in toolboxes/builtin/src/backends/
// MStdAbs_*.cpp (SIMD-backed) and only its declaration is in
// math/elementary/rounding.hpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/builtin/elfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>
#include "rounding.hpp"
#include "mod_simd.hpp"
#include "_unary_hint.hpp"
#include "rounding_detail.hpp"


namespace numkit::builtin {

// Public 2-arg wrapper — delegates to the 3-arg overload in the SIMD
// backends with no buffer hint.
Value abs(const Value &x, keel::Allocator *mr) { return abs(x, nullptr, mr); }



Value floor(const Value &x, keel::Allocator *mr)
{
    return roundLikeDispatch(x, [](keel::f64 v) { return keel::floor(v); }, ::numkit::builtin::detail::doubleFloorLoop, mr);
}

Value ceil(const Value &x, keel::Allocator *mr)
{
    return roundLikeDispatch(x, [](keel::f64 v) { return keel::ceil(v); }, ::numkit::builtin::detail::doubleCeilLoop, mr);
}

Value round(const Value &x, keel::Allocator *mr)
{
    return roundLikeDispatch(x, [](keel::f64 v) { return keel::round(v); }, ::numkit::builtin::detail::doubleRoundLoop, mr);
}

Value roundN(const Value &x, keel::i32 n, bool significant, keel::Allocator *mr)
{
    return unaryDouble(x, [n, significant](keel::f64 v) { return roundNScalar(v, n, significant); }, mr);
}

Value fix(const Value &x, keel::Allocator *mr)
{
    return roundLikeDispatch(x, [](keel::f64 v) { return keel::trunc(v); }, ::numkit::builtin::detail::doubleFixLoop, mr);
}

Value sign(const Value &x, keel::Allocator *mr)
{
    auto signOp = [](keel::f64 v) {
        return keel::is_nan(v) ? v : (v > 0) ? 1.0 : (v < 0 ? -1.0 : 0.0);
    };
    // Complex: sign(z) = z/|z| for z != 0, else 0 (MATLAB R2025b);
    // sign(3-4i) = 0.6-0.8i. The double path below can't take complex.
    if (x.isComplex()) {
        auto unit = [](keel::c64 z) -> keel::c64 {
            const keel::f64 m = keel::abs(z);
            return m == 0.0 ? keel::c64(0.0, 0.0) : z / m;
        };
        if (x.isScalar())
            return Value::complexScalar(unit(x.toComplex()), mr);
        Value out = createLike(x, ValueType::COMPLEX, mr);
        const keel::c64 *src = x.complexData();
        keel::c64 *dst = out.complexDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i) dst[i] = unit(src[i]);
        return out;
    }
    // Integer types keep their class (sign(int8(-5))=-1 int8). Promote to
    // double first (unaryDouble's array path needs doubleData), then cast the
    // -1/0/1 result back to the integer class.
    if (isIntegerType(x.type())) {
        Value d = unaryDouble(toDoubleValue(x, mr), signOp, mr);
        return doubleToIntegerExact(d, x.type(), mr);
    }
    return unaryDouble(x, signOp, mr);
}

Value subplus(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x,
                       [](keel::f64 v) {
                           return keel::is_nan(v) ? v : keel::max(v, 0.0);
                       },
                       mr);
}

namespace {
template <typename Op>
Value modRemImpl(const Value &a, const Value &b, Op op,
                 keel::Allocator *mr)
{
    const bool ai = isIntegerType(a.type());
    const bool bi = isIntegerType(b.type());
    if (!ai && !bi)
        return elementwiseDouble(a, b, op, mr);
    Value ad = ai ? toDoubleValue(a, mr) : a;
    Value bd = bi ? toDoubleValue(b, mr) : b;
    Value r = elementwiseDouble(ad, bd, op, mr);
    return doubleToIntegerExact(r, ai ? a.type() : b.type(), mr);
}
} // namespace

Value mod(const Value &a, const Value &b, keel::Allocator *mr)
{
    if (a.type() == ValueType::DOUBLE && b.type() == ValueType::DOUBLE
        && !a.isEmpty() && !b.isEmpty()) {
        const bool aScalar = a.isScalar();
        const bool bScalar = b.isScalar();
        if (!aScalar && !bScalar && a.dims() == b.dims()) {
            Value r = createLike(a, ValueType::DOUBLE, mr);
            detail::modLoopVV(a.doubleData(), b.doubleData(), r.doubleDataMut(), a.numel());
            return r;
        }
        if (!aScalar && bScalar) {
            Value r = createLike(a, ValueType::DOUBLE, mr);
            detail::modLoopVS(a.doubleData(), b.toScalar(), r.doubleDataMut(), a.numel());
            return r;
        }
        if (aScalar && !bScalar) {
            Value r = createLike(b, ValueType::DOUBLE, mr);
            detail::modLoopSV(a.toScalar(), b.doubleData(), r.doubleDataMut(), b.numel());
            return r;
        }
    }
    return modRemImpl(a, b,
                      [](keel::f64 aa, keel::f64 bb) {
                          return bb != 0 ? aa - keel::floor(aa / bb) * bb : aa;
                      },
                      mr);
}

Value rem(const Value &a, const Value &b, keel::Allocator *mr)
{
    return modRemImpl(a, b,
                      [](keel::f64 aa, keel::f64 bb) { return keel::fmod(aa, bb); }, mr);
}

} // namespace numkit::builtin
