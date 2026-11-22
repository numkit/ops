// src/builtin/src/elfun/trig_portable.cpp
//
// Reference scalar implementations of trig family for numkit::builtin.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/builtin/elfun.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/helpers.hpp>
#include "sinpi_kernel.hpp"


namespace numkit::builtin {

namespace {

template <typename ScalarOp, typename ComplexOp>
Value unaryRealDoubleHint(const Value &x, Value *hint, ScalarOp scalarOp, ComplexOp complexOp, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, complexOp, mr);
    if (x.isScalar())
        return Value::scalar(scalarOp(x.toScalar()), mr);

    if (hint && hint->isHeapDouble() && hint->heapRefCount() == 1
        && hint->dims() == x.dims()) {
        Value r = std::move(*hint);
        const keel::f64 *in  = x.doubleData();
        keel::f64       *out = r.doubleDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            out[i] = scalarOp(in[i]);
        return r;
    }
    return unaryDouble(x, scalarOp, mr);
}

constexpr keel::f64 kDeg2Rad_p = keel::kPi / 180.0;
constexpr keel::f64 kRad2Deg_p = 180.0 / keel::kPi;
constexpr keel::f64 kPi_p      = keel::kPi;

inline keel::f64 reduceDeg360_p(keel::f64 x) { return keel::fmod(x, 360.0); }

inline keel::f64 sind_scalar_p(keel::f64 x)
{
    if (!keel::is_finite(x)) return keel::Limits<keel::f64>::quiet_nan();
    const keel::f64 xr = reduceDeg360_p(x);
    if (xr == 0.0 || xr == 180.0 || xr == -180.0) return 0.0;
    if (xr == 90.0)  return  1.0;
    if (xr == -90.0) return -1.0;
    return keel::sin(xr * kDeg2Rad_p);
}

inline keel::f64 cosd_scalar_p(keel::f64 x)
{
    if (!keel::is_finite(x)) return keel::Limits<keel::f64>::quiet_nan();
    const keel::f64 xr = reduceDeg360_p(x);
    if (xr == 90.0 || xr == -90.0 || xr == 270.0 || xr == -270.0) return 0.0;
    if (xr == 0.0) return  1.0;
    if (xr == 180.0 || xr == -180.0) return -1.0;
    return keel::cos(xr * kDeg2Rad_p);
}

inline keel::f64 tand_scalar_p(keel::f64 x)
{
    if (!keel::is_finite(x)) return keel::Limits<keel::f64>::quiet_nan();
    const keel::f64 xr = reduceDeg360_p(x);
    if (xr == 0.0 || xr == 180.0 || xr == -180.0) return 0.0;
    if (xr == 90.0)  return  keel::Limits<keel::f64>::infinity();
    if (xr == -90.0) return -keel::Limits<keel::f64>::infinity();
    if (xr == 270.0) return  keel::Limits<keel::f64>::infinity();
    if (xr == -270.0)return -keel::Limits<keel::f64>::infinity();
    return keel::tan(xr * kDeg2Rad_p);
}

inline keel::f64 asind_scalar_p(keel::f64 x)
{
    if (x < -1.0 || x > 1.0) return keel::Limits<keel::f64>::quiet_nan();
    if (x ==  1.0) return  90.0;
    if (x == -1.0) return -90.0;
    if (x ==  0.0) return   0.0;
    return keel::asin(x) * kRad2Deg_p;
}

inline keel::f64 acosd_scalar_p(keel::f64 x)
{
    if (x < -1.0 || x > 1.0) return keel::Limits<keel::f64>::quiet_nan();
    if (x ==  1.0) return   0.0;
    if (x ==  0.0) return  90.0;
    if (x == -1.0) return 180.0;
    return keel::acos(x) * kRad2Deg_p;
}

inline keel::f64 atand_scalar_p(keel::f64 x)
{
    if (keel::is_nan(x)) return x;
    if (x == 0.0) return 0.0;
    return keel::atan(x) * kRad2Deg_p;
}

inline keel::f64 atan2d_scalar_p(keel::f64 y, keel::f64 x)
{
    return keel::atan2(y, x) * kRad2Deg_p;
}

inline keel::f64 sinpi_scalar_p(keel::f64 x) { return detail::sinpi_kernel(x); }
inline keel::f64 cospi_scalar_p(keel::f64 x) { return detail::cospi_kernel(x); }

} // namespace

Value sin(const Value &x, Value *hint, keel::Allocator *mr)
{
    return unaryRealDoubleHint(x, hint, [](keel::f64 v) { return keel::sin(v); }, [](const keel::c64 &c) { return keel::sin(c); }, mr);
}

Value cos(const Value &x, Value *hint, keel::Allocator *mr)
{
    return unaryRealDoubleHint(x, hint, [](keel::f64 v) { return keel::cos(v); }, [](const keel::c64 &c) { return keel::cos(c); }, mr);
}

Value tan(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::tan(c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return keel::tan(v); }, mr);
}

Value sinh(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::sinh(c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return keel::sinh(v); }, mr);
}

Value cosh(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::cosh(c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return keel::cosh(v); }, mr);
}

Value tanh(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::tanh(c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return keel::tanh(v); }, mr);
}

Value asin(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::asin(c); }, mr);
    // Real out-of-domain generates complex result in MATLAB semantics
    for (keel::usize i = 0; i < x.numel(); ++i) {
        keel::f64 v = x.elemAsDouble(i);
        if (v < -1.0 || v > 1.0) {
            return unaryComplex(x, [](const keel::c64 &c) { return keel::asin(c); }, mr);
        }
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::asin(v); }, mr);
}

Value acos(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::acos(c); }, mr);
    for (keel::usize i = 0; i < x.numel(); ++i) {
        keel::f64 v = x.elemAsDouble(i);
        if (v < -1.0 || v > 1.0) {
            return unaryComplex(x, [](const keel::c64 &c) { return keel::acos(c); }, mr);
        }
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::acos(v); }, mr);
}

Value atan(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::atan(c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return keel::atan(v); }, mr);
}

Value atan2(const Value &y, const Value &x, keel::Allocator *mr)
{
    return elementwiseDouble(y, x, [](keel::f64 yy, keel::f64 xx) { return keel::atan2(yy, xx); }, mr);
}

Value asinh(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::asinh(c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return keel::asinh(v); }, mr);
}

Value acosh(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::acosh(c); }, mr);
    for (keel::usize i = 0; i < x.numel(); ++i) {
        if (x.elemAsDouble(i) < 1.0) {
            return unaryComplex(x, [](const keel::c64 &c) { return keel::acosh(c); }, mr);
        }
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::acosh(v); }, mr);
}

Value atanh(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::atanh(c); }, mr);
    for (keel::usize i = 0; i < x.numel(); ++i) {
        keel::f64 v = x.elemAsDouble(i);
        if (v < -1.0 || v > 1.0) {
            return unaryComplex(x, [](const keel::c64 &c) { return keel::atanh(c); }, mr);
        }
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::atanh(v); }, mr);
}

Value sind(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::sin(c * kDeg2Rad_p); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return sind_scalar_p(v); }, mr);
}

Value cosd(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::cos(c * kDeg2Rad_p); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return cosd_scalar_p(v); }, mr);
}

Value tand(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::tan(c * kDeg2Rad_p); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return tand_scalar_p(v); }, mr);
}

Value asind(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::asin(c) * kRad2Deg_p; }, mr);
    return unaryDouble(x, [](keel::f64 v) { return asind_scalar_p(v); }, mr);
}

Value acosd(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::acos(c) * kRad2Deg_p; }, mr);
    return unaryDouble(x, [](keel::f64 v) { return acosd_scalar_p(v); }, mr);
}

Value atand(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::atan(c) * kRad2Deg_p; }, mr);
    return unaryDouble(x, [](keel::f64 v) { return atand_scalar_p(v); }, mr);
}

Value atan2d(const Value &y, const Value &x, keel::Allocator *mr)
{
    return elementwiseDouble(y, x, [](keel::f64 yy, keel::f64 xx) { return atan2d_scalar_p(yy, xx); }, mr);
}

Value sinpi(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::sin(kPi_p * c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return sinpi_scalar_p(v); }, mr);
}

Value cospi(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::cos(kPi_p * c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return cospi_scalar_p(v); }, mr);
}

} // namespace numkit::builtin
