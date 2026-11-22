// toolboxes/builtin/src/math/trig/trig_recip_portable.cpp
//
// Reference scalar implementations of the reciprocal trig family
// (sec / csc / cot / sech / csch / coth / secd / cscd / cotd and their
// inverses). Compiled when NUMKIT_WITH_SIMD=OFF; the Highway-dispatched
// variant lives in trig_recip_highway.cpp and matches this file
// bit-for-bit on complex / scalar inputs (SIMD only helps real-vector).

#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/builtin/elfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>


namespace numkit::builtin {

namespace {
constexpr keel::f64 kDeg2Rad = keel::kPi / 180.0;
constexpr keel::f64 kRad2Deg = 180.0 / keel::kPi;

inline keel::f64 sind_scalar(keel::f64 x)
{
    if (!keel::is_finite(x)) return keel::Limits<keel::f64>::quiet_nan();
    const keel::f64 xr = keel::fmod(x, 360.0);
    if (xr == 0.0 || xr == 180.0 || xr == -180.0) return 0.0;
    if (xr == 90.0)  return  1.0;
    if (xr == -90.0) return -1.0;
    return keel::sin(xr * kDeg2Rad);
}

inline keel::f64 cosd_scalar(keel::f64 x)
{
    if (!keel::is_finite(x)) return keel::Limits<keel::f64>::quiet_nan();
    const keel::f64 xr = keel::fmod(x, 360.0);
    if (xr == 90.0 || xr == -90.0 || xr == 270.0 || xr == -270.0) return 0.0;
    if (xr == 0.0) return  1.0;
    if (xr == 180.0 || xr == -180.0) return -1.0;
    return keel::cos(xr * kDeg2Rad);
}
} // namespace

// ── Forward reciprocal ─────────────────────────────────────────────

Value sec(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::c64(1.0) / keel::cos(c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return 1.0 / keel::cos(v); }, mr);
}

Value csc(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::c64(1.0) / keel::sin(c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return 1.0 / keel::sin(v); }, mr);
}

Value cot(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::cos(c) / keel::sin(c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return keel::cos(v) / keel::sin(v); }, mr);
}

// ── Hyperbolic reciprocal ──────────────────────────────────────────

Value sech(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::c64(1.0) / keel::cosh(c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return 1.0 / keel::cosh(v); }, mr);
}

Value csch(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::c64(1.0) / keel::sinh(c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return 1.0 / keel::sinh(v); }, mr);
}

Value coth(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::cosh(c) / keel::sinh(c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return keel::cosh(v) / keel::sinh(v); }, mr);
}

// ── Degree reciprocal ──────────────────────────────────────────────

Value secd(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::c64(1.0) / keel::cos(c * kDeg2Rad); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return 1.0 / cosd_scalar(v); }, mr);
}

Value cscd(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::c64(1.0) / keel::sin(c * kDeg2Rad); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return 1.0 / sind_scalar(v); }, mr);
}

Value cotd(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) {
            return keel::cos(c * kDeg2Rad) / keel::sin(c * kDeg2Rad);
        }, mr);
    return unaryDouble(x, [](keel::f64 v) { return cosd_scalar(v) / sind_scalar(v); }, mr);
}

// ── Inverse reciprocal ─────────────────────────────────────────────

Value asec(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::acos(keel::c64(1.0) / c); }, mr);
    if (x.isScalar()) {
        const keel::f64 v = x.toScalar();
        if (v > -1.0 && v < 1.0)
            return Value::complexScalar(keel::acos(keel::c64(1.0 / v, 0.0)), mr);
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::acos(1.0 / v); }, mr);
}

Value acsc(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::asin(keel::c64(1.0) / c); }, mr);
    if (x.isScalar()) {
        const keel::f64 v = x.toScalar();
        if (v > -1.0 && v < 1.0 && v != 0.0)
            return Value::complexScalar(keel::asin(keel::c64(1.0 / v, 0.0)), mr);
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::asin(1.0 / v); }, mr);
}

Value acot(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::atan(keel::c64(1.0) / c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return keel::atan(1.0 / v); }, mr);
}

// ── Inverse hyperbolic reciprocal ──────────────────────────────────

Value asech(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::acosh(keel::c64(1.0) / c); }, mr);
    if (x.isScalar()) {
        const keel::f64 v = x.toScalar();
        if (v <= 0.0 || v > 1.0)
            return Value::complexScalar(keel::acosh(keel::c64(1.0 / v, 0.0)), mr);
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::acosh(1.0 / v); }, mr);
}

Value acsch(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::asinh(keel::c64(1.0) / c); }, mr);
    return unaryDouble(x, [](keel::f64 v) { return keel::asinh(1.0 / v); }, mr);
}

Value acoth(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::atanh(keel::c64(1.0) / c); }, mr);
    if (x.isScalar()) {
        const keel::f64 v = x.toScalar();
        if (v >= -1.0 && v <= 1.0)
            return Value::complexScalar(keel::atanh(keel::c64(1.0 / v, 0.0)), mr);
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::atanh(1.0 / v); }, mr);
}

// ── Inverse degree reciprocal ──────────────────────────────────────

Value asecd(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) {
            return keel::acos(keel::c64(1.0) / c) * kRad2Deg;
        }, mr);
    if (x.isScalar()) {
        const keel::f64 v = x.toScalar();
        if (v > -1.0 && v < 1.0)
            return Value::complexScalar(keel::acos(keel::c64(1.0 / v, 0.0)) * kRad2Deg, mr);
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::acos(1.0 / v) * kRad2Deg; }, mr);
}

Value acscd(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) {
            return keel::asin(keel::c64(1.0) / c) * kRad2Deg;
        }, mr);
    if (x.isScalar()) {
        const keel::f64 v = x.toScalar();
        if (v > -1.0 && v < 1.0 && v != 0.0)
            return Value::complexScalar(keel::asin(keel::c64(1.0 / v, 0.0)) * kRad2Deg, mr);
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::asin(1.0 / v) * kRad2Deg; }, mr);
}

Value acotd(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) {
            return keel::atan(keel::c64(1.0) / c) * kRad2Deg;
        }, mr);
    return unaryDouble(x, [](keel::f64 v) { return keel::atan(1.0 / v) * kRad2Deg; }, mr);
}

} // namespace numkit::builtin
