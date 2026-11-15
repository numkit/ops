// toolboxes/builtin/src/math/trig/trig_recip_portable.cpp
//
// Reference scalar implementations of the reciprocal trig family
// (sec / csc / cot / sech / csch / coth / secd / cscd / cotd and their
// inverses). Compiled when NUMKIT_WITH_SIMD=OFF; the Highway-dispatched
// variant lives in trig_recip_highway.cpp and matches this file
// bit-for-bit on complex / scalar inputs (SIMD only helps real-vector).

#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/builtin/elfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>

#include <cmath>

namespace numkit::builtin {

namespace {
constexpr double kDeg2Rad = 3.14159265358979323846 / 180.0;
constexpr double kRad2Deg = 180.0 / 3.14159265358979323846;

inline double sind_scalar(double x)
{
    if (!std::isfinite(x)) return std::numeric_limits<double>::quiet_NaN();
    const double xr = std::fmod(x, 360.0);
    if (xr == 0.0 || xr == 180.0 || xr == -180.0) return 0.0;
    if (xr == 90.0)  return  1.0;
    if (xr == -90.0) return -1.0;
    return std::sin(xr * kDeg2Rad);
}

inline double cosd_scalar(double x)
{
    if (!std::isfinite(x)) return std::numeric_limits<double>::quiet_NaN();
    const double xr = std::fmod(x, 360.0);
    if (xr == 90.0 || xr == -90.0 || xr == 270.0 || xr == -270.0) return 0.0;
    if (xr == 0.0) return  1.0;
    if (xr == 180.0 || xr == -180.0) return -1.0;
    return std::cos(xr * kDeg2Rad);
}
} // namespace

// ── Forward reciprocal ─────────────────────────────────────────────

Value sec(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::c64(1.0) / keel::cos(c); }, mr);
    return unaryDouble(x, [](double v) { return 1.0 / std::cos(v); }, mr);
}

Value csc(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::c64(1.0) / keel::sin(c); }, mr);
    return unaryDouble(x, [](double v) { return 1.0 / std::sin(v); }, mr);
}

Value cot(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::cos(c) / keel::sin(c); }, mr);
    return unaryDouble(x, [](double v) { return std::cos(v) / std::sin(v); }, mr);
}

// ── Hyperbolic reciprocal ──────────────────────────────────────────

Value sech(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::c64(1.0) / keel::cosh(c); }, mr);
    return unaryDouble(x, [](double v) { return 1.0 / std::cosh(v); }, mr);
}

Value csch(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::c64(1.0) / keel::sinh(c); }, mr);
    return unaryDouble(x, [](double v) { return 1.0 / std::sinh(v); }, mr);
}

Value coth(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::cosh(c) / keel::sinh(c); }, mr);
    return unaryDouble(x, [](double v) { return std::cosh(v) / std::sinh(v); }, mr);
}

// ── Degree reciprocal ──────────────────────────────────────────────

Value secd(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::c64(1.0) / keel::cos(c * kDeg2Rad); }, mr);
    return unaryDouble(x, [](double v) { return 1.0 / cosd_scalar(v); }, mr);
}

Value cscd(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::c64(1.0) / keel::sin(c * kDeg2Rad); }, mr);
    return unaryDouble(x, [](double v) { return 1.0 / sind_scalar(v); }, mr);
}

Value cotd(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) {
            return keel::cos(c * kDeg2Rad) / keel::sin(c * kDeg2Rad);
        }, mr);
    return unaryDouble(x, [](double v) { return cosd_scalar(v) / sind_scalar(v); }, mr);
}

// ── Inverse reciprocal ─────────────────────────────────────────────

Value asec(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::acos(keel::c64(1.0) / c); }, mr);
    if (x.isScalar()) {
        const double v = x.toScalar();
        if (v > -1.0 && v < 1.0)
            return Value::complexScalar(keel::acos(keel::c64(1.0 / v, 0.0)), mr);
    }
    return unaryDouble(x, [](double v) { return std::acos(1.0 / v); }, mr);
}

Value acsc(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::asin(keel::c64(1.0) / c); }, mr);
    if (x.isScalar()) {
        const double v = x.toScalar();
        if (v > -1.0 && v < 1.0 && v != 0.0)
            return Value::complexScalar(keel::asin(keel::c64(1.0 / v, 0.0)), mr);
    }
    return unaryDouble(x, [](double v) { return std::asin(1.0 / v); }, mr);
}

Value acot(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::atan(keel::c64(1.0) / c); }, mr);
    return unaryDouble(x, [](double v) { return std::atan(1.0 / v); }, mr);
}

// ── Inverse hyperbolic reciprocal ──────────────────────────────────

Value asech(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::acosh(keel::c64(1.0) / c); }, mr);
    if (x.isScalar()) {
        const double v = x.toScalar();
        if (v <= 0.0 || v > 1.0)
            return Value::complexScalar(keel::acosh(keel::c64(1.0 / v, 0.0)), mr);
    }
    return unaryDouble(x, [](double v) { return std::acosh(1.0 / v); }, mr);
}

Value acsch(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::asinh(keel::c64(1.0) / c); }, mr);
    return unaryDouble(x, [](double v) { return std::asinh(1.0 / v); }, mr);
}

Value acoth(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::atanh(keel::c64(1.0) / c); }, mr);
    if (x.isScalar()) {
        const double v = x.toScalar();
        if (v >= -1.0 && v <= 1.0)
            return Value::complexScalar(keel::atanh(keel::c64(1.0 / v, 0.0)), mr);
    }
    return unaryDouble(x, [](double v) { return std::atanh(1.0 / v); }, mr);
}

// ── Inverse degree reciprocal ──────────────────────────────────────

Value asecd(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) {
            return keel::acos(keel::c64(1.0) / c) * kRad2Deg;
        }, mr);
    if (x.isScalar()) {
        const double v = x.toScalar();
        if (v > -1.0 && v < 1.0)
            return Value::complexScalar(keel::acos(keel::c64(1.0 / v, 0.0)) * kRad2Deg, mr);
    }
    return unaryDouble(x, [](double v) { return std::acos(1.0 / v) * kRad2Deg; }, mr);
}

Value acscd(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) {
            return keel::asin(keel::c64(1.0) / c) * kRad2Deg;
        }, mr);
    if (x.isScalar()) {
        const double v = x.toScalar();
        if (v > -1.0 && v < 1.0 && v != 0.0)
            return Value::complexScalar(keel::asin(keel::c64(1.0 / v, 0.0)) * kRad2Deg, mr);
    }
    return unaryDouble(x, [](double v) { return std::asin(1.0 / v) * kRad2Deg; }, mr);
}

Value acotd(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) {
            return keel::atan(keel::c64(1.0) / c) * kRad2Deg;
        }, mr);
    return unaryDouble(x, [](double v) { return std::atan(1.0 / v) * kRad2Deg; }, mr);
}

} // namespace numkit::builtin
