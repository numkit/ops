// src/builtin/src/elfun/trigonometry.cpp
//
// Trigonometric coordinate transforms, degree conversions, angle wrapping.

#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/math.h>
#include <numkit/builtin/elfun.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/helpers.hpp>

#include "_unary_hint.hpp"

namespace numkit::builtin {

Value sin(const Value &x, keel::Allocator *mr) { return sin(x, nullptr, mr); }
Value cos(const Value &x, keel::Allocator *mr) { return cos(x, nullptr, mr); }

// ── Degree / Radian conversions ─────────────────────────────────────────────

Value deg2rad(const Value &x, keel::Allocator *mr)
{
    constexpr keel::f64 k = keel::kPi / 180.0;
    return unaryDouble(x, [](keel::f64 v) { return v * k; }, mr);
}

Value rad2deg(const Value &x, keel::Allocator *mr)
{
    constexpr keel::f64 k = 180.0 / keel::kPi;
    return unaryDouble(x, [](keel::f64 v) { return v * k; }, mr);
}

// ── Angle Wrapping ──────────────────────────────────────────────────────────

namespace {
constexpr keel::f64 kPi = keel::kPi;
constexpr keel::f64 kTwoPi = 2.0 * kPi;

inline keel::f64 wrapMod(keel::f64 x, keel::f64 period)
{
    return x - keel::floor(x / period) * period;
}

inline keel::f64 wrapToUpper(keel::f64 x, keel::f64 period)
{
    const bool positiveInput = x > 0.0;
    keel::f64 m = wrapMod(x, period);
    if (m == 0.0 && positiveInput) m = period;
    return m;
}

inline keel::f64 wrapToSym(keel::f64 x, keel::f64 half, keel::f64 period)
{
    if (x < -half || half < x) return wrapToUpper(x + half, period) - half;
    return x;
}
} // namespace

Value wrapToPi(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](keel::f64 v) { return wrapToSym(v, kPi, kTwoPi); }, mr);
}

Value wrapTo2Pi(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](keel::f64 v) { return wrapToUpper(v, kTwoPi); }, mr);
}

Value wrapTo180(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](keel::f64 v) { return wrapToSym(v, 180.0, 360.0); }, mr);
}

Value wrapTo360(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](keel::f64 v) { return wrapToUpper(v, 360.0); }, mr);
}

// ── Coordinate transforms ───────────────────────────────────────────────────

PolarPair cart2pol(const Value &x, const Value &y, keel::Allocator *mr)
{
    Value theta = atan2(y, x, mr);
    Value rho   = hypot(x, y, mr);
    return { std::move(theta), std::move(rho) };
}

CylTriple cart2pol(const Value &x, const Value &y, const Value &z, keel::Allocator *mr)
{
    auto [theta, rho] = cart2pol(x, y, mr);
    return { std::move(theta), std::move(rho), z };
}

CartPair pol2cart(const Value &theta, const Value &rho, keel::Allocator *mr)
{
    Value xv = elementwiseDouble(rho, theta,
        [](keel::f64 r, keel::f64 t) { return r * keel::cos(t); }, mr);
    Value yv = elementwiseDouble(rho, theta,
        [](keel::f64 r, keel::f64 t) { return r * keel::sin(t); }, mr);
    return { std::move(xv), std::move(yv) };
}

CartTriple pol2cart(const Value &theta, const Value &rho, const Value &z, keel::Allocator *mr)
{
    auto [xv, yv] = pol2cart(theta, rho, mr);
    return { std::move(xv), std::move(yv), z };
}

SphTriple cart2sph(const Value &x, const Value &y, const Value &z, keel::Allocator *mr)
{
    Value az = elementwiseDouble(y, x,
        [](keel::f64 yy, keel::f64 xx) { return keel::atan2(yy + 0.0, xx + 0.0); }, mr);
    Value rxy = elementwiseDouble(x, y,
        [](keel::f64 xx, keel::f64 yy) { return keel::hypot(xx, yy); }, mr);
    Value el = elementwiseDouble(z, rxy,
        [](keel::f64 zz, keel::f64 rr) { return keel::atan2(zz + 0.0, rr + 0.0); }, mr);
    Value r = elementwiseDouble(rxy, z,
        [](keel::f64 rxy_v, keel::f64 zz) { return keel::hypot(rxy_v, zz); }, mr);
    return { std::move(az), std::move(el), std::move(r) };
}

CartTriple sph2cart(const Value &az, const Value &el, const Value &r, keel::Allocator *mr)
{
    Value rcos_el = elementwiseDouble(r, el,
        [](keel::f64 rr, keel::f64 ee) { return rr * keel::cos(ee); }, mr);
    Value xv = elementwiseDouble(rcos_el, az,
        [](keel::f64 rce, keel::f64 aa) { return rce * keel::cos(aa); }, mr);
    Value yv = elementwiseDouble(rcos_el, az,
        [](keel::f64 rce, keel::f64 aa) { return rce * keel::sin(aa); }, mr);
    Value zv = elementwiseDouble(r, el,
        [](keel::f64 rr, keel::f64 ee) { return rr * keel::sin(ee); }, mr);
    return { std::move(xv), std::move(yv), std::move(zv) };
}

Value cart2pol(keel::ArrayRef<Value> args, keel::Allocator *mr)
{
    if (args.size() < 2)
        throw Error("cart2pol: requires at least 2 arguments (x, y)",
                     0, 0, "cart2pol", "", "numkit:cart2pol:nargin");
    return cart2pol(args[0], args[1], mr).theta;
}

Value pol2cart(keel::ArrayRef<Value> args, keel::Allocator *mr)
{
    if (args.size() < 2)
        throw Error("pol2cart: requires at least 2 arguments (theta, rho)",
                     0, 0, "pol2cart", "", "numkit:pol2cart:nargin");
    return pol2cart(args[0], args[1], mr).x;
}

Value cart2sph(keel::ArrayRef<Value> args, keel::Allocator *mr)
{
    if (args.size() < 3)
        throw Error("cart2sph: requires 3 arguments (x, y, z)",
                     0, 0, "cart2sph", "", "numkit:cart2sph:nargin");
    return cart2sph(args[0], args[1], args[2], mr).az;
}

Value sph2cart(keel::ArrayRef<Value> args, keel::Allocator *mr)
{
    if (args.size() < 3)
        throw Error("sph2cart: requires 3 arguments (az, el, r)",
                     0, 0, "sph2cart", "", "numkit:sph2cart:nargin");
    return sph2cart(args[0], args[1], args[2], mr).x;
}

} // namespace numkit::builtin
