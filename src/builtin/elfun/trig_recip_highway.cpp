// toolboxes/builtin/src/math/trig/trig_recip_highway.cpp
//
// Highway dynamic-dispatch reciprocal-trig family: sec, csc, cot,
// sech, csch, coth, secd, cscd, cotd, and their inverse forms (asec,
// acsc, acot, asech, acsch, acoth, asecd, acscd, acotd). All composed
// from Highway's contrib/math primitives (Cos, Sin, Sinh, Acos, Asin,
// Atan, Acosh, Asinh, Atanh) â€” the reciprocal forms emit a SIMD divide
// or a primary-on-1/x.
//
// Out-of-domain elements (e.g. asec(0.5) which MATLAB returns as
// complex) get NaN here, matching the existing scalar fallback for
// non-scalar inputs in trigonometry.cpp. The scalar fast-path keeps
// the complex fallback so `asec(0.5)` returns the MATLAB-spec complex
// value.
//
// See BUGS.md note attached to trig SIMD coverage (was: sec/csc/cot
// family fell back to scalar libm). Parity vs MATLAB R2025b verified
// at ULP <= 1e-10 for the test domains.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/builtin/elfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/ops/parallel_for.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>

#include <cmath>
#include <cstddef>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "builtin/elfun/trig_recip_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>
#include <hwy/contrib/math/math-inl.h>

HWY_BEFORE_NAMESPACE();
namespace numkit::builtin {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

namespace recip_consts {
constexpr keel::f64 kDeg2Rad = keel::kPi / 180.0;
constexpr keel::f64 kRad2Deg = 180.0 / keel::kPi;
}

// â”€â”€ Forward reciprocal: sec / csc / cot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

void SecLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Div(one, hn::Cos(d, v)), d, out + i);
    }
    for (; i < n; ++i) out[i] = 1.0 / std::cos(in[i]);
}

void CscLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Div(one, hn::Sin(d, v)), d, out + i);
    }
    for (; i < n; ++i) out[i] = 1.0 / std::sin(in[i]);
}

void CotLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Div(hn::Cos(d, v), hn::Sin(d, v)), d, out + i);
    }
    for (; i < n; ++i) out[i] = std::cos(in[i]) / std::sin(in[i]);
}

// â”€â”€ Hyperbolic reciprocal: sech / csch / coth â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Highway has Sinh and Tanh but not Cosh â€” compose cosh from Exp.

static HWY_INLINE auto CoshVec(const hn::ScalableTag<keel::f64> d, hn::Vec<hn::ScalableTag<keel::f64>> v)
{
    const auto half = hn::Set(d, 0.5);
    auto ep = hn::Exp(d, v);
    auto en = hn::Exp(d, hn::Neg(v));
    return hn::Mul(half, hn::Add(ep, en));
}

void SechLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Div(one, CoshVec(d, v)), d, out + i);
    }
    for (; i < n; ++i) out[i] = 1.0 / std::cosh(in[i]);
}

void CschLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Div(one, hn::Sinh(d, v)), d, out + i);
    }
    for (; i < n; ++i) out[i] = 1.0 / std::sinh(in[i]);
}

void CothLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Div(CoshVec(d, v), hn::Sinh(d, v)), d, out + i);
    }
    for (; i < n; ++i) out[i] = std::cosh(in[i]) / std::sinh(in[i]);
}

// â”€â”€ Degree reciprocal: secd / cscd / cotd â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Vector path skips the integer-multiple-of-90Â° snap (matches the
// degree forms in trig_highway.cpp). Scalar fast-path in trig_recip_*.cpp
// keeps the snap.

void SecdLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto k = hn::Set(d, recip_consts::kDeg2Rad);
    const auto one = hn::Set(d, 1.0);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Div(one, hn::Cos(d, hn::Mul(v, k))), d, out + i);
    }
    for (; i < n; ++i) out[i] = 1.0 / std::cos(in[i] * recip_consts::kDeg2Rad);
}

void CscdLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto k = hn::Set(d, recip_consts::kDeg2Rad);
    const auto one = hn::Set(d, 1.0);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Div(one, hn::Sin(d, hn::Mul(v, k))), d, out + i);
    }
    for (; i < n; ++i) out[i] = 1.0 / std::sin(in[i] * recip_consts::kDeg2Rad);
}

void CotdLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto k = hn::Set(d, recip_consts::kDeg2Rad);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v   = hn::LoadU(d, in + i);
        auto vrad = hn::Mul(v, k);
        hn::StoreU(hn::Div(hn::Cos(d, vrad), hn::Sin(d, vrad)), d, out + i);
    }
    for (; i < n; ++i) out[i] = std::cos(in[i] * recip_consts::kDeg2Rad)
                                / std::sin(in[i] * recip_consts::kDeg2Rad);
}

// â”€â”€ Inverse reciprocal: asec / acsc / acot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// MATLAB returns complex for out-of-domain inputs (e.g. asec(0.5)).
// Vector path produces NaN there; scalar fallback in trig_recip_*.cpp
// detects this and returns a complex Value.

void AsecLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Acos(d, hn::Div(one, v)), d, out + i);
    }
    for (; i < n; ++i) out[i] = std::acos(1.0 / in[i]);
}

void AcscLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Asin(d, hn::Div(one, v)), d, out + i);
    }
    for (; i < n; ++i) out[i] = std::asin(1.0 / in[i]);
}

void AcotLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Atan(d, hn::Div(one, v)), d, out + i);
    }
    for (; i < n; ++i) out[i] = std::atan(1.0 / in[i]);
}

// â”€â”€ Inverse hyperbolic reciprocal: asech / acsch / acoth â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

void AsechLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Acosh(d, hn::Div(one, v)), d, out + i);
    }
    for (; i < n; ++i) out[i] = std::acosh(1.0 / in[i]);
}

void AcschLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Asinh(d, hn::Div(one, v)), d, out + i);
    }
    for (; i < n; ++i) out[i] = std::asinh(1.0 / in[i]);
}

void AcothLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Atanh(d, hn::Div(one, v)), d, out + i);
    }
    for (; i < n; ++i) out[i] = std::atanh(1.0 / in[i]);
}

// â”€â”€ Inverse degree reciprocal: asecd / acscd / acotd â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

void AsecdLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    const auto r2d = hn::Set(d, recip_consts::kRad2Deg);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Mul(hn::Acos(d, hn::Div(one, v)), r2d), d, out + i);
    }
    for (; i < n; ++i) out[i] = std::acos(1.0 / in[i]) * recip_consts::kRad2Deg;
}

void AcscdLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    const auto r2d = hn::Set(d, recip_consts::kRad2Deg);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Mul(hn::Asin(d, hn::Div(one, v)), r2d), d, out + i);
    }
    for (; i < n; ++i) out[i] = std::asin(1.0 / in[i]) * recip_consts::kRad2Deg;
}

void AcotdLoop(const keel::f64 *HWY_RESTRICT in, keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto one = hn::Set(d, 1.0);
    const auto r2d = hn::Set(d, recip_consts::kRad2Deg);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        hn::StoreU(hn::Mul(hn::Atan(d, hn::Div(one, v)), r2d), d, out + i);
    }
    for (; i < n; ++i) out[i] = std::atan(1.0 / in[i]) * recip_consts::kRad2Deg;
}

} // namespace HWY_NAMESPACE
} // namespace numkit::builtin
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace numkit::builtin {

HWY_EXPORT(SecLoop);
HWY_EXPORT(CscLoop);
HWY_EXPORT(CotLoop);
HWY_EXPORT(SechLoop);
HWY_EXPORT(CschLoop);
HWY_EXPORT(CothLoop);
HWY_EXPORT(SecdLoop);
HWY_EXPORT(CscdLoop);
HWY_EXPORT(CotdLoop);
HWY_EXPORT(AsecLoop);
HWY_EXPORT(AcscLoop);
HWY_EXPORT(AcotLoop);
HWY_EXPORT(AsechLoop);
HWY_EXPORT(AcschLoop);
HWY_EXPORT(AcothLoop);
HWY_EXPORT(AsecdLoop);
HWY_EXPORT(AcscdLoop);
HWY_EXPORT(AcotdLoop);

namespace {

constexpr keel::f64 kDeg2Rad = keel::kPi / 180.0;
constexpr keel::f64 kRad2Deg = 180.0 / keel::kPi;

// Snap helpers reused from the degree forms (sind/cosd/tand) for the
// scalar fast-path of secd/cscd/cotd. MATLAB exposes exact 0 / Â±1 /
// Â±Inf at integer multiples of 90Â° on scalar input.
inline keel::f64 sind_scalar(keel::f64 x)
{
    if (!std::isfinite(x)) return std::numeric_limits<keel::f64>::quiet_NaN();
    const keel::f64 xr = std::fmod(x, 360.0);
    if (xr == 0.0 || xr == 180.0 || xr == -180.0) return 0.0;
    if (xr == 90.0)  return  1.0;
    if (xr == -90.0) return -1.0;
    return std::sin(xr * kDeg2Rad);
}

inline keel::f64 cosd_scalar(keel::f64 x)
{
    if (!std::isfinite(x)) return std::numeric_limits<keel::f64>::quiet_NaN();
    const keel::f64 xr = std::fmod(x, 360.0);
    if (xr == 90.0 || xr == -90.0 || xr == 270.0 || xr == -270.0) return 0.0;
    if (xr == 0.0) return  1.0;
    if (xr == 180.0 || xr == -180.0) return -1.0;
    return std::cos(xr * kDeg2Rad);
}

// Generic SIMD-dispatched real-vector path. Mirror of unaryRealDouble
// in trig_highway.cpp; complex / scalar inputs delegate to the
// reference path via scalarOp / complexOp lambdas.
template <typename LoopDispatch, typename ScalarOp, typename ComplexOp>
Value unaryRealDoubleRecip(const Value &x, LoopDispatch loop, ScalarOp scalarOp, ComplexOp complexOp, keel::Allocator *mr)
{
    if (x.isComplex()) {
        if (x.isScalar())
            return Value::complexScalar(complexOp(x.toComplex()), mr);
        return unaryComplex(x, complexOp, mr);
    }
    if (x.isScalar())
        return Value::scalar(scalarOp(x.toScalar()), mr);

    Value r = createLike(x, ValueType::DOUBLE, mr);
    if (x.numel() == 0) return r;
    const keel::f64 *in  = x.doubleData();
    keel::f64       *out = r.doubleDataMut();
    numkit::detail::parallel_for(x.numel(), numkit::detail::kTranscendentalThreshold,
        [=](keel::usize s, keel::usize e) {
            loop(in + s, out + s, e - s);
        });
    return r;
}

} // namespace

// â”€â”€ Forward reciprocal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

Value sec(const Value &x, keel::Allocator *mr)
{
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(SecLoop)(in, out, n);
        }, [](keel::f64 v) { return 1.0 / std::cos(v); }, [](const keel::c64 &c) { return keel::c64(1.0) / keel::cos(c); }, mr);
}

Value csc(const Value &x, keel::Allocator *mr)
{
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(CscLoop)(in, out, n);
        }, [](keel::f64 v) { return 1.0 / std::sin(v); }, [](const keel::c64 &c) { return keel::c64(1.0) / keel::sin(c); }, mr);
}

Value cot(const Value &x, keel::Allocator *mr)
{
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(CotLoop)(in, out, n);
        }, [](keel::f64 v) { return std::cos(v) / std::sin(v); }, [](const keel::c64 &c) { return keel::cos(c) / keel::sin(c); }, mr);
}

// â”€â”€ Hyperbolic reciprocal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

Value sech(const Value &x, keel::Allocator *mr)
{
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(SechLoop)(in, out, n);
        }, [](keel::f64 v) { return 1.0 / std::cosh(v); }, [](const keel::c64 &c) { return keel::c64(1.0) / keel::cosh(c); }, mr);
}

Value csch(const Value &x, keel::Allocator *mr)
{
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(CschLoop)(in, out, n);
        }, [](keel::f64 v) { return 1.0 / std::sinh(v); }, [](const keel::c64 &c) { return keel::c64(1.0) / keel::sinh(c); }, mr);
}

Value coth(const Value &x, keel::Allocator *mr)
{
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(CothLoop)(in, out, n);
        }, [](keel::f64 v) { return std::cosh(v) / std::sinh(v); }, [](const keel::c64 &c) { return keel::cosh(c) / keel::sinh(c); }, mr);
}

// â”€â”€ Degree reciprocal: scalar path keeps the integer-multiple-of-90Â° snap.

Value secd(const Value &x, keel::Allocator *mr)
{
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(SecdLoop)(in, out, n);
        }, [](keel::f64 v) { return 1.0 / cosd_scalar(v); }, [](const keel::c64 &c) { return keel::c64(1.0) / keel::cos(c * kDeg2Rad); }, mr);
}

Value cscd(const Value &x, keel::Allocator *mr)
{
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(CscdLoop)(in, out, n);
        }, [](keel::f64 v) { return 1.0 / sind_scalar(v); }, [](const keel::c64 &c) { return keel::c64(1.0) / keel::sin(c * kDeg2Rad); }, mr);
}

Value cotd(const Value &x, keel::Allocator *mr)
{
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(CotdLoop)(in, out, n);
        }, [](keel::f64 v) { return cosd_scalar(v) / sind_scalar(v); }, [](const keel::c64 &c) {
            return keel::cos(c * kDeg2Rad) / keel::sin(c * kDeg2Rad);
        }, mr);
}

// â”€â”€ Inverse reciprocal: scalar fast-path keeps the in-domain
// complex fallback (asec(0.5) â†’ complex in MATLAB).

Value asec(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::acos(keel::c64(1.0) / c); }, mr);
    if (x.isScalar()) {
        const keel::f64 v = x.toScalar();
        if (v > -1.0 && v < 1.0)
            return Value::complexScalar(keel::acos(keel::c64(1.0 / v, 0.0)), mr);
    }
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(AsecLoop)(in, out, n);
        }, [](keel::f64 v) { return std::acos(1.0 / v); }, [](const keel::c64 &c) { return keel::acos(keel::c64(1.0) / c); }, mr);
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
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(AcscLoop)(in, out, n);
        }, [](keel::f64 v) { return std::asin(1.0 / v); }, [](const keel::c64 &c) { return keel::asin(keel::c64(1.0) / c); }, mr);
}

Value acot(const Value &x, keel::Allocator *mr)
{
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(AcotLoop)(in, out, n);
        }, [](keel::f64 v) { return std::atan(1.0 / v); }, [](const keel::c64 &c) { return keel::atan(keel::c64(1.0) / c); }, mr);
}

// â”€â”€ Inverse hyperbolic reciprocal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

Value asech(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::acosh(keel::c64(1.0) / c); }, mr);
    if (x.isScalar()) {
        const keel::f64 v = x.toScalar();
        if (v <= 0.0 || v > 1.0)
            return Value::complexScalar(keel::acosh(keel::c64(1.0 / v, 0.0)), mr);
    }
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(AsechLoop)(in, out, n);
        }, [](keel::f64 v) { return std::acosh(1.0 / v); }, [](const keel::c64 &c) { return keel::acosh(keel::c64(1.0) / c); }, mr);
}

Value acsch(const Value &x, keel::Allocator *mr)
{
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(AcschLoop)(in, out, n);
        }, [](keel::f64 v) { return std::asinh(1.0 / v); }, [](const keel::c64 &c) { return keel::asinh(keel::c64(1.0) / c); }, mr);
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
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(AcothLoop)(in, out, n);
        }, [](keel::f64 v) { return std::atanh(1.0 / v); }, [](const keel::c64 &c) { return keel::atanh(keel::c64(1.0) / c); }, mr);
}

// â”€â”€ Inverse degree reciprocal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(AsecdLoop)(in, out, n);
        }, [](keel::f64 v) { return std::acos(1.0 / v) * kRad2Deg; }, [](const keel::c64 &c) { return keel::acos(keel::c64(1.0) / c) * kRad2Deg; }, mr);
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
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(AcscdLoop)(in, out, n);
        }, [](keel::f64 v) { return std::asin(1.0 / v) * kRad2Deg; }, [](const keel::c64 &c) { return keel::asin(keel::c64(1.0) / c) * kRad2Deg; }, mr);
}

Value acotd(const Value &x, keel::Allocator *mr)
{
    return unaryRealDoubleRecip(x, [](const keel::f64 *in, keel::f64 *out, keel::usize n) {
            HWY_DYNAMIC_DISPATCH(AcotdLoop)(in, out, n);
        }, [](keel::f64 v) { return std::atan(1.0 / v) * kRad2Deg; }, [](const keel::c64 &c) { return keel::atan(keel::c64(1.0) / c) * kRad2Deg; }, mr);
}

} // namespace numkit::builtin

#endif // HWY_ONCE
