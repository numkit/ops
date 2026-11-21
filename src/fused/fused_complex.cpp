// ops/src/fused/fused_complex.cpp
//
// Complex-input fused kernels. Unlike the real kernels these are plain scalar
// std::complex<double> loops — Highway has no complex transcendentals, and
// nk's own per-op complex path is scalar std::complex too, so there is no
// SIMD form to mirror. Each kernel reproduces nk's per-op composition with
// the SAME std::complex operators / std:: functions on the SAME compiler, so the
// fused result is bit-identical to the unfused one by construction (the parity
// harness checks it via isequaln). The win here is temporary elimination (fewer
// passes over memory), not SIMD. Complex arithmetic / transcendentals are total
// (no real-domain promotion), so no kernel needs a domain guard.
//
// One file (not split per kernel like the Highway TUs): scalar loops carry no
// per-TU inliner-budget cost. Lives in the always-on ops sources so it is in
// both the SIMD and portable builds.

#include <keel/core.h>
#include <keel/math.h>
#include <numkit/ops/fused/fused_kernels.hpp>

#include <cmath>
#include <cstddef>

namespace numkit::ops {

// The inner of a complex f(inner): affine (genuine complex multiply, for the
// product-coefficient kinds) or a pure shift/neg (scale is exactly ±1 → add or
// negate, NOT a complex mul by (±1+0i), which would make 0*Inf=NaN and diverge
// from the per-op bare add/sub on a non-finite z).
static inline keel::c64 cxInner(const keel::c64 &z, const keel::c64 &scale, const keel::c64 &offset,
                         bool affine) {
    if (affine) return scale * z + offset;
    return (scale.real() > 0.0 ? z : -z) + offset;
}

void fusedAffineCx(const keel::c64 *x, keel::c64 scale, keel::c64 offset, keel::c64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = scale * x[i] + offset;
}

void fusedAxpbyCx(const keel::c64 *x, keel::c64 a, const keel::c64 *y, keel::c64 b, keel::c64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = a * x[i] + b * y[i];
}

void fusedShiftScaleMulCx(const keel::c64 *x, keel::c64 sub, keel::c64 mul, keel::c64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = (x[i] - sub) * mul;
}

void fusedShiftScaleDivCx(const keel::c64 *x, keel::c64 sub, keel::c64 div, keel::c64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = (x[i] - sub) / div;
}

// z.^2 on a COMPLEX base is std::pow(z, 2) in nk (the .^2 == z.*z fast-path
// is real-only — for complex, elementPower takes the std::pow branch), so the
// complex square mirrors that, NOT v*v (std::pow(z,2) != z*z for complex).
void fusedSqAffineCx(const keel::c64 *x, keel::c64 scale, keel::c64 offset, bool affine,
                     keel::c64 *out, keel::usize n) {
    const keel::c64 two(2.0, 0.0);
    for (keel::usize i = 0; i < n; ++i)
        out[i] = keel::pow(cxInner(x[i], scale, offset, affine), two);
}

void fusedSqDiffCx(const keel::c64 *x, const keel::c64 *y, keel::c64 *out, keel::usize n) {
    const keel::c64 two(2.0, 0.0);
    for (keel::usize i = 0; i < n; ++i)
        out[i] = keel::pow(x[i] - y[i], two);
}

// (z-c).^2 / (c-z).^2 — `rev` picks the subtract order. pow(-w,2) != pow(w,2)
// for complex (branch), so order is preserved (not folded like a real square).
void fusedSqDiffScalarCx(const keel::c64 *z, keel::c64 c, bool rev, keel::c64 *out, keel::usize n) {
    const keel::c64 two(2.0, 0.0);
    for (keel::usize i = 0; i < n; ++i)
        out[i] = keel::pow(rev ? c - z[i] : z[i] - c, two);
}

void fusedSqrtSumSqCx(const keel::c64 *x, const keel::c64 *y, keel::c64 *out, keel::usize n) {
    const keel::c64 two(2.0, 0.0);
    for (keel::usize i = 0; i < n; ++i)
        out[i] = keel::sqrt(keel::pow(x[i], two) + keel::pow(y[i], two));
}

void fusedSqShiftDivCx(const keel::c64 *x, keel::c64 sub, keel::c64 div, keel::c64 *out, keel::usize n) {
    const keel::c64 two(2.0, 0.0);
    for (keel::usize i = 0; i < n; ++i)
        out[i] = keel::pow((x[i] - sub) / div, two);
}

void fusedSqrtAffineCx(const keel::c64 *x, keel::c64 scale, keel::c64 offset, bool affine,
                       keel::c64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i)
        out[i] = keel::sqrt(cxInner(x[i], scale, offset, affine));
}

void fusedSqrtShiftDivCx(const keel::c64 *x, keel::c64 sub, keel::c64 div, keel::c64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = keel::sqrt((x[i] - sub) / div);
}

// floor/ceil/fix/round on a COMPLEX value apply the scalar rounder to the real
// and imaginary parts INDEPENDENTLY (MATLAB R2025b; nk's per-op
// roundLikeDispatch does Complex(scalar(re), scalar(im)) with the SAME std::
// functions). Mirrors that exactly → bit-identical to the unfused per-op result.
// (Sqrt is whole-complex, not component-wise — it has its own kernel.)
static inline keel::f64 roundLikeCxScalar(UnaryAffineFn fn, keel::f64 d) {
    switch (fn) {
        case UnaryAffineFn::Floor: return std::floor(d);
        case UnaryAffineFn::Ceil:  return std::ceil(d);
        case UnaryAffineFn::Fix:   return std::trunc(d);
        case UnaryAffineFn::Round: return std::round(d);   // half-away (std::round)
        case UnaryAffineFn::Sqrt:  break;                  // not a roundLike fn
    }
    return d;
}

void fusedRoundLikeAffineCx(const keel::c64 *x, keel::c64 scale, keel::c64 offset, bool affine,
                            UnaryAffineFn fn, keel::c64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) {
        const keel::c64 v = cxInner(x[i], scale, offset, affine);
        out[i] = keel::c64(roundLikeCxScalar(fn, v.real()), roundLikeCxScalar(fn, v.imag()));
    }
}

void fusedRoundLikeShiftDivCx(const keel::c64 *x, keel::c64 sub, keel::c64 div, UnaryAffineFn fn,
                              keel::c64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) {
        const keel::c64 v = (x[i] - sub) / div;
        out[i] = keel::c64(roundLikeCxScalar(fn, v.real()), roundLikeCxScalar(fn, v.imag()));
    }
}

// Per-element complex transcendental, mirroring nk's complex op for each fn
// (the complexOp lambdas in trig/exp_log): mostly std::F(z); log2 = log(z)/log2,
// log1p = log(1+z), expm1 = exp(z)-1 (matches the per-op complex expm1).
static inline keel::c64 transCxApply(TransAffineFn fn, keel::c64 v) {
    switch (fn) {
        case TransAffineFn::Exp:   return keel::exp(v);
        case TransAffineFn::Log:   return keel::log(v);
        case TransAffineFn::Log2:  return keel::log(v) / std::log(2.0);
        case TransAffineFn::Log10: return keel::log10(v);
        case TransAffineFn::Log1p: return keel::log(keel::c64(1.0, 0.0) + v);
        case TransAffineFn::Sin:   return keel::sin(v);
        case TransAffineFn::Cos:   return keel::cos(v);
        case TransAffineFn::Tan:   return keel::tan(v);
        case TransAffineFn::Tanh:  return keel::tanh(v);
        case TransAffineFn::Sinh:  return keel::sinh(v);
        case TransAffineFn::Cosh:  return keel::cosh(v);
        case TransAffineFn::Atan:  return keel::atan(v);
        case TransAffineFn::Asinh: return keel::asinh(v);
        case TransAffineFn::Asin:  return keel::asin(v);
        case TransAffineFn::Acos:  return keel::acos(v);
        case TransAffineFn::Acosh: return keel::acosh(v);
        case TransAffineFn::Atanh: return keel::atanh(v);
        case TransAffineFn::Expm1: return keel::exp(v) - 1.0;  // exp(z)-1 (per-op match)
    }
    return v;
}

void fusedTransAffineCx(const keel::c64 *x, keel::c64 scale, keel::c64 offset, bool affine,
                        TransAffineFn fn, keel::c64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i)
        out[i] = transCxApply(fn, cxInner(x[i], scale, offset, affine));
}

void fusedTransShiftDivCx(const keel::c64 *x, keel::c64 sub, keel::c64 div, TransAffineFn fn,
                          keel::c64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i)
        out[i] = transCxApply(fn, (x[i] - sub) / div);
}

// abs of complex → the REAL magnitude std::abs(z) (mirrors nk's abs(complex),
// which is per-element std::abs). out is double.
void fusedAbsAffineCx(const keel::c64 *x, keel::c64 scale, keel::c64 offset, bool affine,
                      keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i)
        out[i] = keel::abs(cxInner(x[i], scale, offset, affine));
}

void fusedAbsShiftDivCx(const keel::c64 *x, keel::c64 sub, keel::c64 div, keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = keel::abs((x[i] - sub) / div);
}

void fusedAbsDiffCx(const keel::c64 *x, const keel::c64 *y, keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = keel::abs(x[i] - y[i]);
}

// |z - c| (= |c - z|, since |w| == |-w|) → real magnitude.
void fusedAbsDiffScalarCx(const keel::c64 *z, keel::c64 c, keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = keel::abs(z[i] - c);
}

// complex soft-threshold: sign(z) .* max(0, |z| - t). sign(z) = z/|z| (0 at
// z==0, MATLAB R2025b); the .* by the real max promotes it to Complex(mx,0)
// (full complex mul), exactly as nk's per-op sign(z).*max(...).
void fusedSoftThresholdCx(const keel::c64 *x, keel::f64 t, keel::c64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) {
        const keel::c64 z = x[i];
        const keel::f64 m = keel::abs(z);
        const keel::c64 s = (m == 0.0) ? keel::c64(0.0, 0.0) : z / m;   // sign(z)
        out[i] = s * keel::c64(std::fmax(0.0, m - t), 0.0);
    }
}

} // namespace numkit::ops
