// ops/include/numkit/ops/fused/fused_kernels.hpp
//
// Fused element-wise kernels: each is a hand-written, fixed-structure SIMD
// loop that computes a whole common idiom (an element-wise chain) in ONE pass
// — every array leaf read once, the output written once, intermediates kept in
// registers. This is the form that actually wins: a runtime micro-VM can't keep
// intermediates in registers (variable slot indices spill to the stack), so it
// loses to the per-op SIMD path; a fixed-structure kernel does not.
//
// These are the primitives behind VM element-wise fusion (the compiler/walker
// recognise an idiom and dispatch here, falling back to the per-op path when
// the runtime types don't fit — see the fusion rule registry). Real double
// only; `out` may alias an array input (each element reads its inputs before
// its own store). Large arrays are threaded internally via parallel_for.
//
// One Highway TU per kernel (fused_<name>_highway.cpp) keeps MSVC's per-TU
// inliner budget from leaking between kernels (same reason the FFT kernels are
// split); the scalar fallbacks share fused_kernels_portable.cpp.

#pragma once

#include <keel/core.h>
#include <keel/complex.h>

namespace numkit::ops {

// out[i] = max(lo, min(hi, scale*x[i] + offset))   — affine then clamp.
// Covers normalize+saturate idioms: max(0,min(1, a.*x+b)), rescale, ReLU(+aff).
void fusedAffineClamp(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                      keel::f64 lo, keel::f64 hi, keel::f64 *out, keel::usize n);

// out[i] = min(hi, max(lo, scale*x[i] + offset))   — the min-outer spelling of
// the same clamp. Identical to fusedAffineClamp for finite inputs, but differs
// on NaN (min-outer saturates a NaN to lo, max-outer to hi — matching the
// respective per-op min/max nesting), so it is a separate kernel, not a flag.
void fusedAffineClampMinOuter(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                              keel::f64 lo, keel::f64 hi, keel::f64 *out, keel::usize n);

// Divide-inner clamps: out[i] = clamp((x[i] - sub)/div, lo, hi). (x-c)./d is a
// distinct rounding from scale*x+offset (1/d would round), so these are their
// own kernels — the canonical rescale-then-saturate `max(0,min(1,(x-lo)./rng))`.
void fusedAffineClampShiftDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                              keel::f64 lo, keel::f64 hi, keel::f64 *out, keel::usize n);
void fusedAffineClampMinOuterShiftDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                                      keel::f64 lo, keel::f64 hi, keel::f64 *out,
                                      keel::usize n);

// out[i] = scale*x[i] + offset   — plain affine, NO clamp, NaN-preserving.
// Covers two-op scale-and-shift chains: a.*x+b, x.*a-b, b+a.*x (rescale,
// negate, unit conversion). A separate kernel from fusedAffineClamp because a
// ±inf "clamp" can NOT stand in for "no clamp": fmin(+inf, NaN) = +inf would
// erase a NaN, whereas scale*NaN+offset = NaN flows through here untouched.
void fusedAffine(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                 keel::f64 *out, keel::usize n);

// out[i] = a*x[i] + b*y[i]   — two-array linear combination (x, y same size).
// Covers blend/lerp (b=1-a), weighted sum, scaled difference (b<0), a.*x+b.*y.
// Mul, Mul, Add (no FMA) so it matches the per-op `a.*x + b.*y` bit-for-bit.
void fusedAxpby(const keel::f64 *x, keel::f64 a, const keel::f64 *y, keel::f64 b,
                keel::f64 *out, keel::usize n);

// out[i] = (x[i] - sub) * mul   — center then scale (rescale, normalize,
// z-score by a precomputed reciprocal). Sub then Mul (two roundings) so it is
// bit-identical to the per-op `(x - c).*s`. NaN/Inf propagate naturally.
void fusedShiftScaleMul(const keel::f64 *x, keel::f64 sub, keel::f64 mul,
                        keel::f64 *out, keel::usize n);

// out[i] = (x[i] - sub) / div   — center then divide ((x-mu)./sigma,
// (x-lo)./range). Sub then Div (two roundings) so it is bit-identical to the
// per-op `(x - c)./d` (which can't be folded into a *mul: 1/d would round).
void fusedShiftScaleDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                        keel::f64 *out, keel::usize n);

// out[i] = |scale*x[i] + offset|   — magnitude of an affine. Mul, Add, then
// abs (abs is exact — a sign-bit clear), so bit-identical to `abs(a.*x ± b)`.
// Also serves `abs(x - c)` (scale 1) since |1*x + (-c)| == |x - c|.
void fusedAbsAffine(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                    keel::f64 *out, keel::usize n);

// out[i] = |x[i] - y[i]|   — absolute difference of two arrays (L1 residual /
// error). Sub then abs (exact) → bit-identical to `abs(x - y)`.
void fusedAbsDiff(const keel::f64 *x, const keel::f64 *y, keel::f64 *out, keel::usize n);

// out[i] = |(x[i] - sub)/div|   — abs of a divide-inner (abs(x./d),
// abs((x-c)./d)); the divide rounding differs from scale*x+offset, abs exact.
void fusedAbsShiftDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                      keel::f64 *out, keel::usize n);

// Unary function applied to an affine: out[i] = f(scale*x[i] + offset).
// Restricted to the functions whose SIMD form is bit-identical to the scalar
// libm one — sqrt is correctly-rounded; floor/ceil are exact — so the result
// matches the per-op `f(a.*x ± b)` regardless of how the array is chunked.
// (exp/log/sin… are NOT here: Highway's polynomial differs from libm by a few
// ULP, so they'd only match if the per-op tail-vs-SIMD split were reproduced.)
// Sqrt's negative-input domain (where MATLAB promotes to complex) is handled by
// the caller, which declines before invoking this kernel. Fix is hn::Trunc
// (toward zero, exact); Round is round-half-away-from-zero computed as
// Trunc(v + CopySign(0.5, v)) — NOT hn::Round (which is round-to-nearest-even,
// the wrong tie rule for MATLAB) — mirroring nk's RoundLoop bit-for-bit.
enum class UnaryAffineFn { Sqrt, Floor, Ceil, Fix, Round };
void fusedUnaryAffine(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                      UnaryAffineFn fn, keel::f64 *out, keel::usize n);

// out[i] = f((x[i] - sub) / div) — the divide-inner variant (f(x./d),
// f((x-c)./d)). Sub-then-Div is a distinct rounding from scale*x+offset, so it
// can't reuse fusedUnaryAffine; same f-set (sqrt correctly-rounded, floor/ceil
// exact) → bit-identical to the per-op f((x-c)./d).
void fusedUnaryShiftDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                        UnaryAffineFn fn, keel::f64 *out, keel::usize n);

// out[i] = (scale*x[i] + offset)^2   — squared affine (energy, squared error,
// squared deviation `(x-c).^2`). Mul, Add, then a final Mul (the square); three
// roundings, matching the per-op `(a.*x ± b).^2` now that x.^2 is x.*x.
void fusedSqAffine(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                   keel::f64 *out, keel::usize n);

// out[i] = (x[i] - y[i])^2   — squared difference of two arrays (SSE term).
// Sub then Mul → matches `(x - y).^2`.
void fusedSqDiff(const keel::f64 *x, const keel::f64 *y, keel::f64 *out, keel::usize n);

// out[i] = ((x[i] - sub)/div)^2   — square of a divide-inner ((x./d).^2,
// ((x-c)./d).^2 = squared z-score); divide rounding then a plain-Mul square.
void fusedSqShiftDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                     keel::f64 *out, keel::usize n);

// out[i] = sqrt(x[i]*x[i] + y[i]*y[i])   — magnitude / 2-D length / gradient
// magnitude, the literal `sqrt(x.^2 + y.^2)`. x*x, y*y, Add, Sqrt (no FMA); a
// sum of real squares is never negative, so the result stays real (no complex
// promotion) and sqrt is correctly-rounded → bit-identical to per-op.
void fusedSqrtSumSq(const keel::f64 *x, const keel::f64 *y, keel::f64 *out, keel::usize n);

// out[i] = sign(x[i]) * max(0, |x[i]| - t)   — soft-threshold / wavelet
// shrinkage / L1 proximal. sign matches MATLAB (NaN→NaN, ±0→0, ±Inf→±1) and
// max(0,·) omits NaN; every step is exact/IEEE, so it is bit-identical to the
// per-op `sign(x) .* max(0, abs(x) - t)`.
void fusedSoftThreshold(const keel::f64 *x, keel::f64 t, keel::f64 *out, keel::usize n);

// out[i] = f(scale*x[i] + offset) for a transcendental f. Unlike sqrt/floor/
// ceil, these Highway forms (hwy/contrib/math) differ from libm by a few ULP,
// so to stay bit-identical to the per-op f(a.*x ± b) this kernel mirrors
// nk's exp/log/sin/… loop exactly: the same hn:: on the SIMD body, the same
// std:: on the per-chunk scalar tail, chunked by the same kTranscendentalThreshold.
// Cosh has no Highway primitive — it is composed 0.5*(Exp(v)+Exp(-v)) exactly as
// nk's CoshLoop. Tan has none either: it mirrors nk's TanLoop — SLEEF's
// xtan kernel (TanVec) on lanes with |inner| < 1e6, a per-block std::tan fallback
// otherwise (so the per-block decision lands on the same inner values as per-op).
// Always-real (no domain guard): Exp, Expm1, Sin, Cos, Tanh, Sinh, Atan, Asinh,
// Cosh, Tan. Complex-promoting in MATLAB — the caller declines on the offending
// range (a pre-scan) and lets the per-op path produce the complex result:
// {Log,Log2,Log10} on a negative; Log1p on < -1; Acosh on < 1; {Asin,Acos,Atanh}
// on |·| > 1.
enum class TransAffineFn { Exp, Expm1, Log, Log2, Log10, Sin, Cos, Tanh,
                           Sinh, Atan, Asinh, Asin, Acos, Acosh, Atanh,
                           Log1p, Cosh, Tan };
void fusedTransAffine(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                      TransAffineFn fn, keel::f64 *out, keel::usize n);

// out[i] = f((x[i] - sub) / div) — the divide-inner variant (f(x./d),
// f((x-c)./d)) of fusedTransAffine. Same mirror-the-nk-loop discipline.
void fusedTransShiftDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                        TransAffineFn fn, keel::f64 *out, keel::usize n);

// ---- complex-input kernels ---------------------------------------------
// When the array operand is complex, fusion routes here. These are scalar
// std::complex<double> loops (Highway has no complex transcendentals, and
// nk's per-op complex path is itself scalar std::complex), so each kernel
// mirrors nk's per-op composition with the SAME std::complex operators /
// std:: functions on the SAME compiler — bit-identical by construction. They
// still eliminate the intermediate temporaries (the DRAM-traffic win), just
// without SIMD. `out` may alias `x`. Complex is "total" — no domain declines.
// out[i] = scale*x[i] + offset (mirrors `a.*z` then `+b`).
void fusedAffineCx(const keel::c64 *x, keel::c64 scale, keel::c64 offset, keel::c64 *out, keel::usize n);
// out[i] = a*x[i] + b*y[i].
void fusedAxpbyCx(const keel::c64 *x, keel::c64 a, const keel::c64 *y, keel::c64 b, keel::c64 *out, keel::usize n);
// out[i] = (x[i] - sub) * mul  /  (x[i] - sub) / div.
void fusedShiftScaleMulCx(const keel::c64 *x, keel::c64 sub, keel::c64 mul, keel::c64 *out, keel::usize n);
void fusedShiftScaleDivCx(const keel::c64 *x, keel::c64 sub, keel::c64 div, keel::c64 *out, keel::usize n);
// The complex f(inner) kernels share an inner shape selected by `affine`:
//   affine=true  → scale*x[i] + offset  (genuine complex multiply: the
//                  product-coefficient inner kinds a.*z[±b]).
//   affine=false → (scale==+1 ? x[i] : -x[i]) + offset  (a pure shift/neg, where
//                  scale is exactly ±1: add/negate, NOT a complex mul by (±1+0i),
//                  whose 0*Inf=NaN would diverge from the per-op bare add/sub).
// This lets every affine spelling (product, shift x±c, c-x, -x) fuse on complex.

// out[i] = std::pow(<inner>, 2). On a COMPLEX base, nk's `.^2` takes the
// std::pow branch (the .^2 == z.*z fast-path is real-only), so these mirror
// std::pow(v, 2) — NOT v*v (they differ for complex).
void fusedSqAffineCx(const keel::c64 *x, keel::c64 scale, keel::c64 offset, bool affine,
                     keel::c64 *out, keel::usize n);
void fusedSqDiffCx(const keel::c64 *x, const keel::c64 *y, keel::c64 *out, keel::usize n);
// (z-c).^2 / (c-z).^2 — array-scalar squared diff. `rev` picks the subtract
// order: complex pow(-w,2) != pow(w,2), so the order is NOT foldable (unlike a
// real square). The subtract is genuine (no mul-by-1).
void fusedSqDiffScalarCx(const keel::c64 *z, keel::c64 c, bool rev, keel::c64 *out, keel::usize n);
// sqrt(x.^2 + y.^2) on complex arrays → std::sqrt(pow(x,2)+pow(y,2)) (complex).
void fusedSqrtSumSqCx(const keel::c64 *x, const keel::c64 *y, keel::c64 *out, keel::usize n);
// ((x-sub)/div).^2 — divide-inner square on complex → std::pow((x-sub)/div, 2).
void fusedSqShiftDivCx(const keel::c64 *x, keel::c64 sub, keel::c64 div, keel::c64 *out, keel::usize n);

// sqrt of a complex affine / divide-inner: std::sqrt(scale*x+offset) /
// std::sqrt((x-sub)/div) — whole-complex (not component-wise).
void fusedSqrtAffineCx(const keel::c64 *x, keel::c64 scale, keel::c64 offset, bool affine,
                       keel::c64 *out, keel::usize n);
void fusedSqrtShiftDivCx(const keel::c64 *x, keel::c64 sub, keel::c64 div, keel::c64 *out, keel::usize n);

// floor/ceil/fix/round of a complex affine / divide-inner — applied COMPONENT-
// WISE to re and im (nk's per-op roundLikeDispatch) with the same std::
// floor/ceil/trunc/round. fn must be one of {Floor,Ceil,Fix,Round} (Sqrt has its
// own whole-complex kernel above).
void fusedRoundLikeAffineCx(const keel::c64 *x, keel::c64 scale, keel::c64 offset, bool affine,
                            UnaryAffineFn fn, keel::c64 *out, keel::usize n);
void fusedRoundLikeShiftDivCx(const keel::c64 *x, keel::c64 sub, keel::c64 div, UnaryAffineFn fn,
                              keel::c64 *out, keel::usize n);

// transcendental of a complex affine / divide-inner: f(scale*x+offset) /
// f((x-sub)/div), mirroring nk's complex op per fn — mostly std::F(z), with
// log2 = log(z)/log(2), log1p = log(1+z), expm1 = exp(z)-1.
void fusedTransAffineCx(const keel::c64 *x, keel::c64 scale, keel::c64 offset, bool affine,
                        TransAffineFn fn, keel::c64 *out, keel::usize n);
void fusedTransShiftDivCx(const keel::c64 *x, keel::c64 sub, keel::c64 div, TransAffineFn fn,
                          keel::c64 *out, keel::usize n);

// abs of a complex affine / div-inner / two-array diff → the REAL magnitude
// std::abs(z) (= hypot), so `out` is double. |a.*z+b|, |z±c|, |-z|, |z./d|, |z-w|.
void fusedAbsAffineCx(const keel::c64 *x, keel::c64 scale, keel::c64 offset, bool affine,
                      keel::f64 *out, keel::usize n);
void fusedAbsShiftDivCx(const keel::c64 *x, keel::c64 sub, keel::c64 div, keel::f64 *out, keel::usize n);
void fusedAbsDiffCx(const keel::c64 *x, const keel::c64 *y, keel::f64 *out, keel::usize n);
// |z - c| array-scalar (= |c - z|, order-agnostic) → real magnitude.
void fusedAbsDiffScalarCx(const keel::c64 *z, keel::c64 c, keel::f64 *out, keel::usize n);

// complex soft-threshold: out[i] = sign(z) .* max(0, |z| - t), sign(z) = z/|z|
// (0 at z==0) — MATLAB's complex shrinkage. t real; out complex.
void fusedSoftThresholdCx(const keel::c64 *x, keel::f64 t, keel::c64 *out, keel::usize n);

// ---- shared scalar bodies (small-N gate ↔ portable fallback) -----------
// Internal — NOT part of the fusion API (declared alongside the kernels, in
// numkit::ops rather than ::detail, because the public fused kernels live here
// too and a bare `detail::` in the Highway TUs already means numkit::detail).
// The kernels that carry a small-N inline gate (fusedAffine, fusedAbs*,
// fusedSq*) share their scalar body here: the portable fallback and the Highway
// gate both call these, so they are one definition (can't drift) and the loop
// lives in a lambda-free TU (fused_scalar.cpp) that auto-vectorizes regardless
// of the dispatcher's parallel_for capture. See fused_scalar.cpp +
// bugs/ops/cheap-elementwise-simd-small-n. Bit-identical to the per-op path
// (one IEEE chain each).
void fusedAffineScalar(const keel::f64 *x, keel::f64 scale, keel::f64 offset, keel::f64 *out, keel::usize n);
void fusedAbsAffineScalar(const keel::f64 *x, keel::f64 scale, keel::f64 offset, keel::f64 *out, keel::usize n);
void fusedAbsShiftDivScalar(const keel::f64 *x, keel::f64 sub, keel::f64 div, keel::f64 *out, keel::usize n);
void fusedAbsDiffScalar(const keel::f64 *x, const keel::f64 *y, keel::f64 *out, keel::usize n);
void fusedSqAffineScalar(const keel::f64 *x, keel::f64 scale, keel::f64 offset, keel::f64 *out, keel::usize n);
void fusedSqShiftDivScalar(const keel::f64 *x, keel::f64 sub, keel::f64 div, keel::f64 *out, keel::usize n);
void fusedSqDiffScalar(const keel::f64 *x, const keel::f64 *y, keel::f64 *out, keel::usize n);

} // namespace numkit::ops
