// ops/src/fused/fused_abs_highway.cpp
//
// SIMD kernels for the absolute-value idioms:
//   fusedAbsAffine: out[i] = |scale*x[i] + offset|   — `abs(a.*x ± b)`, `abs(x-c)`
//   fusedAbsDiff:   out[i] = |x[i] - y[i]|           — `abs(x - y)` (L1 residual)
// abs is exact (a sign-bit clear, no rounding), so the only roundings are the
// Mul+Add / Sub before it — making each kernel bit-identical to its per-op
// spelling. NaN/Inf flow through naturally (|NaN| = NaN, |±Inf| = Inf).

#include <keel/core.h>
#include <keel/math.h>
#include <numkit/ops/fused/fused_kernels.hpp>
#include <numkit/ops/parallel_for.hpp>

#include <cstddef>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fused/fused_abs_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace numkit::ops {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// inner is affine (p0*x + p1) or shift-divide ((x - p0)/p1) per `divide` (a
// loop-invariant branch). abs is exact (a sign-bit clear), so each form is
// bit-identical to its per-op spelling.
void AbsAffineImpl(const keel::f64 *x, keel::f64 p0, keel::f64 p1, bool divide,
                   keel::f64 *out, keel::usize n) {
    const hn::ScalableTag<keel::f64> d;
    const keel::usize L = hn::Lanes(d);
    const auto vp0 = hn::Set(d, p0), vp1 = hn::Set(d, p1);
    keel::usize i = 0;
    for (; i + L <= n; i += L) {
        const auto xv = hn::LoadU(d, x + i);
        const auto v = divide ? hn::Div(hn::Sub(xv, vp0), vp1)
                              : hn::Add(hn::Mul(xv, vp0), vp1);
        hn::StoreU(hn::Abs(v), d, out + i);
    }
    for (; i < n; ++i)
        out[i] = keel::abs(divide ? (x[i] - p0) / p1 : p0 * x[i] + p1);
}

void AbsDiffImpl(const keel::f64 *x, const keel::f64 *y, keel::f64 *out, keel::usize n) {
    const hn::ScalableTag<keel::f64> d;
    const keel::usize L = hn::Lanes(d);
    keel::usize i = 0;
    for (; i + L <= n; i += L)
        hn::StoreU(hn::Abs(hn::Sub(hn::LoadU(d, x + i), hn::LoadU(d, y + i))), d, out + i);
    for (; i < n; ++i) out[i] = keel::abs(x[i] - y[i]);
}

} // namespace HWY_NAMESPACE
} // namespace numkit::ops
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace numkit::ops {

HWY_EXPORT(AbsAffineImpl);
HWY_EXPORT(AbsDiffImpl);

void fusedAbsAffine(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                    keel::f64 *out, keel::usize n) {
    if (n == 0) return;
    // Small native arrays delegate to the shared scalar kernel (fused_scalar.cpp)
    // rather than an inline loop: one body shared with the portable fallback
    // (can't drift), and living in a lambda-free TU it vectorizes regardless of
    // the worker capture below — the [&]-escape alias trap can't recur (see
    // bugs/ops/cheap-elementwise-simd-small-n).
    if (n < numkit::detail::kSimdInlineThreshold) {
        fusedAbsAffineScalar(x, scale, offset, out, n);
        return;
    }
    detail::parallel_for(n, keel::usize{1} << 16, [=](keel::usize s, keel::usize e) {
        HWY_DYNAMIC_DISPATCH(AbsAffineImpl)(x + s, scale, offset, /*divide=*/false,
                                            out + s, e - s);
    });
}

void fusedAbsShiftDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                      keel::f64 *out, keel::usize n) {
    if (n == 0) return;
    if (n < numkit::detail::kSimdInlineThreshold) {   // delegate (see fusedAbsAffine)
        fusedAbsShiftDivScalar(x, sub, div, out, n);
        return;
    }
    detail::parallel_for(n, keel::usize{1} << 16, [=](keel::usize s, keel::usize e) {
        HWY_DYNAMIC_DISPATCH(AbsAffineImpl)(x + s, sub, div, /*divide=*/true,
                                            out + s, e - s);
    });
}

void fusedAbsDiff(const keel::f64 *x, const keel::f64 *y, keel::f64 *out, keel::usize n) {
    if (n == 0) return;
    if (n < numkit::detail::kSimdInlineThreshold) {   // delegate (see fusedAbsAffine)
        fusedAbsDiffScalar(x, y, out, n);
        return;
    }
    detail::parallel_for(n, keel::usize{1} << 16, [=](keel::usize s, keel::usize e) {
        HWY_DYNAMIC_DISPATCH(AbsDiffImpl)(x + s, y + s, out + s, e - s);
    });
}

} // namespace numkit::ops
#endif // HWY_ONCE
