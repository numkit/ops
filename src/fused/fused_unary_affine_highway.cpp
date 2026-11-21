// ops/src/fused/fused_unary_affine_highway.cpp
//
// SIMD kernel for fusedUnaryAffine: out[i] = f(scale*x[i] + offset), one
// streaming pass instead of the per-op affine temporary + the unary's own pass.
// Only f ∈ {sqrt, floor, ceil} — each has a SIMD form bit-identical to the
// scalar libm one (sqrt maps to vsqrtpd = correctly-rounded IEEE; floor/ceil
// are exact directed roundings), so the fused result matches `f(a.*x ± b)`
// bit-for-bit no matter where the array is split. The affine is Mul-then-Add
// (two roundings, NOT FMA) to match the affine fallback.

#include <keel/core.h>
#include <numkit/ops/fused/fused_kernels.hpp>
#include <numkit/ops/parallel_for.hpp>

#include <cmath>
#include <cstddef>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fused/fused_unary_affine_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace numkit::ops {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// One inner+unary pass. The inner is either affine (p0*x + p1) or shift-divide
// ((x - p0)/p1) per `divide` — a loop-invariant branch, so the affine path is
// unchanged. VecOp maps a vector → vector (the Highway unary); ScalarOp maps the
// tail double → double (the matching libm unary).
template <class VecOp, class ScalarOp>
void unaryInnerLoop(const keel::f64 *x, keel::f64 p0, keel::f64 p1, bool divide,
                    keel::f64 *out, keel::usize n, VecOp vop, ScalarOp sop) {
    const hn::ScalableTag<keel::f64> d;
    const keel::usize L = hn::Lanes(d);
    const auto vp0 = hn::Set(d, p0), vp1 = hn::Set(d, p1);
    keel::usize i = 0;
    for (; i + L <= n; i += L) {
        const auto v = hn::LoadU(d, x + i);
        const auto inner = divide ? hn::Div(hn::Sub(v, vp0), vp1)
                                  : hn::Add(hn::Mul(v, vp0), vp1);
        hn::StoreU(vop(inner), d, out + i);
    }
    for (; i < n; ++i)
        out[i] = sop(divide ? (x[i] - p0) / p1 : p0 * x[i] + p1);
}

void UnaryAffineImpl(const keel::f64 *x, keel::f64 p0, keel::f64 p1, bool divide,
                     UnaryAffineFn fn, keel::f64 *out, keel::usize n) {
    switch (fn) {
        case UnaryAffineFn::Sqrt:
            unaryInnerLoop(x, p0, p1, divide, out, n,
                           [](auto v) { return hn::Sqrt(v); },
                           [](keel::f64 v) { return std::sqrt(v); });
            break;
        case UnaryAffineFn::Floor:
            unaryInnerLoop(x, p0, p1, divide, out, n,
                           [](auto v) { return hn::Floor(v); },
                           [](keel::f64 v) { return std::floor(v); });
            break;
        case UnaryAffineFn::Ceil:
            unaryInnerLoop(x, p0, p1, divide, out, n,
                           [](auto v) { return hn::Ceil(v); },
                           [](keel::f64 v) { return std::ceil(v); });
            break;
        case UnaryAffineFn::Fix:
            unaryInnerLoop(x, p0, p1, divide, out, n,
                           [](auto v) { return hn::Trunc(v); },
                           [](keel::f64 v) { return std::trunc(v); });
            break;
        case UnaryAffineFn::Round:
            // MATLAB round = half-away-from-zero: Trunc(v + CopySign(0.5, v)),
            // mirroring nk's RoundLoop (NOT hn::Round = round-to-even).
            unaryInnerLoop(x, p0, p1, divide, out, n,
                           [](auto v) {
                               const hn::DFromV<decltype(v)> d;
                               return hn::Trunc(hn::Add(v, hn::CopySign(hn::Set(d, 0.5), v)));
                           },
                           [](keel::f64 v) { return std::round(v); });
            break;
    }
}

} // namespace HWY_NAMESPACE
} // namespace numkit::ops
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace numkit::ops {

HWY_EXPORT(UnaryAffineImpl);

void fusedUnaryAffine(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                      UnaryAffineFn fn, keel::f64 *out, keel::usize n) {
    if (n == 0) return;
    detail::parallel_for(n, keel::usize{1} << 16, [&](keel::usize s, keel::usize e) {
        HWY_DYNAMIC_DISPATCH(UnaryAffineImpl)(x + s, scale, offset, /*divide=*/false,
                                              fn, out + s, e - s);
    });
}

void fusedUnaryShiftDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                        UnaryAffineFn fn, keel::f64 *out, keel::usize n) {
    if (n == 0) return;
    detail::parallel_for(n, keel::usize{1} << 16, [&](keel::usize s, keel::usize e) {
        HWY_DYNAMIC_DISPATCH(UnaryAffineImpl)(x + s, sub, div, /*divide=*/true,
                                              fn, out + s, e - s);
    });
}

} // namespace numkit::ops
#endif // HWY_ONCE
