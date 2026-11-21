// toolboxes/builtin/src/math/arithmetic/mod_highway.cpp
//
// Highway SIMD for the hot real-double paths of mod(a, b): same-shape
// arrays, array-by-scalar, scalar-by-array. The kernel is exactly the
// scalar formula  r = (b != 0) ? a - floor(a/b)*b : a  evaluated with a
// separate multiply and subtract (no FMA) so the SIMD result is
// bit-identical to the scalar reference in math/arithmetic/misc.cpp.
// Integer-typed operands, complex, empties and broadcasting fall back to
// the scalar elementwiseDouble path in misc.cpp (these never call here).
// rem stays scalar â€” it needs an exact fmod that Highway has no primitive
// for; the a - trunc(a/b)*b approximation drifts from std::fmod.

#include <keel/core.h>
#include "mod_simd.hpp"

#include <cmath>
#include <cstddef>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "builtin/elfun/mod_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace numkit::builtin {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// r = a - floor(a/b)*b, with b == 0 -> a (separate mul/sub, no FMA).
template <class D, class V>
HWY_INLINE V ModVec(D d, V va, V vb)
{
    const auto q = hn::Floor(hn::Div(va, vb));
    const auto r = hn::Sub(va, hn::Mul(q, vb));
    return hn::IfThenElse(hn::Eq(vb, hn::Zero(d)), va, r);
}

void ModLoopVV(const keel::f64 *HWY_RESTRICT a, const keel::f64 *HWY_RESTRICT b,
               keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    keel::usize i = 0;
    for (; i + N <= n; i += N)
        hn::StoreU(ModVec(d, hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, out + i);
    for (; i < n; ++i) out[i] = (b[i] != 0) ? a[i] - std::floor(a[i] / b[i]) * b[i] : a[i];
}

void ModLoopVS(const keel::f64 *HWY_RESTRICT a, keel::f64 s,
               keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto vs = hn::Set(d, s);
    keel::usize i = 0;
    for (; i + N <= n; i += N)
        hn::StoreU(ModVec(d, hn::LoadU(d, a + i), vs), d, out + i);
    for (; i < n; ++i) out[i] = (s != 0) ? a[i] - std::floor(a[i] / s) * s : a[i];
}

void ModLoopSV(keel::f64 s, const keel::f64 *HWY_RESTRICT b,
               keel::f64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto vs = hn::Set(d, s);
    keel::usize i = 0;
    for (; i + N <= n; i += N)
        hn::StoreU(ModVec(d, vs, hn::LoadU(d, b + i)), d, out + i);
    for (; i < n; ++i) out[i] = (b[i] != 0) ? s - std::floor(s / b[i]) * b[i] : s;
}

} // namespace HWY_NAMESPACE
} // namespace numkit::builtin
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace numkit::builtin::detail {

HWY_EXPORT(ModLoopVV);
HWY_EXPORT(ModLoopVS);
HWY_EXPORT(ModLoopSV);

// Serial SIMD (no parallel_for): mod is memory-bandwidth bound, and on
// Arrow Lake the worker-dispatch overhead loses to the single-threaded
// SIMD loop at these array sizes (matches the project's threading note).
void modLoopVV(const keel::f64 *a, const keel::f64 *b, keel::f64 *out, keel::usize n)
{
    HWY_DYNAMIC_DISPATCH(ModLoopVV)(a, b, out, n);
}

void modLoopVS(const keel::f64 *a, keel::f64 scalar, keel::f64 *out, keel::usize n)
{
    HWY_DYNAMIC_DISPATCH(ModLoopVS)(a, scalar, out, n);
}

void modLoopSV(keel::f64 scalar, const keel::f64 *b, keel::f64 *out, keel::usize n)
{
    HWY_DYNAMIC_DISPATCH(ModLoopSV)(scalar, b, out, n);
}

} // namespace numkit::builtin::detail

#endif // HWY_ONCE
