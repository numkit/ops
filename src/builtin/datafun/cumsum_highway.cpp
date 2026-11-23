// toolboxes/builtin/src/math/elementary/backends/cumsum_highway.cpp
//
// Highway dynamic-dispatch prefix-op family (Hillis-Steele) for cumsum,
// cumprod, cummax, cummin. Per N-lane SIMD vector: log2(N) `(shift +
// op)` steps compute the inclusive prefix within the vector, then the
// op is applied with a broadcasted carry from the previous vector's
// last lane.
//
// The shift step in Hillis-Steele defaults the filled lanes to zero
// (Highway `ShiftLeftLanes` semantics). For Add this is the identity;
// for Mul / Max / Min it's wrong (multiplying by 0 zeroes the result;
// max-against-0 is incorrect for negative inputs; etc.). The generic
// kernel below uses a per-step `IfThenElse(Iota < K, identity, shifted)`
// to overwrite the filled lanes with the correct identity. cumsum
// keeps the original zero-fill version since it skips that overhead.
//
// For cummax / cummin: NaN inputs are pre-masked to the identity
// (-inf / +inf) so they don't contaminate the running aggregate. The
// leading-NaN prefix in src is preserved as NaN in dst by handling it
// scalar-ly before entering the SIMD loop.

#include <keel/core.h>
#include <keel/math.h>
#include "cumsum.hpp"


#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "builtin/datafun/cumsum_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace numkit::builtin {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// IMPORTANT: AVX2 `ShiftLeftLanes` lowers to `_mm256_slli_si256` which
// shifts each 128-bit half independently â€” so for a 4-lane double
// vector `[a, b, c, d]`, ShiftLeftLanes<1> gives `[0, a, 0, c]`, NOT
// `[0, a, b, c]`. We need a true cross-lane slide. Highway's
// `SlideUpLanes(d, v, amt)` handles this correctly across all targets.

// Sum-specific scan: identity is 0, which matches SlideUpLanes' zero-
// fill â€” no IfThenElse correction needed.
template <class D>
HWY_INLINE auto inVectorScanSum(D d, hn::VFromD<D> v)
{
    constexpr keel::usize N = hn::MaxLanes(d);
    if constexpr (N >= 2) v = hn::Add(v, hn::SlideUpLanes(d, v, 1));
    if constexpr (N >= 4) v = hn::Add(v, hn::SlideUpLanes(d, v, 2));
    if constexpr (N >= 8) v = hn::Add(v, hn::SlideUpLanes(d, v, 4));
    return v;
}

// Generic Hillis-Steele scan with a non-zero identity. After each
// SlideUpLanes<K>, the K filled lanes (which carry zeros) are
// overwritten with `identity` via IfThenElse(FirstN(K), idVec, slid).
// Used by cumprod (identity=1), cummax (identity=-inf), cummin
// (identity=+inf).
template <class D, class Op>
HWY_INLINE auto inVectorScanGen(D d, hn::VFromD<D> v, keel::f64 identity, Op op)
{
    constexpr keel::usize N = hn::MaxLanes(d);
    const auto idVec = hn::Set(d, identity);
    if constexpr (N >= 2) {
        auto s = hn::SlideUpLanes(d, v, 1);
        s = hn::IfThenElse(hn::FirstN(d, 1), idVec, s);
        v = op(v, s);
    }
    if constexpr (N >= 4) {
        auto s = hn::SlideUpLanes(d, v, 2);
        s = hn::IfThenElse(hn::FirstN(d, 2), idVec, s);
        v = op(v, s);
    }
    if constexpr (N >= 8) {
        auto s = hn::SlideUpLanes(d, v, 4);
        s = hn::IfThenElse(hn::FirstN(d, 4), idVec, s);
        v = op(v, s);
    }
    return v;
}

void CumsumScanLoop(const keel::f64 *HWY_RESTRICT src, keel::f64 *HWY_RESTRICT dst,
                    keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    keel::f64 carry = 0.0;
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, src + i);
        const auto scanned = inVectorScanSum(d, v);
        const auto carriedV = hn::Add(scanned, hn::Set(d, carry));
        hn::StoreU(carriedV, d, dst + i);
        carry = hn::ExtractLane(carriedV, N - 1);
    }
    for (; i < n; ++i) {
        carry += src[i];
        dst[i] = carry;
    }
}

void CumprodScanLoop(const keel::f64 *HWY_RESTRICT src, keel::f64 *HWY_RESTRICT dst,
                     keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    keel::f64 carry = 1.0;
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        const auto v = hn::LoadU(d, src + i);
        const auto scanned = inVectorScanGen(d, v, 1.0,
            [](auto a, auto b) { return hn::Mul(a, b); });
        const auto carriedV = hn::Mul(scanned, hn::Set(d, carry));
        hn::StoreU(carriedV, d, dst + i);
        alignas(64) keel::f64 tmp[hn::MaxLanes(d)];
        hn::Store(carriedV, d, tmp);
        carry = tmp[N - 1];
    }
    for (; i < n; ++i) {
        carry *= src[i];
        dst[i] = carry;
    }
}

// cummax / cummin: NaN-skip by replacing NaN with identity per lane.
// Pre-masked vector goes into the standard prefix-Max scan.
void CummaxScanLoopBody(const keel::f64 *HWY_RESTRICT src,
                        keel::f64 *HWY_RESTRICT dst, keel::usize n,
                        keel::f64 initialCarry)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    constexpr keel::f64 NEG_INF = -keel::Limits<keel::f64>::infinity();
    const auto neg_inf_v = hn::Set(d, NEG_INF);

    keel::f64 carry = initialCarry;
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, src + i);
        v = hn::IfThenElse(hn::IsNaN(v), neg_inf_v, v);
        const auto scanned = inVectorScanGen(d, v, NEG_INF,
            [](auto a, auto b) { return hn::Max(a, b); });
        const auto carriedV = hn::Max(scanned, hn::Set(d, carry));
        hn::StoreU(carriedV, d, dst + i);
        carry = hn::ExtractLane(carriedV, N - 1);
    }
    for (; i < n; ++i) {
        const keel::f64 xi = src[i];
        if (!keel::is_nan(xi) && xi > carry) carry = xi;
        dst[i] = carry;
    }
}

void CumminScanLoopBody(const keel::f64 *HWY_RESTRICT src,
                        keel::f64 *HWY_RESTRICT dst, keel::usize n,
                        keel::f64 initialCarry)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    constexpr keel::f64 POS_INF = keel::Limits<keel::f64>::infinity();
    const auto pos_inf_v = hn::Set(d, POS_INF);

    keel::f64 carry = initialCarry;
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, src + i);
        v = hn::IfThenElse(hn::IsNaN(v), pos_inf_v, v);
        const auto scanned = inVectorScanGen(d, v, POS_INF,
            [](auto a, auto b) { return hn::Min(a, b); });
        const auto carriedV = hn::Min(scanned, hn::Set(d, carry));
        hn::StoreU(carriedV, d, dst + i);
        carry = hn::ExtractLane(carriedV, N - 1);
    }
    for (; i < n; ++i) {
        const keel::f64 xi = src[i];
        if (!keel::is_nan(xi) && xi < carry) carry = xi;
        dst[i] = carry;
    }
}

} // namespace HWY_NAMESPACE
} // namespace numkit::builtin
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace numkit::builtin {

HWY_EXPORT(CumsumScanLoop);
HWY_EXPORT(CumprodScanLoop);
HWY_EXPORT(CummaxScanLoopBody);
HWY_EXPORT(CumminScanLoopBody);

void cumsumScan(const keel::f64 *src, keel::f64 *dst, keel::usize n)
{
    if (n == 0) return;
    HWY_DYNAMIC_DISPATCH(CumsumScanLoop)(src, dst, n);
}

void cumprodScan(const keel::f64 *src, keel::f64 *dst, keel::usize n)
{
    if (n == 0) return;
    HWY_DYNAMIC_DISPATCH(CumprodScanLoop)(src, dst, n);
}

namespace {

// Walk the leading-NaN prefix scalar-ly and emit NaN; returns the index
// of the first non-NaN element (or n if the input is all-NaN).
keel::usize copyLeadingNaN(const keel::f64 *src, keel::f64 *dst, keel::usize n)
{
    keel::usize i = 0;
    for (; i < n && keel::is_nan(src[i]); ++i)
        dst[i] = keel::kNaN;
    return i;
}

} // namespace

void cummaxScan(const keel::f64 *src, keel::f64 *dst, keel::usize n)
{
    if (n == 0) return;
    const keel::usize start = copyLeadingNaN(src, dst, n);
    if (start == n) return; // all-NaN â€” done
    // First non-NaN seeds the running max (matches scalar semantics).
    dst[start] = src[start];
    if (start + 1 == n) return;
    HWY_DYNAMIC_DISPATCH(CummaxScanLoopBody)(src + start + 1,
                                             dst + start + 1,
                                             n - start - 1,
                                             src[start]);
}

void cumminScan(const keel::f64 *src, keel::f64 *dst, keel::usize n)
{
    if (n == 0) return;
    const keel::usize start = copyLeadingNaN(src, dst, n);
    if (start == n) return;
    dst[start] = src[start];
    if (start + 1 == n) return;
    HWY_DYNAMIC_DISPATCH(CumminScanLoopBody)(src + start + 1,
                                             dst + start + 1,
                                             n - start - 1,
                                             src[start]);
}

} // namespace numkit::builtin
#endif // HWY_ONCE
