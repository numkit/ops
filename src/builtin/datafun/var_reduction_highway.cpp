// toolboxes/builtin/src/math/elementary/backends/var_reduction_highway.cpp
//
// Highway dynamic-dispatch two-pass variance kernel (P5). Two scans:
//   pass 1: lane-parallel sum â†’ mean
//   pass 2: lane-parallel sum of (x - mean)^2 via FMA (Highway MulAdd)
//          â†’ variance after dividing by n or n-1.
// 4Ã— unrolled accumulators per pass keep enough independent ops in
// flight to hide load-to-use latency on the wide pipelines.

#include <keel/core.h>
#include <keel/math.h>
#include "var_reduction.hpp"


#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "builtin/datafun/var_reduction_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace numkit::builtin {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

keel::f64 SumScanLoop(const keel::f64 *HWY_RESTRICT p, keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    auto a0 = hn::Zero(d), a1 = hn::Zero(d), a2 = hn::Zero(d), a3 = hn::Zero(d);

    keel::usize i = 0;
    for (; i + 4 * N <= n; i += 4 * N) {
        a0 = hn::Add(a0, hn::LoadU(d, p + i + 0 * N));
        a1 = hn::Add(a1, hn::LoadU(d, p + i + 1 * N));
        a2 = hn::Add(a2, hn::LoadU(d, p + i + 2 * N));
        a3 = hn::Add(a3, hn::LoadU(d, p + i + 3 * N));
    }
    for (; i + N <= n; i += N)
        a0 = hn::Add(a0, hn::LoadU(d, p + i));
    keel::f64 s = hn::ReduceSum(d, hn::Add(hn::Add(a0, a1), hn::Add(a2, a3)));
    for (; i < n; ++i) s += p[i];
    return s;
}

// In-place accumulate: dst += src. 4Ã— unrolled for ILP, regular
// cached stores (caller reads dst again on the next column-pass iter).
void AddIntoLoop(keel::f64 *HWY_RESTRICT dst, const keel::f64 *HWY_RESTRICT src,
                 keel::usize n)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);

    keel::usize i = 0;
    for (; i + 4 * N <= n; i += 4 * N) {
        const auto d0 = hn::Add(hn::LoadU(d, dst + i + 0 * N),
                                hn::LoadU(d, src + i + 0 * N));
        const auto d1 = hn::Add(hn::LoadU(d, dst + i + 1 * N),
                                hn::LoadU(d, src + i + 1 * N));
        const auto d2 = hn::Add(hn::LoadU(d, dst + i + 2 * N),
                                hn::LoadU(d, src + i + 2 * N));
        const auto d3 = hn::Add(hn::LoadU(d, dst + i + 3 * N),
                                hn::LoadU(d, src + i + 3 * N));
        hn::StoreU(d0, d, dst + i + 0 * N);
        hn::StoreU(d1, d, dst + i + 1 * N);
        hn::StoreU(d2, d, dst + i + 2 * N);
        hn::StoreU(d3, d, dst + i + 3 * N);
    }
    for (; i + N <= n; i += N)
        hn::StoreU(hn::Add(hn::LoadU(d, dst + i), hn::LoadU(d, src + i)),
                   d, dst + i);
    for (; i < n; ++i) dst[i] += src[i];
}

keel::f64 SumSquaredDeviationsScanLoop(const keel::f64 *HWY_RESTRICT p, keel::usize n,
                                    keel::f64 mean)
{
    const hn::ScalableTag<keel::f64> d;
    const keel::usize N = hn::Lanes(d);
    const auto vmean = hn::Set(d, mean);
    auto a0 = hn::Zero(d), a1 = hn::Zero(d), a2 = hn::Zero(d), a3 = hn::Zero(d);

    keel::usize i = 0;
    for (; i + 4 * N <= n; i += 4 * N) {
        const auto v0 = hn::Sub(hn::LoadU(d, p + i + 0 * N), vmean);
        const auto v1 = hn::Sub(hn::LoadU(d, p + i + 1 * N), vmean);
        const auto v2 = hn::Sub(hn::LoadU(d, p + i + 2 * N), vmean);
        const auto v3 = hn::Sub(hn::LoadU(d, p + i + 3 * N), vmean);
        // MulAdd(a, b, c) = a*b + c with one rounding (FMA when available).
        a0 = hn::MulAdd(v0, v0, a0);
        a1 = hn::MulAdd(v1, v1, a1);
        a2 = hn::MulAdd(v2, v2, a2);
        a3 = hn::MulAdd(v3, v3, a3);
    }
    for (; i + N <= n; i += N) {
        const auto v = hn::Sub(hn::LoadU(d, p + i), vmean);
        a0 = hn::MulAdd(v, v, a0);
    }
    keel::f64 ss = hn::ReduceSum(d, hn::Add(hn::Add(a0, a1), hn::Add(a2, a3)));
    for (; i < n; ++i) {
        const keel::f64 dv = p[i] - mean;
        ss += dv * dv;
    }
    return ss;
}

} // namespace HWY_NAMESPACE
} // namespace numkit::builtin
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace numkit::builtin {

HWY_EXPORT(SumScanLoop);
HWY_EXPORT(SumSquaredDeviationsScanLoop);
HWY_EXPORT(AddIntoLoop);

keel::f64 sumScan(const keel::f64 *p, keel::usize n)
{
    if (n == 0) return 0.0;
    return HWY_DYNAMIC_DISPATCH(SumScanLoop)(p, n);
}

keel::f64 sumSquaredDeviationsScan(const keel::f64 *p, keel::usize n, keel::f64 mean)
{
    if (n == 0) return 0.0;
    return HWY_DYNAMIC_DISPATCH(SumSquaredDeviationsScanLoop)(p, n, mean);
}

void addInto(keel::f64 *dst, const keel::f64 *src, keel::usize n)
{
    if (n == 0) return;
    HWY_DYNAMIC_DISPATCH(AddIntoLoop)(dst, src, n);
}

keel::f64 varianceTwoPass(const keel::f64 *p, keel::usize n, keel::i32 normFlag)
{
    if (n == 0) return keel::kNaN;
    // MATLAB: variance of a single observation is 0 for BOTH the N-1
    // (default) and N normalizations â€” not NaN from the 0/0 of N-1.
    if (n == 1) return 0.0;
    const keel::f64 mean  = sumScan(p, n) / keel::as<keel::f64>(n);
    const keel::f64 ss    = sumSquaredDeviationsScan(p, n, mean);
    const keel::f64 denom = (normFlag == 1) ? keel::as<keel::f64>(n)
                                         : keel::as<keel::f64>(n - 1);
    return ss / denom;
}

} // namespace numkit::builtin
#endif // HWY_ONCE
