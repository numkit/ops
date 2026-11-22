// ops/src/nan_reductions_portable.cpp
//
// Scalar single-pass nansum / nanmean. Compiled when NUMKIT_WITH_SIMD=OFF.
// Same single-read pattern as the Highway variant — branch-free predicate
// (`std::isnan` + ternary) keeps it autovectorisable by recent compilers
// even on the portable build.

#include <keel/core.h>
#include <keel/math.h>
#include <numkit/ops/nan_reductions.hpp>

#include <cstddef>

namespace numkit::ops {

keel::f64 nanSumScan(const keel::f64 *p, keel::usize n)
{
    keel::f64 sum = 0.0;
    for (keel::usize i = 0; i < n; ++i)
        if (!keel::is_nan(p[i])) sum += p[i];
    return sum;
}

NanSumCount nanSumCountScan(const keel::f64 *p, keel::usize n)
{
    keel::f64 sum = 0.0;
    keel::usize cnt = 0;
    for (keel::usize i = 0; i < n; ++i) {
        if (!keel::is_nan(p[i])) {
            sum += p[i];
            ++cnt;
        }
    }
    return {sum, cnt};
}

keel::f64 nanMaxScan(const keel::f64 *p, keel::usize n)
{
    keel::f64 best = -keel::Limits<keel::f64>::infinity();
    bool seen = false;
    for (keel::usize i = 0; i < n; ++i) {
        if (!keel::is_nan(p[i])) {
            if (!seen || p[i] > best) best = p[i];
            seen = true;
        }
    }
    return seen ? best : keel::kNaN;
}

keel::f64 nanMinScan(const keel::f64 *p, keel::usize n)
{
    keel::f64 best = keel::Limits<keel::f64>::infinity();
    bool seen = false;
    for (keel::usize i = 0; i < n; ++i) {
        if (!keel::is_nan(p[i])) {
            if (!seen || p[i] < best) best = p[i];
            seen = true;
        }
    }
    return seen ? best : keel::kNaN;
}

keel::f64 nanVarianceTwoPass(const keel::f64 *p, keel::usize n, keel::i32 normFlag)
{
    if (n == 0) return keel::kNaN;
    const auto sc = nanSumCountScan(p, n);
    if (sc.count == 0) return keel::kNaN;
    // MATLAB: variance over a single non-NaN value is 0 for BOTH the N-1
    // (default) and N normalizations — e.g. nanvar([NaN 5 NaN]) == 0, not NaN.
    if (sc.count == 1) return 0.0;
    const keel::f64 mean = sc.sum / keel::as<keel::f64>(sc.count);
    keel::f64 ss = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        if (!keel::is_nan(p[i])) {
            const keel::f64 d = p[i] - mean;
            ss += d * d;
        }
    }
    const keel::f64 denom = (normFlag == 1) ? keel::as<keel::f64>(sc.count)
                                         : keel::as<keel::f64>(sc.count - 1);
    return ss / denom;
}

} // namespace numkit::ops
