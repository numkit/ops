// ops/src/quantile_select.cpp
//
// nth_element-based linear-interpolation quantile, moved verbatim from stats'
// descriptive_extras_detail.hpp (sliceQuantile) so the select primitive lives
// in the kernel layer. See quantile_select.hpp.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/math.h>
#include <numkit/ops/quantile_select.hpp>


namespace numkit::ops {

keel::f64 sliceQuantile(keel::f64 *s, keel::usize n, keel::f64 p)
{
    if (n == 0) return keel::Limits<keel::f64>::quiet_nan();
    if (n == 1) return s[0];
    // The MATLAB "+0.5" rule needs at most two adjacent order statistics, so
    // partition with nth_element (O(n) average) instead of a full O(n log n)
    // sort. The selected values are exactly the sorted ones, so results are
    // bit-for-bit unchanged — only the cost drops (a two-call iqr over ~10^7
    // elements goes from ~1.7 s of sorting to well under 0.1 s). Shared by
    // iqr / mad / median-of-slice.
    const keel::f64 q = p * keel::as<keel::f64>(n) + 0.5;
    if (q <= 1.0) return *keel::min_element(s, s + n);
    if (q >= keel::as<keel::f64>(n)) return *keel::max_element(s, s + n);
    const keel::usize lo   = keel::as<keel::usize>(keel::floor(q)) - 1;
    const keel::f64      frac = q - keel::floor(q);
    // Place the lo-th smallest at s[lo]; everything in (lo, n) is then >= it,
    // so the (lo+1)-th smallest is the minimum of that right partition.
    keel::nth_element(s, s + lo, s + n);
    const keel::f64 a = s[lo];
    const keel::f64 b = *keel::min_element(s + lo + 1, s + n);
    return a + frac * (b - a);
}

} // namespace numkit::ops
