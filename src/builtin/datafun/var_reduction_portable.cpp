// toolboxes/builtin/src/math/elementary/backends/var_reduction_portable.cpp
//
// Scalar two-pass variance. Compiled when NUMKIT_WITH_SIMD=OFF. Same
// numerical recipe as the Highway variant — a tight scalar loop is
// reliably autovectorised by recent compilers and the FMA in the
// inner-loop accumulator gives the same single-rounding behaviour.

#include <keel/core.h>
#include <keel/math.h>
#include "var_reduction.hpp"


namespace numkit::builtin {

keel::f64 sumScan(const keel::f64 *p, keel::usize n)
{
    keel::f64 s = 0.0;
    for (keel::usize i = 0; i < n; ++i) s += p[i];
    return s;
}

keel::f64 sumSquaredDeviationsScan(const keel::f64 *p, keel::usize n, keel::f64 mean)
{
    keel::f64 ss = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 d = p[i] - mean;
        ss += d * d;
    }
    return ss;
}

void addInto(keel::f64 *dst, const keel::f64 *src, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i) dst[i] += src[i];
}

keel::f64 varianceTwoPass(const keel::f64 *p, keel::usize n, keel::i32 normFlag)
{
    if (n == 0) return keel::kNaN;
    // MATLAB: variance of a single observation is 0 for BOTH the N-1
    // (default) and N normalizations — not NaN from the 0/0 of N-1.
    if (n == 1) return 0.0;
    const keel::f64 mean  = sumScan(p, n) / keel::as<keel::f64>(n);
    const keel::f64 ss    = sumSquaredDeviationsScan(p, n, mean);
    const keel::f64 denom = (normFlag == 1) ? keel::as<keel::f64>(n)
                                         : keel::as<keel::f64>(n - 1);
    return ss / denom;
}

} // namespace numkit::builtin
