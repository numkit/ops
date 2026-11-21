// toolboxes/builtin/src/math/elementary/backends/cumsum_portable.cpp
//
// Scalar inclusive prefix-sum. Compiled when NUMKIT_WITH_SIMD=OFF.
// The loop has a serial data dependency (`s += src[i]; dst[i] = s;`)
// so this is fundamentally one-add-per-cycle on the portable build.

#include <keel/core.h>
#include "cumsum.hpp"

#include <cmath>
#include <cstddef>

namespace numkit::builtin {

void cumsumScan(const keel::f64 *src, keel::f64 *dst, keel::usize n)
{
    keel::f64 s = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        s += src[i];
        dst[i] = s;
    }
}

void cumprodScan(const keel::f64 *src, keel::f64 *dst, keel::usize n)
{
    keel::f64 s = 1.0;
    for (keel::usize i = 0; i < n; ++i) {
        s *= src[i];
        dst[i] = s;
    }
}

void cummaxScan(const keel::f64 *src, keel::f64 *dst, keel::usize n)
{
    if (n == 0) return;
    // Walk leading NaN, preserve as NaN.
    keel::usize i = 0;
    for (; i < n && std::isnan(src[i]); ++i) dst[i] = std::nan("");
    if (i == n) return;
    keel::f64 acc = src[i];
    dst[i] = acc;
    for (++i; i < n; ++i) {
        const keel::f64 xi = src[i];
        if (!std::isnan(xi) && xi > acc) acc = xi;
        dst[i] = acc;
    }
}

void cumminScan(const keel::f64 *src, keel::f64 *dst, keel::usize n)
{
    if (n == 0) return;
    keel::usize i = 0;
    for (; i < n && std::isnan(src[i]); ++i) dst[i] = std::nan("");
    if (i == n) return;
    keel::f64 acc = src[i];
    dst[i] = acc;
    for (++i; i < n; ++i) {
        const keel::f64 xi = src[i];
        if (!std::isnan(xi) && xi < acc) acc = xi;
        dst[i] = acc;
    }
}

} // namespace numkit::builtin
