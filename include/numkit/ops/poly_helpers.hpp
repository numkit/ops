// ops/include/numkit/ops/poly_helpers.hpp
//
// Polynomial helpers shared between the public roots() builtin and the
// DSP-side filter design code (tf2sos / tf2zp / butter). All-inline; no
// matching .cpp.

#pragma once


// Complex comes from <complex> below; this header is value+ops-only (no core).

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace numkit::ops {

// Horner evaluation of a polynomial p(z) and its derivative p'(z). The
// coefficients are in MATLAB / numpy order: c[0] is the leading
// coefficient (highest power), c[n] is the constant. Pointer + size
// instead of a container ref so it composes with std::vector,
// std::pmr::vector, or raw arrays.
inline void polyEvalHorner(const keel::c64 *c, keel::usize n,
                           keel::c64 z, keel::c64 &outP, keel::c64 &outDP)
{
    keel::c64 p  = c[0];
    keel::c64 dp(0.0, 0.0);
    for (keel::usize i = 1; i < n; ++i) {
        dp = dp * z + p;
        p  = p  * z + c[i];
    }
    outP  = p;
    outDP = dp;
}

// Durand-Kerner simultaneous root finder for a real-coefficient
// polynomial. Coefficients in MATLAB order. Returns a vector of length
// equal to the polynomial degree; complex-conjugate pairs are not
// guaranteed to come out exactly conjugate (they're each found
// independently), but typically agree to ~1e-13.
//
// Behaviour at edge cases:
//   * Leading zeros are stripped (degree drop).
//   * Trailing zeros become explicit roots at the origin.
//   * Constant or empty input → no roots.
//
// ~50-200 iterations to ~1e-14 tolerance for moderate polynomial
// degrees (≤ 32). Higher degrees may need a tighter inner loop, but
// practical SOS / roots() usage rarely exceeds this.
inline keel::Array<keel::c64>
polyRootsDurandKerner(const keel::c64 *coeffs, keel::usize coeffsN,
                      keel::Allocator *mr = nullptr)
{
    keel::usize lo = 0, hi = coeffsN;
    while (lo < hi && coeffs[lo] == keel::c64(0.0, 0.0)) ++lo;
    keel::usize origZeros = 0;
    while (hi > lo && coeffs[hi - 1] == keel::c64(0.0, 0.0)) { --hi; ++origZeros; }
    if (hi - lo <= 1)
        return keel::Array<keel::c64>(origZeros, keel::c64(0.0, 0.0), mr);

    keel::Array<keel::c64> c(hi - lo, mr);
    for (keel::usize i = 0; i < c.size(); ++i)
        c[i] = coeffs[lo + i] / coeffs[lo];
    const keel::usize n = c.size() - 1;

    keel::f64 maxCoef = 0.0;
    for (keel::usize i = 1; i <= n; ++i)
        maxCoef = std::max(maxCoef, keel::abs(c[i]));
    const keel::f64 R = 1.0 + maxCoef;
    keel::Array<keel::c64> roots(n, mr);
    constexpr keel::f64 kTwoPi = 6.28318530717958647692;
    for (keel::usize k = 0; k < n; ++k) {
        const keel::f64 theta = kTwoPi * static_cast<keel::f64>(k)
                             / static_cast<keel::f64>(n) + 0.4;
        roots[k] = keel::c64(R * std::cos(theta), R * std::sin(theta));
    }

    constexpr int    kMaxIter = 200;
    constexpr keel::f64 kTol     = 1e-14;
    for (int it = 0; it < kMaxIter; ++it) {
        keel::f64 maxStep = 0.0;
        for (keel::usize k = 0; k < n; ++k) {
            keel::c64 p, dp;
            polyEvalHorner(c.data(), c.size(), roots[k], p, dp);
            keel::c64 denom(1.0, 0.0);
            for (keel::usize j = 0; j < n; ++j)
                if (j != k) denom *= (roots[k] - roots[j]);
            if (keel::abs(denom) < 1e-300) continue;
            const keel::c64 step = p / denom;
            roots[k] -= step;
            const keel::f64 s = keel::abs(step);
            if (s > maxStep) maxStep = s;
        }
        if (maxStep < kTol) break;
    }
    if (origZeros > 0) {
        keel::Array<keel::c64> out(origZeros + roots.size(), mr);
        for (keel::usize i = 0; i < origZeros; ++i)
            out[i] = keel::c64(0.0, 0.0);
        for (keel::usize i = 0; i < roots.size(); ++i)
            out[origZeros + i] = roots[i];
        return out;
    }
    return roots;
}

inline keel::Array<keel::c64>
polyRootsDurandKerner(const keel::f64 *coeffs, keel::usize coeffsN,
                      keel::Allocator *mr = nullptr)
{
    keel::Array<keel::c64> cc(coeffsN, mr);
    for (keel::usize i = 0; i < coeffsN; ++i)
        cc[i] = keel::c64(coeffs[i], 0.0);
    return polyRootsDurandKerner(cc.data(), coeffsN, mr);
}

// Expand a list of polynomial roots into a real-coefficient polynomial
// via repeated convolution by (x - r). Output length = rootsN + 1
// in MATLAB order (leading coefficient first). Imaginary residue is
// dropped — caller is responsible for ensuring the roots come in
// complex-conjugate pairs (otherwise the imaginary parts won't cancel
// to zero).
inline keel::Array<keel::f64>
polyExpandFromRoots(const keel::c64 *roots, keel::usize rootsN,
                    keel::Allocator *mr = nullptr)
{
    const int n = static_cast<int>(rootsN);
    keel::Array<keel::c64> poly(static_cast<keel::usize>(n + 1), mr);
    poly[0] = keel::c64(1.0, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j >= 1; --j)
            poly[j] = poly[j] - roots[i] * poly[j - 1];
    keel::Array<keel::f64> r(static_cast<keel::usize>(n + 1), mr);
    for (int i = 0; i <= n; ++i)
        r[i] = poly[i].real();
    return r;
}

} // namespace numkit::ops
