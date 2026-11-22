// src/builtin/src/elfun/sinpi_kernel.hpp
//
// Accurate sinpi / cospi scalar kernel.
#pragma once

#include <keel/core.h>
#include <keel/math.h>
#include <cstdint>

namespace numkit::builtin::detail {

// sinpik / cospik share one coefficient table; only the o-selection and
// the final sign differ. Index 0 is the leading (highest-degree) term.
inline constexpr keel::f64 kSinpiPoly_o[8] = {
     9.94480387626843774090208e-16,
    -3.89796226062932799164047e-13,
     1.150115825399960352669010e-10,
    -2.46113695010446974953590e-08,
     3.590860448590527540050620e-06,
    -0.000325991886927389905997954,
     0.0158543442438155018914259,
    -0.308425137534042437259529,
};
inline constexpr keel::f64 kSinpiPoly_e[8] = {
    -2.02461120785182399295868e-14,
     6.948218305801794613277840e-12,
    -1.75724749952853179952664e-09,
     3.133616889668683928784220e-07,
    -3.65762041821615519203610e-05,
     0.0024903945701927185027435600,
    -0.0807455121882807852484731,
     0.785398163397448278999491,
};

// Beyond 2^53 a double is an even integer: sinpi == 0, cospi == 1.
inline constexpr keel::f64 kSinpiIntThreshold = 9007199254740992.0; // 2^53

inline keel::f64 sinpi_poly(bool o, keel::f64 s)
{
    const keel::f64 *c = o ? kSinpiPoly_o : kSinpiPoly_e;
    keel::f64 x = c[0];
    for (keel::i32 k = 1; k < 8; ++k)
        x = x * s + c[k];
    return x;
}

// Nearest even integer to u, via SLEEF's (trunc(u) + (u>=0)) & ~1.
inline keel::i64 sinpi_octant(keel::f64 u)
{
    keel::i64 q = keel::as<keel::i64>(u); // truncate toward zero
    q += (q < 0) ? 0 : 1;
    return q & ~keel::as<keel::i64>(1);
}

inline keel::f64 sinpi_kernel(keel::f64 d)
{
    if (keel::is_nan(d)) return d;
    if (keel::is_inf(d)) return keel::Limits<keel::f64>::quiet_nan();
    if (keel::abs(d) >= kSinpiIntThreshold) return keel::copysign(0.0, d);

    const keel::f64        u = d * 4.0;
    const keel::i64  q = sinpi_octant(u);
    const bool          o = (q & 2) != 0;
    const keel::f64        t = u - keel::as<keel::f64>(q);
    const keel::f64        s = t * t;

    keel::f64 x = sinpi_poly(o, s) * (o ? s : t);
    if (o) x += 1.0;
    x = (q & 4) ? -x : x;
    // sinpi(integer) is an exact zero whose sign MATLAB takes from the
    // input (sinpi(1)==+0, sinpi(-1)==-0), not from the quadrant.
    return (x == 0.0) ? keel::copysign(0.0, d) : x;
}

inline keel::f64 cospi_kernel(keel::f64 d)
{
    if (keel::is_nan(d)) return d;
    if (keel::is_inf(d)) return keel::Limits<keel::f64>::quiet_nan();
    if (keel::abs(d) >= kSinpiIntThreshold) return 1.0;

    const keel::f64        u = d * 4.0;
    const keel::i64  q = sinpi_octant(u);
    const bool          o = (q & 2) == 0;
    const keel::f64        t = u - keel::as<keel::f64>(q);
    const keel::f64        s = t * t;

    keel::f64 x = sinpi_poly(o, s) * (o ? s : t);
    if (o) x += 1.0;
    x = ((q + 2) & 4) ? -x : x;
    // cospi(half-integer) is +0 in MATLAB regardless of side.
    return (x == 0.0) ? 0.0 : x;
}

} // namespace numkit::builtin::detail
