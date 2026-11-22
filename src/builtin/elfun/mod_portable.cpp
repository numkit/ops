// toolboxes/builtin/src/math/arithmetic/mod_portable.cpp
//
// Scalar reference for the mod() fast-path entry points. Compiled when
// NUMKIT_WITH_SIMD=OFF; the Highway-dispatched variant lives in
// mod_highway.cpp and is bit-identical (same separate-mul/sub formula).

#include <keel/core.h>
#include <keel/math.h>
#include "mod_simd.hpp"


namespace numkit::builtin::detail {

void modLoopVV(const keel::f64 *a, const keel::f64 *b, keel::f64 *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i)
        out[i] = (b[i] != 0) ? a[i] - keel::floor(a[i] / b[i]) * b[i] : a[i];
}

void modLoopVS(const keel::f64 *a, keel::f64 s, keel::f64 *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i)
        out[i] = (s != 0) ? a[i] - keel::floor(a[i] / s) * s : a[i];
}

void modLoopSV(keel::f64 s, const keel::f64 *b, keel::f64 *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i)
        out[i] = (b[i] != 0) ? s - keel::floor(s / b[i]) * b[i] : s;
}

} // namespace numkit::builtin::detail
