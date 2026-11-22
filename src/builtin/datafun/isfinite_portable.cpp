// toolboxes/builtin/src/math/datafun/isfinite_portable.cpp
//
// Reference scalar isnan/isinf/isfinite over a double array. Compiled
// when NUMKIT_WITH_SIMD=OFF; the Highway-dispatched variant lives in
// isfinite_highway.cpp.

#include <keel/core.h>
#include <keel/math.h>
#include "isfinite.hpp"

#include <cstddef>
#include <cstdint>

namespace numkit::builtin::detail {

void doubleIsNaNLoop(const keel::f64 *in, keel::u8 *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i)
        out[i] = keel::is_nan(in[i]) ? 1 : 0;
}

void doubleIsInfLoop(const keel::f64 *in, keel::u8 *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i)
        out[i] = keel::is_inf(in[i]) ? 1 : 0;
}

void doubleIsFiniteLoop(const keel::f64 *in, keel::u8 *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i)
        out[i] = keel::is_finite(in[i]) ? 1 : 0;
}

} // namespace numkit::builtin::detail
