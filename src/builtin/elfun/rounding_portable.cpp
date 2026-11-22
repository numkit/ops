// toolboxes/builtin/src/math/arithmetic/rounding_portable.cpp
//
// Reference scalar ceil/floor/round/fix over a double array.
// Compiled when NUMKIT_WITH_SIMD=OFF.

#include <keel/core.h>
#include <keel/math.h>
#include "rounding_detail.hpp"

#include <cstddef>

namespace numkit::builtin::detail {

void doubleCeilLoop(const keel::f64 *in, keel::f64 *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i) out[i] = keel::ceil(in[i]);
}

void doubleFloorLoop(const keel::f64 *in, keel::f64 *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i) out[i] = keel::floor(in[i]);
}

void doubleRoundLoop(const keel::f64 *in, keel::f64 *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i) out[i] = keel::round(in[i]);
}

void doubleFixLoop(const keel::f64 *in, keel::f64 *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i) out[i] = keel::trunc(in[i]);
}

} // namespace numkit::builtin::detail
