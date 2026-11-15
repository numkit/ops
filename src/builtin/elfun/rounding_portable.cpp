// toolboxes/builtin/src/math/arithmetic/rounding_portable.cpp
//
// Reference scalar ceil/floor/round/fix over a double array.
// Compiled when NUMKIT_WITH_SIMD=OFF.

#include <keel/core.h>
#include "rounding_detail.hpp"

#include <cmath>
#include <cstddef>

namespace numkit::builtin::detail {

void doubleCeilLoop(const double *in, double *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i) out[i] = std::ceil(in[i]);
}

void doubleFloorLoop(const double *in, double *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i) out[i] = std::floor(in[i]);
}

void doubleRoundLoop(const double *in, double *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i) out[i] = std::round(in[i]);
}

void doubleFixLoop(const double *in, double *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i) out[i] = std::trunc(in[i]);
}

} // namespace numkit::builtin::detail
