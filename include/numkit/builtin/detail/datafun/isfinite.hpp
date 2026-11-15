// toolboxes/builtin/src/math/arithmetic/isfinite.hpp
//
// Internal header for the isnan/isinf/isfinite backend. The double
// fast-path lives in isfinite_{highway,portable}.cpp; consumed from
// language/types/types.cpp.

#pragma once

#include <keel/core.h>
#include <cstddef>
#include <cstdint>

namespace numkit::builtin::detail {

void doubleIsNaNLoop(const double *in, keel::u8 *out, keel::usize n);
void doubleIsInfLoop(const double *in, keel::u8 *out, keel::usize n);
void doubleIsFiniteLoop(const double *in, keel::u8 *out, keel::usize n);

} // namespace numkit::builtin::detail
