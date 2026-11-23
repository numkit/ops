// toolboxes/builtin/src/math/arithmetic/isfinite.hpp
//
// Internal header for the isnan/isinf/isfinite backend. The double
// fast-path lives in isfinite_{highway,portable}.cpp; consumed from
// language/types/types.cpp.

#pragma once

#include <keel/core.h>

namespace numkit::builtin::detail {

void doubleIsNaNLoop(const keel::f64 *in, keel::u8 *out, keel::usize n);
void doubleIsInfLoop(const keel::f64 *in, keel::u8 *out, keel::usize n);
void doubleIsFiniteLoop(const keel::f64 *in, keel::u8 *out, keel::usize n);

} // namespace numkit::builtin::detail
