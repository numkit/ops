// src/builtin/src/elfun/rounding.hpp
//
// Internal header for the ceil/floor/round/fix backend in numkit::builtin.
#pragma once

#include <keel/core.h>
#include <cstddef>

namespace numkit::builtin::detail {

void doubleCeilLoop  (const double *in, double *out, keel::usize n);
void doubleFloorLoop (const double *in, double *out, keel::usize n);
void doubleRoundLoop (const double *in, double *out, keel::usize n);
void doubleFixLoop   (const double *in, double *out, keel::usize n);

} // namespace numkit::builtin::detail
