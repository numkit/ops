// src/builtin/src/elfun/mod_simd.hpp
//
// Backend-split entry points for the SIMD-accelerated mod() fast path in numkit::builtin.
#pragma once

#include <keel/core.h>
#include <cstddef>

namespace numkit::builtin::detail {

void modLoopVV(const double *a, const double *b, double *out, keel::usize n);
void modLoopVS(const double *a, double scalar, double *out, keel::usize n);
void modLoopSV(double scalar, const double *b, double *out, keel::usize n);

} // namespace numkit::builtin::detail
