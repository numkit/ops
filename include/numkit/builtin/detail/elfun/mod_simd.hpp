// src/builtin/src/elfun/mod_simd.hpp
//
// Backend-split entry points for the SIMD-accelerated mod() fast path in numkit::builtin.
#pragma once

#include <keel/core.h>
#include <cstddef>

namespace numkit::builtin::detail {

void modLoopVV(const keel::f64 *a, const keel::f64 *b, keel::f64 *out, keel::usize n);
void modLoopVS(const keel::f64 *a, keel::f64 scalar, keel::f64 *out, keel::usize n);
void modLoopSV(keel::f64 scalar, const keel::f64 *b, keel::f64 *out, keel::usize n);

} // namespace numkit::builtin::detail
