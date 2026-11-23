// toolboxes/builtin/src/language/types/casts.hpp
//
// Internal header shared by casts_highway.cpp / casts_portable.cpp
// and consumed by types.cpp's numericConstructor fast-path.
//
// Each `doubleToIntXX` function performs MATLAB-spec int cast on a
// double array: NaN→0, round half-away-from-zero, saturate to target.

#pragma once

#include <keel/core.h>

namespace numkit::builtin::detail {

void doubleToInt8 (const keel::f64 *in, keel::i8   *out, keel::usize n);
void doubleToInt16(const keel::f64 *in, keel::i16  *out, keel::usize n);
void doubleToInt32(const keel::f64 *in, keel::i32  *out, keel::usize n);
void doubleToInt64(const keel::f64 *in, keel::i64  *out, keel::usize n);
void doubleToUInt8 (const keel::f64 *in, keel::u8  *out, keel::usize n);
void doubleToUInt16(const keel::f64 *in, keel::u16 *out, keel::usize n);
void doubleToUInt32(const keel::f64 *in, keel::u32 *out, keel::usize n);
void doubleToUInt64(const keel::f64 *in, keel::u64 *out, keel::usize n);

} // namespace numkit::builtin::detail
