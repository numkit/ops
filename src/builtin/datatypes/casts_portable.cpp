// toolboxes/builtin/src/language/datatypes/casts_portable.cpp
//
// Reference scalar implementation of the double→int* fast path.
// Compiled when NUMKIT_WITH_SIMD=OFF; the Highway-dispatched variant
// lives in casts_highway.cpp and matches this file bit-for-bit.
//
// MATLAB's `int*(x)` casts: NaN→0, then round half-away-from-zero,
// then saturate to the target's [min, max]. Matches std::round +
// saturate.

#include <keel/core.h>
#include "casts.hpp"

#include <numkit/value/error.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace numkit::builtin::detail {

namespace {

template <typename T>
inline T saturate_cast(double v)
{
    if (std::isnan(v)) return 0;
    v = std::round(v);
    constexpr double lo = keel::as<double>(std::numeric_limits<T>::min());
    constexpr double hi = keel::as<double>(std::numeric_limits<T>::max());
    if (v < lo) return std::numeric_limits<T>::min();
    if (v > hi) return std::numeric_limits<T>::max();
    return static_cast<T>(v);
}

template <typename T>
void doubleToIntLoopScalar(const double *in, T *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i) out[i] = saturate_cast<T>(in[i]);
}

} // namespace

void doubleToInt8 (const double *in, keel::i8   *out, keel::usize n) { doubleToIntLoopScalar<keel::i8  >(in, out, n); }
void doubleToInt16(const double *in, keel::i16  *out, keel::usize n) { doubleToIntLoopScalar<keel::i16 >(in, out, n); }
void doubleToInt32(const double *in, keel::i32  *out, keel::usize n) { doubleToIntLoopScalar<keel::i32 >(in, out, n); }
void doubleToInt64(const double *in, keel::i64  *out, keel::usize n) { doubleToIntLoopScalar<keel::i64 >(in, out, n); }
void doubleToUInt8 (const double *in, keel::u8  *out, keel::usize n) { doubleToIntLoopScalar<keel::u8 >(in, out, n); }
void doubleToUInt16(const double *in, keel::u16 *out, keel::usize n) { doubleToIntLoopScalar<keel::u16>(in, out, n); }
void doubleToUInt32(const double *in, keel::u32 *out, keel::usize n) { doubleToIntLoopScalar<keel::u32>(in, out, n); }
void doubleToUInt64(const double *in, keel::u64 *out, keel::usize n) { doubleToIntLoopScalar<keel::u64>(in, out, n); }

} // namespace numkit::builtin::detail
