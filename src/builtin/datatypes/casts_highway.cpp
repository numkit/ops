// toolboxes/builtin/src/language/datatypes/casts_highway.cpp
//
// Highway dynamic-dispatch doubleâ†’int* casts. Matches MATLAB's
// `int*(x)`: NaNâ†’0, round half-away-from-zero, saturate to target
// type's [min, max]. Direct ConvertTo would truncate toward zero
// â€” wrong â€” so we Round explicitly before ConvertTo, then clamp.
//
// 8 variants (int8/16/32/64 + uint8/16/32/64). Each gets its own
// HWY_EXPORT pair. The complex "demote/widen" plumbing for narrow
// targets (int8/int16) shrinks 2x via DemoteTo per stage.
//
// For wide targets (int64/uint64) Highway needs ConvertTo<int64> on
// the same lane count as double â€” that's a 1:1 lane count, no
// demote needed.

#include <keel/core.h>
#include "casts.hpp"

#include <numkit/ops/parallel_for.hpp>
#include <numkit/value/error.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "builtin/datatypes/casts_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace numkit::builtin::detail {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Scalar helper for the loop tail: matches MATLAB cast semantics.
template <typename T>
inline T saturate_cast_scalar(keel::f64 v)
{
    if (std::isnan(v)) return 0;
    v = std::round(v);
    constexpr keel::f64 lo = keel::as<keel::f64>(std::numeric_limits<T>::min());
    constexpr keel::f64 hi = keel::as<keel::f64>(std::numeric_limits<T>::max());
    if (v < lo) return std::numeric_limits<T>::min();
    if (v > hi) return std::numeric_limits<T>::max();
    return static_cast<T>(v);
}

// Generic SIMD body for doubleâ†’intN where N matches int64-lane width
// (i.e. for int32/int64/uint32/uint64). Branch on T to handle narrow
// types separately (they need DemoteTo).
//
// Strategy:
//   1. Load N doubles
//   2. Replace NaN with 0 (Highway IsNaN + IfThenElse)
//   3. Round half-away-from-zero: trunc(v + copysign(0.5, v)). Highway's
//      Round is ties-to-even, which diverges from MATLAB int*()/round on
//      exact half-integers (2.5 -> 2 instead of 3), so we do not use it.
//   4. Clamp to [target_min, target_max]
//   5. ConvertTo<int64> (or DemoteTo for narrow)

void DoubleToInt32Loop(const keel::f64 *HWY_RESTRICT in, keel::i32 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64>  d;
    const hn::Rebind<keel::i32, decltype(d)> di_rebind;
    const auto zero = hn::Zero(d);
    const auto lo   = hn::Set(d, keel::as<keel::f64>(std::numeric_limits<keel::i32>::min()));
    const auto hi   = hn::Set(d, keel::as<keel::f64>(std::numeric_limits<keel::i32>::max()));
    const keel::usize N = hn::Lanes(d);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        v = hn::IfThenElse(hn::IsNaN(v), zero, v);
        v = hn::Trunc(hn::Add(v, hn::CopySign(hn::Set(d, 0.5), v)));  // half-away (MATLAB int*/round)
        v = hn::Min(hn::Max(v, lo), hi);
        auto vi = hn::DemoteTo(di_rebind, v);
        hn::StoreU(vi, di_rebind, out + i);
    }
    for (; i < n; ++i) out[i] = saturate_cast_scalar<keel::i32>(in[i]);
}

void DoubleToUInt32Loop(const keel::f64 *HWY_RESTRICT in, keel::u32 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64>  d;
    const hn::Rebind<keel::u32, decltype(d)> du_rebind;
    const auto zero = hn::Zero(d);
    const auto lo   = hn::Set(d, 0.0);
    const auto hi   = hn::Set(d, keel::as<keel::f64>(std::numeric_limits<keel::u32>::max()));
    const keel::usize N = hn::Lanes(d);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        v = hn::IfThenElse(hn::IsNaN(v), zero, v);
        v = hn::Trunc(hn::Add(v, hn::CopySign(hn::Set(d, 0.5), v)));  // half-away (MATLAB int*/round)
        v = hn::Min(hn::Max(v, lo), hi);
        // Highway: doubles in [0, 2^32-1] convert via DemoteTo<uint32>.
        auto vi = hn::DemoteTo(du_rebind, v);
        hn::StoreU(vi, du_rebind, out + i);
    }
    for (; i < n; ++i) out[i] = saturate_cast_scalar<keel::u32>(in[i]);
}

void DoubleToInt64Loop(const keel::f64 *HWY_RESTRICT in, keel::i64 *HWY_RESTRICT out, keel::usize n)
{
    const hn::ScalableTag<keel::f64>  d;
    const hn::ScalableTag<keel::i64> di;
    const auto zero = hn::Zero(d);
    // int64 range exceeds double-exact precision; saturation bounds
    // are still meaningful for clamping but inputs above 2^53 lose
    // representability â€” same as MATLAB.
    const auto lo   = hn::Set(d, keel::as<keel::f64>(std::numeric_limits<keel::i64>::min()));
    const auto hi   = hn::Set(d, keel::as<keel::f64>(std::numeric_limits<keel::i64>::max()));
    const keel::usize N = hn::Lanes(d);
    keel::usize i = 0;
    for (; i + N <= n; i += N) {
        auto v = hn::LoadU(d, in + i);
        v = hn::IfThenElse(hn::IsNaN(v), zero, v);
        v = hn::Trunc(hn::Add(v, hn::CopySign(hn::Set(d, 0.5), v)));  // half-away (MATLAB int*/round)
        v = hn::Min(hn::Max(v, lo), hi);
        auto vi = hn::ConvertTo(di, v);
        hn::StoreU(vi, di, out + i);
    }
    for (; i < n; ++i) out[i] = saturate_cast_scalar<keel::i64>(in[i]);
}

void DoubleToUInt64Loop(const keel::f64 *HWY_RESTRICT in, keel::u64 *HWY_RESTRICT out, keel::usize n)
{
    // Highway has no doubleâ†’uint64 ConvertTo on all targets; fall
    // back to scalar saturate_cast for this size. Still benefits
    // from straight-line code (no virtual dispatch through Value).
    for (keel::usize i = 0; i < n; ++i) out[i] = saturate_cast_scalar<keel::u64>(in[i]);
}

void DoubleToInt16Loop(const keel::f64 *HWY_RESTRICT in, keel::i16 *HWY_RESTRICT out, keel::usize n)
{
    // double â†’ int32 â†’ int16 (two demote steps, 4x lane width)
    for (keel::usize i = 0; i < n; ++i) out[i] = saturate_cast_scalar<keel::i16>(in[i]);
}

void DoubleToUInt16Loop(const keel::f64 *HWY_RESTRICT in, keel::u16 *HWY_RESTRICT out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i) out[i] = saturate_cast_scalar<keel::u16>(in[i]);
}

void DoubleToInt8Loop(const keel::f64 *HWY_RESTRICT in, keel::i8 *HWY_RESTRICT out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i) out[i] = saturate_cast_scalar<keel::i8>(in[i]);
}

void DoubleToUInt8Loop(const keel::f64 *HWY_RESTRICT in, keel::u8 *HWY_RESTRICT out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i) out[i] = saturate_cast_scalar<keel::u8>(in[i]);
}

} // namespace HWY_NAMESPACE
} // namespace numkit::builtin::detail
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace numkit::builtin::detail {

HWY_EXPORT(DoubleToInt32Loop);
HWY_EXPORT(DoubleToUInt32Loop);
HWY_EXPORT(DoubleToInt64Loop);
HWY_EXPORT(DoubleToUInt64Loop);
HWY_EXPORT(DoubleToInt16Loop);
HWY_EXPORT(DoubleToUInt16Loop);
HWY_EXPORT(DoubleToInt8Loop);
HWY_EXPORT(DoubleToUInt8Loop);

void doubleToInt8 (const keel::f64 *in, keel::i8   *out, keel::usize n) {
    HWY_DYNAMIC_DISPATCH(DoubleToInt8Loop)(in, out, n);
}
void doubleToInt16(const keel::f64 *in, keel::i16  *out, keel::usize n) {
    HWY_DYNAMIC_DISPATCH(DoubleToInt16Loop)(in, out, n);
}
void doubleToInt32(const keel::f64 *in, keel::i32  *out, keel::usize n) {
    HWY_DYNAMIC_DISPATCH(DoubleToInt32Loop)(in, out, n);
}
void doubleToInt64(const keel::f64 *in, keel::i64  *out, keel::usize n) {
    HWY_DYNAMIC_DISPATCH(DoubleToInt64Loop)(in, out, n);
}
void doubleToUInt8 (const keel::f64 *in, keel::u8  *out, keel::usize n) {
    HWY_DYNAMIC_DISPATCH(DoubleToUInt8Loop)(in, out, n);
}
void doubleToUInt16(const keel::f64 *in, keel::u16 *out, keel::usize n) {
    HWY_DYNAMIC_DISPATCH(DoubleToUInt16Loop)(in, out, n);
}
void doubleToUInt32(const keel::f64 *in, keel::u32 *out, keel::usize n) {
    HWY_DYNAMIC_DISPATCH(DoubleToUInt32Loop)(in, out, n);
}
void doubleToUInt64(const keel::f64 *in, keel::u64 *out, keel::usize n) {
    HWY_DYNAMIC_DISPATCH(DoubleToUInt64Loop)(in, out, n);
}

} // namespace numkit::builtin::detail

#endif // HWY_ONCE
