// ops/include/numkit/ops/iir_filter.hpp
//
// Direct-Form-II-transposed IIR filter recurrence — the numerical core of
// signal's filter()/filtfilt() (and the zi/zf initial-condition forms exposed
// through the register layer). Raw-buffer, core-free; the recurrence is
// inherently sequential (each output feeds the next), so there is no SIMD
// backend — the value of living in ops is dedup (one kernel for filter/filtfilt/
// the register half) plus a single native codegen-lowering target for filter().
//
// b/a are expected a0-normalised by the caller (the Value-API divides by a(1)).

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <numkit/value/value.hpp>    // numkit::Complex

#include <cstddef>

namespace numkit::ops {

// Direct-form-II transposed IIR filter (real), applied to a flat input buffer.
// Optional `zi` (length ziLen) seeds the initial delay state; when `zfOut` is
// non-null the final state (length max(nb,na)-1) is written there — this backs
// MATLAB's filter(b,a,x,zi) and [y,zf] = filter(...).
keel::Array<keel::f64> applyFilterDf2t(const keel::f64 *bn, keel::usize nb, const keel::f64 *an,
                                   keel::usize na, const keel::f64 *input, keel::usize len,
                                   const keel::f64 *zi = nullptr, keel::usize ziLen = 0,
                                   keel::f64 *zfOut = nullptr,
                                   keel::Allocator *mr = nullptr);

// Complex DF2T variant. filter() is bilinear (the recursive a-part mixes
// terms), so a real/imag split does not apply — the recurrence runs over
// Complex.
keel::Array<keel::c64> applyFilterDf2tComplex(const keel::c64 *bn, keel::usize nb,
                                           const keel::c64 *an, keel::usize na,
                                           const keel::c64 *input, keel::usize len,
                                           const keel::c64 *zi = nullptr,
                                           keel::usize ziLen = 0,
                                           keel::c64 *zfOut = nullptr,
                                           keel::Allocator *mr = nullptr);

// ── Second-order section (biquad) DF2T ────────────────────────────────────
// One biquad section applied in place to a single-channel signal, Direct-Form-II
// transposed. b/a are pre-normalised so a0 = 1:
//   y[n] = b0*x[n] + s1;  s1 = b1*x[n] - a1*y[n] + s2;  s2 = b2*x[n] - a2*y[n].
// The building block of an SOS cascade (signal sosfilt/sosfiltfilt). Like the
// general DF2T above it is a sequential recurrence — no SIMD.
void biquadDf2t(keel::f64 b0, keel::f64 b1, keel::f64 b2, keel::f64 a1, keel::f64 a2,
                const keel::f64 *x, keel::f64 *y, keel::usize n);

// Same recurrence seeded with an explicit initial state (s1_init, s2_init) —
// used by the forward/backward passes of sosfiltfilt.
void biquadDf2tWithState(keel::f64 b0, keel::f64 b1, keel::f64 b2, keel::f64 a1, keel::f64 a2,
                         const keel::f64 *x, keel::f64 *y, keel::usize n,
                         keel::f64 s1_init, keel::f64 s2_init);

} // namespace numkit::ops
