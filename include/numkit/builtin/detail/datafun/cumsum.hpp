// toolboxes/builtin/src/math/elementary/backends/cumsum.hpp
//
// Internal contract for SIMD prefix-op family (cumsum / cumprod /
// cummax / cummin) on contiguous double vectors. SIMD-dispatched
// (Highway) when NUMKIT_WITH_SIMD=ON, scalar otherwise. The data
// dependency `acc = op(acc, x[i]); r[i] = acc;` makes the scalar loop
// fundamentally serial — Hillis-Steele parallel scan within each SIMD
// vector breaks the chain into log2(lanes) steps, then propagates a
// carry across vectors.
//
// The cum-max / cum-min entries skip NaN inputs (matches MATLAB
// 'omitnan' default since R2018a). Leading-NaN prefix in src is
// preserved as NaN in dst — once the first non-NaN appears, the
// running max/min is well-defined and subsequent NaN inputs leave it
// unchanged.

#pragma once

#include <keel/core.h>

namespace numkit::builtin {

// Inclusive prefix sum over [src, src+n) into [dst, dst+n).
void cumsumScan(const keel::f64 *src, keel::f64 *dst, keel::usize n);

// Inclusive prefix product. Identity = 1.0; NaN propagates per IEEE
// (matches MATLAB cumprod which has no NaN-skip option).
void cumprodScan(const keel::f64 *src, keel::f64 *dst, keel::usize n);

// Inclusive prefix max / min, skipping NaN. Identity = ±inf. Leading
// NaN positions in src are preserved as NaN in dst (running aggregate
// is undefined until the first non-NaN appears).
void cummaxScan(const keel::f64 *src, keel::f64 *dst, keel::usize n);
void cumminScan(const keel::f64 *src, keel::f64 *dst, keel::usize n);

} // namespace numkit::builtin
