// ops/include/numkit/ops/quantile_select.hpp
//
// nth_element-based quantile selection over a raw double buffer — the numerical
// primitive behind stats iqr / mad / median-of-slice. Raw-buffer, core-free.
// Promoted to the kernel layer so any toolbox needing the quantile of a flat
// buffer shares one O(n)-average implementation instead of re-deriving it.

#pragma once

#include <keel/core.h>

namespace numkit::ops {

// Linear-interpolation quantile of s[0,n) at probability p — MATLAB R2025b
// default (type 5: order-statistic positions (k-0.5)/N, i.e. q = p*N + 0.5
// clamped to [1, N]). Selects with nth_element (O(n) average); the picked order
// statistics are exactly the sorted ones, so the result is bit-identical to a
// full-sort quantile, only cheaper.
//
// MUTATES the buffer (nth_element reorders s in place) — callers pass a scratch
// copy. Returns NaN for n == 0.
keel::f64 sliceQuantile(keel::f64 *s, keel::usize n, keel::f64 p);

} // namespace numkit::ops
