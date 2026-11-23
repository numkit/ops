// ops/root_solve.hpp
//
// Scalar / low-dim iterative solver KERNELS (raw double precision, FnHandle
// objective) — the numerical core behind fzero / fminbnd / fminsearch. Core-free
// (value layer only); the toolbox Value-API wrappers and the engine adapters live
// above. Solver kernels belong in ops per LAYERING_TARGET_ARCHITECTURE §2/§3/§9.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <numkit/value/fn_handle.hpp>
#include <utility>

namespace numkit::ops {

/// Expand a bracket around x0 by stepping outward until a sign change; returns
/// {a, b} with f(a)*f(b) <= 0 (a==b when a root is hit exactly). Throws if none
/// is found within the iteration cap.
std::pair<keel::f64, keel::f64> findBracket(FnHandle fn, keel::f64 x0, keel::Allocator *mr = nullptr);

/// Brent's method root on [a, b] (requires f(a)*f(b) < 0, or one endpoint == 0).
keel::f64 brent(FnHandle fn, keel::f64 a, keel::f64 b, keel::Allocator *mr = nullptr);

/// Bounded scalar minimum on [a, b] via Brent's golden-section + parabolic
/// interpolation hybrid (the fminbnd kernel).
keel::f64 brentMin(FnHandle fn, keel::f64 a, keel::f64 b, keel::f64 tol, keel::Allocator *mr = nullptr);

/// Unconstrained multi-dim minimum from x0[0..n) via Nelder-Mead (the fminsearch
/// kernel). Returns the best vertex (length n), allocated on `mr`.
keel::Array<keel::f64> nelderMead(FnHandle fn, const keel::f64 *x0, keel::usize n, keel::f64 tol,
                              keel::Allocator *mr = nullptr);

} // namespace numkit::ops
