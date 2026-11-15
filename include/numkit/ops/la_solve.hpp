// ops/include/numkit/ops/la_solve.hpp
//
// Internal linear-solve kernel: solve `A · X = B` (LU for square A,
// Householder QR for tall A). An L0.5 ops primitive — raw column-major
// double buffers + a memory_resource, no Value, no engine. It backs both the
// `\` operator in toolboxes/builtin (mldivide / mrdivide) AND the user-facing
// wrappers (inv / linsolve / pageinv / eig …) in toolboxes/linalg.
//
// It used to live in toolboxes/builtin/include/numkit/builtin/internal/ precisely
// because "builtin must not include linalg" left nowhere shared for it. ops/
// (below both builtin and linalg) is its proper home; builtin's internal
// header now re-exports it for back-compat.
//
// Wide systems (m < n) are NOT supported here — MATLAB's mldivide returns the
// minimum-norm solution for wide matrices (needs SVD or QR-of-A'); deferred.
// All scratch goes through std::pmr::memory_resource (HARD RULE).

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <cstddef>

namespace numkit::ops {

// Solve A · X = B for X.
//   A : m×n column-major (m >= n required; wide systems return false)
//   B : m×nrhs column-major
//   X : n×nrhs column-major (caller-allocated)
// Returns false on: m < n (unsupported), singular A (square, zero pivot),
// or rank-deficient A (tall, zero column norm or zero R diagonal). Internal
// scratch is allocated via the supplied memory_resource.
bool la_solve(const keel::f64 *A, keel::usize m, keel::usize n,
              const keel::f64 *B, keel::usize nrhs,
              keel::f64 *X,
              keel::Allocator *mr = nullptr);

bool la_solve(const keel::c64 *A, keel::usize m, keel::usize n,
              const keel::c64 *B, keel::usize nrhs,
              keel::c64 *X,
              keel::Allocator *mr = nullptr);

bool lu_factor_inplace(keel::f64 *A, keel::usize lda, keel::i32 *piv, keel::usize m, keel::usize n);
bool lu_factor_inplace(keel::c64 *A, keel::usize lda, keel::i32 *piv, keel::usize m, keel::usize n);

} // namespace numkit::ops
