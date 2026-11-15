// ops/include/numkit/ops/blas2.hpp
#pragma once

#include <keel/core.h>
#include <cstddef>

namespace numkit::ops {

/// @brief General matrix-vector multiply y = alpha * A * x + beta * y (real double).
void gemv(keel::usize m, keel::usize n,
          keel::f64 alpha, const keel::f64 *A, keel::usize lda,
          const keel::f64 *x, keel::usize incx,
          keel::f64 beta, keel::f64 *y, keel::usize incy);

/// @brief Rank-1 update A = alpha * x * y^T + A (real double).
void ger(keel::usize m, keel::usize n,
         keel::f64 alpha, const keel::f64 *x, keel::usize incx,
         const keel::f64 *y, keel::usize incy,
         keel::f64 *A, keel::usize lda);

} // namespace numkit::ops
