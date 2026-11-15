// ops/include/numkit/ops/blas1.hpp
#pragma once

#include <keel/core.h>
#include <cstddef>

namespace numkit::ops {

/// @brief Vector add scaled y = alpha * x + y (real double).
void axpy(keel::usize n, keel::f64 alpha, const keel::f64 *x, keel::f64 *y);

} // namespace numkit::ops
