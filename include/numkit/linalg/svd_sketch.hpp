/// @file svd_sketch.hpp
/// @ingroup group_matfun
// toolboxes/linalg/include/numkit/linalg/svd_sketch.hpp
//
// svdsketch & svdappend: Randomized and Incremental SVD.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <tuple>
#include <numkit/value/value.hpp>

namespace numkit::linalg {

/// @addtogroup group_matfun
/// @{


/// @brief Randomized low-rank SVD approximation (`[U, S, V] = svdsketch(A, tol)`).
/// Uses Halkoâ€“Martinssonâ€“Tropp algorithm with Gaussian random test matrix.
std::tuple<Value, Value, Value> svdsketch(const Value &A, keel::f64 tol = 1e-6,
                                          keel::Allocator *mr = nullptr);

/// @brief Incremental SVD update (`[U, S, V] = svdappend(U, S, V, A_new)`).
/// Appends new columns to existing SVD decomposition using Brand's algorithm.
std::tuple<Value, Value, Value> svdappend(const Value &U, const Value &S, const Value &V,
                                          const Value &A_new,
                                          keel::Allocator *mr = nullptr);


/// @}
} // namespace numkit::linalg
