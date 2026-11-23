/// @file eigs.hpp
/// @ingroup group_matfun
// toolboxes/linalg/include/numkit/linalg/eigs.hpp
//
// eigs & svds: Subset of eigenvalues / singular values via Arnoldi iteration.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/tuple.h>
#include <numkit/value/value.hpp>

namespace numkit::linalg {

/// @addtogroup group_matfun
/// @{


/// @brief Compute top k eigenvalues of A as a column vector.
Value eigs_values(const Value &A, keel::usize k = 6, keel::Allocator *mr = nullptr);
Value eigs_values(const Value &A, keel::usize k, const Value &sigma, keel::Allocator *mr = nullptr);

/// @brief Compute top k eigenvalues and eigenvectors of A: [V, D] = eigs(A, k[, sigma]).
keel::Tuple<Value, Value> eigs(const Value &A, keel::usize k = 6, keel::Allocator *mr = nullptr);
keel::Tuple<Value, Value> eigs(const Value &A, keel::usize k, const Value &sigma, keel::Allocator *mr = nullptr);

/// @brief Generalized subset eigenvalues / eigenvectors: eigs(A, B, k[, sigma]).
Value eigs_generalized_values(const Value &A, const Value &B, keel::usize k = 6,
                              const Value &sigma = Value(), keel::Allocator *mr = nullptr);
keel::Tuple<Value, Value> eigs_generalized(const Value &A, const Value &B, keel::usize k = 6,
                                          const Value &sigma = Value(), keel::Allocator *mr = nullptr);

/// @brief Compute top k singular values of A as a column vector.
Value svds_values(const Value &A, keel::usize k = 6, keel::Allocator *mr = nullptr);

/// @brief Compute top k singular values and vectors of A: [U, S, V] = svds(A, k).
keel::Tuple<Value, Value, Value> svds(const Value &A, keel::usize k = 6, keel::Allocator *mr = nullptr);


/// @}
} // namespace numkit::linalg
