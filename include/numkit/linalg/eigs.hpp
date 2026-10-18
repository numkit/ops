/// @file eigs.hpp
/// @ingroup group_matfun
// toolboxes/linalg/include/nk/linalg/eigs.hpp
//
// eigs & svds: Subset of eigenvalues / singular values via Arnoldi iteration.

#pragma once

#include <memory_resource>
#include <tuple>
#include <numkit/value/value.hpp>

namespace numkit::linalg {

/// @addtogroup group_matfun
/// @{


/// @brief Compute top k eigenvalues of A as a column vector.
Value eigs_values(const Value &A, std::size_t k = 6, std::pmr::memory_resource *mr = nullptr);
Value eigs_values(const Value &A, std::size_t k, const Value &sigma, std::pmr::memory_resource *mr = nullptr);

/// @brief Compute top k eigenvalues and eigenvectors of A: [V, D] = eigs(A, k[, sigma]).
std::tuple<Value, Value> eigs(const Value &A, std::size_t k = 6, std::pmr::memory_resource *mr = nullptr);
std::tuple<Value, Value> eigs(const Value &A, std::size_t k, const Value &sigma, std::pmr::memory_resource *mr = nullptr);

/// @brief Generalized subset eigenvalues / eigenvectors: eigs(A, B, k[, sigma]).
Value eigs_generalized_values(const Value &A, const Value &B, std::size_t k = 6,
                              const Value &sigma = Value(), std::pmr::memory_resource *mr = nullptr);
std::tuple<Value, Value> eigs_generalized(const Value &A, const Value &B, std::size_t k = 6,
                                          const Value &sigma = Value(), std::pmr::memory_resource *mr = nullptr);

/// @brief Compute top k singular values of A as a column vector.
Value svds_values(const Value &A, std::size_t k = 6, std::pmr::memory_resource *mr = nullptr);

/// @brief Compute top k singular values and vectors of A: [U, S, V] = svds(A, k).
std::tuple<Value, Value, Value> svds(const Value &A, std::size_t k = 6, std::pmr::memory_resource *mr = nullptr);


/// @}
} // namespace numkit::linalg
