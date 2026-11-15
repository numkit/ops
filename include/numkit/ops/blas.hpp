// ops/include/numkit/ops/blas.hpp
//
// Umbrella header for BLAS-like operations.

#pragma once

#include <keel/core.h>
#include <keel/complex.h>
#include <numkit/ops/blas1.hpp>
#include <numkit/ops/blas2.hpp>
#include <numkit/ops/blas3.hpp>
#include <atomic>
#include <cstdint>

namespace numkit::ops {

/// @brief Native Highway SIMD LU panel factorization kernel.
bool lu_panel(keel::f64 *A, keel::usize lda, keel::i32 *piv, keel::usize m, keel::usize n, keel::usize offset_row);

/// @brief Native Highway SIMD LU panel factorization kernel for complex.
bool lu_panel(keel::c64 *A, keel::usize lda, keel::i32 *piv, keel::usize m, keel::usize n, keel::usize offset_row);

/// @brief Returns the number of threads used in the most recent gemm call.
inline std::atomic<keel::usize> g_last_gemm_threads_used{0};
inline keel::usize get_last_gemm_threads_used() {
    return g_last_gemm_threads_used.load();
}

} // namespace numkit::ops
