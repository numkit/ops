// ops/include/numkit/ops/blas.hpp
//
// Umbrella header for BLAS-like operations (re-exported from keel::linalg).

#pragma once

#include <keel/linalg.h>
#include <numkit/ops/blas1.hpp>
#include <numkit/ops/blas2.hpp>
#include <numkit/ops/blas3.hpp>

namespace numkit::ops {

using keel::lu_panel;

/// @brief Returns the number of threads used in the most recent gemm call.
inline keel::usize get_last_gemm_threads_used() noexcept {
    return keel::linalg_last_gemm_threads_used();
}

} // namespace numkit::ops
