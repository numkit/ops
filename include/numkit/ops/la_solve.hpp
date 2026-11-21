// ops/include/numkit/ops/la_solve.hpp
//
// Linear-solve and blocked LU kernels (re-exported from keel::linalg).

#pragma once

#include <keel/linalg.h>

namespace numkit::ops {

using keel::linalg::la_solve;

inline bool lu_factor_inplace(keel::f64 *A, keel::usize lda, keel::i32 *piv, keel::usize m, keel::usize n) noexcept {
    return keel::linalg::lu_factor(A, lda, piv, m, n);
}

inline bool lu_factor_inplace(keel::c64 *A, keel::usize lda, keel::i32 *piv, keel::usize m, keel::usize n) noexcept {
    return keel::linalg::lu_factor(A, lda, piv, m, n);
}

} // namespace numkit::ops
