/// @file solvers.hpp
/// @ingroup group_matfun
// toolboxes/linalg/include/numkit/linalg/solvers.hpp
//
// Linear-system solvers: linsolve, lsqminnorm, lsqnonneg.
// Migrated from toolboxes/builtin/src/language/arrays/{matrix,lsq}.cpp.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/string.h>
#include <numkit/value/value.hpp>

namespace numkit::linalg {

/// @addtogroup group_matfun
/// @{


/// @brief Solve `AÂ·X = B` (`X = linsolve(A, B)`).
///
/// LU for square `A`, Householder QR for tall `A` (least-squares).
/// Backs `mldivide` / `\` â€” but `mldivide` lives in toolboxes/builtin (it's
/// the operator implementation). This is the user-facing `linsolve()`.
Value linsolve(const Value &A, const Value &B, keel::Allocator *mr = nullptr);

/// @brief Minimum-norm least-squares solution (`X = lsqminnorm(A, B)`).
///
/// Computes `pinv(A, tol) Â· B`. Full-rank A collapses to A\B; rank-
/// deficient A returns the unique min-norm solution.
Value lsqminnorm(const Value &A, const Value &B, bool have_tol, keel::f64 tol_user,
                 keel::Allocator *mr = nullptr);

/// @brief Result of @ref lsqnonneg_impl.
struct NnlsResult {
    Value x;             ///< Non-negative solution column.
    keel::f64 resnorm;      ///< Squared 2-norm of the residual `â€–CÂ·x âˆ’ dâ€–Â²`.
    Value residual;      ///< Residual column `d âˆ’ CÂ·x`.
    keel::i32 exitflag;        ///< 1 on convergence.
    keel::i32 iterations;      ///< Outer-loop iteration count.
    keel::String algorithm; ///< Always "active-set".
    keel::String message; ///< Termination message.
};

/// @brief Non-negative least-squares (`x = lsqnonneg(C, d)`).
///
/// Lawson-Hanson active-set NNLS: minimises `â€–CÂ·x âˆ’ dâ€–â‚‚` subject to
/// `x â‰¥ 0`. Returns the full @ref NnlsResult for the multi-output forms.
NnlsResult lsqnonneg_impl(const Value &C, const Value &d,
                          keel::Allocator *mr = nullptr);


/// @}
} // namespace numkit::linalg
