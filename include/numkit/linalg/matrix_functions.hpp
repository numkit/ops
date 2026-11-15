/// @file matrix_functions.hpp
/// @ingroup group_matfun
// toolboxes/linalg/include/numkit/linalg/matrix_functions.hpp
//
// Matrix functions: expm, logm, sqrtm. Migrated from
// toolboxes/builtin/src/language/arrays/matrix.cpp.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/complex.h>
#include <functional>
#include <numkit/value/value.hpp>

namespace numkit::linalg {

/// @addtogroup group_matfun
/// @{


/// @brief Matrix exponential (`B = expm(A)`).
///
/// PadÃ©(6) approximation with scaling-and-squaring (Higham 2005).
Value expm(const Value &A, keel::Allocator *mr = nullptr);

/// @brief Matrix logarithm for symmetric positive-definite A.
Value logm_sym(const Value &A, keel::Allocator *mr = nullptr);

/// @brief Matrix square root for symmetric PSD A.
Value sqrtm_sym(const Value &A, keel::Allocator *mr = nullptr);

/// @brief Matrix square root for general square matrix A (`R = sqrtm(A)`).
/// Via BjÃ¶rckâ€“Hammarling recurrence on Schur form.
Value sqrtm(const Value &A, keel::Allocator *mr = nullptr);

/// @brief Matrix logarithm for general square matrix A (`L = logm(A)`).
/// Via Schur decomposition and inverse scaling-and-squaring.
Value logm(const Value &A, keel::Allocator *mr = nullptr);

/// @brief Solve Sylvester equation `A Â· X + X Â· B = C` (`X = sylvester(A, B, C)`).
/// Via Bartelsâ€“Stewart algorithm on Schur forms.
Value sylvester(const Value &A, const Value &B, const Value &C,
                keel::Allocator *mr = nullptr);

/// @brief Action of matrix exponential on a vector (`w = expmv(t, A, v)`).
///
/// Computes `w â‰ˆ exp(t Â· A) Â· v` without forming the full matrix
/// exponential. Krylov subspace via Arnoldi + small expm on the
/// resulting upper Hessenberg matrix; fixed Krylov dimension
/// `m = min(30, n)` in v1 (adaptive refinement deferred).
///
/// Faster than `expm(t*A) * v` for large `n` when v is a single
/// vector; identical to it at machine precision for small n.
Value expmv(keel::f64 t, const Value &A, const Value &v,
            keel::Allocator *mr = nullptr);

/// @brief General matrix function evaluator `funm(A, fun)`.
/// Evaluates scalar function f on square matrix A via Schurâ€“Parlett algorithm.
Value funm(const Value &A, std::function<keel::c64(keel::c64)> f,
           keel::Allocator *mr = nullptr);

/// @brief General matrix function evaluator by string name (`'exp'`, `'sin'`, `'cos'`, `'sinh'`, `'cosh'`).
Value funm(const Value &A, keel::StringRef fnName,
           keel::Allocator *mr = nullptr);


/// @}
} // namespace numkit::linalg
