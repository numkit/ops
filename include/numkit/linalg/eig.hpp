/// @file eig.hpp
/// @ingroup group_matfun
// toolboxes/linalg/include/nk/linalg/eig.hpp
//
// Eigenvalue family â€” symmetric & general eig, Hessenberg reduction,
// symmetric Schur, characteristic polynomial, Sylvester equation.
// Migrated from toolboxes/builtin/src/language/arrays/matrix.cpp.

#pragma once

#include <memory_resource>
#include <tuple>
#include <numkit/value/value.hpp>

namespace numkit::linalg {

/// @addtogroup group_matfun
/// @{


/// @brief Symmetric eigendecomposition (`[V, D] = eig(A)`).
///
/// Classical Jacobi rotations. Returns `(V, D)` such that `AÂ·V == VÂ·D`,
/// `V` orthogonal, `D` diagonal (ascending real eigenvalues). Throws if
/// `A` is not symmetric within tolerance.
std::tuple<Value, Value>
eig_symmetric(const Value &A, std::pmr::memory_resource *mr = nullptr);

/// @brief Eigenvalues only (`e = eig(A)` single-output form, symmetric).
Value eig_values(const Value &A, std::pmr::memory_resource *mr = nullptr);

/// @brief Approximate-symmetry test used to pick the symmetric (Jacobi)
/// vs general (char-poly) eig path.
///
/// True iff `A` is square and every off-diagonal pair matches within
/// `tolÂ·(1 + max(|A(i,j)|, |A(j,i)|))`. Shared by `eig_symmetric`'s own
/// guard and the eig register-side auto-dispatch.
bool isSymmetricApprox(const Value &A, double tol);

/// @brief Characteristic polynomial of a matrix (`p = poly(A)`).
///
/// Souriau-Faddeev-LeVerrier. `roots(p) == eig(A)`.
Value poly_of_matrix(const Value &A, std::pmr::memory_resource *mr = nullptr);

/// @brief General (non-symmetric) eigenvalues via characteristic polynomial.
///
/// Less numerically stable than QR iteration but works for moderate `n`.
Value eig_general_values(const Value &A, std::pmr::memory_resource *mr = nullptr);

/// @brief General `[V, D]` eig for real-eigenvalue asymmetric `A`.
///
/// Throws if any eigenvalue has non-zero imaginary part (Francis QR
/// for complex eigvec extraction is deferred).
std::tuple<Value, Value>
eig_general_VD(const Value &A, std::pmr::memory_resource *mr = nullptr);

/// @brief Generalized eigenvalues `e = eig(A, B)` as a column vector.
Value eig_generalized_values(const Value &A, const Value &B,
                             std::pmr::memory_resource *mr = nullptr);

/// @brief Generalized eigendecomposition `[V, D] = eig(A, B)` solving `A*V = B*V*D`.
std::tuple<Value, Value>
eig_generalized_VD(const Value &A, const Value &B,
                   std::pmr::memory_resource *mr = nullptr);

/// @brief Fast Cholesky-Lanczos shift-invert solver for `eigs(A, B, k, 'smallestabs')`
/// when A, B are real symmetric, A is SPD, and k << n.
bool try_eigs_generalized_smallestabs(const Value &A, const Value &B, std::size_t k,
                                      Value &Vout, Value &Dout,
                                      std::pmr::memory_resource *mr = nullptr);

/// @brief Hessenberg reduction (`[P, H] = hess(A)`).
///
/// `A = PÂ·HÂ·P'`, `P` orthogonal, `H` upper-Hessenberg.
std::tuple<Value, Value>
hess(const Value &A, std::pmr::memory_resource *mr = nullptr);

/// @brief Hessenberg-only output (`H = hess(A)` single-output form).
Value hess_H_only(const Value &A, std::pmr::memory_resource *mr = nullptr);

/// @brief Schur decomposition for symmetric A (`[U, T] = schur(A)`).
///
/// Equivalent to eigendecomposition: `A = UÂ·TÂ·U'`, `T` diagonal.
std::tuple<Value, Value>
schur_sym(const Value &A, std::pmr::memory_resource *mr = nullptr);

/// @brief Real Schur decomposition for general (nonsymmetric) A
/// (`[U, T] = schur(A)`).
///
/// `A = UÂ·TÂ·Uáµ€` with `U` orthogonal and `T` the real Schur form â€” upper
/// quasi-triangular, with 1Ã—1 diagonal blocks for real eigenvalues and 2Ã—2
/// blocks for complex-conjugate pairs. Hessenberg reduction + Francis
/// double-shift QR. (The Schur form is not unique; eigenvalues + `UÂ·TÂ·Uáµ€`
/// are the invariants.)
std::tuple<Value, Value>
schur_general(const Value &A, std::pmr::memory_resource *mr = nullptr);

/// @brief Sylvester equation for symmetric A and B (`X = sylvester_sym(A, B, C)`).
///
/// Solves `AÂ·X + XÂ·B = C` via simultaneous diagonalisation.
Value sylvester_sym(const Value &A, const Value &B, const Value &C,
                    std::pmr::memory_resource *mr = nullptr);

/// @brief Polynomial eigenvalue problem
/// (`e = polyeig(A0, A1, ..., Ak)`, possibly `[V, e]`).
///
/// Finds scalars `Î»` and non-zero vectors `x` such that
/// `(A0 + Î»Â·A1 + Î»Â²Â·A2 + â€¦ + Î»^kÂ·Ak) Â· x = 0`.
///
/// Implementation: linearisation via the standard companion matrix
/// of size `kÂ·n Ã— kÂ·n`; eigenvalues of the companion are the
/// polynomial eigenvalues. Right eigenvectors of the companion
/// have a Kronecker structure `[x; Î» x; Î»Â² x; â€¦]`; the first `n`
/// rows give `x`.
///
/// Eigenvalue-only form returns a column of `kÂ·n` values; the matrix
/// form `[V, e] = polyeig(...)` returns the eigenvectors in an
/// `n Ã— kÂ·n` matrix. The matrix form requires real eigenvalues
/// (Francis QR for complex pairs is deferred).
///
/// @param coeffs   Span of `k+1` polynomial coefficient matrices,
///                 ordered constant first: `[A0, A1, ..., Ak]`.
///                 All `n Ã— n`, same size. `Ak` (leading) must be
///                 non-singular.
/// @param mr       Memory resource.
/// @return         Eigenvalue column of length `kÂ·n` (possibly COMPLEX).
Value polyeig_values(Span<const Value> coeffs, std::pmr::memory_resource *mr = nullptr);

/// @brief Polynomial eigenvalue problem, eigenvectors + eigenvalues
/// (`[V, e] = polyeig(A0, ..., Ak)`).
///
/// Throws if any eigenvalue is complex (needs Francis QR).
std::tuple<Value, Value>
polyeig_VE(Span<const Value> coeffs, std::pmr::memory_resource *mr = nullptr);

/// @brief Eigenvalues from a (quasi-)triangular Schur factor
/// (`e = ordeig(T)`).
///
/// Returns eigenvalues in the order they appear in `T`. For a true
/// triangular `T` (real eigenvalues only â€” e.g. diagonal symmetric
/// Schur factor) returns `diag(T)`. For a real quasi-triangular
/// Schur factor, 2Ã—2 sub-diagonal blocks emit conjugate-pair complex
/// eigenvalues from `(a Â± âˆš(cÂ·d))` where the block is `[[a c]; [d a]]`.
///
/// Unlike `eig(T)`, `ordeig` does NOT sort; the result preserves the
/// diagonal/block ordering of `T` exactly â€” useful for reading off
/// eigenvalues from a manually-reordered Schur factor.
Value ordeig(const Value &T, std::pmr::memory_resource *mr = nullptr);


/// @}
} // namespace numkit::linalg
