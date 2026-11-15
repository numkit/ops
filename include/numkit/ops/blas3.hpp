// ops/include/numkit/ops/blas3.hpp
#pragma once

#include <keel/core.h>
#include <keel/complex.h>
#include <cstddef>

namespace numkit::ops {

enum class MatrixSide { Left, Right };
enum class MatrixUplo { Upper, Lower };
enum class MatrixTranspose { NoTrans, Trans, ConjTrans };
enum class MatrixDiag { NonUnit, Unit };

/// @brief General matrix-matrix multiply C = alpha * op(A) * op(B) + beta * C (real double).
/// Column-major matrices: op(A) is (m x k), op(B) is (k x n), C is (m x n).
void gemm(MatrixTranspose transa, MatrixTranspose transb,
          keel::usize m, keel::usize n, keel::usize k,
          keel::f64 alpha, const keel::f64 *A, keel::usize lda,
          const keel::f64 *B, keel::usize ldb,
          keel::f64 beta, keel::f64 *C, keel::usize ldc);

/// @brief General matrix-matrix multiply C = alpha * op(A) * op(B) + beta * C (complex double).
void gemm(MatrixTranspose transa, MatrixTranspose transb,
          keel::usize m, keel::usize n, keel::usize k,
          keel::c64 alpha, const keel::c64 *A, keel::usize lda,
          const keel::c64 *B, keel::usize ldb,
          keel::c64 beta, keel::c64 *C, keel::usize ldc);

/// @brief Triangular matrix solve (real double).
void trsm(MatrixSide side, MatrixUplo uplo, MatrixTranspose trans, MatrixDiag diag,
          keel::usize m, keel::usize n,
          keel::f64 alpha, const keel::f64 *A, keel::usize lda,
          keel::f64 *B, keel::usize ldb);

/// @brief Triangular matrix solve (complex double).
void trsm(MatrixSide side, MatrixUplo uplo, MatrixTranspose trans, MatrixDiag diag,
          keel::usize m, keel::usize n,
          keel::c64 alpha, const keel::c64 *A, keel::usize lda,
          keel::c64 *B, keel::usize ldb);

/// @brief Symmetric rank-k update C = alpha * A * A^T + beta * C (real double).
void syrk(MatrixUplo uplo, MatrixTranspose trans,
          keel::usize n, keel::usize k,
          keel::f64 alpha, const keel::f64 *A, keel::usize lda,
          keel::f64 beta, keel::f64 *C, keel::usize ldc);

/// @brief Hermitian/Symmetric rank-k update C = alpha * A * A^H + beta * C (complex double).
void syrk(MatrixUplo uplo, MatrixTranspose trans,
          keel::usize n, keel::usize k,
          keel::c64 alpha, const keel::c64 *A, keel::usize lda,
          keel::c64 beta, keel::c64 *C, keel::usize ldc);

} // namespace numkit::ops
