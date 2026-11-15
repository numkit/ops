// ops/src/la_solve.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/ops/la_solve.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <numkit/ops/blas.hpp>
#include <numkit/ops/parallel_for.hpp>

namespace numkit::ops {

namespace {

template <typename T>
constexpr bool is_complex_v = false;
template <typename T>
constexpr bool is_complex_v<keel::Complex<T>> = true;

template <typename T>
inline double abs_val(const T &x) {
    if constexpr (is_complex_v<T>) return keel::abs(x);
    else return std::fabs(x);
}

template <typename T>
inline double abs_sq(const T &x) {
    if constexpr (is_complex_v<T>) return keel::norm(x);
    else return x * x;
}

template <typename T>
inline T conj_val(const T &x) {
    if constexpr (is_complex_v<T>) return keel::conj(x);
    else return x;
}

// ── Blocked LU with partial pivoting (square A, n×n) using SIMD GEMM ──
template <typename T>
bool lu_recursive_inplace(T *A, keel::usize lda, keel::i32 *piv, keel::usize m, keel::usize n, keel::usize offset_row = 0)
{
    if (m == 0 || n == 0) return true;

    const keel::usize base_n = (m >= 256) ? 32 : 128;
    if (n <= base_n) {
        if constexpr (std::is_same_v<T, double>) {
            return ::numkit::ops::lu_panel(reinterpret_cast<double*>(A), lda, piv, m, n, offset_row);
        } else {
            return ::numkit::ops::lu_panel(reinterpret_cast<keel::c64*>(A), lda, piv, m, n, offset_row);
        }
    }

    // Divide & Conquer
    const keel::usize n1 = n / 2;
    const keel::usize n2 = n - n1;

    // 1. Recursive LU on left panel A1 (m x n1)
    if (!lu_recursive_inplace(A, lda, piv, m, n1, offset_row)) return false;

    // 2. Apply permutations piv[0..n1-1] to right panel A2 (m x n2)
    const keel::usize p2_thresh = (n2 >= 64) ? 32 : (n2 + 1);
    numkit::detail::parallel_for(n2, p2_thresh, [=](keel::usize c_start, keel::usize c_end) {
        for (keel::usize col = c_start; col < c_end; ++col) {
            T *col_ptr = A + (n1 + col) * lda;
            for (keel::usize i = 0; i < n1; ++i) {
                keel::usize p = static_cast<keel::usize>(piv[i] - offset_row);
                if (p != i) {
                    std::swap(col_ptr[i], col_ptr[p]);
                }
            }
        }
    });

    // 3. Solve L11 * U12 = A12 using multithreaded SIMD trsm (n1 x n2)
    T *L11 = A;
    T *A12 = A + n1 * lda;
    if constexpr (is_complex_v<T>) {
        ::numkit::ops::trsm(MatrixSide::Left, MatrixUplo::Lower, MatrixTranspose::NoTrans, MatrixDiag::Unit,
                           n1, n2, keel::c64(1.0, 0.0),
                           reinterpret_cast<const keel::c64*>(L11), lda,
                           reinterpret_cast<keel::c64*>(A12), lda);
    } else {
        ::numkit::ops::trsm(MatrixSide::Left, MatrixUplo::Lower, MatrixTranspose::NoTrans, MatrixDiag::Unit,
                           n1, n2, 1.0, reinterpret_cast<double*>(L11), lda, reinterpret_cast<double*>(A12), lda);
    }

    // 4. Trailing matrix GEMM update: A22 -= L21 * U12 ((m - n1) x n2)
    T *L21 = A + n1;
    T *U12 = A + n1 * lda;
    T *A22 = A + n1 + n1 * lda;
    const keel::usize rem_rows = m - n1;

    if constexpr (is_complex_v<T>) {
        ::numkit::ops::gemm(MatrixTranspose::NoTrans, MatrixTranspose::NoTrans, rem_rows, n2, n1, keel::c64(-1.0, 0.0),
                           reinterpret_cast<const keel::c64*>(L21), lda,
                           reinterpret_cast<const keel::c64*>(U12), lda,
                           keel::c64(1.0, 0.0),
                           reinterpret_cast<keel::c64*>(A22), lda);
    } else {
        ::numkit::ops::gemm(MatrixTranspose::NoTrans, MatrixTranspose::NoTrans, rem_rows, n2, n1, -1.0,
                           reinterpret_cast<const double*>(L21), lda,
                           reinterpret_cast<const double*>(U12), lda,
                           1.0,
                           reinterpret_cast<double*>(A22), lda);
    }

    // 5. Recursive LU on right submatrix A22 ((m - n1) x n2)
    if (!lu_recursive_inplace(A22, lda, piv + n1, rem_rows, n2, offset_row + n1)) return false;

    // 6. Apply permutations piv[n1..n-1] to left panel L21 ((m - n1) x n1)
    const keel::usize p6_thresh = (n1 >= 64) ? 32 : (n1 + 1);
    numkit::detail::parallel_for(n1, p6_thresh, [=](keel::usize c_start, keel::usize c_end) {
        for (keel::usize col = c_start; col < c_end; ++col) {
            T *col_ptr = A + n1 + col * lda;
            for (keel::usize i = 0; i < n2; ++i) {
                keel::usize p = static_cast<keel::usize>(piv[n1 + i] - (offset_row + n1));
                if (p != i) {
                    std::swap(col_ptr[i], col_ptr[p]);
                }
            }
        }
    });

    return true;
}

template <typename T>
bool lu_pivot_inplace(T *LU, keel::i32 *piv, keel::usize n)
{
    return lu_recursive_inplace(LU, n, piv, n, n, 0);
}

// ── QR via Householder (tall A, m×n with m >= n) ─────────────────────
template <typename T>
bool qr_solve_house(T *A, keel::usize m, keel::usize n, T *B, keel::usize nrhs, T *X, keel::Allocator *mr)
{
    keel::Array<T> v(m, mr);

    for (keel::usize k = 0; k < n; ++k) {
        double norm_sq = 0.0;
        for (keel::usize i = k; i < m; ++i) {
            norm_sq += abs_sq(A[i + k * m]);
        }
        if (norm_sq == 0.0) return false;

        const T xk = A[k + k * m];
        const double norm = std::sqrt(norm_sq);

        T alpha;
        if constexpr (is_complex_v<T>) {
            double ax = keel::abs(xk);
            T phase = (ax > 0.0) ? (xk / ax) : T(1.0, 0.0);
            alpha = -phase * norm;
        } else {
            alpha = (xk >= 0.0) ? -norm : norm;
        }

        v[k] = xk - alpha;
        for (keel::usize i = k + 1; i < m; ++i) v[i] = A[i + k * m];

        double v_norm_sq = abs_sq(v[k]);
        for (keel::usize i = k + 1; i < m; ++i) v_norm_sq += abs_sq(v[i]);
        if (v_norm_sq == 0.0) {
            A[k + k * m] = alpha;
            continue;
        }
        const double tau = 2.0 / v_norm_sq;

        for (keel::usize j = k + 1; j < n; ++j) {
            T dot(0);
            for (keel::usize i = k; i < m; ++i) dot += conj_val(v[i]) * A[i + j * m];
            const T s = T(tau) * dot;
            for (keel::usize i = k; i < m; ++i) A[i + j * m] -= s * v[i];
        }
        for (keel::usize j = 0; j < nrhs; ++j) {
            T dot(0);
            for (keel::usize i = k; i < m; ++i) dot += conj_val(v[i]) * B[i + j * m];
            const T s = T(tau) * dot;
            for (keel::usize i = k; i < m; ++i) B[i + j * m] -= s * v[i];
        }

        A[k + k * m] = alpha;
    }

    for (keel::usize j = 0; j < nrhs; ++j) {
        for (keel::usize i = n; i-- > 0;) {
            T s = B[i + j * m];
            for (keel::usize k = i + 1; k < n; ++k)
                s -= A[i + k * m] * X[k + j * n];
            const T r_ii = A[i + i * m];
            if (abs_val(r_ii) == 0.0) return false;
            X[i + j * n] = s / r_ii;
        }
    }
    return true;
}

template <typename T>
bool la_solve_small_fastpath(const T *A, keel::usize lda,
                             const T *B, keel::usize ldb,
                             T *X, keel::usize ldx, keel::usize n, keel::usize nrhs)
{
    alignas(64) T LU_stack[128 * 128];
    alignas(64) keel::i32 piv_stack[128];

    for (keel::usize col = 0; col < n; ++col) {
        std::memcpy(LU_stack + col * n, A + col * lda, n * sizeof(T));
    }
    for (keel::usize col = 0; col < nrhs; ++col) {
        std::memcpy(X + col * ldx, B + col * ldb, n * sizeof(T));
    }

    if constexpr (std::is_same_v<T, double>) {
        if (!::numkit::ops::lu_panel(LU_stack, n, piv_stack, n, n, 0)) return false;
    } else {
        if (!lu_recursive_inplace(LU_stack, n, piv_stack, n, n, 0)) return false;
    }

    for (keel::usize i = 0; i < n; ++i) {
        keel::usize p = static_cast<keel::usize>(piv_stack[i]);
        if (p != i) {
            T *r1 = X + i;
            T *r2 = X + p;
            for (keel::usize col = 0; col < nrhs; ++col) {
                std::swap(r1[col * ldx], r2[col * ldx]);
            }
        }
    }

    keel::usize j = 0;
    if constexpr (std::is_same_v<T, double>) {
        // Unroll RHS by 4 to minimize L1 bandwidth and maximize ILP
        for (; j + 3 < nrhs; j += 4) {
            double *x_col0 = reinterpret_cast<double*>(X + (j + 0) * ldx);
            double *x_col1 = reinterpret_cast<double*>(X + (j + 1) * ldx);
            double *x_col2 = reinterpret_cast<double*>(X + (j + 2) * ldx);
            double *x_col3 = reinterpret_cast<double*>(X + (j + 3) * ldx);
            
            // Forward substitution
            for (keel::usize k = 0; k < n; ++k) {
                const double x0 = x_col0[k];
                const double x1 = x_col1[k];
                const double x2 = x_col2[k];
                const double x3 = x_col3[k];
                if (x0 == 0.0 && x1 == 0.0 && x2 == 0.0 && x3 == 0.0) continue;
                
                const double *l_col = reinterpret_cast<const double*>(LU_stack) + k * n;
                double *y0 = x_col0; double *y1 = x_col1; double *y2 = x_col2; double *y3 = x_col3;
                const double n0 = -x0; const double n1 = -x1; const double n2 = -x2; const double n3 = -x3;
                
                keel::usize i = k + 1;
                for (; i + 3 < n; i += 4) {
                    const double u0 = l_col[i + 0];
                    const double u1 = l_col[i + 1];
                    const double u2 = l_col[i + 2];
                    const double u3 = l_col[i + 3];

                    y0[i+0] += u0 * n0; y0[i+1] += u1 * n0; y0[i+2] += u2 * n0; y0[i+3] += u3 * n0;
                    y1[i+0] += u0 * n1; y1[i+1] += u1 * n1; y1[i+2] += u2 * n1; y1[i+3] += u3 * n1;
                    y2[i+0] += u0 * n2; y2[i+1] += u1 * n2; y2[i+2] += u2 * n2; y2[i+3] += u3 * n2;
                    y3[i+0] += u0 * n3; y3[i+1] += u1 * n3; y3[i+2] += u2 * n3; y3[i+3] += u3 * n3;
                }
                for (; i < n; ++i) {
                    const double u = l_col[i];
                    y0[i] += u * n0; y1[i] += u * n1; y2[i] += u * n2; y3[i] += u * n3;
                }
            }
            
            // Backward substitution
            for (std::intptr_t k_idx = static_cast<std::intptr_t>(n) - 1; k_idx >= 0; --k_idx) {
                const keel::usize k = static_cast<keel::usize>(k_idx);
                const double diag = reinterpret_cast<const double*>(LU_stack)[k + k * n];
                x_col0[k] /= diag; x_col1[k] /= diag; x_col2[k] /= diag; x_col3[k] /= diag;
                const double x0 = x_col0[k];
                const double x1 = x_col1[k];
                const double x2 = x_col2[k];
                const double x3 = x_col3[k];
                if (x0 == 0.0 && x1 == 0.0 && x2 == 0.0 && x3 == 0.0) continue;
                
                const double *u_col = reinterpret_cast<const double*>(LU_stack) + k * n;
                double *y0 = x_col0; double *y1 = x_col1; double *y2 = x_col2; double *y3 = x_col3;
                const double n0 = -x0; const double n1 = -x1; const double n2 = -x2; const double n3 = -x3;
                
                keel::usize i = 0;
                for (; i + 3 < k; i += 4) {
                    const double u0 = u_col[i + 0];
                    const double u1 = u_col[i + 1];
                    const double u2 = u_col[i + 2];
                    const double u3 = u_col[i + 3];

                    y0[i+0] += u0 * n0; y0[i+1] += u1 * n0; y0[i+2] += u2 * n0; y0[i+3] += u3 * n0;
                    y1[i+0] += u0 * n1; y1[i+1] += u1 * n1; y1[i+2] += u2 * n1; y1[i+3] += u3 * n1;
                    y2[i+0] += u0 * n2; y2[i+1] += u1 * n2; y2[i+2] += u2 * n2; y2[i+3] += u3 * n2;
                    y3[i+0] += u0 * n3; y3[i+1] += u1 * n3; y3[i+2] += u2 * n3; y3[i+3] += u3 * n3;
                }
                for (; i < k; ++i) {
                    const double u = u_col[i];
                    y0[i] += u * n0; y1[i] += u * n1; y2[i] += u * n2; y3[i] += u * n3;
                }
            }
        }
    }
    
    // Cleanup remaining columns (for double: columns nrhs - (nrhs % 4) .. nrhs - 1; for complex: all columns 0 .. nrhs - 1)
    for (; j < nrhs; ++j) {
        T *x_col = X + j * ldx;
        for (keel::usize k = 0; k < n; ++k) {
            const T xkj = x_col[k];
            if (xkj == T(0)) continue;
            if constexpr (std::is_same_v<T, double>) {
                const double *l_col = LU_stack + k * n;
                double *y_ptr = x_col;
                const keel::usize rem = n - (k + 1);
                const double neg_x = -xkj;
                numkit::ops::axpy(rem, neg_x, l_col + (k + 1), y_ptr + (k + 1));
            } else {
                for (keel::usize i = k + 1; i < n; ++i) {
                    x_col[i] -= LU_stack[i + k * n] * xkj;
                }
            }
        }
        for (std::intptr_t k = static_cast<std::intptr_t>(n) - 1; k >= 0; --k) {
            x_col[k] /= LU_stack[k + k * n];
            const T xkj = x_col[k];
            if (xkj == T(0)) continue;
            if constexpr (std::is_same_v<T, double>) {
                const double *u_col = LU_stack + k * n;
                double *y_ptr = x_col;
                const keel::usize rem = static_cast<keel::usize>(k);
                const double neg_x = -xkj;
                numkit::ops::axpy(rem, neg_x, u_col, y_ptr);
            } else {
                for (std::intptr_t i = k - 1; i >= 0; --i) {
                    x_col[i] -= LU_stack[i + k * n] * xkj;
                }
            }
        }
    }

    return true;
}

template <typename T>
void trsm_L_recursive(keel::usize m, keel::usize n, const T* A, keel::usize lda, T* B, keel::usize ldb) {
    if (m <= 16) {
        if constexpr (is_complex_v<T>) {
            ops::trsm(MatrixSide::Left, MatrixUplo::Lower, MatrixTranspose::NoTrans, MatrixDiag::Unit, m, n, keel::c64(1,0), reinterpret_cast<const keel::c64*>(A), lda, reinterpret_cast<keel::c64*>(B), ldb);
        } else {
            ops::trsm(MatrixSide::Left, MatrixUplo::Lower, MatrixTranspose::NoTrans, MatrixDiag::Unit, m, n, 1.0, A, lda, B, ldb);
        }
        return;
    }
    keel::usize m1 = m / 2;
    keel::usize m2 = m - m1;
    trsm_L_recursive(m1, n, A, lda, B, ldb);
    if constexpr (is_complex_v<T>) {
        ops::gemm(ops::MatrixTranspose::NoTrans, ops::MatrixTranspose::NoTrans, m2, n, m1, keel::c64(-1,0), reinterpret_cast<const keel::c64*>(A + m1), lda, reinterpret_cast<const keel::c64*>(B), ldb, keel::c64(1,0), reinterpret_cast<keel::c64*>(B + m1), ldb);
    } else {
        ops::gemm(ops::MatrixTranspose::NoTrans, ops::MatrixTranspose::NoTrans, m2, n, m1, -1.0, A + m1, lda, B, ldb, 1.0, B + m1, ldb);
    }
    trsm_L_recursive(m2, n, A + m1 + m1 * lda, lda, B + m1, ldb);
}

template <typename T>
void trsm_U_recursive(keel::usize m, keel::usize n, const T* A, keel::usize lda, T* B, keel::usize ldb) {
    if (m <= 16) {
        if constexpr (is_complex_v<T>) {
            ops::trsm(MatrixSide::Left, MatrixUplo::Upper, MatrixTranspose::NoTrans, MatrixDiag::NonUnit, m, n, keel::c64(1,0), reinterpret_cast<const keel::c64*>(A), lda, reinterpret_cast<keel::c64*>(B), ldb);
        } else {
            ops::trsm(MatrixSide::Left, MatrixUplo::Upper, MatrixTranspose::NoTrans, MatrixDiag::NonUnit, m, n, 1.0, A, lda, B, ldb);
        }
        return;
    }
    keel::usize m1 = m / 2;
    keel::usize m2 = m - m1;
    trsm_U_recursive(m2, n, A + m1 + m1 * lda, lda, B + m1, ldb);
    if constexpr (is_complex_v<T>) {
        ops::gemm(ops::MatrixTranspose::NoTrans, ops::MatrixTranspose::NoTrans, m1, n, m2, keel::c64(-1,0), reinterpret_cast<const keel::c64*>(A + m1 * lda), lda, reinterpret_cast<const keel::c64*>(B + m1), ldb, keel::c64(1,0), reinterpret_cast<keel::c64*>(B), ldb);
    } else {
        ops::gemm(ops::MatrixTranspose::NoTrans, ops::MatrixTranspose::NoTrans, m1, n, m2, -1.0, A + m1 * lda, lda, B + m1, ldb, 1.0, B, ldb);
    }
    trsm_U_recursive(m1, n, A, lda, B, ldb);
}

template <typename T>
void trsm_L_blocked_seq(keel::usize n, keel::usize nrhs, const T *A, keel::usize lda, T *B, keel::usize ldb) {
    constexpr keel::usize nb = 64;
    for (keel::usize j = 0; j < n; j += nb) {
        const keel::usize jb = std::min(nb, n - j);
        // Solve L_jj * X_j = B_j
        for (keel::usize k = 0; k < jb; ++k) {
            const T* l_col = A + (j + k) * lda;
            for (keel::usize c = 0; c < nrhs; ++c) {
                const T xkc = B[j + k + c * ldb];
                if (xkc != T(0)) {
                    if constexpr (std::is_same_v<T, double>) {
                        if (jb > k + 1) {
                            numkit::ops::axpy(jb - (k + 1), -xkc, reinterpret_cast<const double*>(l_col + j + k + 1), reinterpret_cast<double*>(B + j + k + 1 + c * ldb));
                        }
                    } else {
                        for (keel::usize i = k + 1; i < jb; ++i) {
                            B[j + i + c * ldb] -= l_col[j + i] * xkc;
                        }
                    }
                }
            }
        }
        // Update trailing matrix: B_trailing -= L_trailing * X_j
        if (j + jb < n) {
            const keel::usize rem_m = n - (j + jb);
            const T* L21 = A + j * lda + (j + jb);
            T* B2 = B + (j + jb);
            
            if constexpr (std::is_same_v<T, double>) {
                numkit::ops::gemm(ops::MatrixTranspose::NoTrans, ops::MatrixTranspose::NoTrans, rem_m, nrhs, jb, -1.0, 
                                  reinterpret_cast<const double*>(L21), lda, 
                                  reinterpret_cast<const double*>(B + j), ldb, 
                                  1.0, 
                                  reinterpret_cast<double*>(B2), ldb);
            } else {
                numkit::ops::gemm(ops::MatrixTranspose::NoTrans, ops::MatrixTranspose::NoTrans, rem_m, nrhs, jb, keel::c64(-1.0, 0.0), 
                                  reinterpret_cast<const keel::c64*>(L21), lda, 
                                  reinterpret_cast<const keel::c64*>(B + j), ldb, 
                                  keel::c64(1.0, 0.0), 
                                  reinterpret_cast<keel::c64*>(B2), ldb);
            }
        }
    }
}

template <typename T>
void trsm_U_blocked_seq(keel::usize n, keel::usize nrhs, const T *A, keel::usize lda, T *B, keel::usize ldb) {
    constexpr keel::usize nb = 64;
    for (std::intptr_t j_idx = static_cast<std::intptr_t>(n) - 1; j_idx >= 0; j_idx -= nb) {
        const keel::usize j = static_cast<keel::usize>(std::max(std::intptr_t{0}, j_idx - static_cast<std::intptr_t>(nb) + 1));
        const keel::usize jb = static_cast<keel::usize>(j_idx) - j + 1;
        
        // Solve U_jj * X_j = B_j
        for (std::intptr_t k_idx = static_cast<std::intptr_t>(jb) - 1; k_idx >= 0; --k_idx) {
            const keel::usize k = static_cast<keel::usize>(k_idx);
            const T* u_col = A + (j + k) * lda;
            const T akk = u_col[j + k];
            for (keel::usize c = 0; c < nrhs; ++c) {
                B[j + k + c * ldb] /= akk;
                const T xkc = B[j + k + c * ldb];
                if (xkc != T(0)) {
                    if constexpr (std::is_same_v<T, double>) {
                        if (k > 0) {
                            numkit::ops::axpy(k, -xkc, reinterpret_cast<const double*>(u_col + j), reinterpret_cast<double*>(B + j + c * ldb));
                        }
                    } else {
                        for (keel::usize i = 0; i < k; ++i) {
                            B[j + i + c * ldb] -= u_col[j + i] * xkc;
                        }
                    }
                }
            }
        }
        // Update leading matrix: B_leading -= U_leading * X_j
        if (j > 0) {
            const keel::usize rem_m = j;
            const T* U12 = A + j * lda;
            T* B1 = B;
            
            if constexpr (std::is_same_v<T, double>) {
                numkit::ops::gemm(ops::MatrixTranspose::NoTrans, ops::MatrixTranspose::NoTrans, rem_m, nrhs, jb, -1.0, 
                                  reinterpret_cast<const double*>(U12), lda, 
                                  reinterpret_cast<const double*>(B + j), ldb, 
                                  1.0, 
                                  reinterpret_cast<double*>(B1), ldb);
            } else {
                numkit::ops::gemm(ops::MatrixTranspose::NoTrans, ops::MatrixTranspose::NoTrans, rem_m, nrhs, jb, keel::c64(-1.0, 0.0), 
                                  reinterpret_cast<const keel::c64*>(U12), lda, 
                                  reinterpret_cast<const keel::c64*>(B + j), ldb, 
                                  keel::c64(1.0, 0.0), 
                                  reinterpret_cast<keel::c64*>(B1), ldb);
            }
        }
    }
}

template <typename T>
bool lu_blocked_inplace(T *A, keel::usize lda, keel::i32 *piv, keel::usize m, keel::usize n, keel::usize offset_row = 0)
{
    if (m == 0 || n == 0) return true;

    if (m <= 128 || n <= 128) {
        return lu_recursive_inplace(A, lda, piv, m, n, offset_row);
    }

    constexpr keel::usize nb = 256;
    const keel::usize min_mn = std::min(m, n);

    for (keel::usize j = 0; j < min_mn; j += nb) {
        const keel::usize jb = std::min(nb, min_mn - j);
        const keel::usize rem_m = m - j;
        T *A_jj = A + j + j * lda;

        // Factor panel: recursive LU handles tall-skinny panels perfectly
        if (!lu_recursive_inplace(A_jj, lda, piv + j, rem_m, jb, offset_row + j)) return false;

        // Apply permutations piv[j..j+jb-1] to left panel columns 0..j-1
        if (j > 0) {
            const keel::usize p_left_thresh = (j >= 64) ? 32 : (j + 1);
            numkit::detail::parallel_for(j, p_left_thresh, [=](keel::usize c_start, keel::usize c_end) {
                for (keel::usize col = c_start; col < c_end; ++col) {
                    T *col_ptr = A + j + col * lda;
                    for (keel::usize i = 0; i < jb; ++i) {
                        keel::usize p = static_cast<keel::usize>(piv[j + i] - (offset_row + j));
                        if (p != i) {
                            std::swap(col_ptr[i], col_ptr[p]);
                        }
                    }
                }
            });
        }

        if (j + jb < n) {
            const keel::usize rem_n = n - (j + jb);
            T *A_right = A + j + (j + jb) * lda;

            // Apply permutations to trailing matrix A_right (cache-friendly column major traversal)
            const keel::usize p_thresh = (rem_n >= 64) ? 32 : (rem_n + 1);
            numkit::detail::parallel_for(rem_n, p_thresh, [=](keel::usize c_start, keel::usize c_end) {
                for (keel::usize col = c_start; col < c_end; ++col) {
                    T *col_ptr = A_right + col * lda;
                    for (keel::usize i = 0; i < jb; ++i) {
                        keel::usize p = static_cast<keel::usize>(piv[j + i] - (offset_row + j));
                        if (p != i) {
                            std::swap(col_ptr[i], col_ptr[p]);
                        }
                    }
                }
            });

            // TRSM: L11 \ U12
            T *L11 = A_jj;
            T *U12 = A_right;
            if constexpr (is_complex_v<T>) {
                ops::trsm(ops::MatrixSide::Left, ops::MatrixUplo::Lower, ops::MatrixTranspose::NoTrans, ops::MatrixDiag::Unit,
                          jb, rem_n, keel::c64(1.0, 0.0), reinterpret_cast<const keel::c64*>(L11), lda,
                          reinterpret_cast<keel::c64*>(U12), lda);
                if (rem_m > jb) {
                    // GEMM: A22 -= L21 * U12
                    ops::gemm(ops::MatrixTranspose::NoTrans, ops::MatrixTranspose::NoTrans,
                              rem_m - jb, rem_n, jb, keel::c64(-1.0, 0.0),
                              reinterpret_cast<const keel::c64*>(L11 + jb), lda,
                              reinterpret_cast<const keel::c64*>(U12), lda,
                              keel::c64(1.0, 0.0),
                              reinterpret_cast<keel::c64*>(U12 + jb), lda);
                }
            } else {
                ops::trsm(ops::MatrixSide::Left, ops::MatrixUplo::Lower, ops::MatrixTranspose::NoTrans, ops::MatrixDiag::Unit,
                          jb, rem_n, 1.0, reinterpret_cast<const double*>(L11), lda,
                          reinterpret_cast<double*>(U12), lda);
                if (rem_m > jb) {
                    // GEMM: A22 -= L21 * U12
                    ops::gemm(ops::MatrixTranspose::NoTrans, ops::MatrixTranspose::NoTrans,
                              rem_m - jb, rem_n, jb, -1.0,
                              reinterpret_cast<const double*>(L11 + jb), lda,
                              reinterpret_cast<const double*>(U12), lda,
                              1.0,
                              reinterpret_cast<double*>(U12 + jb), lda);
                }
            }
        }
    }

    return true;
}

template <typename T>
bool la_solve_impl(const T *A, keel::usize m, keel::usize n, const T *B, keel::usize nrhs, T *X, keel::Allocator *mr)
{
    if (m < n || m == 0 || n == 0 || nrhs == 0) return false;

    keel::ArenaTemp arena = keel::get_scratch(mr);

    if (m == n) {
        if (n <= 96 && nrhs <= 96) {
            return la_solve_small_fastpath(A, m, B, m, X, m, n, nrhs);
        }
        keel::Array<T> A_lu(n * n, arena.allocator());
        std::copy(A, A + n * n, A_lu.begin());
        keel::Array<keel::i32> piv(n, arena.allocator());
        if (!lu_blocked_inplace(A_lu.data(), n, piv.data(), n, n, 0)) return false;

        std::memcpy(X, B, n * nrhs * sizeof(T));

        const keel::i32 *piv_ptr = piv.data();
        const keel::usize p_thresh = (nrhs >= 64) ? 32 : (nrhs + 1);
        numkit::detail::parallel_for(nrhs, p_thresh, [=](keel::usize c_start, keel::usize c_end) {
            for (keel::usize col = c_start; col < c_end; ++col) {
                T *col_ptr = X + col * n;
                for (keel::usize k = 0; k < n; ++k) {
                    keel::usize p = static_cast<keel::usize>(piv_ptr[k]);
                    if (p != k) {
                        std::swap(col_ptr[k], col_ptr[p]);
                    }
                }
            }
        });

        numkit::detail::parallel_for(nrhs, (nrhs >= 64) ? 16 : 1, [=, &A_lu](keel::usize c_start, keel::usize c_end) {
            keel::usize cols = c_end - c_start;
            T* X_slice = X + c_start * n;
            trsm_L_blocked_seq(n, cols, A_lu.data(), n, X_slice, n);
            trsm_U_blocked_seq(n, cols, A_lu.data(), n, X_slice, n);
        });

        return true;
    }

    keel::Array<T> A_qr(m * n, arena.allocator());
    keel::Array<T> B_qr(m * nrhs, arena.allocator());
    std::copy(A, A + m * n, A_qr.begin());
    std::copy(B, B + m * nrhs, B_qr.begin());
    return qr_solve_house(A_qr.data(), m, n, B_qr.data(), nrhs, X, arena.allocator());
}

} // anonymous namespace

bool la_solve(const double *A, keel::usize m, keel::usize n, const double *B, keel::usize nrhs, double *X, keel::Allocator *mr)
{
    return la_solve_impl(A, m, n, B, nrhs, X, mr);
}

bool la_solve(const keel::c64 *A, keel::usize m, keel::usize n, const keel::c64 *B, keel::usize nrhs, keel::c64 *X, keel::Allocator *mr)
{
    return la_solve_impl(A, m, n, B, nrhs, X, mr);
}

bool lu_factor_inplace(double *A, keel::usize lda, keel::i32 *piv, keel::usize m, keel::usize n)
{
    return lu_blocked_inplace(A, lda, piv, m, n, 0);
}

bool lu_factor_inplace(keel::c64 *A, keel::usize lda, keel::i32 *piv, keel::usize m, keel::usize n)
{
    return lu_blocked_inplace(A, lda, piv, m, n, 0);
}

} // namespace numkit::ops
