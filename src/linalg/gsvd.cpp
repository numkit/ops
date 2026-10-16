// toolboxes/linalg/src/gsvd.cpp
//
// gsvd â€” Generalized Singular Value Decomposition.

#include <numkit/linalg/gsvd.hpp>
#include <numkit/linalg/decompositions.hpp>
#include <numkit/linalg/pseudo_subspace.hpp>
#include <numkit/linalg/solvers.hpp>
#include "linalg_detail.hpp"

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace numkit::linalg {

namespace {

using Complex = std::complex<double>;

} // anonymous namespace

Value gsvd_values(const Value &A, const Value &B, std::pmr::memory_resource *mr)
{
    // The reference solves LOGICAL operands by coercion (mixed
    // logical/double pairs answer a double result) but rejects integer
    // and char classes outright â€” coerce only logical, let the gate
    // voice everything else (R2025b probes).
    Value Ac = (A.type() == ValueType::LOGICAL)
                   ? detail::coerceToDoubleMatrix(A, mr) : A;
    Value Bc = (B.type() == ValueType::LOGICAL)
                   ? detail::coerceToDoubleMatrix(B, mr) : B;
    detail::requireDoubleInput(Ac, "gsvd", "A", 1);
    detail::requireDoubleInput(Bc, "gsvd", "B", 2);
    if (Ac.dims().ndim() != 2 || Bc.dims().ndim() != 2)
        throw Error("gsvd: A and B must be 2D matrices", 0, 0, "gsvd", "", "numkit:gsvd:notMatrix");
    const std::size_t m = static_cast<std::size_t>(Ac.dims().dim(0));
    const std::size_t n = static_cast<std::size_t>(Ac.dims().dim(1));
    const std::size_t p = static_cast<std::size_t>(Bc.dims().dim(0));
    if (static_cast<std::size_t>(Bc.dims().dim(1)) != n)
        throw Error("gsvd: A and B must have the same number of columns", 0, 0, "gsvd", "", "numkit:gsvd:badDims");

    if (m == 0 || n == 0 || p == 0) return Value::matrix(0, 1, ValueType::DOUBLE, mr);

    // Step 1: Stack K = [A; B] ((m+p) x n)
    auto K = Value::matrix(m + p, n, (Ac.isComplex() || Bc.isComplex()) ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    if (K.isComplex()) {
        Complex *kd = K.complexDataMut();
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                kd[i + j * (m + p)] = Ac.isComplex() ? Ac.complexData()[i + j * m] : Complex(Ac.doubleData()[i + j * m], 0.0);
            }
            for (std::size_t i = 0; i < p; ++i) {
                kd[m + i + j * (m + p)] = Bc.isComplex() ? Bc.complexData()[i + j * p] : Complex(Bc.doubleData()[i + j * p], 0.0);
            }
        }
    } else {
        double *kd = K.doubleDataMut();
        const double *ad = Ac.doubleData();
        const double *bd = Bc.doubleData();
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                kd[i + j * (m + p)] = ad[i + j * m];
            }
            for (std::size_t i = 0; i < p; ++i) {
                kd[m + i + j * (m + p)] = bd[i + j * p];
            }
        }
    }

    // Step 2: QR of K: K = Qk * Rk
    auto [Qk, Rk_full] = qr_decompose(K, mr);

    // Q1 is top m x n of Qk
    auto Q1 = Value::matrix(m, n, Qk.type(), mr);
    if (Qk.isComplex()) {
        const Complex *qd = Qk.complexData();
        Complex *q1d = Q1.complexDataMut();
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                q1d[i + j * m] = qd[i + j * (m + p)];
            }
        }
    } else {
        const double *qd = Qk.doubleData();
        double *q1d = Q1.doubleDataMut();
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                q1d[i + j * m] = qd[i + j * (m + p)];
            }
        }
    }

    // Singular values of Q1
    Value C_vals = svd_values(Q1, mr);
    const std::size_t num_vals = C_vals.numel();

    Value sigmas = Value::matrix(num_vals, 1, ValueType::DOUBLE, mr);
    double *sd = sigmas.doubleDataMut();

    for (std::size_t i = 0; i < num_vals; ++i) {
        double c = C_vals.doubleData()[i];
        if (c > 1.0) c = 1.0;
        if (c < 0.0) c = 0.0;
        double s = std::sqrt(std::max(0.0, 1.0 - c * c));
        if (s < 1e-14) {
            sd[i] = std::numeric_limits<double>::infinity();
        } else {
            sd[i] = c / s;
        }
    }

    std::sort(sd, sd + num_vals);
    return sigmas;
}

std::tuple<Value, Value, Value, Value, Value> gsvd(const Value &A, const Value &B,
                                                   std::pmr::memory_resource *mr)
{
    // See gsvd_values: ONLY logical coerces to double; integer/char
    // reach the gate and are rejected (R2025b probes).
    Value Ac = (A.type() == ValueType::LOGICAL)
                   ? detail::coerceToDoubleMatrix(A, mr) : A;
    Value Bc = (B.type() == ValueType::LOGICAL)
                   ? detail::coerceToDoubleMatrix(B, mr) : B;
    detail::requireDoubleInput(Ac, "gsvd", "A", 1);
    detail::requireDoubleInput(Bc, "gsvd", "B", 2);
    if (Ac.dims().ndim() != 2 || Bc.dims().ndim() != 2)
        throw Error("gsvd: A and B must be 2D matrices", 0, 0, "gsvd", "", "numkit:gsvd:notMatrix");
    const std::size_t m = static_cast<std::size_t>(Ac.dims().dim(0));
    const std::size_t n = static_cast<std::size_t>(Ac.dims().dim(1));
    const std::size_t p = static_cast<std::size_t>(Bc.dims().dim(0));
    if (static_cast<std::size_t>(Bc.dims().dim(1)) != n)
        throw Error("gsvd: A and B must have the same number of columns", 0, 0, "gsvd", "", "numkit:gsvd:badDims");

    if (m == 0 || n == 0 || p == 0) {
        return {Value::matrix(m, m, ValueType::DOUBLE, mr),
                Value::matrix(p, p, ValueType::DOUBLE, mr),
                Value::matrix(n, n, ValueType::DOUBLE, mr),
                Value::matrix(m, n, ValueType::DOUBLE, mr),
                Value::matrix(p, n, ValueType::DOUBLE, mr)};
    }

    // Step 1: Stack K = [A; B] ((m+p) x n)
    auto K = Value::matrix(m + p, n, (Ac.isComplex() || Bc.isComplex()) ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    if (K.isComplex()) {
        Complex *kd = K.complexDataMut();
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                kd[i + j * (m + p)] = Ac.isComplex() ? Ac.complexData()[i + j * m] : Complex(Ac.doubleData()[i + j * m], 0.0);
            }
            for (std::size_t i = 0; i < p; ++i) {
                kd[m + i + j * (m + p)] = Bc.isComplex() ? Bc.complexData()[i + j * p] : Complex(Bc.doubleData()[i + j * p], 0.0);
            }
        }
    } else {
        double *kd = K.doubleDataMut();
        const double *ad = Ac.doubleData();
        const double *bd = Bc.doubleData();
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                kd[i + j * (m + p)] = ad[i + j * m];
            }
            for (std::size_t i = 0; i < p; ++i) {
                kd[m + i + j * (m + p)] = bd[i + j * p];
            }
        }
    }

    // Step 2: QR of K: K = Qk * Rk (Qk is (m+p) x (m+p), Rk is n x n)
    auto [Qk, Rk_full] = qr_decompose(K, mr);

    // Extract top n x n from Rk_full as Rk
    auto Rk = Value::matrix(n, n, Rk_full.type(), mr);
    if (Rk_full.isComplex()) {
        const Complex *rfd = Rk_full.complexData();
        Complex *rd = Rk.complexDataMut();
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                rd[i + j * n] = rfd[i + j * (m + p)];
            }
        }
    } else {
        const double *rfd = Rk_full.doubleData();
        double *rd = Rk.doubleDataMut();
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                rd[i + j * n] = rfd[i + j * (m + p)];
            }
        }
    }

    // Split Qk = [Q1; Q2] (first n columns of Qk)
    auto Q1 = Value::matrix(m, n, Qk.type(), mr);
    auto Q2 = Value::matrix(p, n, Qk.type(), mr);
    if (Qk.isComplex()) {
        const Complex *qd = Qk.complexData();
        Complex *q1d = Q1.complexDataMut();
        Complex *q2d = Q2.complexDataMut();
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                q1d[i + j * m] = qd[i + j * (m + p)];
            }
            for (std::size_t i = 0; i < p; ++i) {
                q2d[i + j * p] = qd[m + i + j * (m + p)];
            }
        }
    } else {
        const double *qd = Qk.doubleData();
        double *q1d = Q1.doubleDataMut();
        double *q2d = Q2.doubleDataMut();
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                q1d[i + j * m] = qd[i + j * (m + p)];
            }
            for (std::size_t i = 0; i < p; ++i) {
                q2d[i + j * p] = qd[m + i + j * (m + p)];
            }
        }
    }

    // Step 3: SVD of Q1: Q1 = U * C * W^H
    auto [U, C, W] = svd_decompose(Q1, mr);

    // Step 4: SVD of Q2 * W = V * S * Z^H
    // First compute Q2_W = Q2 * W
    auto Q2_W = Value::matrix(p, n, (Q2.isComplex() || W.isComplex()) ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    if (Q2_W.isComplex()) {
        Complex *q2wd = Q2_W.complexDataMut();
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                Complex s(0.0, 0.0);
                for (std::size_t k = 0; k < n; ++k) {
                    Complex q2val = Q2.isComplex() ? Q2.complexData()[i + k * p] : Complex(Q2.doubleData()[i + k * p], 0.0);
                    Complex wval  = W.isComplex()  ? W.complexData()[k + j * n]  : Complex(W.doubleData()[k + j * n], 0.0);
                    s += q2val * wval;
                }
                q2wd[i + j * p] = s;
            }
        }
    } else {
        double *q2wd = Q2_W.doubleDataMut();
        const double *q2d = Q2.doubleData();
        const double *wd  = W.doubleData();
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (std::size_t k = 0; k < n; ++k) {
                    s += q2d[i + k * p] * wd[k + j * n];
                }
                q2wd[i + j * p] = s;
            }
        }
    }

    // Step 4: V = (Q2 * W) * S^{-1} with S = sqrt(I âˆ’ CÂ²).
    // (Q2Â·W)'Â·(Q2Â·W) = W'(I âˆ’ Q1'Q1)W = I âˆ’ CÂ² = SÂ², so normalizing the
    // columns of Q2Â·W by s_j yields the orthonormal V of the CS
    // decomposition DIRECTLY â€” a second SVD introduces a spurious Z that
    // breaks the A = UÂ·CÂ·X' identity whenever it is not identity
    // (repeated singular values). Zero s_j columns (B rank-deficient
    // there) contribute ~0 to B = VÂ·SÂ·X'; they stay zero.
    auto V = Value::matrix(p, n, Q2_W.isComplex() ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    auto S_out = Value::matrix(p, n, Q2_W.type(), mr);
    {
        // C is the mÃ—n diagonal factor of Q1's SVD; read its diagonal.
        const bool C_is_cx = C.isComplex();
        const Complex *C_cx = C_is_cx ? C.complexData() : nullptr;
        const double *C_re = C_is_cx ? nullptr : C.doubleData();
        const std::size_t minCS = std::min(m, n);
        if (V.isComplex()) {
            const Complex *qw = Q2_W.complexData();
            Complex *vd = V.complexDataMut();
            Complex *sd = S_out.complexDataMut();
            for (std::size_t j = 0; j < n; ++j) {
                const double cj = (j < minCS)
                    ? std::abs(C_cx ? C_cx[j + j * m] : Complex(C_re[j + j * m], 0.0))
                    : 0.0;
                const double sj = std::sqrt(std::max(0.0, 1.0 - cj * cj));
                if (j < std::min(p, n))
                    sd[j + j * p] = Complex(sj, 0.0);
                for (std::size_t i = 0; i < p; ++i) {
                    const Complex q = qw[i + j * p];
                    vd[i + j * p] = (sj > 1e-15) ? q / sj : Complex(0.0, 0.0);
                }
            }
        } else {
            const double *qw = Q2_W.doubleData();
            double *vd = V.doubleDataMut();
            double *sd = S_out.doubleDataMut();
            for (std::size_t j = 0; j < n; ++j) {
                const double cj = (j < minCS) ? std::abs(C_re[j + j * m]) : 0.0;
                const double sj = std::sqrt(std::max(0.0, 1.0 - cj * cj));
                if (j < std::min(p, n))
                    sd[j + j * p] = sj;
                for (std::size_t i = 0; i < p; ++i) {
                    const double q = qw[i + j * p];
                    vd[i + j * p] = (sj > 1e-15) ? q / sj : 0.0;
                }
            }
        }
    }

    // Step 5: X = Rk^H * W (shared W â€” see Step 4 note).
    auto X = Value::matrix(n, n, (Rk.isComplex() || W.isComplex()) ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    if (X.isComplex()) {
        Complex *xd = X.complexDataMut();
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                Complex s(0.0, 0.0);
                for (std::size_t k = 0; k < n; ++k) {
                    Complex rkval = Rk.isComplex()  ? std::conj(Rk.complexData()[k + i * n])  : Complex(Rk.doubleData()[k + i * n], 0.0);
                    Complex wval  = W.isComplex()   ? W.complexData()[k + j * n]             : Complex(W.doubleData()[k + j * n], 0.0);
                    s += rkval * wval;
                }
                xd[i + j * n] = s;
            }
        }
    } else {
        double *xd = X.doubleDataMut();
        const double *rkd = Rk.doubleData();
        const double *wd = W.doubleData();
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (std::size_t k = 0; k < n; ++k)
                    s += rkd[k + i * n] * wd[k + j * n]; // Rk^H is transposed
                xd[i + j * n] = s;
            }
        }
    }

    return {detail::narrow_if_real(U, mr),
            detail::narrow_if_real(V, mr),
            detail::narrow_if_real(X, mr),
            detail::narrow_if_real(C, mr),
            detail::narrow_if_real(S_out, mr)};
}

} // namespace numkit::linalg
