// toolboxes/linalg/src/gsvd.cpp
//
// gsvd â€” Generalized Singular Value Decomposition.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/linalg/gsvd.hpp>
#include <numkit/linalg/decompositions.hpp>
#include <numkit/linalg/pseudo_subspace.hpp>
#include <numkit/linalg/solvers.hpp>
#include "linalg_detail.hpp"

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>

namespace numkit::linalg {

namespace {

} // anonymous namespace

Value gsvd_values(const Value &A, const Value &B, keel::Allocator *mr)
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
    const keel::usize m = static_cast<keel::usize>(Ac.dims().dim(0));
    const keel::usize n = static_cast<keel::usize>(Ac.dims().dim(1));
    const keel::usize p = static_cast<keel::usize>(Bc.dims().dim(0));
    if (static_cast<keel::usize>(Bc.dims().dim(1)) != n)
        throw Error("gsvd: A and B must have the same number of columns", 0, 0, "gsvd", "", "numkit:gsvd:badDims");

    const bool wantSingle = Ac.type() == ValueType::SINGLE || Bc.type() == ValueType::SINGLE;
    if (m == 0 || n == 0 || p == 0)
        return Value::matrix(0, 1, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    Ac = detail::widenSingleToDouble(Ac, mr);
    Bc = detail::widenSingleToDouble(Bc, mr);

    // Step 1: Stack K = [A; B] ((m+p) x n)
    auto K = Value::matrix(m + p, n, (Ac.isComplex() || Bc.isComplex()) ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    if (K.isComplex()) {
        keel::c64 *kd = K.complexDataMut();
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < m; ++i) {
                kd[i + j * (m + p)] = Ac.isComplex() ? Ac.complexData()[i + j * m] : keel::c64(Ac.doubleData()[i + j * m], 0.0);
            }
            for (keel::usize i = 0; i < p; ++i) {
                kd[m + i + j * (m + p)] = Bc.isComplex() ? Bc.complexData()[i + j * p] : keel::c64(Bc.doubleData()[i + j * p], 0.0);
            }
        }
    } else {
        double *kd = K.doubleDataMut();
        const double *ad = Ac.doubleData();
        const double *bd = Bc.doubleData();
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < m; ++i) {
                kd[i + j * (m + p)] = ad[i + j * m];
            }
            for (keel::usize i = 0; i < p; ++i) {
                kd[m + i + j * (m + p)] = bd[i + j * p];
            }
        }
    }

    // Step 2: QR of K: K = Qk * Rk
    auto [Qk, Rk_full] = qr_decompose(K, mr);

    // Q1 is top m x n of Qk
    auto Q1 = Value::matrix(m, n, Qk.type(), mr);
    if (Qk.isComplex()) {
        const keel::c64 *qd = Qk.complexData();
        keel::c64 *q1d = Q1.complexDataMut();
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < m; ++i) {
                q1d[i + j * m] = qd[i + j * (m + p)];
            }
        }
    } else {
        const double *qd = Qk.doubleData();
        double *q1d = Q1.doubleDataMut();
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < m; ++i) {
                q1d[i + j * m] = qd[i + j * (m + p)];
            }
        }
    }

    // Singular values of Q1
    Value C_vals = svd_values(Q1, mr);
    const keel::usize num_vals = C_vals.numel();

    Value sigmas = Value::matrix(num_vals, 1, ValueType::DOUBLE, mr);
    double *sd = sigmas.doubleDataMut();

    for (keel::usize i = 0; i < num_vals; ++i) {
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
    return detail::narrowDoubleToSingleIf(sigmas, wantSingle, mr);
}

std::tuple<Value, Value, Value, Value, Value> gsvd(const Value &A, const Value &B,
                                                   keel::Allocator *mr)
{
    Value Ac = (A.type() == ValueType::LOGICAL)
                   ? detail::coerceToDoubleMatrix(A, mr) : A;
    Value Bc = (B.type() == ValueType::LOGICAL)
                   ? detail::coerceToDoubleMatrix(B, mr) : B;
    detail::requireDoubleInput(Ac, "gsvd", "A", 1);
    detail::requireDoubleInput(Bc, "gsvd", "B", 2);
    if (Ac.dims().ndim() != 2 || Bc.dims().ndim() != 2)
        throw Error("gsvd: A and B must be 2D matrices", 0, 0, "gsvd", "", "numkit:gsvd:notMatrix");
    const keel::usize m = static_cast<keel::usize>(Ac.dims().dim(0));
    const keel::usize n = static_cast<keel::usize>(Ac.dims().dim(1));
    const keel::usize p = static_cast<keel::usize>(Bc.dims().dim(0));
    if (static_cast<keel::usize>(Bc.dims().dim(1)) != n)
        throw Error("gsvd: A and B must have the same number of columns", 0, 0, "gsvd", "", "numkit:gsvd:badDims");

    const bool wantSingle = Ac.type() == ValueType::SINGLE || Bc.type() == ValueType::SINGLE;
    const ValueType rtype = wantSingle ? ValueType::SINGLE : ValueType::DOUBLE;
    if (m == 0 || n == 0 || p == 0) {
        return {Value::matrix(m, m, rtype, mr),
                Value::matrix(p, p, rtype, mr),
                Value::matrix(n, n, rtype, mr),
                Value::matrix(m, n, rtype, mr),
                Value::matrix(p, n, rtype, mr)};
    }

    Ac = detail::widenSingleToDouble(Ac, mr);
    Bc = detail::widenSingleToDouble(Bc, mr);

    // Step 1: Stack K = [A; B] ((m+p) x n)
    auto K = Value::matrix(m + p, n, (Ac.isComplex() || Bc.isComplex()) ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    if (K.isComplex()) {
        keel::c64 *kd = K.complexDataMut();
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < m; ++i) {
                kd[i + j * (m + p)] = Ac.isComplex() ? Ac.complexData()[i + j * m] : keel::c64(Ac.doubleData()[i + j * m], 0.0);
            }
            for (keel::usize i = 0; i < p; ++i) {
                kd[m + i + j * (m + p)] = Bc.isComplex() ? Bc.complexData()[i + j * p] : keel::c64(Bc.doubleData()[i + j * p], 0.0);
            }
        }
    } else {
        double *kd = K.doubleDataMut();
        const double *ad = Ac.doubleData();
        const double *bd = Bc.doubleData();
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < m; ++i) {
                kd[i + j * (m + p)] = ad[i + j * m];
            }
            for (keel::usize i = 0; i < p; ++i) {
                kd[m + i + j * (m + p)] = bd[i + j * p];
            }
        }
    }

    // Step 2: QR of K: K = Qk * Rk (Qk is (m+p) x (m+p), Rk is n x n)
    auto [Qk, Rk_full] = qr_decompose(K, mr);

    // Extract top n x n from Rk_full as Rk
    auto Rk = Value::matrix(n, n, Rk_full.type(), mr);
    if (Rk_full.isComplex()) {
        const keel::c64 *rfd = Rk_full.complexData();
        keel::c64 *rd = Rk.complexDataMut();
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < n; ++i) {
                rd[i + j * n] = rfd[i + j * (m + p)];
            }
        }
    } else {
        const double *rfd = Rk_full.doubleData();
        double *rd = Rk.doubleDataMut();
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < n; ++i) {
                rd[i + j * n] = rfd[i + j * (m + p)];
            }
        }
    }

    // Split Qk = [Q1; Q2] (first n columns of Qk)
    auto Q1 = Value::matrix(m, n, Qk.type(), mr);
    auto Q2 = Value::matrix(p, n, Qk.type(), mr);
    if (Qk.isComplex()) {
        const keel::c64 *qd = Qk.complexData();
        keel::c64 *q1d = Q1.complexDataMut();
        keel::c64 *q2d = Q2.complexDataMut();
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < m; ++i) {
                q1d[i + j * m] = qd[i + j * (m + p)];
            }
            for (keel::usize i = 0; i < p; ++i) {
                q2d[i + j * p] = qd[m + i + j * (m + p)];
            }
        }
    } else {
        const double *qd = Qk.doubleData();
        double *q1d = Q1.doubleDataMut();
        double *q2d = Q2.doubleDataMut();
        for (keel::usize j = 0; j < n; ++j) {
            for (keel::usize i = 0; i < m; ++i) {
                q1d[i + j * m] = qd[i + j * (m + p)];
            }
            for (keel::usize i = 0; i < p; ++i) {
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
        keel::c64 *q2wd = Q2_W.complexDataMut();
        for (keel::usize i = 0; i < p; ++i) {
            for (keel::usize j = 0; j < n; ++j) {
                keel::c64 s(0.0, 0.0);
                for (keel::usize k = 0; k < n; ++k) {
                    keel::c64 q2val = Q2.isComplex() ? Q2.complexData()[i + k * p] : keel::c64(Q2.doubleData()[i + k * p], 0.0);
                    keel::c64 wval  = W.isComplex()  ? W.complexData()[k + j * n]  : keel::c64(W.doubleData()[k + j * n], 0.0);
                    s += q2val * wval;
                }
                q2wd[i + j * p] = s;
            }
        }
    } else {
        double *q2wd = Q2_W.doubleDataMut();
        const double *q2d = Q2.doubleData();
        const double *wd  = W.doubleData();
        for (keel::usize i = 0; i < p; ++i) {
            for (keel::usize j = 0; j < n; ++j) {
                double s = 0.0;
                for (keel::usize k = 0; k < n; ++k) {
                    s += q2d[i + k * p] * wd[k + j * n];
                }
                q2wd[i + j * p] = s;
            }
        }
    }

    auto V = Value::matrix(p, n, Q2_W.isComplex() ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    auto S_out = Value::matrix(p, n, Q2_W.type(), mr);
    {
        const bool C_is_cx = C.isComplex();
        const keel::c64 *C_cx = C_is_cx ? C.complexData() : nullptr;
        const double *C_re = C_is_cx ? nullptr : C.doubleData();
        const keel::usize minCS = std::min(m, n);
        if (V.isComplex()) {
            const keel::c64 *qw = Q2_W.complexData();
            keel::c64 *vd = V.complexDataMut();
            keel::c64 *sd = S_out.complexDataMut();
            for (keel::usize j = 0; j < n; ++j) {
                const double cj = (j < minCS)
                    ? keel::abs(C_cx ? C_cx[j + j * m] : keel::c64(C_re[j + j * m], 0.0))
                    : 0.0;
                const double sj = std::sqrt(std::max(0.0, 1.0 - cj * cj));
                if (j < std::min(p, n))
                    sd[j + j * p] = keel::c64(sj, 0.0);
                for (keel::usize i = 0; i < p; ++i) {
                    const keel::c64 q = qw[i + j * p];
                    vd[i + j * p] = (sj > 1e-15) ? q / sj : keel::c64(0.0, 0.0);
                }
            }
        } else {
            const double *qw = Q2_W.doubleData();
            double *vd = V.doubleDataMut();
            double *sd = S_out.doubleDataMut();
            for (keel::usize j = 0; j < n; ++j) {
                const double cj = (j < minCS) ? keel::abs(C_re[j + j * m]) : 0.0;
                const double sj = std::sqrt(std::max(0.0, 1.0 - cj * cj));
                if (j < std::min(p, n))
                    sd[j + j * p] = sj;
                for (keel::usize i = 0; i < p; ++i) {
                    const double q = qw[i + j * p];
                    vd[i + j * p] = (sj > 1e-15) ? q / sj : 0.0;
                }
            }
        }
    }

    // Step 5: X = Rk^H * W (shared W — see Step 4 note).
    auto X = Value::matrix(n, n, (Rk.isComplex() || W.isComplex()) ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    if (X.isComplex()) {
        keel::c64 *xd = X.complexDataMut();
        for (keel::usize i = 0; i < n; ++i) {
            for (keel::usize j = 0; j < n; ++j) {
                keel::c64 s(0.0, 0.0);
                for (keel::usize k = 0; k < n; ++k) {
                    keel::c64 rkval = Rk.isComplex()  ? keel::conj(Rk.complexData()[k + i * n])  : keel::c64(Rk.doubleData()[k + i * n], 0.0);
                    keel::c64 wval  = W.isComplex()   ? W.complexData()[k + j * n]             : keel::c64(W.doubleData()[k + j * n], 0.0);
                    s += rkval * wval;
                }
                xd[i + j * n] = s;
            }
        }
    } else {
        double *xd = X.doubleDataMut();
        const double *rkd = Rk.doubleData();
        const double *wd = W.doubleData();
        for (keel::usize i = 0; i < n; ++i) {
            for (keel::usize j = 0; j < n; ++j) {
                double s = 0.0;
                for (keel::usize k = 0; k < n; ++k)
                    s += rkd[k + i * n] * wd[k + j * n]; // Rk^H is transposed
                xd[i + j * n] = s;
            }
        }
    }

    return {detail::narrowDoubleToSingleIf(detail::narrow_if_real(U, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(V, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(X, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(C, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(S_out, mr), wantSingle, mr)};
}

} // namespace numkit::linalg
