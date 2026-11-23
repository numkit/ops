// toolboxes/linalg/src/svd_sketch.cpp
//
// svdsketch & svdappend â€” Randomized low-rank SVD and Incremental SVD.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/tuple.h>
#include <keel/memory.h>
#include <keel/random.h>
#include <numkit/linalg/svd_sketch.hpp>
#include <numkit/linalg/decompositions.hpp>
#include "linalg_detail.hpp"

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::linalg {

namespace {

// Multiply two real matrices C = A * B
Value matMulReal(const Value &A, const Value &B, keel::Allocator *mr)
{
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize k = keel::as<keel::usize>(A.dims().dim(1));
    const keel::usize n = keel::as<keel::usize>(B.dims().dim(1));

    auto C = Value::matrix(m, n, ValueType::DOUBLE, mr);
    keel::f64 *cd = C.doubleDataMut();
    keel::fill(cd, cd + m * n, 0.0);

    const keel::f64 *ad = A.doubleData();
    const keel::f64 *bd = B.doubleData();

    for (keel::usize j = 0; j < n; ++j) {
        for (keel::usize l = 0; l < k; ++l) {
            const keel::f64 b_lj = bd[l + j * k];
            if (b_lj == 0.0) continue;
            for (keel::usize i = 0; i < m; ++i) {
                cd[i + j * m] += ad[i + l * m] * b_lj;
            }
        }
    }
    return C;
}

// Transpose of a real matrix
Value matTranspose(const Value &A, keel::Allocator *mr)
{
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    auto At = Value::matrix(n, m, ValueType::DOUBLE, mr);
    keel::f64 *atd = At.doubleDataMut();
    const keel::f64 *ad = A.doubleData();
    for (keel::usize i = 0; i < m; ++i) {
        for (keel::usize j = 0; j < n; ++j) {
            atd[j + i * n] = ad[i + j * m];
        }
    }
    return At;
}

} // anonymous namespace

keel::Tuple<Value, Value, Value> svdsketch(const Value &A, keel::f64 tol, keel::Allocator *mr)
{
    if (A.dims().ndim() != 2)
        throw Error("svdsketch: input A must be a 2D matrix", 0, 0, "svdsketch", "", "numkit:svdsketch:notMatrix");

    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    const keel::usize min_mn = keel::min(m, n);

    const bool wantSingle = A.type() == ValueType::SINGLE;
    const ValueType rtype = wantSingle ? ValueType::SINGLE : ValueType::DOUBLE;
    if (min_mn == 0) {
        return {Value::matrix(m, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr),
                Value::matrix(n, 0, rtype, mr)};
    }

    Value Aw = detail::widenSingleToDouble(A, mr);

    // Step 1: Draw Gaussian random test matrix Omega (n x l)
    keel::usize l = keel::min(min_mn, keel::as<keel::usize>(15));
    auto Omega = Value::matrix(n, l, ValueType::DOUBLE, mr);
    keel::f64 *od = Omega.doubleDataMut();

    keel::random::Mt19937 gen(42); // fixed seed for deterministic reproducibility
    for (keel::usize i = 0; i < n * l; ++i) od[i] = keel::random::normal_f64(gen);

    // Step 2: Y = A * Omega (m x l)
    Value Y = matMulReal(Aw, Omega, mr);

    // Step 3: QR of Y: Q * R = Y => Q (m x l)
    auto [Q_full, R_y] = qr_decompose(Y, mr);
    auto Q = Value::matrix(m, l, ValueType::DOUBLE, mr);
    const keel::f64 *qfd = Q_full.doubleData();
    keel::f64 *qd = Q.doubleDataMut();
    for (keel::usize j = 0; j < l; ++j) {
        for (keel::usize i = 0; i < m; ++i) {
            qd[i + j * m] = qfd[i + j * m];
        }
    }

    // Step 4: B = Q^T * A (l x n)
    Value Qt = matTranspose(Q, mr);
    Value B = matMulReal(Qt, Aw, mr);

    // Step 5: SVD of B: B = U_b * S * V^T
    auto [Ub, S, V] = svd_decompose(B, mr);

    // Step 6: U = Q * Ub
    Value U = matMulReal(Q, Ub, mr);

    // Step 7: Truncate rank by tolerance
    const keel::f64 s0 = S.doubleData()[0];
    const keel::f64 cutoff = tol * s0;
    keel::usize rank = 0;
    for (keel::usize i = 0; i < keel::min(l, min_mn); ++i) {
        if (S.doubleData()[i + i * l] >= cutoff) {
            rank++;
        }
    }
    if (rank == 0) rank = 1;

    // Slice top rank columns/rows
    auto Ur = Value::matrix(m, rank, ValueType::DOUBLE, mr);
    auto Sr = Value::matrix(rank, rank, ValueType::DOUBLE, mr);
    auto Vr = Value::matrix(n, rank, ValueType::DOUBLE, mr);

    keel::f64 *urd = Ur.doubleDataMut();
    keel::f64 *srd = Sr.doubleDataMut();
    keel::f64 *vrd = Vr.doubleDataMut();

    keel::fill(srd, srd + rank * rank, 0.0);

    const keel::f64 *ud = U.doubleData();
    const keel::f64 *sd = S.doubleData();
    const keel::f64 *vd = V.doubleData();

    for (keel::usize j = 0; j < rank; ++j) {
        for (keel::usize i = 0; i < m; ++i) urd[i + j * m] = ud[i + j * m];
        for (keel::usize i = 0; i < n; ++i) vrd[i + j * n] = vd[i + j * n];
        srd[j + j * rank] = sd[j + j * l];
    }

    return {detail::narrowDoubleToSingleIf(detail::narrow_if_real(Ur, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Sr, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Vr, mr), wantSingle, mr)};
}

keel::Tuple<Value, Value, Value> svdappend(const Value &U, const Value &S, const Value &V,
                                          const Value &A_new, keel::Allocator *mr)
{
    if (U.dims().ndim() != 2 || S.dims().ndim() != 2 || V.dims().ndim() != 2 || A_new.dims().ndim() != 2)
        throw Error("svdappend: inputs must be 2D matrices", 0, 0, "svdappend", "", "numkit:svdappend:notMatrix");

    const keel::usize m = keel::as<keel::usize>(U.dims().dim(0));
    const keel::usize k = keel::as<keel::usize>(U.dims().dim(1));
    const keel::usize n = keel::as<keel::usize>(V.dims().dim(0));
    const keel::usize p = keel::as<keel::usize>(A_new.dims().dim(1));

    if (keel::as<keel::usize>(A_new.dims().dim(0)) != m)
        throw Error("svdappend: A_new must have the same number of rows as U", 0, 0, "svdappend", "", "numkit:svdappend:badDims");

    const bool wantSingle = U.type() == ValueType::SINGLE || S.type() == ValueType::SINGLE
        || V.type() == ValueType::SINGLE || A_new.type() == ValueType::SINGLE;
    Value Uw = detail::widenSingleToDouble(U, mr);
    Value Sw = detail::widenSingleToDouble(S, mr);
    Value Vw = detail::widenSingleToDouble(V, mr);
    Value A_neww = detail::widenSingleToDouble(A_new, mr);

    // Brand algorithm:
    // m_k = U^T * A_new  (k x p)
    Value Ut = matTranspose(Uw, mr);
    Value m_k = matMulReal(Ut, A_neww, mr);

    // P = A_new - U * m_k  (m x p)
    Value Umk = matMulReal(Uw, m_k, mr);
    Value P = Value::matrix(m, p, ValueType::DOUBLE, mr);
    keel::f64 *pd = P.doubleDataMut();
    const keel::f64 *anewd = A_neww.doubleData();
    const keel::f64 *umkd = Umk.doubleData();
    for (keel::usize i = 0; i < m * p; ++i) pd[i] = anewd[i] - umkd[i];

    // Economic QR of P: Qp * Rp = P (Qp is m x p, Rp is p x p)
    auto [Qp_full, Rp_full] = qr_decompose(P, mr);
    auto Qp = Value::matrix(m, p, ValueType::DOUBLE, mr);
    auto Rp = Value::matrix(p, p, ValueType::DOUBLE, mr);

    const keel::f64 *qpfd = Qp_full.doubleData();
    const keel::f64 *rpfd = Rp_full.doubleData();
    keel::f64 *qpd = Qp.doubleDataMut();
    keel::f64 *rpd = Rp.doubleDataMut();

    for (keel::usize j = 0; j < p; ++j) {
        for (keel::usize i = 0; i < m; ++i) qpd[i + j * m] = qpfd[i + j * m];
        for (keel::usize i = 0; i < p; ++i) rpd[i + j * p] = rpfd[i + j * (m + p)];
    }

    // Build block matrix K ((k+p) x (k+p)):
    // [ S    m_k ]
    // [ 0    R_p ]
    const keel::usize dim = k + p;
    Value K = Value::matrix(dim, dim, ValueType::DOUBLE, mr);
    keel::f64 *kd = K.doubleDataMut();
    keel::fill(kd, kd + dim * dim, 0.0);

    const keel::f64 *sd = Sw.doubleData();
    const keel::f64 *mkd = m_k.doubleData();

    // S block (k x k)
    for (keel::usize j = 0; j < k; ++j) {
        for (keel::usize i = 0; i < k; ++i) {
            kd[i + j * dim] = sd[i + j * k];
        }
    }
    // m_k block (k x p)
    for (keel::usize j = 0; j < p; ++j) {
        for (keel::usize i = 0; i < k; ++i) {
            kd[i + (k + j) * dim] = mkd[i + j * k];
        }
    }
    // Rp block (p x p)
    for (keel::usize j = 0; j < p; ++j) {
        for (keel::usize i = 0; i < p; ++i) {
            kd[(k + i) + (k + j) * dim] = rpd[i + j * p];
        }
    }

    // SVD of K: K = Uk * S_new * Vk^T
    auto [Uk, S_new, Vk] = svd_decompose(K, mr);

    // Form U_large = [U Qp] (m x (k+p))
    Value U_large = Value::matrix(m, dim, ValueType::DOUBLE, mr);
    keel::f64 *uld = U_large.doubleDataMut();
    const keel::f64 *ud = Uw.doubleData();
    for (keel::usize j = 0; j < k; ++j) {
        for (keel::usize i = 0; i < m; ++i) uld[i + j * m] = ud[i + j * m];
    }
    for (keel::usize j = 0; j < p; ++j) {
        for (keel::usize i = 0; i < m; ++i) uld[i + (k + j) * m] = qpd[i + j * m];
    }

    // Form V_large = [V 0; 0 I_p] ((n+p) x (k+p))
    const keel::usize n_new = n + p;
    Value V_large = Value::matrix(n_new, dim, ValueType::DOUBLE, mr);
    keel::f64 *vld = V_large.doubleDataMut();
    keel::fill(vld, vld + n_new * dim, 0.0);

    const keel::f64 *vd = Vw.doubleData();
    for (keel::usize j = 0; j < k; ++j) {
        for (keel::usize i = 0; i < n; ++i) vld[i + j * n_new] = vd[i + j * n];
    }
    for (keel::usize j = 0; j < p; ++j) {
        vld[(n + j) + (k + j) * n_new] = 1.0;
    }

    // Final U_new = U_large * Uk
    Value U_new = matMulReal(U_large, Uk, mr);

    // Final V_new = V_large * Vk
    Value V_new = matMulReal(V_large, Vk, mr);

    return {detail::narrowDoubleToSingleIf(detail::narrow_if_real(U_new, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(S_new, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(V_new, mr), wantSingle, mr)};
}

} // namespace numkit::linalg
