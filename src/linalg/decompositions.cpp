// toolboxes/linalg/src/decompositions.cpp
//
// chol / lu / qr / svd / qrupdate / qrinsert / qrdelete / cholupdate —
// Value adapters over keel::linalg kernels.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/tuple.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/linalg.h>
#include <keel/math.h>
#include <numkit/linalg/decompositions.hpp>
#include <numkit/ops/blas.hpp>
#include <numkit/ops/la_solve.hpp>
#include "decompositions_detail.hpp"   // shared raw-buffer kernels (private)

// Compute-only TU: Value substrate + Error, no engine. The chol / lu / qr /
// svd / qrupdate / qrinsert / qrdelete / cholupdate builtins (CallContext
// wrappers) live in decompositions_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::linalg {

// ────────────────────────────────────────────────────────────────────────
// Cholesky
// ────────────────────────────────────────────────────────────────────────

Value chol(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "chol");
    if (A.dims().ndim() != 2)
        throw Error("chol: input must be a 2D matrix",
                    0, 0, "chol", "", "numkit:chol:notMatrix");
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    if (m != n)
        throw Error("chol: matrix must be square",
                    0, 0, "chol", "", "numkit:chol:notSquare");
    if (m == 0)
        return Value::matrix(0, 0, A.type(), mr);

    const bool wantSingle = (A.type() == ValueType::SINGLE);
    const Value Aw = detail::widenSingleToDouble(A, mr);

    if (Aw.isComplex()) {
        const auto *ad = Aw.complexData();
        for (keel::usize j = 0; j < n; ++j) {
            if (keel::abs(ad[j + j * n].imag()) > 1e-11)
                throw Error("chol: matrix is not positive-definite", 0, 0, "chol", "", "numkit:chol:notPosDef");
            for (keel::usize i = j + 1; i < n; ++i) {
                if (keel::abs(ad[i + j * n] - keel::conj(ad[j + i * n])) > 1e-11)
                    throw Error("chol: matrix is not positive-definite", 0, 0, "chol", "", "numkit:chol:notPosDef");
            }
        }
        auto R = Value::complexMatrix(n, n, mr);
        if (keel::linalg::cholesky_upper(Aw.complexData(), R.complexDataMut(), n) != 0)
            throw Error("chol: matrix is not positive-definite",
                        0, 0, "chol", "", "numkit:chol:notPosDef");
        return detail::narrowDoubleToSingleIf(detail::narrow_if_real(R), wantSingle, mr);
    } else {
        auto R = Value::matrix(n, n, ValueType::DOUBLE, mr);
        if (keel::linalg::cholesky_upper(Aw.doubleData(), R.doubleDataMut(), n) != 0)
            throw Error("chol: matrix is not positive-definite",
                        0, 0, "chol", "", "numkit:chol:notPosDef");
        return detail::narrowDoubleToSingleIf(R, wantSingle, mr);
    }
}

// ────────────────────────────────────────────────────────────────────────
// LU
// ────────────────────────────────────────────────────────────────────────

namespace {

template <typename T>
keel::Tuple<Value, Value, Value>
lu_decompose_impl(const Value &A, keel::Allocator *mr)
{
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    const keel::usize k = keel::min(m, n);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<T> LU(m * n, scratch.allocator());
    keel::Array<keel::i32> piv(keel::max<keel::usize>(k, 1), scratch.allocator());
    const T *src = detail::get_data<T>(A);
    keel::copy(src, src + m * n, LU.begin());
    if (m >= n) {
        if (!keel::linalg::lu_factor(LU.data(), m, piv.data(), m, n)) {
            keel::copy(src, src + m * n, LU.begin());
            (void)keel::linalg::lu_factor_general(LU.data(), m, n, piv.data());
        }
    } else {
        (void)keel::linalg::lu_factor_general(LU.data(), m, n, piv.data());
    }

    Value Lout = detail::make_matrix<T>(m, k, mr);
    Value Uout = detail::make_matrix<T>(k, n, mr);
    Value Pout = Value::matrix(m, m, ValueType::DOUBLE, mr);
    T *L = detail::get_data_mut<T>(Lout);
    T *U = detail::get_data_mut<T>(Uout);
    keel::f64 *P = Pout.doubleDataMut();
    keel::fill(L, L + m * k, T(0));
    keel::fill(U, U + k * n, T(0));
    keel::fill(P, P + m * m, 0.0);

    for (keel::usize j = 0; j < k; ++j) {
        L[j + j * m] = T(1);
        for (keel::usize i = j + 1; i < m; ++i)
            L[i + j * m] = LU[i + j * m];
    }
    for (keel::usize j = 0; j < n; ++j) {
        const keel::usize i_max = keel::min(j + 1, k);
        for (keel::usize i = 0; i < i_max; ++i)
            U[i + j * k] = LU[i + j * m];
    }
    keel::Array<keel::usize> perm(m, scratch.allocator());
    for (keel::usize i = 0; i < m; ++i) perm[i] = i;
    for (keel::usize j = 0; j < k; ++j)
        keel::swap(perm[j], perm[piv[j]]);
    for (keel::usize i = 0; i < m; ++i)
        P[i + perm[i] * m] = 1.0;

    return keel::make_tuple(detail::narrow_if_real(Lout), detail::narrow_if_real(Uout), keel::move(Pout));
}

template <typename T>
Value lu_combined_impl(const Value &A, keel::Allocator *mr)
{
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    const keel::usize k = keel::min(m, n);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i32> piv(keel::max<keel::usize>(k, 1), scratch.allocator());
    Value out = detail::make_matrix<T>(m, n, mr);
    T *LU = detail::get_data_mut<T>(out);
    const T *src = detail::get_data<T>(A);
    keel::copy(src, src + m * n, LU);
    if (m >= n) {
        if (!keel::linalg::lu_factor(LU, m, piv.data(), m, n)) {
            keel::copy(src, src + m * n, LU);
            (void)keel::linalg::lu_factor_general(LU, m, n, piv.data());
        }
    } else {
        (void)keel::linalg::lu_factor_general(LU, m, n, piv.data());
    }
    return detail::narrow_if_real(out);
}

} // anonymous namespace

keel::Tuple<Value, Value, Value>
lu_decompose(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "lu");
    if (A.dims().ndim() != 2)
        throw Error("lu: input must be a 2D matrix",
                    0, 0, "lu", "", "numkit:lu:notMatrix");

    const bool wantSingle = (A.type() == ValueType::SINGLE);
    const Value Aw = detail::widenSingleToDouble(A, mr);

    auto [L, U, P] = Aw.isComplex()
        ? lu_decompose_impl<keel::c64>(Aw, mr)
        : lu_decompose_impl<keel::f64>(Aw, mr);
    if (wantSingle) {
        L = detail::narrowDoubleToSingle(L, mr);
        U = detail::narrowDoubleToSingle(U, mr);
        P = detail::narrowDoubleToSingle(P, mr);
    }
    return keel::make_tuple(keel::move(L), keel::move(U), keel::move(P));
}

Value lu_combined(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "lu");
    if (A.dims().ndim() != 2)
        throw Error("lu: input must be a 2D matrix",
                    0, 0, "lu", "", "numkit:lu:notMatrix");

    const bool wantSingle = (A.type() == ValueType::SINGLE);
    const Value Aw = detail::widenSingleToDouble(A, mr);

    if (Aw.isComplex()) {
        return detail::narrowDoubleToSingleIf(lu_combined_impl<keel::c64>(Aw, mr), wantSingle, mr);
    }
    return detail::narrowDoubleToSingleIf(lu_combined_impl<keel::f64>(Aw, mr), wantSingle, mr);
}

// ────────────────────────────────────────────────────────────────────────
// QR (Householder, full Q)
// ────────────────────────────────────────────────────────────────────────

keel::Tuple<Value, Value>
qr_decompose(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "qr");
    if (A.dims().ndim() != 2)
        throw Error("qr: input must be a 2D matrix",
                    0, 0, "qr", "", "numkit:qr:notMatrix");
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    if (m < n)
        throw Error("qr: number of rows must be >= number of columns "
                    "(wide matrices via row-pivoted QR are deferred)",
                    0, 0, "qr", "", "numkit:qr:wide");
    const bool wantSingle = (A.type() == ValueType::SINGLE);
    const Value Aw = detail::widenSingleToDouble(A, mr);
    if (Aw.isComplex()) {
        auto Q = Value::complexMatrix(m, m, mr);
        auto R = Value::complexMatrix(m, n, mr);
        keel::linalg::qr_householder(Aw.complexData(), m, n, Q.complexDataMut(), R.complexDataMut(), mr);
        return keel::make_tuple(
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Q, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(R, mr), wantSingle, mr));
    }
    auto Q = Value::matrix(m, m, ValueType::DOUBLE, mr);
    auto R = Value::matrix(m, n, ValueType::DOUBLE, mr);
    keel::linalg::qr_householder(Aw.doubleData(), m, n, Q.doubleDataMut(), R.doubleDataMut(), mr);
    return keel::make_tuple(
        detail::narrowDoubleToSingleIf(Q, wantSingle, mr),
        detail::narrowDoubleToSingleIf(R, wantSingle, mr));
}

Value qr_R_only(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "qr");
    if (A.dims().ndim() != 2)
        throw Error("qr: input must be a 2D matrix",
                    0, 0, "qr", "", "numkit:qr:notMatrix");
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    if (m < n)
        throw Error("qr: number of rows must be >= number of columns",
                    0, 0, "qr", "", "numkit:qr:wide");
    const bool wantSingle = (A.type() == ValueType::SINGLE);
    const Value Aw = detail::widenSingleToDouble(A, mr);
    if (Aw.isComplex()) {
        auto R = Value::complexMatrix(m, n, mr);
        keel::linalg::qr_householder(Aw.complexData(), m, n, nullptr, R.complexDataMut(), mr);
        return detail::narrowDoubleToSingleIf(detail::narrow_if_real(R, mr), wantSingle, mr);
    }
    auto R = Value::matrix(m, n, ValueType::DOUBLE, mr);
    keel::linalg::qr_householder(Aw.doubleData(), m, n, nullptr, R.doubleDataMut(), mr);
    return detail::narrowDoubleToSingleIf(R, wantSingle, mr);
}

keel::Tuple<Value, Value>
qr_pivoted(const Value &A, keel::Array<keel::usize> &perm,
           keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "qr");
    if (A.dims().ndim() != 2)
        throw Error("qr: input must be a 2D matrix",
                    0, 0, "qr", "", "numkit:qr:notMatrix");
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    if (m < n)
        throw Error("qr: number of rows must be >= number of columns "
                    "(wide matrices via row-pivoted QR are deferred)",
                    0, 0, "qr", "", "numkit:qr:wide");
    const bool wantSingle = (A.type() == ValueType::SINGLE);
    const Value Aw = detail::widenSingleToDouble(A, mr);
    perm.assign(n, 0);
    if (Aw.isComplex()) {
        auto Q = Value::complexMatrix(m, m, mr);
        auto R = Value::complexMatrix(m, n, mr);
        keel::linalg::qr_pivoted(Aw.complexData(), m, n,
                         Q.complexDataMut(), R.complexDataMut(), perm.data(), mr);
        return keel::make_tuple(
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Q, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(R, mr), wantSingle, mr));
    }
    auto Q = Value::matrix(m, m, ValueType::DOUBLE, mr);
    auto R = Value::matrix(m, n, ValueType::DOUBLE, mr);
    keel::linalg::qr_pivoted(Aw.doubleData(), m, n,
                     Q.doubleDataMut(), R.doubleDataMut(), perm.data(), mr);
    return keel::make_tuple(
        detail::narrowDoubleToSingleIf(Q, wantSingle, mr),
        detail::narrowDoubleToSingleIf(R, wantSingle, mr));
}

// ────────────────────────────────────────────────────────────────────────
// SVD (one-sided Jacobi via keel::linalg::svd_full)
// ────────────────────────────────────────────────────────────────────────

keel::Tuple<Value, Value, Value>
svd_decompose(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "svd");
    if (A.dims().ndim() != 2)
        throw Error("svd: input must be a 2D matrix",
                    0, 0, "svd", "", "numkit:svd:notMatrix");
    const keel::usize m_in = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n_in = keel::as<keel::usize>(A.dims().dim(1));
    const bool wantSingle = (A.type() == ValueType::SINGLE);

    if (m_in == 0 || n_in == 0) {
        return keel::make_tuple(
            detail::narrowDoubleToSingleIf(detail::make_matrix<keel::f64>(m_in, m_in, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::make_matrix<keel::f64>(m_in, n_in, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::make_matrix<keel::f64>(n_in, n_in, mr), wantSingle, mr));
    }

    const Value Aw = detail::widenSingleToDouble(A, mr);
    const keel::usize k = keel::min(m_in, n_in);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> sigma(k, 0.0, scratch.allocator());

    auto Sout = Value::matrix(m_in, n_in, ValueType::DOUBLE, mr);
    keel::f64 *S = Sout.doubleDataMut();
    keel::fill(S, S + m_in * n_in, 0.0);

    if (Aw.isComplex()) {
        auto Uout = Value::complexMatrix(m_in, m_in, mr);
        auto Vout = Value::complexMatrix(n_in, n_in, mr);
        keel::linalg::svd_full(Aw.complexData(), m_in, n_in,
                       Uout.complexDataMut(), sigma.data(), Vout.complexDataMut(), mr);
        for (keel::usize i = 0; i < k; ++i) S[i + i * m_in] = sigma[i];
        return keel::make_tuple(
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Uout, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(Sout, wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Vout, mr), wantSingle, mr));
    }

    auto Uout = Value::matrix(m_in, m_in, ValueType::DOUBLE, mr);
    auto Vout = Value::matrix(n_in, n_in, ValueType::DOUBLE, mr);
    keel::linalg::svd_full(Aw.doubleData(), m_in, n_in,
                   Uout.doubleDataMut(), sigma.data(), Vout.doubleDataMut(), mr);
    for (keel::usize i = 0; i < k; ++i) S[i + i * m_in] = sigma[i];
    return keel::make_tuple(
        detail::narrowDoubleToSingleIf(Uout, wantSingle, mr),
        detail::narrowDoubleToSingleIf(Sout, wantSingle, mr),
        detail::narrowDoubleToSingleIf(Vout, wantSingle, mr));
}

Value svd_values(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "svd");
    const bool wantSingle = (A.type() == ValueType::SINGLE);
    auto [U, S, V] = svd_decompose(detail::widenSingleToDouble(A, mr), mr);
    const keel::usize m = keel::as<keel::usize>(S.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(S.dims().dim(1));
    const keel::usize k = keel::min(m, n);
    auto sv = Value::matrix(k, 1, ValueType::DOUBLE, mr);
    const keel::f64 *S_data = S.doubleData();
    keel::f64 *out = sv.doubleDataMut();
    for (keel::usize i = 0; i < k; ++i)
        out[i] = S_data[i + i * m];
    return detail::narrowDoubleToSingleIf(sv, wantSingle, mr);
}

// ────────────────────────────────────────────────────────────────────────
// QR rank-1 update / column & row insert / delete (via keel::qr_*)
// ────────────────────────────────────────────────────────────────────────

keel::Tuple<Value, Value>
qrupdate(const Value &Q, const Value &R, const Value &u, const Value &v,
         keel::Allocator *mr)
{
    if (Q.dims().ndim() != 2 || R.dims().ndim() != 2)
        throw Error("qrupdate: Q and R must be 2D matrices",
                    0, 0, "qrupdate", "", "numkit:qrupdate:notMatrix");
    const keel::usize m = keel::as<keel::usize>(Q.dims().dim(0));
    const keel::usize mq = keel::as<keel::usize>(Q.dims().dim(1));
    const keel::usize mr2 = keel::as<keel::usize>(R.dims().dim(0));
    const keel::usize n  = keel::as<keel::usize>(R.dims().dim(1));
    if (mq != m || mr2 != m)
        throw Error("qrupdate: Q must be m×m and R must be m×n",
                    0, 0, "qrupdate", "", "numkit:qrupdate:badShape");
    if (u.numel() != m || v.numel() != n)
        throw Error("qrupdate: length(u) must equal size(R, 1) and length(v) must equal size(R, 2)",
                    0, 0, "qrupdate", "", "numkit:qrupdate:badVec");

    const bool wantSingle = (Q.type() == ValueType::SINGLE || R.type() == ValueType::SINGLE
                             || u.type() == ValueType::SINGLE || v.type() == ValueType::SINGLE);
    const Value Qw = detail::widenSingleToDouble(Q, mr);
    const Value Rw = detail::widenSingleToDouble(R, mr);

    auto Q1 = Value::matrix(m, m, ValueType::DOUBLE, mr);
    auto R1 = Value::matrix(m, n, ValueType::DOUBLE, mr);
    keel::copy(Qw.doubleData(), Qw.doubleData() + m * m, Q1.doubleDataMut());
    keel::copy(Rw.doubleData(), Rw.doubleData() + m * n, R1.doubleDataMut());

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> ub(m, scratch.allocator());
    keel::Array<keel::f64> vb(n, scratch.allocator());
    for (keel::usize i = 0; i < m; ++i) ub[i] = u.elemAsDouble(i);
    for (keel::usize j = 0; j < n; ++j) vb[j] = v.elemAsDouble(j);

    keel::linalg::qr_update(Q1.doubleDataMut(), R1.doubleDataMut(), m, n, ub.data(), vb.data(), mr);
    return keel::make_tuple(
        detail::narrowDoubleToSingleIf(Q1, wantSingle, mr),
        detail::narrowDoubleToSingleIf(R1, wantSingle, mr));
}

keel::Tuple<Value, Value>
qrinsert(const Value &Q, const Value &R, keel::usize k_1based, const Value &x,
         keel::Allocator *mr)
{
    if (Q.dims().ndim() != 2 || R.dims().ndim() != 2)
        throw Error("qrinsert: Q and R must be 2D matrices",
                    0, 0, "qrinsert", "", "numkit:qrinsert:notMatrix");
    const keel::usize m = keel::as<keel::usize>(Q.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(R.dims().dim(1));
    if (keel::as<keel::usize>(Q.dims().dim(1)) != m
        || keel::as<keel::usize>(R.dims().dim(0)) != m)
        throw Error("qrinsert: Q must be m×m and R must be m×n",
                    0, 0, "qrinsert", "", "numkit:qrinsert:badShape");
    if (x.numel() != m)
        throw Error("qrinsert: x must have length size(Q, 1)",
                    0, 0, "qrinsert", "", "numkit:qrinsert:badX");
    if (k_1based < 1 || k_1based > n + 1)
        throw Error("qrinsert: k must be in 1..n+1",
                    0, 0, "qrinsert", "", "numkit:qrinsert:badK");

    const bool wantSingle = (Q.type() == ValueType::SINGLE || R.type() == ValueType::SINGLE
                             || x.type() == ValueType::SINGLE);
    const Value Qw = detail::widenSingleToDouble(Q, mr);
    const Value Rw = detail::widenSingleToDouble(R, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xb(m, scratch.allocator());
    for (keel::usize i = 0; i < m; ++i) xb[i] = x.elemAsDouble(i);

    auto Q1 = Value::matrix(m, m, ValueType::DOUBLE, mr);
    auto R1 = Value::matrix(m, n + 1, ValueType::DOUBLE, mr);
    keel::linalg::qr_insert_col(Qw.doubleData(), Rw.doubleData(), m, n, k_1based - 1,
                        xb.data(), Q1.doubleDataMut(), R1.doubleDataMut(), mr);
    return keel::make_tuple(
        detail::narrowDoubleToSingleIf(Q1, wantSingle, mr),
        detail::narrowDoubleToSingleIf(R1, wantSingle, mr));
}

keel::Tuple<Value, Value>
qrdelete(const Value &Q, const Value &R, keel::usize k_1based,
         keel::Allocator *mr)
{
    if (Q.dims().ndim() != 2 || R.dims().ndim() != 2)
        throw Error("qrdelete: Q and R must be 2D matrices",
                    0, 0, "qrdelete", "", "numkit:qrdelete:notMatrix");
    const keel::usize m = keel::as<keel::usize>(Q.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(R.dims().dim(1));
    if (keel::as<keel::usize>(Q.dims().dim(1)) != m
        || keel::as<keel::usize>(R.dims().dim(0)) != m)
        throw Error("qrdelete: Q must be m×m and R must be m×n",
                    0, 0, "qrdelete", "", "numkit:qrdelete:badShape");
    if (n == 0)
        throw Error("qrdelete: R has no columns to delete",
                    0, 0, "qrdelete", "", "numkit:qrdelete:empty");
    if (k_1based < 1 || k_1based > n)
        throw Error("qrdelete: k must be in 1..n",
                    0, 0, "qrdelete", "", "numkit:qrdelete:badK");

    const bool wantSingle = (Q.type() == ValueType::SINGLE || R.type() == ValueType::SINGLE);
    const Value Qw = detail::widenSingleToDouble(Q, mr);
    const Value Rw = detail::widenSingleToDouble(R, mr);

    auto Q1 = Value::matrix(m, m, ValueType::DOUBLE, mr);
    auto R1 = Value::matrix(m, n - 1, ValueType::DOUBLE, mr);
    keel::linalg::qr_delete_col(Qw.doubleData(), Rw.doubleData(), m, n, k_1based - 1,
                        Q1.doubleDataMut(), R1.doubleDataMut());
    return keel::make_tuple(
        detail::narrowDoubleToSingleIf(Q1, wantSingle, mr),
        detail::narrowDoubleToSingleIf(R1, wantSingle, mr));
}

keel::Tuple<Value, Value>
qrinsert_row(const Value &Q, const Value &R, keel::usize k_1based, const Value &x,
             keel::Allocator *mr)
{
    if (Q.dims().ndim() != 2 || R.dims().ndim() != 2)
        throw Error("qrinsert: Q and R must be 2D matrices",
                    0, 0, "qrinsert", "", "numkit:qrinsert:notMatrix");
    const keel::usize m = keel::as<keel::usize>(Q.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(R.dims().dim(1));
    if (keel::as<keel::usize>(Q.dims().dim(1)) != m
        || keel::as<keel::usize>(R.dims().dim(0)) != m)
        throw Error("qrinsert: Q must be m by m and R must be m by n",
                    0, 0, "qrinsert", "", "numkit:qrinsert:badShape");
    if (x.numel() != n)
        throw Error("qrinsert: x must have length size(R, 2) for 'row'",
                    0, 0, "qrinsert", "", "numkit:qrinsert:badX");
    if (k_1based < 1 || k_1based > m + 1)
        throw Error("qrinsert: j must be in 1..m+1 for 'row'",
                    0, 0, "qrinsert", "", "numkit:qrinsert:badK");

    const bool wantSingle = (Q.type() == ValueType::SINGLE || R.type() == ValueType::SINGLE
                             || x.type() == ValueType::SINGLE);
    const Value Qw = detail::widenSingleToDouble(Q, mr);
    const Value Rw = detail::widenSingleToDouble(R, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xb(n, scratch.allocator());
    for (keel::usize j = 0; j < n; ++j) xb[j] = x.elemAsDouble(j);

    auto Q1 = Value::matrix(m + 1, m + 1, ValueType::DOUBLE, mr);
    auto R1 = Value::matrix(m + 1, n, ValueType::DOUBLE, mr);
    keel::linalg::qr_insert_row(Qw.doubleData(), Rw.doubleData(), m, n, k_1based - 1,
                        xb.data(), Q1.doubleDataMut(), R1.doubleDataMut());
    return keel::make_tuple(
        detail::narrowDoubleToSingleIf(Q1, wantSingle, mr),
        detail::narrowDoubleToSingleIf(R1, wantSingle, mr));
}

keel::Tuple<Value, Value>
qrdelete_row(const Value &Q, const Value &R, keel::usize k_1based,
             keel::Allocator *mr)
{
    if (Q.dims().ndim() != 2 || R.dims().ndim() != 2)
        throw Error("qrdelete: Q and R must be 2D matrices",
                    0, 0, "qrdelete", "", "numkit:qrdelete:notMatrix");
    const keel::usize m = keel::as<keel::usize>(Q.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(R.dims().dim(1));
    if (keel::as<keel::usize>(Q.dims().dim(1)) != m
        || keel::as<keel::usize>(R.dims().dim(0)) != m)
        throw Error("qrdelete: Q must be m by m and R must be m by n",
                    0, 0, "qrdelete", "", "numkit:qrdelete:badShape");
    if (m == 0)
        throw Error("qrdelete: R has no rows to delete",
                    0, 0, "qrdelete", "", "numkit:qrdelete:empty");
    if (k_1based < 1 || k_1based > m)
        throw Error("qrdelete: j must be in 1..m for 'row'",
                    0, 0, "qrdelete", "", "numkit:qrdelete:badK");

    const bool wantSingle = (Q.type() == ValueType::SINGLE || R.type() == ValueType::SINGLE);
    const Value Qw = detail::widenSingleToDouble(Q, mr);
    const Value Rw = detail::widenSingleToDouble(R, mr);

    auto Q1 = Value::matrix(m - 1, m - 1, ValueType::DOUBLE, mr);
    auto R1 = Value::matrix(m - 1, n, ValueType::DOUBLE, mr);
    keel::linalg::qr_delete_row(Qw.doubleData(), Rw.doubleData(), m, n, k_1based - 1,
                        Q1.doubleDataMut(), R1.doubleDataMut(), mr);
    return keel::make_tuple(
        detail::narrowDoubleToSingleIf(Q1, wantSingle, mr),
        detail::narrowDoubleToSingleIf(R1, wantSingle, mr));
}

// ────────────────────────────────────────────────────────────────────────
// cholupdate — rank-1 Cholesky update/downdate (via keel::linalg::cholesky_update)
// ────────────────────────────────────────────────────────────────────────

Value cholupdate(const Value &R, const Value &x, keel::i32 sign,
                 keel::Allocator *mr)
{
    detail::requireDoubleInput(R, "cholupdate", "R", 1);
    detail::requireDoubleInput(x, "cholupdate", "x", 2);
    if (R.dims().ndim() != 2)
        throw Error("cholupdate: R must be a 2D matrix",
                    0, 0, "cholupdate", "", "numkit:cholupdate:notMatrix");
    const keel::usize n = keel::as<keel::usize>(R.dims().dim(0));
    if (n != keel::as<keel::usize>(R.dims().dim(1)))
        throw Error("cholupdate: R must be square",
                    0, 0, "cholupdate", "", "numkit:cholupdate:notSquare");
    if (x.numel() != n)
        throw Error("cholupdate: x must have length equal to size(R, 1)",
                    0, 0, "cholupdate", "", "numkit:cholupdate:badX");
    if (sign != 1 && sign != -1)
        throw Error("cholupdate: sign must be '+' (1) or '-' (-1)",
                    0, 0, "cholupdate", "", "numkit:cholupdate:badSign");
    const bool wantSingle = (R.type() == ValueType::SINGLE || x.type() == ValueType::SINGLE);
    if (n == 0)
        return Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    const Value Rw = detail::widenSingleToDouble(R, mr);
    auto out = Value::matrix(n, n, ValueType::DOUBLE, mr);
    keel::copy(Rw.doubleData(), Rw.doubleData() + n * n, out.doubleDataMut());

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xb(n, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) xb[i] = x.elemAsDouble(i);

    if (!keel::linalg::cholesky_update(out.doubleDataMut(), n, xb.data(), /*downdate=*/(sign == -1), mr)) {
        throw Error("cholupdate: downdate would break positive-definiteness",
                    0, 0, "cholupdate", "", "numkit:cholupdate:downdateFailed");
    }
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

} // namespace numkit::linalg
