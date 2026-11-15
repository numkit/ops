// toolboxes/linalg/src/ldl.cpp
//
// MATLAB ldl: block LDL' factorization. v1 implements the Crout LDL'
// algorithm WITHOUT pivoting -- works for all positive- and negative-
// definite matrices, and for indefinite matrices whose principal
// minors avoid zero. For matrices that strictly need Bunch-Kaufman
// 2x2 pivoting (very rare for small n), throws "needs Bunch-Kaufman
// pivoting" -- see KNOWN GAP at end.
//
//   L * D * L' = A
//
// Migrated 2026-05-25 from toolboxes/builtin/src/language/arrays/ldl.cpp.
//
// KNOWN GAP: complex Hermitian, sparse input, [L,D,P,C] 4-output sparse
// scaling form, 'tol' arg, and Bunch-Kaufman pivoting (P != I) all
// deferred to v2.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/linalg/ldl.hpp>

#include "linalg_detail.hpp"

// Compute-only TU: Value substrate + Error, no engine. The ldl builtin
// (CallContext wrapper) lives in ldl_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace numkit::linalg {

namespace {

// Run Crout LDL' factorization on a square symmetric matrix.
void croutLDL(const double *A_in, keel::usize n,
              double *L_out, double *D_out)
{
    double anorm = 0.0;
    for (keel::usize k = 0; k < n * n; ++k) anorm += A_in[k] * A_in[k];
    anorm = std::sqrt(anorm);
    const double tol = std::max(1e-300,
                                anorm * std::numeric_limits<double>::epsilon()
                                    * static_cast<double>(n));

    std::fill(L_out, L_out + n * n, 0.0);

    for (keel::usize k = 0; k < n; ++k) {
        double s = A_in[k + k * n];
        for (keel::usize i = 0; i < k; ++i)
            s -= L_out[k + i * n] * L_out[k + i * n] * D_out[i];
        D_out[k] = s;

        if (keel::abs(D_out[k]) < tol)
            throw Error("ldl: zero pivot encountered (matrix needs "
                        "Bunch-Kaufman pivoting; not supported in v1)",
                        0, 0, "ldl", "", "numkit:ldl:NeedsBKPivoting");

        L_out[k + k * n] = 1.0;

        const double inv_dk = 1.0 / D_out[k];
        for (keel::usize j = k + 1; j < n; ++j) {
            double t = A_in[j + k * n];
            for (keel::usize i = 0; i < k; ++i)
                t -= L_out[j + i * n] * L_out[k + i * n] * D_out[i];
            L_out[j + k * n] = t * inv_dk;
        }
    }
}

void symmetrise(double *A, keel::usize n)
{
    for (keel::usize j = 0; j < n; ++j)
        for (keel::usize i = 0; i < j; ++i) {
            double avg = 0.5 * (A[i + j * n] + A[j + i * n]);
            A[i + j * n] = avg;
            A[j + i * n] = avg;
        }
}

void croutLDLComplex(const keel::c64 *A_in, keel::usize n,
                     keel::c64 *L_out, double *D_out)
{
    double anorm = 0.0;
    for (keel::usize k = 0; k < n * n; ++k) anorm += keel::norm(A_in[k]);
    anorm = std::sqrt(anorm);
    const double tol = std::max(1e-300,
                                anorm * std::numeric_limits<double>::epsilon()
                                    * static_cast<double>(n));

    std::fill(L_out, L_out + n * n, keel::c64(0.0, 0.0));

    for (keel::usize k = 0; k < n; ++k) {
        double s = A_in[k + k * n].real();
        for (keel::usize i = 0; i < k; ++i)
            s -= keel::norm(L_out[k + i * n]) * D_out[i];
        D_out[k] = s;

        if (keel::abs(D_out[k]) < tol)
            throw Error("ldl: zero pivot encountered (matrix needs "
                        "Bunch-Kaufman pivoting; not supported in v1)",
                        0, 0, "ldl", "", "numkit:ldl:NeedsBKPivoting");

        L_out[k + k * n] = keel::c64(1.0, 0.0);

        const double inv_dk = 1.0 / D_out[k];
        for (keel::usize j = k + 1; j < n; ++j) {
            keel::c64 t = A_in[j + k * n];
            for (keel::usize i = 0; i < k; ++i)
                t -= L_out[j + i * n] * keel::conj(L_out[k + i * n]) * D_out[i];
            L_out[j + k * n] = t * inv_dk;
        }
    }
}

void hermitianise(keel::c64 *A, keel::usize n)
{
    for (keel::usize j = 0; j < n; ++j) {
        if (keel::abs(A[j + j * n].imag()) > 0.0)
            throw Error("ldl: matrix must have real diagonal",
                        0, 0, "ldl", "", "numkit:ldl:ComplexDiag");
        for (keel::usize i = 0; i < j; ++i) {
            keel::c64 avg = 0.5 * (A[i + j * n] + keel::conj(A[j + i * n]));
            A[i + j * n] = avg;
            A[j + i * n] = keel::conj(avg);
        }
    }
}

} // namespace

std::tuple<Value, Value, Value>
ldl(const Value &A, bool upper_form, bool p_as_vector, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "ldl");
    if (A.dims().is3D())
        throw Error("ldl: input must be 2D",
                    0, 0, "ldl", "", "numkit:ldl:Not2D");
    const keel::usize n = A.dims().rows();
    if (A.dims().cols() != n)
        throw Error("ldl: matrix must be square",
                    0, 0, "ldl", "", "numkit:ldl:NotSquare");

    const bool wantSingle = (A.type() == ValueType::SINGLE);
    if (n == 0) {
        Value L0 = Value::matrix(0, 0, A.type(), mr);
        Value D0 = Value::matrix(0, 0, A.type(), mr);
        Value P0 = p_as_vector ? Value::matrix(1, 0, ValueType::DOUBLE, mr)
                               : Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
        return {L0, D0, P0};
    }

    const Value Aw = detail::widenSingleToDouble(A, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (Aw.isComplex()) {
        keel::Array<keel::c64> A_herm(n * n, scratch.allocator());
        std::copy(Aw.complexData(), Aw.complexData() + n * n, A_herm.begin());
        hermitianise(A_herm.data(), n);

        keel::Array<keel::c64> L_buf(n * n, scratch.allocator());
        keel::Array<double> D_buf(n, scratch.allocator());
        croutLDLComplex(A_herm.data(), n, L_buf.data(), D_buf.data());

        Value L = Value::complexMatrix(n, n, mr);
        keel::c64 *Ld = L.complexDataMut();
        if (upper_form) {
            std::fill(Ld, Ld + n * n, keel::c64(0.0, 0.0));
            for (keel::usize j = 0; j < n; ++j)
                for (keel::usize i = 0; i <= j; ++i)
                    Ld[i + j * n] = keel::conj(L_buf[j + i * n]);
        } else {
            std::copy(L_buf.begin(), L_buf.end(), Ld);
        }

        Value D = Value::complexMatrix(n, n, mr);
        keel::c64 *Dd = D.complexDataMut();
        std::fill(Dd, Dd + n * n, keel::c64(0.0, 0.0));
        for (keel::usize k = 0; k < n; ++k) Dd[k + k * n] = keel::c64(D_buf[k], 0.0);

        Value P;
        if (p_as_vector) {
            P = Value::matrix(1, n, ValueType::DOUBLE, mr);
            double *pd = P.doubleDataMut();
            for (keel::usize k = 0; k < n; ++k) pd[k] = static_cast<double>(k + 1);
        } else {
            P = Value::matrix(n, n, ValueType::DOUBLE, mr);
            double *pd = P.doubleDataMut();
            std::fill(pd, pd + n * n, 0.0);
            for (keel::usize k = 0; k < n; ++k) pd[k + k * n] = 1.0;
        }
        return {L, D, P};
    }

    keel::Array<double> A_sym(n * n, scratch.allocator());
    std::copy(Aw.doubleData(), Aw.doubleData() + n * n, A_sym.begin());
    symmetrise(A_sym.data(), n);

    keel::Array<double> L_buf(n * n, scratch.allocator());
    keel::Array<double> D_buf(n, scratch.allocator());
    croutLDL(A_sym.data(), n, L_buf.data(), D_buf.data());

    Value L = Value::matrix(n, n, ValueType::DOUBLE, mr);
    double *Ld = L.doubleDataMut();
    if (upper_form) {
        std::fill(Ld, Ld + n * n, 0.0);
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i <= j; ++i)
                Ld[i + j * n] = L_buf[j + i * n];
    } else {
        std::copy(L_buf.begin(), L_buf.end(), Ld);
    }

    Value D = Value::matrix(n, n, ValueType::DOUBLE, mr);
    double *Dd = D.doubleDataMut();
    std::fill(Dd, Dd + n * n, 0.0);
    for (keel::usize k = 0; k < n; ++k) Dd[k + k * n] = D_buf[k];

    Value P;
    if (p_as_vector) {
        P = Value::matrix(1, n, ValueType::DOUBLE, mr);
        double *pd = P.doubleDataMut();
        for (keel::usize k = 0; k < n; ++k) pd[k] = static_cast<double>(k + 1);
    } else {
        P = Value::matrix(n, n, ValueType::DOUBLE, mr);
        double *pd = P.doubleDataMut();
        std::fill(pd, pd + n * n, 0.0);
        for (keel::usize k = 0; k < n; ++k) pd[k + k * n] = 1.0;
        P = detail::narrowDoubleToSingleIf(P, wantSingle, mr);
    }
    return {detail::narrowDoubleToSingleIf(L, wantSingle, mr),
            detail::narrowDoubleToSingleIf(D, wantSingle, mr),
            P};
}

} // namespace numkit::linalg
