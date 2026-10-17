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

#include <numkit/linalg/ldl.hpp>

#include "linalg_detail.hpp"

// Compute-only TU: Value substrate + Error, no engine. The ldl builtin
// (CallContext wrapper) lives in ldl_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace numkit::linalg {

namespace {

// Run Crout LDL' factorization on a square symmetric matrix.
void croutLDL(const double *A_in, size_t n,
              double *L_out, double *D_out)
{
    double anorm = 0.0;
    for (size_t k = 0; k < n * n; ++k) anorm += A_in[k] * A_in[k];
    anorm = std::sqrt(anorm);
    const double tol = std::max(1e-300,
                                anorm * std::numeric_limits<double>::epsilon()
                                    * static_cast<double>(n));

    std::fill(L_out, L_out + n * n, 0.0);

    for (size_t k = 0; k < n; ++k) {
        double s = A_in[k + k * n];
        for (size_t i = 0; i < k; ++i)
            s -= L_out[k + i * n] * L_out[k + i * n] * D_out[i];
        D_out[k] = s;

        if (std::abs(D_out[k]) < tol)
            throw Error("ldl: zero pivot encountered (matrix needs "
                        "Bunch-Kaufman pivoting; not supported in v1)",
                        0, 0, "ldl", "", "numkit:ldl:NeedsBKPivoting");

        L_out[k + k * n] = 1.0;

        const double inv_dk = 1.0 / D_out[k];
        for (size_t j = k + 1; j < n; ++j) {
            double t = A_in[j + k * n];
            for (size_t i = 0; i < k; ++i)
                t -= L_out[j + i * n] * L_out[k + i * n] * D_out[i];
            L_out[j + k * n] = t * inv_dk;
        }
    }
}

void symmetrise(double *A, size_t n)
{
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < j; ++i) {
            double avg = 0.5 * (A[i + j * n] + A[j + i * n]);
            A[i + j * n] = avg;
            A[j + i * n] = avg;
        }
}

void croutLDLComplex(const Complex *A_in, size_t n,
                     Complex *L_out, double *D_out)
{
    double anorm = 0.0;
    for (size_t k = 0; k < n * n; ++k) anorm += std::norm(A_in[k]);
    anorm = std::sqrt(anorm);
    const double tol = std::max(1e-300,
                                anorm * std::numeric_limits<double>::epsilon()
                                    * static_cast<double>(n));

    std::fill(L_out, L_out + n * n, Complex(0.0, 0.0));

    for (size_t k = 0; k < n; ++k) {
        double s = A_in[k + k * n].real();
        for (size_t i = 0; i < k; ++i)
            s -= std::norm(L_out[k + i * n]) * D_out[i];
        D_out[k] = s;

        if (std::abs(D_out[k]) < tol)
            throw Error("ldl: zero pivot encountered (matrix needs "
                        "Bunch-Kaufman pivoting; not supported in v1)",
                        0, 0, "ldl", "", "numkit:ldl:NeedsBKPivoting");

        L_out[k + k * n] = Complex(1.0, 0.0);

        const double inv_dk = 1.0 / D_out[k];
        for (size_t j = k + 1; j < n; ++j) {
            Complex t = A_in[j + k * n];
            for (size_t i = 0; i < k; ++i)
                t -= L_out[j + i * n] * std::conj(L_out[k + i * n]) * D_out[i];
            L_out[j + k * n] = t * inv_dk;
        }
    }
}

void hermitianise(Complex *A, size_t n)
{
    for (size_t j = 0; j < n; ++j) {
        if (std::abs(A[j + j * n].imag()) > 0.0)
            throw Error("ldl: matrix must have real diagonal",
                        0, 0, "ldl", "", "numkit:ldl:ComplexDiag");
        for (size_t i = 0; i < j; ++i) {
            Complex avg = 0.5 * (A[i + j * n] + std::conj(A[j + i * n]));
            A[i + j * n] = avg;
            A[j + i * n] = std::conj(avg);
        }
    }
}

} // namespace

std::tuple<Value, Value, Value>
ldl(const Value &A, bool upper_form, bool p_as_vector, std::pmr::memory_resource *mr)
{
    detail::requireDoubleInput(A, "ldl");
    if (A.dims().is3D())
        throw Error("ldl: input must be 2D",
                    0, 0, "ldl", "", "numkit:ldl:Not2D");
    const size_t n = A.dims().rows();
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

    ScratchArena scratch(mr);
    if (Aw.isComplex()) {
        ScratchVec<Complex> A_herm(n * n, &scratch);
        std::copy(Aw.complexData(), Aw.complexData() + n * n, A_herm.begin());
        hermitianise(A_herm.data(), n);

        ScratchVec<Complex> L_buf(n * n, &scratch);
        ScratchVec<double> D_buf(n, &scratch);
        croutLDLComplex(A_herm.data(), n, L_buf.data(), D_buf.data());

        Value L = Value::complexMatrix(n, n, mr);
        Complex *Ld = L.complexDataMut();
        if (upper_form) {
            std::fill(Ld, Ld + n * n, Complex(0.0, 0.0));
            for (size_t j = 0; j < n; ++j)
                for (size_t i = 0; i <= j; ++i)
                    Ld[i + j * n] = std::conj(L_buf[j + i * n]);
        } else {
            std::copy(L_buf.begin(), L_buf.end(), Ld);
        }

        Value D = Value::complexMatrix(n, n, mr);
        Complex *Dd = D.complexDataMut();
        std::fill(Dd, Dd + n * n, Complex(0.0, 0.0));
        for (size_t k = 0; k < n; ++k) Dd[k + k * n] = Complex(D_buf[k], 0.0);

        Value P;
        if (p_as_vector) {
            P = Value::matrix(1, n, ValueType::DOUBLE, mr);
            double *pd = P.doubleDataMut();
            for (size_t k = 0; k < n; ++k) pd[k] = static_cast<double>(k + 1);
        } else {
            P = Value::matrix(n, n, ValueType::DOUBLE, mr);
            double *pd = P.doubleDataMut();
            std::fill(pd, pd + n * n, 0.0);
            for (size_t k = 0; k < n; ++k) pd[k + k * n] = 1.0;
        }
        return {L, D, P};
    }

    ScratchVec<double> A_sym(n * n, &scratch);
    std::copy(Aw.doubleData(), Aw.doubleData() + n * n, A_sym.begin());
    symmetrise(A_sym.data(), n);

    ScratchVec<double> L_buf(n * n, &scratch);
    ScratchVec<double> D_buf(n, &scratch);
    croutLDL(A_sym.data(), n, L_buf.data(), D_buf.data());

    Value L = Value::matrix(n, n, ValueType::DOUBLE, mr);
    double *Ld = L.doubleDataMut();
    if (upper_form) {
        std::fill(Ld, Ld + n * n, 0.0);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i <= j; ++i)
                Ld[i + j * n] = L_buf[j + i * n];
    } else {
        std::copy(L_buf.begin(), L_buf.end(), Ld);
    }

    Value D = Value::matrix(n, n, ValueType::DOUBLE, mr);
    double *Dd = D.doubleDataMut();
    std::fill(Dd, Dd + n * n, 0.0);
    for (size_t k = 0; k < n; ++k) Dd[k + k * n] = D_buf[k];

    Value P;
    if (p_as_vector) {
        P = Value::matrix(1, n, ValueType::DOUBLE, mr);
        double *pd = P.doubleDataMut();
        for (size_t k = 0; k < n; ++k) pd[k] = static_cast<double>(k + 1);
    } else {
        P = Value::matrix(n, n, ValueType::DOUBLE, mr);
        double *pd = P.doubleDataMut();
        std::fill(pd, pd + n * n, 0.0);
        for (size_t k = 0; k < n; ++k) pd[k + k * n] = 1.0;
        P = detail::narrowDoubleToSingleIf(P, wantSingle, mr);
    }
    return {detail::narrowDoubleToSingleIf(L, wantSingle, mr),
            detail::narrowDoubleToSingleIf(D, wantSingle, mr),
            P};
}

} // namespace numkit::linalg
