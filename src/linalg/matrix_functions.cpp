// toolboxes/linalg/src/matrix_functions.cpp
//
// Matrix functions â€” expm, logm_sym, sqrtm_sym + engine adapters.
// Migrated 2026-05-25 from toolboxes/builtin/src/language/arrays/matrix.cpp.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/linalg/matrix_functions.hpp>
#include <numkit/linalg/decompositions.hpp>
#include "linalg_detail.hpp"

#include <numkit/linalg/eig.hpp>                  // eig_symmetric
#include <numkit/ops/la_solve.hpp>   // numkit::ops::la_solve
// Compute-only TU: Value substrate + Error, no engine. The expm/logm/sqrtm/
// expmv builtins (CallContext wrappers) live in matrix_functions_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>

namespace numkit::linalg {

namespace {

template <typename T>
void matMul(const T *A, const T *B, T *C, keel::usize n)
{
    std::fill(C, C + n * n, T(0.0));
    for (keel::usize j = 0; j < n; ++j)
        for (keel::usize k = 0; k < n; ++k) {
            const T bkj = B[k + j * n];
            if (bkj == T(0.0)) continue;
            for (keel::usize i = 0; i < n; ++i)
                C[i + j * n] += A[i + k * n] * bkj;
        }
}

// 1-norm of an n×n matrix.
template <typename T>
double mat1Norm(const T *A, keel::usize n)
{
    double mx = 0.0;
    for (keel::usize j = 0; j < n; ++j) {
        double s = 0.0;
        for (keel::usize i = 0; i < n; ++i) s += keel::abs(A[i + j * n]);
        mx = std::max(mx, s);
    }
    return mx;
}

} // anonymous namespace

Value expm(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "expm");
    if (A.dims().ndim() != 2)
        throw Error("expm: input must be a 2D matrix",
                    0, 0, "expm", "", "numkit:expm:notMatrix");
    const keel::usize m = static_cast<keel::usize>(A.dims().dim(0));
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(1));
    if (m != n)
        throw Error("expm: matrix must be square",
                    0, 0, "expm", "", "numkit:expm:notSquare");
    if (n == 0) return Value::matrix(0, 0, A.type(), mr);

    const bool wantSingle = A.type() == ValueType::SINGLE;
    Value Aw = detail::widenSingleToDouble(A, mr);

    // Padé(6) coefficients (Higham table 10.4).
    static constexpr double c[7] = {
        1.0, 1.0/2.0, 5.0/44.0, 1.0/66.0, 1.0/792.0, 1.0/15840.0, 1.0/665280.0
    };

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (Aw.isComplex()) {
        keel::Array<keel::c64> A_s(n * n, scratch.allocator());
        std::copy(Aw.complexData(), Aw.complexData() + n * n, A_s.begin());

        const double a_norm = mat1Norm(A_s.data(), n);
        int s = 0;
        if (a_norm > 0.5) {
            s = static_cast<int>(std::ceil(std::log2(a_norm / 0.5)));
            if (s < 0) s = 0;
            const double scale = 1.0 / std::pow(2.0, s);
            for (keel::usize i = 0; i < n * n; ++i) A_s[i] *= scale;
        }

        keel::Array<keel::c64> A2(n * n, scratch.allocator());
        keel::Array<keel::c64> A4(n * n, scratch.allocator());
        keel::Array<keel::c64> A6(n * n, scratch.allocator());
        matMul(A_s.data(), A_s.data(), A2.data(), n);
        matMul(A2.data(), A2.data(), A4.data(), n);
        matMul(A2.data(), A4.data(), A6.data(), n);

        keel::Array<keel::c64> P(n * n, keel::c64(0.0, 0.0), scratch.allocator());
        keel::Array<keel::c64> Q(n * n, keel::c64(0.0, 0.0), scratch.allocator());
        for (keel::usize i = 0; i < n; ++i) {
            P[i + i * n] = keel::c64(c[0], 0.0);
            Q[i + i * n] = keel::c64(c[0], 0.0);
        }
        for (keel::usize i = 0; i < n * n; ++i) {
            P[i] += c[1] * A_s[i];
            Q[i] -= c[1] * A_s[i];
        }
        for (keel::usize i = 0; i < n * n; ++i) {
            P[i] += c[2] * A2[i];
            Q[i] += c[2] * A2[i];
        }
        keel::Array<keel::c64> A3(n * n, scratch.allocator());
        matMul(A_s.data(), A2.data(), A3.data(), n);
        for (keel::usize i = 0; i < n * n; ++i) {
            P[i] += c[3] * A3[i];
            Q[i] -= c[3] * A3[i];
        }
        for (keel::usize i = 0; i < n * n; ++i) {
            P[i] += c[4] * A4[i];
            Q[i] += c[4] * A4[i];
        }
        keel::Array<keel::c64> A5(n * n, scratch.allocator());
        matMul(A_s.data(), A4.data(), A5.data(), n);
        for (keel::usize i = 0; i < n * n; ++i) {
            P[i] += c[5] * A5[i];
            Q[i] -= c[5] * A5[i];
        }
        for (keel::usize i = 0; i < n * n; ++i) {
            P[i] += c[6] * A6[i];
            Q[i] += c[6] * A6[i];
        }

        auto out = Value::complexMatrix(n, n, mr);
        if (!numkit::ops::la_solve(Q.data(), n, n, P.data(), n,
                                   out.complexDataMut(), scratch.allocator()))
            throw Error("expm: Padé denominator is singular",
                        0, 0, "expm", "", "numkit:expm:singular");

        if (s > 0) {
            keel::Array<keel::c64> tmp(n * n, scratch.allocator());
            keel::c64 *X = out.complexDataMut();
            for (int k = 0; k < s; ++k) {
                matMul(X, X, tmp.data(), n);
                std::copy(tmp.begin(), tmp.end(), X);
            }
        }
        return detail::narrow_if_real(out, mr);
    }

    // Padé(6) scaling-and-squaring (real).
    keel::Array<double> A_s(n * n, scratch.allocator());
    std::copy(Aw.doubleData(), Aw.doubleData() + n * n, A_s.begin());

    const double a_norm = mat1Norm(A_s.data(), n);
    int s = 0;
    if (a_norm > 0.5) {
        s = static_cast<int>(std::ceil(std::log2(a_norm / 0.5)));
        if (s < 0) s = 0;
        const double scale = 1.0 / std::pow(2.0, s);
        for (keel::usize i = 0; i < n * n; ++i) A_s[i] *= scale;
    }

    // Powers A^2, A^4, A^6.
    keel::Array<double> A2(n * n, scratch.allocator());
    keel::Array<double> A4(n * n, scratch.allocator());
    keel::Array<double> A6(n * n, scratch.allocator());
    matMul(A_s.data(), A_s.data(), A2.data(), n);
    matMul(A2.data(), A2.data(), A4.data(), n);
    matMul(A2.data(), A4.data(), A6.data(), n);

    keel::Array<double> P(n * n, scratch.allocator());
    keel::Array<double> Q(n * n, scratch.allocator());
    std::fill(P.begin(), P.end(), 0.0);
    std::fill(Q.begin(), Q.end(), 0.0);
    for (keel::usize i = 0; i < n; ++i) {
        P[i + i * n] = c[0];
        Q[i + i * n] = c[0];
    }
    for (keel::usize i = 0; i < n * n; ++i) {
        P[i] += c[1] * A_s[i];
        Q[i] -= c[1] * A_s[i];
    }
    for (keel::usize i = 0; i < n * n; ++i) {
        P[i] += c[2] * A2[i];
        Q[i] += c[2] * A2[i];
    }
    keel::Array<double> A3(n * n, scratch.allocator());
    matMul(A_s.data(), A2.data(), A3.data(), n);
    for (keel::usize i = 0; i < n * n; ++i) {
        P[i] += c[3] * A3[i];
        Q[i] -= c[3] * A3[i];
    }
    for (keel::usize i = 0; i < n * n; ++i) {
        P[i] += c[4] * A4[i];
        Q[i] += c[4] * A4[i];
    }
    keel::Array<double> A5(n * n, scratch.allocator());
    matMul(A_s.data(), A4.data(), A5.data(), n);
    for (keel::usize i = 0; i < n * n; ++i) {
        P[i] += c[5] * A5[i];
        Q[i] -= c[5] * A5[i];
    }
    for (keel::usize i = 0; i < n * n; ++i) {
        P[i] += c[6] * A6[i];
        Q[i] += c[6] * A6[i];
    }

    // Solve Q * X = P for X.
    auto out = Value::matrix(n, n, ValueType::DOUBLE, mr);
    if (!numkit::ops::la_solve(Q.data(), n, n, P.data(), n,
                                            out.doubleDataMut(), scratch.allocator()))
        throw Error("expm: Padé denominator is singular",
                    0, 0, "expm", "", "numkit:expm:singular");

    if (s > 0) {
        keel::Array<double> tmp(n * n, scratch.allocator());
        double *X = out.doubleDataMut();
        for (int k = 0; k < s; ++k) {
            matMul(X, X, tmp.data(), n);
            std::copy(tmp.begin(), tmp.end(), X);
        }
    }
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

namespace {

// Apply scalar function f to symmetric A's eigenvalues and reconstruct:
//   result = V * diag(f(eig)) * V'
Value applyScalarFnSym(const Value &A, double (*f)(double),
                       const char *fnName, const char *errId,
                       keel::Allocator *mr)
{
    const bool wantSingle = A.type() == ValueType::SINGLE;
    Value Aw = detail::widenSingleToDouble(A, mr);
    auto [V, D] = eig_symmetric(Aw, mr);
    const keel::usize n = static_cast<keel::usize>(D.dims().dim(0));
    if (n == 0) return Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    const double *Vdata = V.doubleData();
    const double *Ddata = D.doubleData();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> fD(n, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) {
        const double e = Ddata[i + i * n];
        const double fe = f(e);
        if (!std::isfinite(fe))
            throw Error(keel::format("{}: eigenvalue out of domain (got {})", fnName, e),
                        0, 0, fnName, "", errId);
        fD[i] = fe;
    }

    auto out = Value::matrix(n, n, ValueType::DOUBLE, mr);
    double *R = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i)
        for (keel::usize j = 0; j < n; ++j) {
            double s = 0.0;
            for (keel::usize k = 0; k < n; ++k)
                s += Vdata[i + k * n] * fD[k] * Vdata[j + k * n];
            R[i + j * n] = s;
        }
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

} // anonymous namespace

Value logm_sym(const Value &A, keel::Allocator *mr)
{
    return applyScalarFnSym(A, [](double x) { return std::log(x); },
                            "logm", "numkit:logm:negativeEigenvalue", mr);
}

Value sqrtm_sym(const Value &A, keel::Allocator *mr)
{
    return applyScalarFnSym(A, [](double x) { return std::sqrt(x); },
                            "sqrtm", "numkit:sqrtm:negativeEigenvalue", mr);
}

Value sqrtm(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "sqrtm");
    if (A.dims().ndim() != 2)
        throw Error("sqrtm: input must be a 2D matrix", 0, 0, "sqrtm", "", "numkit:sqrtm:notMatrix");
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(0));
    if (n != static_cast<keel::usize>(A.dims().dim(1)))
        throw Error("sqrtm: matrix must be square", 0, 0, "sqrtm", "", "numkit:sqrtm:notSquare");
    const bool wantSingle = A.type() == ValueType::SINGLE;
    if (n == 0) return Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    Value Aw = detail::widenSingleToDouble(A, mr);

    // Schur decomposition A = U * T * U^H
    auto [U, T] = schur_general(Aw, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> Tc(n * n, scratch.allocator());
    keel::Array<keel::c64> Uc(n * n, scratch.allocator());
    keel::Array<keel::c64> Rc(n * n, keel::c64(0.0, 0.0), scratch.allocator());

    if (T.isComplex()) {
        std::copy(T.complexData(), T.complexData() + n * n, Tc.begin());
    } else {
        const double *td = T.doubleData();
        for (keel::usize i = 0; i < n * n; ++i) Tc[i] = keel::c64(td[i], 0.0);
    }
    if (U.isComplex()) {
        std::copy(U.complexData(), U.complexData() + n * n, Uc.begin());
    } else {
        const double *ud = U.doubleData();
        for (keel::usize i = 0; i < n * n; ++i) Uc[i] = keel::c64(ud[i], 0.0);
    }

    for (keel::usize i = 0; i < n; ++i) {
        Rc[i + i * n] = keel::sqrt(Tc[i + i * n]);
    }

    // Off-diagonal entries:
    for (keel::usize j = 1; j < n; ++j) {
        for (std::intptr_t i = static_cast<std::intptr_t>(j) - 1; i >= 0; --i) {
            keel::usize ui = static_cast<keel::usize>(i);
            keel::c64 sum = Tc[ui + j * n];
            for (keel::usize k = ui + 1; k < j; ++k) {
                sum -= Rc[ui + k * n] * Rc[k + j * n];
            }
            keel::c64 denom = Rc[ui + ui * n] + Rc[j + j * n];
            if (keel::abs(denom) < 1e-14) {
                denom = keel::c64(1e-14, 0.0);
            }
            Rc[ui + j * n] = sum / denom;
        }
    }

    // Reconstruct X = U * R * U^H
    auto out = Value::complexMatrix(n, n, mr);
    keel::c64 *Xd = out.complexDataMut();
    std::fill(Xd, Xd + n * n, keel::c64(0.0, 0.0));

    keel::Array<keel::c64> RUh(n * n, keel::c64(0.0, 0.0), scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) {
        for (keel::usize j = 0; j < n; ++j) {
            keel::c64 s(0.0, 0.0);
            for (keel::usize k = i; k < n; ++k) { // R is upper triangular
                s += Rc[i + k * n] * keel::conj(Uc[j + k * n]);
            }
            RUh[i + j * n] = s;
        }
    }

    for (keel::usize i = 0; i < n; ++i) {
        for (keel::usize j = 0; j < n; ++j) {
            keel::c64 s(0.0, 0.0);
            for (keel::usize k = 0; k < n; ++k) {
                s += Uc[i + k * n] * RUh[k + j * n];
            }
            Xd[i + j * n] = s;
        }
    }

    return detail::narrowDoubleToSingleIf(detail::narrow_if_real(out, mr), wantSingle, mr);
}

Value sylvester(const Value &A, const Value &B, const Value &C, keel::Allocator *mr)
{
    for (const Value *v : {&A, &B, &C}) {
        const ValueType vt = v->type();
        if (vt != ValueType::DOUBLE && vt != ValueType::COMPLEX
            && vt != ValueType::SINGLE)
            throw Error(keel::format("sylvester: inputs must be single or double (got a {} array).",
                                     mtypeName(vt)),
                        0, 0, "sylvester", "", "numkit:sylvester:inputType");
    }
    const bool wantSingle = A.type() == ValueType::SINGLE
        || B.type() == ValueType::SINGLE || C.type() == ValueType::SINGLE;
    for (const Value *v : {&A, &B, &C}) {
        const bool cx = v->isComplex();
        const keel::usize n = v->numel();
        for (keel::usize i = 0; i < n; ++i) {
            const double re = cx ? v->complexElem(i).real()
                                 : v->elemAsDouble(i);
            const double im = cx ? v->complexElem(i).imag() : 0.0;
            if (!std::isfinite(re) || !std::isfinite(im))
                throw Error("sylvester: inputs must be finite (NaN or Inf "
                            "found).",
                            0, 0, "sylvester", "",
                            "numkit:sylvester:inputWithNaNInf");
        }
    }
    if (A.dims().ndim() != 2 || B.dims().ndim() != 2 || C.dims().ndim() != 2)
        throw Error("sylvester: inputs must be 2D matrices", 0, 0, "sylvester", "", "numkit:sylvester:notMatrix");
    const keel::usize m = static_cast<keel::usize>(A.dims().dim(0));
    if (m != static_cast<keel::usize>(A.dims().dim(1)))
        throw Error(keel::format("sylvester: first and second inputs must be square (A is {}x{}).",
                                 A.dims().dim(0), A.dims().dim(1)),
                    0, 0, "sylvester", "", "numkit:sylvester:inputMustBeSquare");
    const keel::usize n = static_cast<keel::usize>(B.dims().dim(0));
    if (n != static_cast<keel::usize>(B.dims().dim(1)))
        throw Error(keel::format("sylvester: first and second inputs must be square (B is {}x{}).",
                                 B.dims().dim(0), B.dims().dim(1)),
                    0, 0, "sylvester", "", "numkit:sylvester:inputMustBeSquare");
    if (static_cast<keel::usize>(C.dims().dim(0)) != m || static_cast<keel::usize>(C.dims().dim(1)) != n)
        throw Error(keel::format("sylvester: inputs must have compatible size (C must be {}x{} to match A and B).",
                                 m, n),
                    0, 0, "sylvester", "", "numkit:sylvester:inputMustBeCompatibleSize");
    if (m == 0 || n == 0)
        throw Error("sylvester: inputs must have compatible size (empty "
                    "operand).",
                    0, 0, "sylvester", "",
                    "numkit:sylvester:inputMustBeCompatibleSize");

    auto widen = [&](const Value &v) -> Value {
        if (v.type() != ValueType::SINGLE)
            return v;
        Value d = Value::matrix(v.dims().rows(), v.dims().cols(),
                                ValueType::DOUBLE, mr);
        const float *sv = v.singleData();
        double *dv = d.doubleDataMut();
        for (keel::usize i = 0; i < v.numel(); ++i)
            dv[i] = static_cast<double>(sv[i]);
        return d;
    };
    const Value Aw = widen(A), Bw = widen(B), Cw = widen(C);

    // Schur decompositions: A = U_A * T_A * U_A^H, B = U_B * T_B * U_B^H
    auto [U_A, T_A] = schur_general(Aw, mr);
    auto [U_B, T_B] = schur_general(Bw, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> Ta(m * m, scratch.allocator());
    keel::Array<keel::c64> Ua(m * m, scratch.allocator());
    keel::Array<keel::c64> Tb(n * n, scratch.allocator());
    keel::Array<keel::c64> Ub(n * n, scratch.allocator());
    keel::Array<keel::c64> Cc(m * n, scratch.allocator());

    auto toComplex = [](const Value &V, keel::usize rows, keel::usize cols, keel::c64 *out) {
        if (V.isComplex()) {
            std::copy(V.complexData(), V.complexData() + rows * cols, out);
        } else {
            const double *vd = V.doubleData();
            for (keel::usize i = 0; i < rows * cols; ++i) out[i] = keel::c64(vd[i], 0.0);
        }
    };

    toComplex(T_A, m, m, Ta.data());
    toComplex(U_A, m, m, Ua.data());
    toComplex(T_B, n, n, Tb.data());
    toComplex(U_B, n, n, Ub.data());
    toComplex(Cw, m, n, Cc.data());

    // C_tilde = U_A^H * C * U_B
    keel::Array<keel::c64> UaH_C(m * n, keel::c64(0.0, 0.0), scratch.allocator());
    for (keel::usize i = 0; i < m; ++i) {
        for (keel::usize j = 0; j < n; ++j) {
            keel::c64 s(0.0, 0.0);
            for (keel::usize k = 0; k < m; ++k) {
                s += keel::conj(Ua[k + i * m]) * Cc[k + j * m];
            }
            UaH_C[i + j * m] = s;
        }
    }
    keel::Array<keel::c64> C_tilde(m * n, keel::c64(0.0, 0.0), scratch.allocator());
    for (keel::usize i = 0; i < m; ++i) {
        for (keel::usize j = 0; j < n; ++j) {
            keel::c64 s(0.0, 0.0);
            for (keel::usize k = 0; k < n; ++k) {
                s += UaH_C[i + k * m] * Ub[k + j * n];
            }
            C_tilde[i + j * m] = s;
        }
    }

    keel::Array<keel::c64> Y(m * n, keel::c64(0.0, 0.0), scratch.allocator());
    for (keel::usize j = 0; j < n; ++j) {
        for (std::intptr_t i = static_cast<std::intptr_t>(m) - 1; i >= 0; --i) {
            keel::usize ui = static_cast<keel::usize>(i);
            keel::c64 s = C_tilde[ui + j * m];
            for (keel::usize k = ui + 1; k < m; ++k) {
                s -= Ta[ui + k * m] * Y[k + j * m];
            }
            for (keel::usize k = 0; k < j; ++k) {
                s -= Y[ui + k * m] * Tb[k + j * n];
            }
            const keel::c64 denom = Ta[ui + ui * m] + Tb[j + j * n];
            Y[ui + j * m] = s / denom;
        }
    }

    // Reconstruct X = U_A * Y * U_B^H
    keel::Array<keel::c64> Ua_Y(m * n, keel::c64(0.0, 0.0), scratch.allocator());
    for (keel::usize i = 0; i < m; ++i) {
        for (keel::usize j = 0; j < n; ++j) {
            keel::c64 s(0.0, 0.0);
            for (keel::usize k = 0; k < m; ++k) {
                s += Ua[i + k * m] * Y[k + j * m];
            }
            Ua_Y[i + j * m] = s;
        }
    }

    auto out = Value::complexMatrix(m, n, mr);
    keel::c64 *Xd = out.complexDataMut();
    for (keel::usize i = 0; i < m; ++i) {
        for (keel::usize j = 0; j < n; ++j) {
            keel::c64 s(0.0, 0.0);
            for (keel::usize k = 0; k < n; ++k) {
                s += Ua_Y[i + k * m] * keel::conj(Ub[j + k * n]);
            }
            Xd[i + j * m] = s;
        }
    }

    Value res = detail::narrow_if_real(std::move(out), mr);
    if (!A.isComplex() && !B.isComplex() && !C.isComplex()
        && res.isComplex()) {
        Value r = Value::matrix(m, n, ValueType::DOUBLE, mr);
        const keel::c64 *cd = res.complexData();
        double *rd = r.doubleDataMut();
        for (keel::usize i = 0; i < m * n; ++i)
            rd[i] = cd[i].real();
        res = std::move(r);
    }
    if (wantSingle && !res.isComplex()) {
        Value s = Value::matrix(res.dims().rows(), res.dims().cols(),
                                ValueType::SINGLE, mr);
        const double *rd = res.doubleData();
        float *sd = s.singleDataMut();
        for (keel::usize i = 0; i < res.numel(); ++i)
            sd[i] = static_cast<float>(rd[i]);
        return s;
    }
    return res;
}

Value logm(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "logm");
    if (A.dims().ndim() != 2)
        throw Error("logm: input must be a 2D matrix", 0, 0, "logm", "", "numkit:logm:notMatrix");
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(0));
    if (n != static_cast<keel::usize>(A.dims().dim(1)))
        throw Error("logm: matrix must be square", 0, 0, "logm", "", "numkit:logm:notSquare");
    const bool wantSingle = A.type() == ValueType::SINGLE;
    if (n == 0) return Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    Value Aw = detail::widenSingleToDouble(A, mr);

    // Schur decomposition A = U * T * U^H
    auto [U, T] = schur_general(Aw, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> Tc(n * n, scratch.allocator());
    keel::Array<keel::c64> Uc(n * n, scratch.allocator());

    if (T.isComplex()) {
        std::copy(T.complexData(), T.complexData() + n * n, Tc.begin());
    } else {
        const double *td = T.doubleData();
        for (keel::usize i = 0; i < n * n; ++i) Tc[i] = keel::c64(td[i], 0.0);
    }
    if (U.isComplex()) {
        std::copy(U.complexData(), U.complexData() + n * n, Uc.begin());
    } else {
        const double *ud = U.doubleData();
        for (keel::usize i = 0; i < n * n; ++i) Uc[i] = keel::c64(ud[i], 0.0);
    }

    int s_count = 0;
    auto matDiffNorm = [](const keel::c64 *M, keel::usize n) -> double {
        double max_col = 0.0;
        for (keel::usize j = 0; j < n; ++j) {
            double col_sum = 0.0;
            for (keel::usize i = 0; i < n; ++i) {
                keel::c64 diff = (i == j) ? (M[i + j * n] - keel::c64(1.0, 0.0)) : M[i + j * n];
                col_sum += keel::abs(diff);
            }
            max_col = std::max(max_col, col_sum);
        }
        return max_col;
    };

    while (s_count < 32 && matDiffNorm(Tc.data(), n) > 0.5) {
        keel::Array<keel::c64> Tc_next(n * n, keel::c64(0.0, 0.0), scratch.allocator());
        for (keel::usize i = 0; i < n; ++i) Tc_next[i + i * n] = keel::sqrt(Tc[i + i * n]);
        for (keel::usize j = 1; j < n; ++j) {
            for (std::intptr_t i = static_cast<std::intptr_t>(j) - 1; i >= 0; --i) {
                keel::usize ui = static_cast<keel::usize>(i);
                keel::c64 sum = Tc[ui + j * n];
                for (keel::usize k = ui + 1; k < j; ++k) {
                    sum -= Tc_next[ui + k * n] * Tc_next[k + j * n];
                }
                keel::c64 denom = Tc_next[ui + ui * n] + Tc_next[j + j * n];
                if (keel::abs(denom) < 1e-14) denom = keel::c64(1e-14, 0.0);
                Tc_next[ui + j * n] = sum / denom;
            }
        }
        std::copy(Tc_next.begin(), Tc_next.end(), Tc.begin());
        s_count++;
    }

    keel::Array<keel::c64> E(n * n, keel::c64(0.0, 0.0), scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) {
        for (keel::usize j = 0; j < n; ++j) {
            E[i + j * n] = (i == j) ? (Tc[i + j * n] - keel::c64(1.0, 0.0)) : Tc[i + j * n];
        }
    }

    keel::Array<keel::c64> logT(n * n, keel::c64(0.0, 0.0), scratch.allocator());
    keel::Array<keel::c64> Ek(n * n, keel::c64(0.0, 0.0), scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) Ek[i + i * n] = keel::c64(1.0, 0.0); // E^0

    for (int k = 1; k <= 24; ++k) {
        keel::Array<keel::c64> Ek_next(n * n, keel::c64(0.0, 0.0), scratch.allocator());
        for (keel::usize i = 0; i < n; ++i) {
            for (keel::usize j = i; j < n; ++j) {
                keel::c64 sum(0.0, 0.0);
                for (keel::usize l = i; l <= j; ++l) {
                    sum += Ek[i + l * n] * E[l + j * n];
                }
                Ek_next[i + j * n] = sum;
            }
        }
        std::copy(Ek_next.begin(), Ek_next.end(), Ek.begin());

        double coef = ((k % 2 == 1) ? 1.0 : -1.0) / static_cast<double>(k);
        for (keel::usize i = 0; i < n * n; ++i) {
            logT[i] += coef * Ek[i];
        }
    }

    double scale = std::pow(2.0, s_count);
    for (keel::usize i = 0; i < n * n; ++i) logT[i] *= scale;

    keel::Array<keel::c64> logT_Uh(n * n, keel::c64(0.0, 0.0), scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) {
        for (keel::usize j = 0; j < n; ++j) {
            keel::c64 s(0.0, 0.0);
            for (keel::usize k = i; k < n; ++k) {
                s += logT[i + k * n] * keel::conj(Uc[j + k * n]);
            }
            logT_Uh[i + j * n] = s;
        }
    }

    auto out = Value::complexMatrix(n, n, mr);
    keel::c64 *Xd = out.complexDataMut();
    for (keel::usize i = 0; i < n; ++i) {
        for (keel::usize j = 0; j < n; ++j) {
            keel::c64 s(0.0, 0.0);
            for (keel::usize k = 0; k < n; ++k) {
                s += Uc[i + k * n] * logT_Uh[k + j * n];
            }
            Xd[i + j * n] = s;
        }
    }

    return detail::narrowDoubleToSingleIf(detail::narrow_if_real(out, mr), wantSingle, mr);
}

Value expmv(double t, const Value &A, const Value &v, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "expmv", "A", 1);
    {
        const ValueType vt = v.type();
        if (vt != ValueType::DOUBLE && vt != ValueType::COMPLEX
            && vt != ValueType::SINGLE && vt != ValueType::LOGICAL
            && vt != ValueType::CHAR && !isIntegerType(vt))
            throw Error(keel::format("expmv: input 3 (v) must be a numeric, logical or"
                                     " char vector (got a {} {} array).",
                                     detail::shapeOf(v), mtypeName(vt)),
                        0, 0, "expmv", "", "numkit:expmv:inputType");
    }
    if (A.dims().ndim() != 2)
        throw Error("expmv: A must be a 2D matrix",
                    0, 0, "expmv", "", "numkit:expmv:notMatrix");
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(0));
    if (static_cast<keel::usize>(A.dims().dim(1)) != n)
        throw Error("expmv: A must be square",
                    0, 0, "expmv", "", "numkit:expmv:notSquare");
    if (v.numel() != n)
        throw Error("expmv: length(v) must equal size(A, 1)",
                    0, 0, "expmv", "", "numkit:expmv:badV");
    const bool wantSingle = A.type() == ValueType::SINGLE || v.type() == ValueType::SINGLE;
    if (n == 0)
        return Value::matrix(0, 1, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    Value Aw = detail::widenSingleToDouble(A, mr);

    double beta = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const double vi = v.elemAsDouble(i);
        beta += vi * vi;
    }
    beta = std::sqrt(beta);
    if (beta == 0.0) {
        auto z = Value::matrix(n, 1, ValueType::DOUBLE, mr);
        std::fill(z.doubleDataMut(), z.doubleDataMut() + n, 0.0);
        return detail::narrowDoubleToSingleIf(z, wantSingle, mr);
    }

    const keel::usize m = std::min<keel::usize>(30, n);   // Krylov dim

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> V(n * (m + 1), 0.0, scratch.allocator());     // Arnoldi basis
    keel::Array<double> H(m * m, 0.0, scratch.allocator());           // square Hessenberg
    keel::Array<double> work(n, scratch.allocator());

    // V[:, 0] = v / beta
    for (keel::usize i = 0; i < n; ++i) V[i + 0 * n] = v.elemAsDouble(i) / beta;

    const double *Ad = Aw.doubleData();
    keel::usize m_eff = m;    // may shrink on lucky breakdown

    for (keel::usize j = 0; j < m; ++j) {
        for (keel::usize i = 0; i < n; ++i) {
            double s = 0.0;
            for (keel::usize k = 0; k < n; ++k) s += Ad[i + k * n] * V[k + j * n];
            work[i] = s;
        }
        // Modified Gram-Schmidt: orthogonalise against V[:, 0..j].
        for (keel::usize i = 0; i <= j; ++i) {
            double hij = 0.0;
            for (keel::usize k = 0; k < n; ++k) hij += V[k + i * n] * work[k];
            if (i < m && j < m) H[i + j * m] = hij;
            for (keel::usize k = 0; k < n; ++k) work[k] -= hij * V[k + i * n];
        }
        // Sub-diagonal entry
        double hjp1 = 0.0;
        for (keel::usize k = 0; k < n; ++k) hjp1 += work[k] * work[k];
        hjp1 = std::sqrt(hjp1);
        if (hjp1 < 1e-14) {
            m_eff = j + 1;
            break;
        }
        if (j + 1 < m) H[(j + 1) + j * m] = hjp1;
        for (keel::usize k = 0; k < n; ++k) V[k + (j + 1) * n] = work[k] / hjp1;
    }

    Value Hsq = Value::matrix(m_eff, m_eff, ValueType::DOUBLE, mr);
    double *Hsd = Hsq.doubleDataMut();
    for (keel::usize j = 0; j < m_eff; ++j)
        for (keel::usize i = 0; i < m_eff; ++i)
            Hsd[i + j * m_eff] = (i < m && j < m) ? H[i + j * m] : 0.0;

    for (keel::usize k = 0; k < m_eff * m_eff; ++k) Hsd[k] *= t;

    auto F = expm(Hsq, mr);
    const double *Fd = F.doubleData();

    auto out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    double *wd = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) {
        double s = 0.0;
        for (keel::usize k = 0; k < m_eff; ++k) s += V[i + k * n] * Fd[k + 0 * m_eff];
        wd[i] = beta * s;
    }
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

static Value funm_eval(const Value &A, keel::FunctionRef<keel::c64(keel::c64)> f, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "funm");
    if (A.dims().ndim() != 2)
        throw Error("funm: input must be a 2D matrix", 0, 0, "funm", "", "numkit:funm:notMatrix");
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(0));
    if (n != static_cast<keel::usize>(A.dims().dim(1)))
        throw Error("funm: matrix must be square", 0, 0, "funm", "", "numkit:funm:notSquare");
    const bool wantSingle = A.type() == ValueType::SINGLE;
    if (n == 0) return Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    Value Aw = detail::widenSingleToDouble(A, mr);

    // Schur decomposition A = U * T * U^H
    auto [U, T] = schur_general(Aw, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> Tc(n * n, scratch.allocator());
    keel::Array<keel::c64> Uc(n * n, scratch.allocator());
    keel::Array<keel::c64> Fc(n * n, keel::c64(0.0, 0.0), scratch.allocator());

    if (T.isComplex()) {
        std::copy(T.complexData(), T.complexData() + n * n, Tc.begin());
    } else {
        const double *td = T.doubleData();
        for (keel::usize i = 0; i < n * n; ++i) Tc[i] = keel::c64(td[i], 0.0);
    }
    if (U.isComplex()) {
        std::copy(U.complexData(), U.complexData() + n * n, Uc.begin());
    } else {
        const double *ud = U.doubleData();
        for (keel::usize i = 0; i < n * n; ++i) Uc[i] = keel::c64(ud[i], 0.0);
    }

    // Parlett recurrence on upper-triangular T:
    // Diagonal entries F_{i,i} = f(T_{i,i})
    for (keel::usize i = 0; i < n; ++i) Fc[i + i * n] = f(Tc[i + i * n]);

    // Off-diagonal entries:
    for (keel::usize j = 1; j < n; ++j) {
        for (std::intptr_t i = static_cast<std::intptr_t>(j) - 1; i >= 0; --i) {
            keel::usize ui = static_cast<keel::usize>(i);
            keel::c64 denom = Tc[ui + ui * n] - Tc[j + j * n];
            if (keel::abs(denom) > 1e-12) {
                keel::c64 sum = Fc[ui + ui * n] * Tc[ui + j * n] - Tc[ui + j * n] * Fc[j + j * n];
                for (keel::usize k = ui + 1; k < j; ++k) {
                    sum += Fc[ui + k * n] * Tc[k + j * n] - Tc[ui + k * n] * Fc[k + j * n];
                }
                Fc[ui + j * n] = sum / denom;
            } else {
                // Confluent / close eigenvalues: numerical derivative via step h = 1e-6
                keel::c64 lambda = 0.5 * (Tc[ui + ui * n] + Tc[j + j * n]);
                keel::c64 h(1e-6, 0.0);
                keel::c64 df = (f(lambda + h) - f(lambda - h)) / (2.0 * h);
                keel::c64 sum = df * Tc[ui + j * n];
                for (keel::usize k = ui + 1; k < j; ++k) {
                    sum += Fc[ui + k * n] * Tc[k + j * n];
                }
                Fc[ui + j * n] = sum;
            }
        }
    }

    // Reconstruct f(A) = U * F * U^H
    keel::Array<keel::c64> F_Uh(n * n, keel::c64(0.0, 0.0), scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) {
        for (keel::usize j = 0; j < n; ++j) {
            keel::c64 s(0.0, 0.0);
            for (keel::usize k = i; k < n; ++k) {
                s += Fc[i + k * n] * keel::conj(Uc[j + k * n]);
            }
            F_Uh[i + j * n] = s;
        }
    }

    auto out = Value::complexMatrix(n, n, mr);
    keel::c64 *Xd = out.complexDataMut();
    for (keel::usize i = 0; i < n; ++i) {
        for (keel::usize j = 0; j < n; ++j) {
            keel::c64 s(0.0, 0.0);
            for (keel::usize k = 0; k < n; ++k) {
                s += Uc[i + k * n] * F_Uh[k + j * n];
            }
            Xd[i + j * n] = s;
        }
    }

    return detail::narrowDoubleToSingleIf(detail::narrow_if_real(out, mr), wantSingle, mr);
}

Value funm(const Value &A, keel::FunctionRef<keel::c64(keel::c64)> f, keel::Allocator *mr)
{
    return funm_eval(A, f, mr);
}

Value funm(const Value &A, keel::StringRef fnName, keel::Allocator *mr)
{
    if (fnName == "exp") return expm(A, mr);
    if (fnName == "log") return logm(A, mr);
    if (fnName == "sqrt") return sqrtm(A, mr);
    if (fnName == "sin") return funm_eval(A, [](keel::c64 x) { return keel::sin(x); }, mr);
    if (fnName == "cos") return funm_eval(A, [](keel::c64 x) { return keel::cos(x); }, mr);
    if (fnName == "sinh") return funm_eval(A, [](keel::c64 x) { return keel::sinh(x); }, mr);
    if (fnName == "cosh") return funm_eval(A, [](keel::c64 x) { return keel::cosh(x); }, mr);

    throw Error(keel::format("funm: unsupported function '{}'", fnName), 0, 0, "funm", "", "numkit:funm:unknownFn");
}

} // namespace numkit::linalg
