// toolboxes/linalg/src/eig.cpp
//
// Eigenvalue family â€” implementations and engine adapters.
// Migrated 2026-05-25 from toolboxes/builtin/src/language/arrays/matrix.cpp.
//
// Includes:
//   - eig_symmetric (classical Jacobi)
//   - eig_values
//   - poly_of_matrix (Souriau-Faddeev-LeVerrier)
//   - eig_general_values (via poly + roots)
//   - eig_general_VD     (via poly + SVD for null-vectors)
//   - hess / hess_H_only (Householder Hessenberg reduction)
//   - schur_sym          (== eig for symmetric A)
//   - sylvester_sym      (simultaneous diagonalisation)

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/linalg/eig.hpp>
#include "linalg_detail.hpp"

#include <numkit/linalg/decompositions.hpp>           // svd_decompose
#include <numkit/linalg/properties.hpp>               // inv (polyeig companion)
#include <numkit/linalg/qz.hpp>                       // qz (generalized eig)
#include <numkit/builtin/polyfun.hpp>   // roots

// Compute-only TU: Value substrate + Error, no engine. The eig / hess /
// schur / sylvester / polyeig / ordeig builtins (CallContext wrappers)
// live in eig_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace numkit::linalg {

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Characteristic polynomial + general eig via roots
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

Value poly_of_matrix(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "poly");
    if (A.dims().ndim() != 2)
        throw Error("poly: input must be a 2D matrix",
                    0, 0, "poly", "", "numkit:poly:notMatrix");
    const keel::usize m = static_cast<keel::usize>(A.dims().dim(0));
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(1));
    if (m != n)
        throw Error("poly: matrix must be square (use poly(roots) for vector input)",
                    0, 0, "poly", "", "numkit:poly:notSquare");
    const bool wantSingle = (A.type() == ValueType::SINGLE);
    if (n == 0) {
        auto out = Value::matrix(1, 1, ValueType::DOUBLE, mr);
        out.doubleDataMut()[0] = 1.0;
        return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
    }

    Value evals = eig_general_values(A, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> coeff(n + 1, keel::c64(0.0, 0.0), scratch.allocator());
    coeff[0] = keel::c64(1.0, 0.0);
    for (keel::usize k = 0; k < n; ++k) {
        const keel::c64 lam = evals.isComplex()
                                ? evals.complexData()[k]
                                : keel::c64(evals.elemAsDouble(k), 0.0);
        for (keel::usize j = k + 1; j >= 1; --j) {
            coeff[j] -= lam * coeff[j - 1];
        }
    }

    if (!A.isComplex()) {
        auto out = Value::matrix(1, n + 1, ValueType::DOUBLE, mr);
        double *c = out.doubleDataMut();
        for (keel::usize j = 0; j <= n; ++j)
            c[j] = coeff[j].real();
        return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
    }

    auto out = Value::complexMatrix(1, n + 1, mr);
    keel::c64 *c = out.complexDataMut();
    for (keel::usize j = 0; j <= n; ++j)
        c[j] = coeff[j];
    if (wantSingle) out.setSingleComplex(true);
    return detail::narrow_if_real(out, mr);
}

static bool isHermitianComplexApprox(const Value &A, double tol = 1e-10)
{
    if (A.dims().ndim() != 2 || !A.isComplex()) return false;
    const keel::usize m = static_cast<keel::usize>(A.dims().dim(0));
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(1));
    if (m != n) return false;
    const keel::c64 *ad = A.complexData();
    double aMax = 1.0;
    for (keel::usize idx = 0; idx < n * n; ++idx) {
        const double v = keel::abs(ad[idx]);
        if (v > aMax) aMax = v;
    }
    const double thresh = tol * aMax;
    for (keel::usize i = 0; i < n; ++i) {
        if (std::fabs(ad[i + i * n].imag()) > thresh) return false;
        for (keel::usize j = i + 1; j < n; ++j) {
            if (keel::abs(ad[i + j * n] - keel::conj(ad[j + i * n])) > thresh)
                return false;
        }
    }
    return true;
}

Value eig_general_values(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "eig");
    if (A.dims().ndim() != 2)
        throw Error("eig: input must be a 2D matrix",
                    0, 0, "eig", "", "numkit:eig:notMatrix");
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(0));
    if (n != static_cast<keel::usize>(A.dims().dim(1)))
        throw Error("eig: matrix must be square",
                    0, 0, "eig", "", "numkit:eig:notSquare");
    if (n == 0) return Value::matrix(0, 1, A.type(), mr);

    const bool wantSingle = (A.type() == ValueType::SINGLE);
    Value Aw = detail::widenSingleToDouble(A, mr);

    auto [U, T] = schur_general(Aw, mr);

    if (isHermitianComplexApprox(Aw)) {
        auto out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        if (T.isComplex()) {
            const keel::c64 *td = T.complexData();
            for (keel::usize i = 0; i < n; ++i) od[i] = td[i + i * n].real();
        } else {
            const double *td = T.doubleData();
            for (keel::usize i = 0; i < n; ++i) od[i] = td[i + i * n];
        }
        std::sort(od, od + n);
        return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
    }

    auto out = Value::complexMatrix(n, 1, mr);
    keel::c64 *o = out.complexDataMut();

    if (T.isComplex()) {
        const keel::c64 *td = T.complexData();
        for (keel::usize i = 0; i < n; ++i) o[i] = td[i + i * n];
    } else {
        const double *td = T.doubleData();
        keel::usize i = 0;
        while (i < n) {
            if (i + 1 < n && td[(i + 1) + i * n] != 0.0) {
                double a = td[i + i * n];
                double b = td[i + (i + 1) * n];
                double c = td[(i + 1) + i * n];
                double d = td[(i + 1) + (i + 1) * n];
                double tr = a + d;
                double det = a * d - b * c;
                double disc = tr * tr - 4.0 * det;
                double re = 0.5 * tr;
                double im = 0.5 * std::sqrt(std::max(0.0, -disc));
                o[i] = keel::c64(re, im);
                o[i + 1] = keel::c64(re, -im);
                i += 2;
            } else {
                o[i] = keel::c64(td[i + i * n], 0.0);
                i += 1;
            }
        }
    }
    return detail::narrowDoubleToSingleIf(detail::narrow_if_real(out, mr), wantSingle, mr);
}

std::tuple<Value, Value>
eig_general_VD(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "eig");
    if (A.dims().ndim() != 2)
        throw Error("eig: input must be a 2D matrix",
                    0, 0, "eig", "", "numkit:eig:notMatrix");
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(0));
    if (n != static_cast<keel::usize>(A.dims().dim(1)))
        throw Error("eig: matrix must be square",
                    0, 0, "eig", "", "numkit:eig:notSquare");
    if (n == 0) {
        return std::make_tuple(
            Value::matrix(0, 0, A.type(), mr),
            Value::matrix(0, 0, A.type(), mr));
    }

    const bool wantSingle = (A.type() == ValueType::SINGLE);
    Value Aw = detail::widenSingleToDouble(A, mr);

    Value A_c = Aw.isComplex() ? Aw : Value::complexMatrix(n, n, mr);
    if (!Aw.isComplex()) {
        const double *ad = Aw.doubleData();
        keel::c64 *acd = A_c.complexDataMut();
        for (keel::usize i = 0; i < n * n; ++i) acd[i] = keel::c64(ad[i], 0.0);
    }

    auto [U, T] = schur_general(A_c, mr);

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

    if (isHermitianComplexApprox(Aw)) {
        keel::Array<keel::usize> order(n, scratch.allocator());
        for (keel::usize i = 0; i < n; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](keel::usize a, keel::usize b) {
            return Tc[a + a * n].real() < Tc[b + b * n].real();
        });
        auto Vout = Value::complexMatrix(n, n, mr);
        auto Dout = Value::matrix(n, n, ValueType::DOUBLE, mr);
        keel::c64 *Vd = Vout.complexDataMut();
        double *Dd = Dout.doubleDataMut();
        std::fill(Dd, Dd + n * n, 0.0);
        for (keel::usize k = 0; k < n; ++k) {
            const keel::usize src = order[k];
            Dd[k + k * n] = Tc[src + src * n].real();
            for (keel::usize i = 0; i < n; ++i)
                Vd[i + k * n] = Uc[i + src * n];
        }
        return std::make_tuple(
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Vout, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(Dout, wantSingle, mr));
    }

    double tMax = 0.0;
    for (keel::usize i = 0; i < n * n; ++i)
        tMax = std::max(tMax, keel::abs(Tc[i]));
    const double eps = std::numeric_limits<double>::epsilon();
    const double backTol = 64.0 * static_cast<double>(n) * eps * std::max(1.0, tMax);
    const double smin = std::max(eps * std::max(1.0, tMax),
                                 std::numeric_limits<double>::min() * (static_cast<double>(n) / eps));

    keel::Array<keel::c64> Yc(n * n, keel::c64(0.0, 0.0), scratch.allocator());

    for (keel::usize k = 0; k < n; ++k) {
        keel::c64 lam = Tc[k + k * n];
        Yc[k + k * n] = keel::c64(1.0, 0.0);
        double maxY = 1.0;
        for (std::intptr_t i = static_cast<std::intptr_t>(k) - 1; i >= 0; --i) {
            const keel::usize ui = static_cast<keel::usize>(i);
            keel::c64 sum(0.0, 0.0);
            for (keel::usize j = ui + 1; j <= k; ++j) {
                sum += Tc[ui + j * n] * Yc[j + k * n];
            }
            keel::c64 denom = Tc[ui + ui * n] - lam;
            if (keel::abs(denom) <= backTol && keel::abs(sum) <= backTol * maxY) {
                // Diagonalizable repeated eigenvalue: free coordinate -> keep independent from col ui
                Yc[ui + k * n] = keel::c64(0.0, 0.0);
            } else {
                if (keel::abs(denom) < smin) denom = keel::c64(smin, 0.0);
                Yc[ui + k * n] = -sum / denom;
                maxY = std::max(maxY, keel::abs(Yc[ui + k * n]));
            }
        }
    }

    auto Vout = Value::complexMatrix(n, n, mr);
    auto Dout = Value::complexMatrix(n, n, mr);
    keel::c64 *Vd = Vout.complexDataMut();
    keel::c64 *Dd = Dout.complexDataMut();
    std::fill(Dd, Dd + n * n, keel::c64(0.0, 0.0));

    for (keel::usize k = 0; k < n; ++k) {
        Dd[k + k * n] = Tc[k + k * n];
        double norm_sq = 0.0;
        for (keel::usize i = 0; i < n; ++i) {
            keel::c64 s(0.0, 0.0);
            for (keel::usize j = 0; j <= k; ++j) {
                s += Uc[i + j * n] * Yc[j + k * n];
            }
            Vd[i + k * n] = s;
            norm_sq += detail::abs_sq(s);
        }
        double norm = std::sqrt(norm_sq);
        if (norm > 0.0) {
            for (keel::usize i = 0; i < n; ++i)
                Vd[i + k * n] /= norm;
        }
    }

    // Reference semantics: for a real input matrix A, any individual real
    // eigenvalue has exact zero imaginary part and a real eigenvector (the
    // complex-Schur pipeline multiplies each eigenvector by an arbitrary
    // unit complex phase and leaves ~1e-16 imaginary dust). When the whole
    // spectrum is real, narrow V and D to double; when mixed real/complex,
    // real columns of V and D have exact 0.0 imaginary part so slicing
    // V(:, k) narrows to real double (bugs: linalg/eig-real-spectrum-complex-typing).
    if (!Aw.isComplex()) {
        double aMax = 0.0;
        for (keel::usize i = 0; i < n * n; ++i)
            aMax = std::max(aMax, keel::abs(A_c.complexData()[i].real()));
        const double tol = 64.0 * static_cast<double>(n) *
                           std::numeric_limits<double>::epsilon() * (1.0 + aMax);
        bool real_spectrum = true;
        for (keel::usize k = 0; k < n; ++k) {
            if (keel::abs(Dd[k + k * n].imag()) <= tol) {
                Dd[k + k * n] = keel::c64(Dd[k + k * n].real(), 0.0);
                // Find largest-magnitude component of V(:, k) and rotate out its phase.
                keel::usize iMax = 0;
                double maxAbs = 0.0;
                for (keel::usize i = 0; i < n; ++i) {
                    double a = keel::abs(Vd[i + k * n]);
                    if (a > maxAbs) {
                        maxAbs = a;
                        iMax = i;
                    }
                }
                if (maxAbs > 0.0) {
                    keel::c64 phase = keel::conj(Vd[iMax + k * n]) / maxAbs;
                    double colImax = 0.0;
                    for (keel::usize i = 0; i < n; ++i) {
                        keel::c64 rotated = Vd[i + k * n] * phase;
                        Vd[i + k * n] = rotated;
                        colImax = std::max(colImax, keel::abs(rotated.imag()));
                    }
                    if (colImax <= 1e3 * tol) {
                        double rNormSq = 0.0;
                        for (keel::usize i = 0; i < n; ++i) {
                            double re = Vd[i + k * n].real();
                            Vd[i + k * n] = keel::c64(re, 0.0);
                            rNormSq += re * re;
                        }
                        double rNorm = std::sqrt(rNormSq);
                        if (rNorm > 0.0) {
                            for (keel::usize i = 0; i < n; ++i)
                                Vd[i + k * n] = keel::c64(Vd[i + k * n].real() / rNorm, 0.0);
                        }
                    }
                }
            } else {
                real_spectrum = false;
            }
        }
        double vImax = 0.0;
        for (keel::usize i = 0; i < n * n; ++i)
            vImax = std::max(vImax, keel::abs(Vd[i].imag()));
        if (real_spectrum && vImax <= 1e3 * tol) {
            auto Vr = Value::matrix(n, n, ValueType::DOUBLE, mr);
            auto Dr = Value::matrix(n, n, ValueType::DOUBLE, mr);
            for (keel::usize i = 0; i < n * n; ++i) {
                Vr.doubleDataMut()[i] = Vd[i].real();
                Dr.doubleDataMut()[i] = Dd[i].real();
            }
            return std::make_tuple(
                detail::narrowDoubleToSingleIf(Vr, wantSingle, mr),
                detail::narrowDoubleToSingleIf(Dr, wantSingle, mr));
        }
    }

    return std::make_tuple(
        detail::narrowDoubleToSingleIf(detail::narrow_if_real(Vout, mr), wantSingle, mr),
        detail::narrowDoubleToSingleIf(detail::narrow_if_real(Dout, mr), wantSingle, mr));
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Symmetric eig (classical Jacobi)
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

namespace {

// // Classical Jacobi for SYMMETRIC A (used for small n <= 16).
void jacobiSymInplace(double *A, keel::usize n, double *V,
                      keel::usize maxSweeps, double tol)
{
    std::fill(V, V + n * n, 0.0);
    for (keel::usize i = 0; i < n; ++i) V[i + i * n] = 1.0;
    if (n <= 1) return;

    double diagSq = 0.0;
    for (keel::usize i = 0; i < n; ++i)
        diagSq += A[i + i * n] * A[i + i * n];

    auto offSum = [&]() {
        double s = 0.0;
        for (keel::usize p = 0; p + 1 < n; ++p)
            for (keel::usize q = p + 1; q < n; ++q)
                s += A[p + q * n] * A[p + q * n];
        return s;
    };

    const double eps = std::numeric_limits<double>::epsilon();
    const double scaleSq = std::max(1.0, diagSq + 2.0 * offSum());
    const double convTolSq = (tol * tol) * scaleSq;

    for (keel::usize sweep = 0; sweep < maxSweeps; ++sweep) {
        if (offSum() < convTolSq) break;
        for (keel::usize p = 0; p + 1 < n; ++p) {
            for (keel::usize q = p + 1; q < n; ++q) {
                const double Apq = A[p + q * n];
                const double App = A[p + p * n];
                const double Aqq = A[q + q * n];
                if (std::fabs(Apq) <= eps * (std::fabs(App) + std::fabs(Aqq))) {
                    A[p + q * n] = 0.0;
                    continue;
                }
                double c, s;
                if (App == Aqq) {
                    c = 0.7071067811865476;
                    s = (Apq >= 0.0 ? 1.0 : -1.0) * c;
                } else {
                    const double tau = (Aqq - App) / (2.0 * Apq);
                    const double t = (tau >= 0.0)
                        ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
                        : 1.0 / (tau - std::sqrt(1.0 + tau * tau));
                    c = 1.0 / std::sqrt(1.0 + t * t);
                    s = t * c;
                }

                A[p + p * n] = c * c * App - 2.0 * c * s * Apq + s * s * Aqq;
                A[q + q * n] = s * s * App + 2.0 * c * s * Apq + c * c * Aqq;
                A[p + q * n] = 0.0;
                A[q + p * n] = 0.0;
                for (keel::usize r = 0; r < n; ++r) {
                    if (r == p || r == q) continue;
                    auto get = [&](keel::usize a, keel::usize b) -> double & {
                        return (a < b) ? A[a + b * n] : A[b + a * n];
                    };
                    const double Arp = get(r, p);
                    const double Arq = get(r, q);
                    get(r, p) = c * Arp - s * Arq;
                    get(r, q) = s * Arp + c * Arq;
                }
                for (keel::usize r = 0; r < n; ++r) {
                    const double Vrp = V[r + p * n];
                    const double Vrq = V[r + q * n];
                    V[r + p * n] = c * Vrp - s * Vrq;
                    V[r + q * n] = s * Vrp + c * Vrq;
                }
            }
        }
    }

    // Mirror upper triangle into lower for clean output.
    for (keel::usize p = 0; p + 1 < n; ++p)
        for (keel::usize q = p + 1; q < n; ++q)
            A[q + p * n] = A[p + q * n];
}

// Householder tridiagonalization (tred2) + implicit Wilkinson-shifted QL (tql2)
// for real symmetric matrices (O(n^3) direct reduction, used for n > 16).
// Input: V holds a copy of A (n x n, column-major).
// Output: d[0..n-1] holds the eigenvalues, V holds the orthonormal eigenvectors.
void symTridiagQL(double *V, keel::usize n, double *d, const keel::ArenaTemp &scratch)
{
    keel::Array<double> e(n, 0.0, scratch.allocator());

    for (keel::usize i = n - 1; i >= 1; --i) {
        const keel::usize l = i - 1;
        double h = 0.0;
        double scale = 0.0;
        if (l > 0) {
            for (keel::usize k = 0; k <= l; ++k)
                scale += std::fabs(V[i + k * n]);
            if (scale == 0.0) {
                e[i] = V[i + l * n];
            } else {
                for (keel::usize k = 0; k <= l; ++k) {
                    V[i + k * n] /= scale;
                    h += V[i + k * n] * V[i + k * n];
                }
                double f = V[i + l * n];
                double g = (f >= 0.0) ? -std::sqrt(h) : std::sqrt(h);
                e[i] = scale * g;
                h -= f * g;
                V[i + l * n] = f - g;
                f = 0.0;
                for (keel::usize j = 0; j <= l; ++j) {
                    V[j + i * n] = V[i + j * n] / h;
                    g = 0.0;
                    for (keel::usize k = 0; k <= j; ++k)
                        g += V[j + k * n] * V[i + k * n];
                    for (keel::usize k = j + 1; k <= l; ++k)
                        g += V[k + j * n] * V[i + k * n];
                    e[j] = g / h;
                    f += e[j] * V[i + j * n];
                }
                const double hh = f / (h + h);
                for (keel::usize j = 0; j <= l; ++j) {
                    f = V[i + j * n];
                    g = e[j] - hh * f;
                    e[j] = g;
                    for (keel::usize k = 0; k <= j; ++k)
                        V[j + k * n] -= (f * e[k] + g * V[i + k * n]);
                }
            }
        } else {
            e[i] = V[i + l * n];
        }
        d[i] = h;
    }

    d[0] = 0.0;
    e[0] = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        if (d[i] != 0.0) {
            for (keel::usize j = 0; j < i; ++j) {
                double g = 0.0;
                for (keel::usize k = 0; k < i; ++k)
                    g += V[i + k * n] * V[k + j * n];
                for (keel::usize k = 0; k < i; ++k)
                    V[k + j * n] -= g * V[k + i * n];
            }
        }
        d[i] = V[i + i * n];
        V[i + i * n] = 1.0;
        for (keel::usize j = 0; j < i; ++j) {
            V[j + i * n] = 0.0;
            V[i + j * n] = 0.0;
        }
    }

    for (keel::usize i = 1; i < n; ++i)
        e[i - 1] = e[i];
    e[n - 1] = 0.0;

    const double eps = std::numeric_limits<double>::epsilon();
    for (keel::usize l = 0; l < n; ++l) {
        keel::usize iter = 0;
        keel::usize m = l;
        for (;;) {
            for (m = l; m + 1 < n; ++m) {
                const double dd = std::fabs(d[m]) + std::fabs(d[m + 1]);
                if (std::fabs(e[m]) <= eps * dd) {
                    e[m] = 0.0;
                    break;
                }
            }
            if (m == l) break;
            if (++iter > 300) break;
            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = std::hypot(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + (g >= 0.0 ? r : -r));
            double s = 1.0, c = 1.0, p = 0.0;
            bool underflow = false;
            for (keel::usize ii = m; ii > l; --ii) {
                const keel::usize i = ii - 1;
                const double f = s * e[i];
                const double b = c * e[i];
                r = std::hypot(f, g);
                e[i + 1] = r;
                if (r == 0.0) {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    underflow = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2.0 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                double *vi1 = V + (i + 1) * n;
                double *vi  = V + i * n;
                for (keel::usize k = 0; k < n; ++k) {
                    const double t = vi1[k];
                    vi1[k] = s * vi[k] + c * t;
                    vi[k]  = c * vi[k] - s * t;
                }
            }
            if (underflow) continue;
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
}

} // anonymous namespace

// Approximate-symmetry predicate — declared in eig.hpp; shared with the
// register-side auto-dispatch in eig_reg.cpp. Element reads are total
// (elemAsDouble) so non-double inputs dispatch cleanly into the entry
// argchecks instead of tripping a raw buffer error here.
bool isSymmetricApprox(const Value &A, double tol)
{
    if (A.dims().ndim() != 2) return false;
    // Only types elemAsDouble can read dispatch here; cell/struct/string
    // fall through to the general path, whose entry argcheck rejects
    // them with the proper diagnostic. Complex matrices also take the
    // general path — the elemAsDouble reads below cannot see imaginary
    // parts, and the complex-Schur route handles them.
    if (A.isComplex()) return false;
    if (!A.isNumeric() && !A.isLogical() && !A.isChar()) return false;
    const keel::usize m = static_cast<keel::usize>(A.dims().dim(0));
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(1));
    if (m != n) return false;
    double aMax = 1.0;
    for (keel::usize idx = 0; idx < n * n; ++idx) {
        const double v = std::fabs(A.elemAsDouble(idx));
        if (v > aMax) aMax = v;
    }
    const double thresh = tol * aMax;
    for (keel::usize i = 0; i < n; ++i)
        for (keel::usize j = i + 1; j < n; ++j) {
            const double aij = A.elemAsDouble(i + j * m);
            const double aji = A.elemAsDouble(j + i * m);
            const double d = std::fabs(aij - aji);
            if (d > thresh) return false;
        }
    return true;
}

std::tuple<Value, Value>
eig_symmetric(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "eig");
    if (A.dims().ndim() != 2)
        throw Error("eig: input must be a 2D matrix",
                    0, 0, "eig", "", "numkit:eig:notMatrix");
    const keel::usize m = static_cast<keel::usize>(A.dims().dim(0));
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(1));
    if (m != n)
        throw Error("eig: matrix must be square",
                    0, 0, "eig", "", "numkit:eig:notSquare");
    if (!isSymmetricApprox(A, 1e-10))
        throw Error("eig: only symmetric matrices supported in this revision "
                    "(general eig via Hessenberg + Francis QR is deferred to Phase 2b)",
                    0, 0, "eig", "", "numkit:eig:notSymmetric");
    if (n == 0) {
        return std::make_tuple(
            Value::matrix(0, 0, A.type(), mr),
            Value::matrix(0, 0, A.type(), mr));
    }

    const bool wantSingle = (A.type() == ValueType::SINGLE);
    Value Aw = detail::widenSingleToDouble(A, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> V_work(n * n, scratch.allocator());
    keel::Array<double> evals(n, scratch.allocator());

    if (n <= 16) {
        keel::Array<double> A_work(n * n, scratch.allocator());
        std::copy(Aw.doubleData(), Aw.doubleData() + n * n, A_work.begin());
        jacobiSymInplace(A_work.data(), n, V_work.data(),
                         /*maxSweeps=*/64, /*tol=*/1e-13);
        for (keel::usize i = 0; i < n; ++i)
            evals[i] = A_work[i + i * n];
    } else {
        std::copy(Aw.doubleData(), Aw.doubleData() + n * n, V_work.begin());
        symTridiagQL(V_work.data(), n, evals.data(), scratch);
    }

    // Sort eigenvalues ASCENDING using LAPACK dsteqr selection sort
    // (matches LAPACK's exact permutation on repeated eigenvalues).
    keel::Array<keel::usize> order(n, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) order[i] = i;
    for (keel::usize ii = 1; ii < n; ++ii) {
        keel::usize i = ii - 1;
        keel::usize k = i;
        double p = evals[order[i]];
        for (keel::usize j = ii; j < n; ++j) {
            if (evals[order[j]] < p) {
                k = j;
                p = evals[order[j]];
            }
        }
        if (k != i) std::swap(order[i], order[k]);
    }

    auto Vout = Value::matrix(n, n, ValueType::DOUBLE, mr);
    auto Dout = Value::matrix(n, n, ValueType::DOUBLE, mr);
    double *V = Vout.doubleDataMut();
    double *D = Dout.doubleDataMut();
    std::fill(D, D + n * n, 0.0);
    for (keel::usize k = 0; k < n; ++k) {
        const keel::usize src = order[k];
        D[k + k * n] = evals[src];
        for (keel::usize i = 0; i < n; ++i)
            V[i + k * n] = V_work[i + src * n];
    }
    return std::make_tuple(
        detail::narrowDoubleToSingleIf(Vout, wantSingle, mr),
        detail::narrowDoubleToSingleIf(Dout, wantSingle, mr));
}

Value eig_values(const Value &A, keel::Allocator *mr)
{
    const bool wantSingle = (A.type() == ValueType::SINGLE);
    Value Aw = detail::widenSingleToDouble(A, mr);
    auto [V, D] = eig_symmetric(Aw, mr);
    const keel::usize n = static_cast<keel::usize>(D.dims().dim(0));
    auto out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    const double *Ddata = D.doubleData();
    double *o = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) o[i] = Ddata[i + i * n];
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

namespace {

// Upper-triangular Cholesky factor R (stored column-major, R(k, j) at R[k + j*n] for k <= j)
// such that R^T * R = M (i.e. L = R^T is the lower Cholesky factor).
// Storing R in column-major order makes column j of R (= row j of L) contiguous in memory!
bool tryCholeskyUpper(const double *M, keel::usize n, double *R)
{
    double diagMax = 0.0;
    for (keel::usize i = 0; i < n; ++i)
        diagMax = std::max(diagMax, std::fabs(M[i + i * n]));
    if (!(diagMax > 0.0)) return false;
    const double tol = 1e-12 * diagMax;

    std::vector<keel::usize> firstNz(n, 0);
    for (keel::usize j = 0; j < n; ++j) {
        const double *mj = M + j * n;
        keel::usize fnz = j;
        for (keel::usize i = 0; i < j; ++i) {
            if (mj[i] != 0.0) { fnz = i; break; }
        }
        firstNz[j] = fnz;
    }

    for (keel::usize j = 0; j < n; ++j) {
        double *rj = R + j * n;
        const keel::usize fnzJ = firstNz[j];
        for (keel::usize i = 0; i < fnzJ; ++i)
            rj[i] = 0.0;
        for (keel::usize i = fnzJ; i < j; ++i) {
            const double *ri = R + i * n;
            double s = M[i + j * n];
            const keel::usize kStart = std::max(firstNz[i], fnzJ);
            for (keel::usize k = kStart; k < i; ++k)
                s -= ri[k] * rj[k];
            rj[i] = s / ri[i];
        }
        double sum = 0.0;
        for (keel::usize k = fnzJ; k < j; ++k)
            sum += rj[k] * rj[k];
        const double d = M[j + j * n] - sum;
        if (!(d > tol)) return false;
        rj[j] = std::sqrt(d);
        for (keel::usize i = j + 1; i < n; ++i)
            rj[i] = 0.0;
    }
    return true;
}

// Compute C = R^-T * M * R^-1 (where R^T * R = B, so L = R^T) using contiguous column dot products.
void formSymmetricReduced(const double *M, const double *R, keel::usize n,
                          double *C, const keel::ArenaTemp &scratch)
{
    keel::Array<double> Y(n * n, scratch.allocator());
    // Solve R^T * Y = M  (for each column col of M, y_i = (M(i,col) - sum_{k<i} R(k,i)*y_k) / R(i,i))
    for (keel::usize col = 0; col < n; ++col) {
        double *ycol = Y.data() + col * n;
        const double *mcol = M + col * n;
        for (keel::usize i = 0; i < n; ++i) {
            const double *ri = R + i * n;
            double s = mcol[i];
            for (keel::usize k = 0; k < i; ++k)
                s -= ri[k] * ycol[k];
            ycol[i] = s / ri[i];
        }
    }
    // Transpose Y into C temporarily so rows of Y become contiguous columns: Yt(k, r) = Y(r, k)
    keel::Array<double> Yt(n * n, scratch.allocator());
    for (keel::usize col = 0; col < n; ++col) {
        const double *ycol = Y.data() + col * n;
        for (keel::usize row = 0; row < n; ++row)
            Yt[col + row * n] = ycol[row];
    }
    // Solve R^T * c_r = Y(r, :)^T for i = 0..r (upper triangle of symmetric C only!)
    for (keel::usize r = 0; r < n; ++r) {
        double *cr = C + r * n;
        const double *ytr = Yt.data() + r * n;
        for (keel::usize i = 0; i <= r; ++i) {
            const double *ri = R + i * n;
            double s = ytr[i];
            for (keel::usize k = 0; k < i; ++k)
                s -= ri[k] * cr[k];
            cr[i] = s / ri[i];
        }
    }
    // Mirror upper triangle of C to lower triangle.
    for (keel::usize r = 0; r < n; ++r) {
        for (keel::usize i = 0; i < r; ++i)
            C[r + i * n] = C[i + r * n];
    }
}

// Solve R * V = Q (i.e. L^T * V = Q) by back-substitution with contiguous column operations.
void backsolveR(const double *R, const double *Q, keel::usize n, double *V)
{
    std::copy(Q, Q + n * n, V);
    for (keel::usize col = 0; col < n; ++col) {
        double *vcol = V + col * n;
        for (keel::usize ii = n; ii > 0; --ii) {
            const keel::usize i = ii - 1;
            const double *ri = R + i * n;
            const double vii = vcol[i] / ri[i];
            vcol[i] = vii;
            if (vii != 0.0) {
                for (keel::usize k = 0; k < i; ++k)
                    vcol[k] -= ri[k] * vii;
            }
        }
    }
}

} // anonymous namespace

std::tuple<Value, Value>
eig_generalized_VD(const Value &A, const Value &B, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "eig");
    detail::requireDoubleInput(B, "eig", "B");
    if (A.dims().ndim() != 2 || B.dims().ndim() != 2)
        throw Error("eig: inputs must be 2D matrices", 0, 0, "eig", "", "numkit:eig:notMatrix");
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(0));
    if (n != static_cast<keel::usize>(A.dims().dim(1)) ||
        n != static_cast<keel::usize>(B.dims().dim(0)) ||
        n != static_cast<keel::usize>(B.dims().dim(1)))
        throw Error("eig: A and B must be square matrices of the same size",
                    0, 0, "eig", "", "numkit:eig:notSquare");

    const bool wantSingle = (A.type() == ValueType::SINGLE || B.type() == ValueType::SINGLE);
    const ValueType rtype = wantSingle ? ValueType::SINGLE : ValueType::DOUBLE;
    if (n == 0) {
        return std::make_tuple(Value::matrix(0, 0, rtype, mr), Value::matrix(0, 0, rtype, mr));
    }

    Value Aw = detail::widenSingleToDouble(A, mr);
    Value Bw = detail::widenSingleToDouble(B, mr);

    if (!Aw.isComplex() && !Bw.isComplex() &&
        isSymmetricApprox(Aw, 1e-8) && isSymmetricApprox(Bw, 1e-8)) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<double> L(n * n, scratch.allocator());

        // Case 1: B is symmetric positive definite -> standard Cholesky reduction
        // (R^-T A R^-1) Q = Q D, V = R^-1 Q (B-orthonormal: V' * B * V = I).
        if (tryCholeskyUpper(Bw.doubleData(), n, L.data())) {
            auto Cval = Value::matrix(n, n, ValueType::DOUBLE, mr);
            formSymmetricReduced(Aw.doubleData(), L.data(), n, Cval.doubleDataMut(), scratch);
            auto [Q, D] = eig_symmetric(Cval, mr);
            auto V = Value::matrix(n, n, ValueType::DOUBLE, mr);
            backsolveR(L.data(), Q.doubleData(), n, V.doubleDataMut());
            return std::make_tuple(
                detail::narrowDoubleToSingleIf(V, wantSingle, mr),
                detail::narrowDoubleToSingleIf(D, wantSingle, mr));
        }

        // Case 2: B is singular/indefinite (e.g. geometric stiffness Kg in buckling),
        // while A is symmetric positive definite -> reciprocal Cholesky reduction
        // (R_A^-T B R_A^-1) Q = Q D_mu, lambda = 1 / mu (Inf for mu == 0).
        if (tryCholeskyUpper(Aw.doubleData(), n, L.data())) {
            auto Cval = Value::matrix(n, n, ValueType::DOUBLE, mr);
            formSymmetricReduced(Bw.doubleData(), L.data(), n, Cval.doubleDataMut(), scratch);
            auto [Q, Dmu] = eig_symmetric(Cval, mr);
            keel::Array<double> Vraw(n * n, scratch.allocator());
            backsolveR(L.data(), Q.doubleData(), n, Vraw.data());

            const double *dmu = Dmu.doubleData();
            double muMax = 0.0;
            for (keel::usize j = 0; j < n; ++j)
                muMax = std::max(muMax, std::fabs(dmu[j + j * n]));
            const double muTol = 1e-11 * std::max(1.0, muMax);

            keel::Array<double> lam(n, scratch.allocator());
            keel::Array<keel::usize> order(n, scratch.allocator());
            for (keel::usize j = 0; j < n; ++j) {
                order[j] = j;
                const double mu = dmu[j + j * n];
                lam[j] = (std::fabs(mu) > muTol) ? (1.0 / mu)
                                                 : std::numeric_limits<double>::infinity();
                // Normalize column j so its largest-magnitude component is +1.0 (LAPACK/QZ convention).
                keel::usize iMax = 0;
                double maxAbs = 0.0;
                for (keel::usize i = 0; i < n; ++i) {
                    const double a = std::fabs(Vraw[i + j * n]);
                    if (a > maxAbs) { maxAbs = a; iMax = i; }
                }
                if (maxAbs > 0.0) {
                    const double scale = Vraw[iMax + j * n];
                    for (keel::usize i = 0; i < n; ++i)
                        Vraw[i + j * n] /= scale;
                }
            }
            std::sort(order.begin(), order.end(), [&](keel::usize a, keel::usize b) {
                return lam[a] < lam[b];
            });

            auto V = Value::matrix(n, n, ValueType::DOUBLE, mr);
            auto D = Value::matrix(n, n, ValueType::DOUBLE, mr);
            double *vd = V.doubleDataMut();
            double *dd = D.doubleDataMut();
            std::fill(dd, dd + n * n, 0.0);
            for (keel::usize col = 0; col < n; ++col) {
                const keel::usize src = order[col];
                dd[col + col * n] = lam[src];
                for (keel::usize row = 0; row < n; ++row)
                    vd[row + col * n] = Vraw[row + src * n];
            }
            return std::make_tuple(
                detail::narrowDoubleToSingleIf(V, wantSingle, mr),
                detail::narrowDoubleToSingleIf(D, wantSingle, mr));
        }
    }

    // General matrix pencil (A, B): generalized Schur QZ decomposition
    // Q * A * Z = AA, Q * B * Z = BB (AA, BB upper triangular), followed by
    // O(n^3) generalized triangular eigenvector back-substitution:
    //   (beta_k * AA - alpha_k * BB) * y_k = 0,  v_k = Z * y_k.
    auto [AA, BB, Q, Z] = qz(Aw, Bw, mr);
    (void)Q;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> AAc(n * n, scratch.allocator());
    keel::Array<keel::c64> BBc(n * n, scratch.allocator());
    keel::Array<keel::c64> Zc(n * n, scratch.allocator());

    auto copyToComplex = [&](const Value &src, keel::Array<keel::c64> &dst) {
        if (src.isComplex()) {
            std::copy(src.complexData(), src.complexData() + n * n, dst.begin());
        } else {
            const double *sd = src.doubleData();
            for (keel::usize i = 0; i < n * n; ++i)
                dst[i] = keel::c64(sd[i], 0.0);
        }
    };
    copyToComplex(AA, AAc);
    copyToComplex(BB, BBc);
    copyToComplex(Z, Zc);

    double aMax = 0.0, bMax = 0.0;
    for (keel::usize i = 0; i < n * n; ++i) {
        aMax = std::max(aMax, keel::abs(AAc[i]));
        bMax = std::max(bMax, keel::abs(BBc[i]));
    }
    const double eps = std::numeric_limits<double>::epsilon();
    const double abScale = std::max(1.0, std::max(aMax, bMax));
    const double backTol = 64.0 * static_cast<double>(n) * eps * abScale;
    const double smin = std::max(eps * abScale,
                                 std::numeric_limits<double>::min() * (static_cast<double>(n) / eps));
    const double bZeroTol = 64.0 * static_cast<double>(n) * eps * std::max(1.0, bMax);
    const double aZeroTol = 64.0 * static_cast<double>(n) * eps * std::max(1.0, aMax);

    keel::Array<keel::c64> Yc(n * n, keel::c64(0.0, 0.0), scratch.allocator());
    for (keel::usize k = 0; k < n; ++k) {
        keel::c64 alpha = AAc[k + k * n];
        keel::c64 beta = BBc[k + k * n];
        const double pairScale = std::max(keel::abs(alpha), keel::abs(beta));
        if (pairScale > 0.0) {
            alpha /= pairScale;
            beta /= pairScale;
        }
        Yc[k + k * n] = keel::c64(1.0, 0.0);
        double maxY = 1.0;
        for (std::intptr_t i = static_cast<std::intptr_t>(k) - 1; i >= 0; --i) {
            const keel::usize ui = static_cast<keel::usize>(i);
            keel::c64 sum(0.0, 0.0);
            for (keel::usize j = ui + 1; j <= k; ++j) {
                const keel::c64 mij = beta * AAc[ui + j * n] - alpha * BBc[ui + j * n];
                sum += mij * Yc[j + k * n];
            }
            keel::c64 denom = beta * AAc[ui + ui * n] - alpha * BBc[ui + ui * n];
            if (keel::abs(denom) <= backTol && keel::abs(sum) <= backTol * maxY) {
                Yc[ui + k * n] = keel::c64(0.0, 0.0);
            } else {
                if (keel::abs(denom) < smin) denom = keel::c64(smin, 0.0);
                Yc[ui + k * n] = -sum / denom;
                maxY = std::max(maxY, keel::abs(Yc[ui + k * n]));
            }
        }
    }

    auto Vout = Value::complexMatrix(n, n, mr);
    auto Dout = Value::complexMatrix(n, n, mr);
    keel::c64 *Vd = Vout.complexDataMut();
    keel::c64 *Dd = Dout.complexDataMut();
    std::fill(Dd, Dd + n * n, keel::c64(0.0, 0.0));

    for (keel::usize k = 0; k < n; ++k) {
        const keel::c64 alpha = AAc[k + k * n];
        const keel::c64 beta = BBc[k + k * n];
        if (keel::abs(beta) > bZeroTol) {
            Dd[k + k * n] = alpha / beta;
        } else if (keel::abs(alpha) > aZeroTol) {
            Dd[k + k * n] = keel::c64(std::numeric_limits<double>::infinity(), 0.0);
        } else {
            Dd[k + k * n] = keel::c64(std::numeric_limits<double>::quiet_NaN(), 0.0);
        }

        double normSq = 0.0;
        for (keel::usize i = 0; i < n; ++i) {
            keel::c64 s(0.0, 0.0);
            for (keel::usize j = 0; j <= k; ++j)
                s += Zc[i + j * n] * Yc[j + k * n];
            Vd[i + k * n] = s;
            normSq += detail::abs_sq(s);
        }
        const double norm = std::sqrt(normSq);
        if (norm > 0.0) {
            for (keel::usize i = 0; i < n; ++i)
                Vd[i + k * n] /= norm;
        }
    }

    if (!Aw.isComplex() && !Bw.isComplex()) {
        const double imagTol = 64.0 * static_cast<double>(n) * eps * (1.0 + abScale);
        bool realSpectrum = true;
        for (keel::usize k = 0; k < n; ++k) {
            const keel::c64 dkk = Dd[k + k * n];
            const bool isRealEig = !std::isfinite(dkk.real()) ||
                                   (keel::abs(dkk.imag()) <= imagTol * std::max(1.0, keel::abs(dkk.real())));
            if (isRealEig) {
                Dd[k + k * n] = keel::c64(dkk.real(), 0.0);
                keel::usize iMax = 0;
                double maxAbs = 0.0;
                for (keel::usize i = 0; i < n; ++i) {
                    const double a = keel::abs(Vd[i + k * n]);
                    if (a > maxAbs) { maxAbs = a; iMax = i; }
                }
                if (maxAbs > 0.0) {
                    const keel::c64 phase = keel::conj(Vd[iMax + k * n]) / maxAbs;
                    double colImax = 0.0;
                    for (keel::usize i = 0; i < n; ++i) {
                        const keel::c64 rotated = Vd[i + k * n] * phase;
                        Vd[i + k * n] = rotated;
                        colImax = std::max(colImax, keel::abs(rotated.imag()));
                    }
                    if (colImax <= 1e3 * imagTol) {
                        double rNormSq = 0.0;
                        for (keel::usize i = 0; i < n; ++i) {
                            const double re = Vd[i + k * n].real();
                            Vd[i + k * n] = keel::c64(re, 0.0);
                            rNormSq += re * re;
                        }
                        const double rNorm = std::sqrt(rNormSq);
                        if (rNorm > 0.0) {
                            for (keel::usize i = 0; i < n; ++i)
                                Vd[i + k * n] = keel::c64(Vd[i + k * n].real() / rNorm, 0.0);
                        }
                    }
                }
            } else {
                realSpectrum = false;
            }
        }
        double vImax = 0.0;
        for (keel::usize i = 0; i < n * n; ++i)
            vImax = std::max(vImax, keel::abs(Vd[i].imag()));
        if (realSpectrum && vImax <= 1e3 * imagTol) {
            auto Vr = Value::matrix(n, n, ValueType::DOUBLE, mr);
            auto Dr = Value::matrix(n, n, ValueType::DOUBLE, mr);
            for (keel::usize i = 0; i < n * n; ++i) {
                Vr.doubleDataMut()[i] = Vd[i].real();
                Dr.doubleDataMut()[i] = Dd[i].real();
            }
            return std::make_tuple(
                detail::narrowDoubleToSingleIf(Vr, wantSingle, mr),
                detail::narrowDoubleToSingleIf(Dr, wantSingle, mr));
        }
    }

    return std::make_tuple(
        detail::narrowDoubleToSingleIf(detail::narrow_if_real(Vout, mr), wantSingle, mr),
        detail::narrowDoubleToSingleIf(detail::narrow_if_real(Dout, mr), wantSingle, mr));
}

Value eig_generalized_values(const Value &A, const Value &B, keel::Allocator *mr)
{
    auto [V, D] = eig_generalized_VD(A, B, mr);
    (void)V;
    const keel::usize n = static_cast<keel::usize>(D.dims().dim(0));
    if (D.isComplex()) {
        auto out = Value::complexMatrix(n, 1, mr);
        const keel::c64 *dd = D.complexData();
        keel::c64 *od = out.complexDataMut();
        for (keel::usize i = 0; i < n; ++i) od[i] = dd[i + i * n];
        return out;
    }
    const bool wantSingle = (D.type() == ValueType::SINGLE);
    auto out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < n; ++i)
        out.doubleDataMut()[i] = D.elemAsDouble(i + i * n);
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

bool try_eigs_generalized_smallestabs(const Value &A, const Value &B, keel::usize k,
                                      Value &Vout, Value &Dout,
                                      keel::Allocator *mr)
{
    if (A.isComplex() || B.isComplex() || A.dims().ndim() != 2 || B.dims().ndim() != 2)
        return false;
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(0));
    if (n == 0 || k == 0 || k > n)
        return false;
    if (n != static_cast<keel::usize>(A.dims().dim(1)) ||
        n != static_cast<keel::usize>(B.dims().dim(0)) ||
        n != static_cast<keel::usize>(B.dims().dim(1)))
        return false;

    const bool wantSingle = (A.type() == ValueType::SINGLE || B.type() == ValueType::SINGLE);
    Value Aw = detail::widenSingleToDouble(A, mr);
    Value Bw = detail::widenSingleToDouble(B, mr);
    if (!isSymmetricApprox(Aw, 1e-8) || !isSymmetricApprox(Bw, 1e-8))
        return false;

    const double *ad = Aw.doubleData();
    const double *bd = Bw.doubleData();

    // Compute skyline envelope of A so upper Cholesky R_A and triangular solves exploit banded structure.
    std::vector<keel::usize> aFirstNz(n, 0);
    for (keel::usize j = 0; j < n; ++j) {
        const double *aj = ad + j * n;
        keel::usize fnz = j;
        for (keel::usize i = 0; i < j; ++i) {
            if (aj[i] != 0.0) { fnz = i; break; }
        }
        aFirstNz[j] = fnz;
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> RA(n * n, scratch.allocator());
    if (!tryCholeskyUpper(ad, n, RA.data()))
        return false;

    // Precompute column non-zero spans of B for fast B * z.
    std::vector<keel::usize> bFirst(n, 0), bLast(n, 0);
    bool bDiagAllPos = true;
    for (keel::usize j = 0; j < n; ++j) {
        const double *bj = bd + j * n;
        if (!(bj[j] > 0.0)) bDiagAllPos = false;
        keel::usize fnz = n, lnz = 0;
        for (keel::usize i = 0; i < n; ++i) {
            if (bj[i] != 0.0) {
                if (fnz == n) fnz = i;
                lnz = i + 1;
            }
        }
        bFirst[j] = fnz;
        bLast[j] = lnz;
    }

    const double *R = RA.data();
    auto solveRT = [&](const double *rhs, double *y) {
        for (keel::usize i = 0; i < n; ++i) {
            const double *ri = R + i * n;
            double s = rhs[i];
            const keel::usize kStart = aFirstNz[i];
            for (keel::usize k = kStart; k < i; ++k)
                s -= ri[k] * y[k];
            y[i] = s / ri[i];
        }
    };
    auto solveR = [&](const double *rhs, double *x) {
        std::copy(rhs, rhs + n, x);
        for (keel::usize ii = n; ii > 0; --ii) {
            const keel::usize i = ii - 1;
            const double *ri = R + i * n;
            const double xi = x[i] / ri[i];
            x[i] = xi;
            if (xi != 0.0) {
                const keel::usize kStart = aFirstNz[i];
                for (keel::usize k = kStart; k < i; ++k)
                    x[k] -= ri[k] * xi;
            }
        }
    };
    auto matvecB = [&](const double *z, double *u) {
        std::fill(u, u + n, 0.0);
        for (keel::usize j = 0; j < n; ++j) {
            const double zj = z[j];
            if (zj == 0.0) continue;
            const keel::usize f = bFirst[j], l = bLast[j];
            const double *bj = bd + j * n;
            for (keel::usize i = f; i < l; ++i)
                u[i] += bj[i] * zj;
        }
    };

    // Lanczos iteration on symmetric operator H = R_A^-T * B * R_A^-1.
    // Eigenvalues mu of H satisfy A v = (1/mu) B v with v = R_A^-1 q,
    // so largest |mu| of H correspond to smallest |lambda| = 1/|mu| of (A, B).
    const keel::usize m = std::min(n, std::max(3 * k + 24, static_cast<keel::usize>(64)));
    keel::Array<double> Q(n * m, scratch.allocator());
    keel::Array<double> alpha(m, scratch.allocator());
    keel::Array<double> beta(m, scratch.allocator());
    keel::Array<double> zBuf(n, scratch.allocator());
    keel::Array<double> uBuf(n, scratch.allocator());
    keel::Array<double> wBuf(n, scratch.allocator());

    // Deterministic starting vector with pseudo-random components to avoid symmetry orthogonality traps.
    double *q0 = Q.data();
    keel::u64 seed = 0x9E3779B97F4A7C15ULL;
    double norm0 = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        const double val = static_cast<double>( static_cast<keel::i64>(seed * 0x2545F4914F6CDD1DULL) )
                           * (1.0 / 9223372036854775808.0);
        q0[i] = val;
        norm0 += val * val;
    }
    norm0 = std::sqrt(norm0);
    for (keel::usize i = 0; i < n; ++i) q0[i] /= norm0;

    keel::usize mActual = 0;
    for (keel::usize j = 0; j < m; ++j) {
        const double *qj = Q.data() + j * n;
        // w = H * qj = R_A^-T * (B * (R_A^-1 * qj))
        solveR(qj, zBuf.data());
        matvecB(zBuf.data(), uBuf.data());
        solveRT(uBuf.data(), wBuf.data());

        double aj = 0.0;
        for (keel::usize i = 0; i < n; ++i) aj += qj[i] * wBuf[i];
        alpha[j] = aj;

        // Two-pass DGKS full reorthogonalization against Q(:, 0..j)
        for (int pass = 0; pass < 2; ++pass) {
            for (keel::usize r = 0; r <= j; ++r) {
                const double *qr = Q.data() + r * n;
                double dot = 0.0;
                for (keel::usize i = 0; i < n; ++i) dot += qr[i] * wBuf[i];
                for (keel::usize i = 0; i < n; ++i) wBuf[i] -= dot * qr[i];
            }
        }

        double bnorm = 0.0;
        for (keel::usize i = 0; i < n; ++i) bnorm += wBuf[i] * wBuf[i];
        bnorm = std::sqrt(bnorm);
        beta[j] = bnorm;
        mActual = j + 1;

        if (j + 1 < m) {
            double *qNext = Q.data() + (j + 1) * n;
            if (bnorm > 1e-14) {
                const double invB = 1.0 / bnorm;
                for (keel::usize i = 0; i < n; ++i) qNext[i] = wBuf[i] * invB;
            } else {
                // Invariant subspace reached; generate a new orthogonal direction.
                for (keel::usize i = 0; i < n; ++i) qNext[i] = ((i + j) % 2 == 0) ? 1.0 : -1.0;
                for (int pass = 0; pass < 2; ++pass) {
                    for (keel::usize r = 0; r <= j; ++r) {
                        const double *qr = Q.data() + r * n;
                        double dot = 0.0;
                        for (keel::usize i = 0; i < n; ++i) dot += qr[i] * qNext[i];
                        for (keel::usize i = 0; i < n; ++i) qNext[i] -= dot * qr[i];
                    }
                }
                double nrm = 0.0;
                for (keel::usize i = 0; i < n; ++i) nrm += qNext[i] * qNext[i];
                nrm = std::sqrt(nrm);
                if (nrm <= 1e-14) break;
                for (keel::usize i = 0; i < n; ++i) qNext[i] /= nrm;
            }
        }
    }

    if (mActual < k) return false;

    // Solve the mActual x mActual symmetric tridiagonal eigenproblem T * S = S * diag(mu).
    keel::Array<double> S(mActual * mActual, scratch.allocator());
    std::fill(S.data(), S.data() + mActual * mActual, 0.0);
    for (keel::usize i = 0; i < mActual; ++i) {
        S[i + i * mActual] = alpha[i];
        if (i + 1 < mActual) {
            S[(i + 1) + i * mActual] = beta[i];
            S[i + (i + 1) * mActual] = beta[i];
        }
    }
    keel::Array<double> dTrid(mActual, scratch.allocator());
    symTridiagQL(S.data(), mActual, dTrid.data(), scratch);

    // Pick the k Ritz pairs with largest |mu| (which map to smallest |lambda| = 1/|mu|).
    double muMax = 0.0;
    for (keel::usize i = 0; i < mActual; ++i)
        muMax = std::max(muMax, std::fabs(dTrid[i]));
    const double muTol = 1e-12 * std::max(1.0, muMax);

    std::vector<keel::usize> idx(mActual);
    for (keel::usize i = 0; i < mActual; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](keel::usize a, keel::usize b) {
        return std::fabs(dTrid[a]) > std::fabs(dTrid[b]);
    });
    idx.resize(k);

    // Sort selected k eigenvalues in ascending algebraic order of lambda = 1/mu (matching MATLAB eigs).
    auto lambdaOf = [&](keel::usize r) {
        const double mu = dTrid[r];
        return (std::fabs(mu) > muTol) ? (1.0 / mu) : std::numeric_limits<double>::infinity();
    };
    std::sort(idx.begin(), idx.end(), [&](keel::usize a, keel::usize b) {
        return lambdaOf(a) < lambdaOf(b);
    });

    auto V = Value::matrix(n, k, ValueType::DOUBLE, mr);
    auto D = Value::matrix(k, k, ValueType::DOUBLE, mr);
    double *vd = V.doubleDataMut();
    double *dd = D.doubleDataMut();
    std::fill(dd, dd + k * k, 0.0);

    for (keel::usize col = 0; col < k; ++col) {
        const keel::usize r = idx[col];
        const double mu = dTrid[r];
        dd[col + col * k] = lambdaOf(r);

        // Compute y = Q(:, 0..mActual-1) * S(:, r)
        const double *sr = S.data() + r * mActual;
        std::fill(zBuf.data(), zBuf.data() + n, 0.0);
        for (keel::usize j = 0; j < mActual; ++j) {
            const double sj = sr[j];
            const double *qj = Q.data() + j * n;
            for (keel::usize i = 0; i < n; ++i)
                zBuf[i] += qj[i] * sj;
        }
        // Solve R_A * v_col = y
        double *vcol = vd + col * n;
        solveR(zBuf.data(), vcol);

        // Normalize v_col:
        // If B has positive diagonal and mu > 0, normalize to unit B-norm (v^T B v = 1).
        // Note: v^T B v = y^T H y = mu * (y^T y) = mu!
        if (bDiagAllPos && mu > muTol) {
            matvecB(vcol, uBuf.data());
            double bNormSq = 0.0;
            for (keel::usize i = 0; i < n; ++i) bNormSq += vcol[i] * uBuf[i];
            if (bNormSq > 0.0) {
                const double invNorm = 1.0 / std::sqrt(bNormSq);
                for (keel::usize i = 0; i < n; ++i) vcol[i] *= invNorm;
            }
        } else {
            // Indefinite/singular B (e.g. buckling): normalize so largest-magnitude entry is +1.0.
            keel::usize iMax = 0;
            double maxAbs = 0.0;
            for (keel::usize i = 0; i < n; ++i) {
                const double a = std::fabs(vcol[i]);
                if (a > maxAbs) { maxAbs = a; iMax = i; }
            }
            if (maxAbs > 0.0) {
                const double scale = vcol[iMax];
                for (keel::usize i = 0; i < n; ++i) vcol[i] /= scale;
            }
        }
    }

    Vout = detail::narrowDoubleToSingleIf(V, wantSingle, mr);
    Dout = detail::narrowDoubleToSingleIf(D, wantSingle, mr);
    return true;
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Hessenberg reduction
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

namespace {

// In-place Hessenberg reduction via Householder reflectors.
template <typename T>
void hessReduceInplace(T *A, keel::usize n, T *P,
                       keel::Allocator *mr)
{
    std::fill(P, P + n * n, T(0));
    for (keel::usize i = 0; i < n; ++i) P[i + i * n] = T(1);
    if (n < 3) return;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<T> v_storage(n, scratch.allocator());
    T *v = v_storage.data();

    for (keel::usize k = 0; k + 2 < n; ++k) {
        double norm_sq = 0.0;
        for (keel::usize i = k + 1; i < n; ++i)
            norm_sq += detail::abs_sq(A[i + k * n]);
        if (norm_sq == 0.0) continue;
        const T xk = A[k + 1 + k * n];
        const double norm = std::sqrt(norm_sq);

        T alpha;
        if constexpr (detail::is_complex_v<T>) {
            const double abs_xk = keel::abs(xk);
            const T phase = (abs_xk > 0.0) ? (xk / abs_xk) : T(1.0, 0.0);
            alpha = -phase * norm;
        } else {
            alpha = (xk >= 0.0) ? -norm : norm;
        }

        v[k + 1] = xk - alpha;
        for (keel::usize i = k + 2; i < n; ++i) v[i] = A[i + k * n];
        double v_norm_sq = 0.0;
        for (keel::usize i = k + 1; i < n; ++i) v_norm_sq += detail::abs_sq(v[i]);
        if (v_norm_sq == 0.0) continue;
        const T tau = T(2.0 / v_norm_sq);

        for (keel::usize j = k; j < n; ++j) {
            T dot = T(0);
            for (keel::usize i = k + 1; i < n; ++i) {
                if constexpr (detail::is_complex_v<T>) {
                    dot += keel::conj(v[i]) * A[i + j * n];
                } else {
                    dot += v[i] * A[i + j * n];
                }
            }
            const T s = tau * dot;
            for (keel::usize i = k + 1; i < n; ++i)
                A[i + j * n] -= s * v[i];
        }
        for (keel::usize i = 0; i < n; ++i) {
            T dot = T(0);
            for (keel::usize j = k + 1; j < n; ++j)
                dot += A[i + j * n] * v[j];
            const T s = tau * dot;
            for (keel::usize j = k + 1; j < n; ++j) {
                if constexpr (detail::is_complex_v<T>) {
                    A[i + j * n] -= s * keel::conj(v[j]);
                } else {
                    A[i + j * n] -= s * v[j];
                }
            }
        }
        for (keel::usize i = 0; i < n; ++i) {
            T dot = T(0);
            for (keel::usize j = k + 1; j < n; ++j)
                dot += P[i + j * n] * v[j];
            const T s = tau * dot;
            for (keel::usize j = k + 1; j < n; ++j) {
                if constexpr (detail::is_complex_v<T>) {
                    P[i + j * n] -= s * keel::conj(v[j]);
                } else {
                    P[i + j * n] -= s * v[j];
                }
            }
        }
        A[k + 1 + k * n] = alpha;
        for (keel::usize i = k + 2; i < n; ++i)
            A[i + k * n] = T(0);
    }
}

void complexSchurQR(keel::c64 *H, keel::c64 *Z, keel::usize n)
{
    if (n < 2) return;
    auto h = [&](keel::usize i, keel::usize j) -> keel::c64 & { return H[i + j * n]; };
    auto z = [&](keel::usize i, keel::usize j) -> keel::c64 & { return Z[i + j * n]; };

    const double eps = std::numeric_limits<double>::epsilon();
    const int N = static_cast<int>(n);

    int p = N - 1;
    int iter = 0;

    while (p > 0) {
        int l = p;
        while (l > 0) {
            double s = keel::abs(h(l - 1, l - 1)) + keel::abs(h(l, l));
            if (s == 0.0) s = 1.0;
            if (keel::abs(h(l, l - 1)) <= eps * s) {
                h(l, l - 1) = keel::c64(0.0, 0.0);
                break;
            }
            --l;
        }
        if (l == p) { p -= 1; iter = 0; continue; }
        if (++iter > 60 * (N + 1))  // LAPACK-style budget; unreachable when converged
            throw Error("eig: QR iteration failed to converge",
                        0, 0, "eig", "", "numkit:eig:qrNoConverge");

        // Wilkinson shift from trailing 2x2 block: H[p-1..p, p-1..p]
        keel::c64 a = h(p - 1, p - 1);
        keel::c64 b = h(p - 1, p);
        keel::c64 c_blk = h(p, p - 1);
        keel::c64 d = h(p, p);

        keel::c64 tr = a + d;
        keel::c64 det = a * d - b * c_blk;
        keel::c64 disc = keel::sqrt(tr * tr - 4.0 * det);

        keel::c64 mu1 = 0.5 * (tr + disc);
        keel::c64 mu2 = 0.5 * (tr - disc);

        keel::c64 shift = (keel::abs(mu1 - d) < keel::abs(mu2 - d)) ? mu1 : mu2;

        if (iter % 30 == 0) {
            shift += keel::c64(keel::abs(c_blk), keel::abs(h(p - 1, std::max(0, p - 2))));
        }

        for (int k = l; k < p; ++k) {
            keel::c64 f = (k == l) ? (h(k, k) - shift) : h(k, k - 1);
            // k > l chases the BULGE created by the previous right rotation:
            // it sits at h(k+1, k-1), NOT at h(k+1, k). Reading the wrong
            // column built a Givens that never annihilated the bulge â€” the
            // iteration budget ran out and the forced triangularisation
            // emitted garbage eigenvalues (bugs: linalg/eig-vector-selection-ahp).
            keel::c64 g = (k == l) ? h(k + 1, k) : h(k + 1, k - 1);

            double norm = std::sqrt(detail::abs_sq(f) + detail::abs_sq(g));
            if (norm == 0.0) continue;

            double c = keel::abs(f) / norm;
            keel::c64 phase = (keel::abs(f) > 0.0) ? (f / keel::abs(f)) : keel::c64(1.0, 0.0);
            keel::c64 s = phase * keel::conj(g) / norm;

            int col0 = (k > 0) ? (k - 1) : 0;
            for (int j = col0; j < N; ++j) {
                keel::c64 hk = h(k, j);
                keel::c64 hk1 = h(k + 1, j);
                h(k, j)     = c * hk + s * hk1;
                h(k + 1, j) = -keel::conj(s) * hk + c * hk1;
            }
            int row1 = std::min(k + 2, N - 1);
            for (int i = 0; i <= row1; ++i) {
                keel::c64 hk = h(i, k);
                keel::c64 hk1 = h(i, k + 1);
                h(i, k)     = c * hk + keel::conj(s) * hk1;
                h(i, k + 1) = -s * hk + c * hk1;
            }
            for (int i = 0; i < N; ++i) {
                keel::c64 zk = z(i, k);
                keel::c64 zk1 = z(i, k + 1);
                z(i, k)     = c * zk + keel::conj(s) * zk1;
                z(i, k + 1) = -s * zk + c * zk1;
            }
        }
    }

    for (int j = 0; j + 1 < N; ++j)
        for (int i = j + 1; i < N; ++i)
            h(static_cast<keel::usize>(i), static_cast<keel::usize>(j)) = keel::c64(0.0, 0.0);
}

} // anonymous namespace

std::tuple<Value, Value>
hess(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "hess");
    if (A.dims().ndim() != 2)
        throw Error("hess: input must be a 2D matrix",
                    0, 0, "hess", "", "numkit:hess:notMatrix");
    const keel::usize m = static_cast<keel::usize>(A.dims().dim(0));
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(1));
    if (m != n)
        throw Error("hess: matrix must be square",
                    0, 0, "hess", "", "numkit:hess:notSquare");
    if (A.isComplex()) {
        auto Hout = Value::complexMatrix(n, n, mr);
        auto Pout = Value::complexMatrix(n, n, mr);
        if (n == 0) return std::make_tuple(std::move(Pout), std::move(Hout));
        std::copy(A.complexData(), A.complexData() + n * n, Hout.complexDataMut());
        hessReduceInplace(Hout.complexDataMut(), n, Pout.complexDataMut(), mr);
        return std::make_tuple(detail::narrow_if_real(Pout, mr), detail::narrow_if_real(Hout, mr));
    }
    const bool wantSingle = (A.type() == ValueType::SINGLE);
    if (n == 0) {
        return std::make_tuple(
            Value::matrix(0, 0, A.type(), mr),
            Value::matrix(0, 0, A.type(), mr));
    }
    Value Aw = detail::widenSingleToDouble(A, mr);
    auto Hout = Value::matrix(n, n, ValueType::DOUBLE, mr);
    auto Pout = Value::matrix(n, n, ValueType::DOUBLE, mr);
    std::copy(Aw.doubleData(), Aw.doubleData() + n * n, Hout.doubleDataMut());
    hessReduceInplace(Hout.doubleDataMut(), n, Pout.doubleDataMut(), mr);
    return std::make_tuple(
        detail::narrowDoubleToSingleIf(Pout, wantSingle, mr),
        detail::narrowDoubleToSingleIf(Hout, wantSingle, mr));
}

Value hess_H_only(const Value &A, keel::Allocator *mr)
{
    auto [P, H] = hess(A, mr);
    return H;
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Schur (symmetric) + Sylvester (symmetric A, B)
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

std::tuple<Value, Value>
schur_sym(const Value &A, keel::Allocator *mr)
{
    // For symmetric A, Schur decomposition is the same as eig. The schur
    // gate voices this entry (not eig) for non-float inputs and shape
    // errors; eig_symmetric's own checks become unreachable no-ops.
    detail::requireDoubleInput(A, "schur");
    if (A.dims().ndim() != 2)
        throw Error("schur: input must be a 2D matrix",
                    0, 0, "schur", "", "numkit:schur:notMatrix");
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(0));
    if (n != static_cast<keel::usize>(A.dims().dim(1)))
        throw Error("schur: matrix must be square",
                    0, 0, "schur", "", "numkit:schur:notSquare");
    return eig_symmetric(A, mr);
}

namespace {

// Francis double-shift QR on an upper-Hessenberg H (nÃ—n, column-major), with
// the orthogonal accumulator Z (pre-seeded with the Hessenberg transform P).
// On exit H is the real (quasi-upper-triangular) Schur form T â€” 1Ã—1 blocks for
// real eigenvalues, 2Ã—2 blocks for complex-conjugate pairs â€” and A == ZÂ·TÂ·Záµ€.
// Classical bulge-chasing implicit double-shift QR (Golub & Van Loan Â§7.5).
void francisSchur(double *H, double *Z, keel::usize n)
{
    if (n < 3) return;   // <=2x2 is already (quasi-)triangular
    auto h = [&](keel::usize i, keel::usize j) -> double & { return H[i + j * n]; };
    auto z = [&](keel::usize i, keel::usize j) -> double & { return Z[i + j * n]; };
    const double eps = std::numeric_limits<double>::epsilon();
    const int    N   = static_cast<int>(n);

    double hnorm = 0.0;
    for (keel::usize j = 0; j < n; ++j) {
        double colSum = 0.0;
        const keel::usize iEnd = std::min(n, j + 2);
        for (keel::usize i = 0; i < iEnd; ++i)
            colSum += std::fabs(h(i, j));
        if (colSum > hnorm) hnorm = colSum;
    }
    if (hnorm == 0.0) return;
    const double smlnum = std::numeric_limits<double>::min() * (static_cast<double>(n) / eps);

    int p = N - 1;          // bottom of the active (un-deflated) block
    int iter = 0;
    while (p >= 0) {
        // Deflation: smallest l so the active block is H[l..p, l..p].
        int l = p;
        while (l > 0) {
            double s = std::fabs(h(l - 1, l - 1)) + std::fabs(h(l, l));
            if (s == 0.0) s = hnorm;
            if (std::fabs(h(l, l - 1)) <= std::max(eps * s, smlnum)) {
                h(l, l - 1) = 0.0;
                break;
            }
            --l;
        }
        if (l == p)       { p -= 1; iter = 0; continue; }   // 1x1 converged
        if (l == p - 1)   { p -= 2; iter = 0; continue; }   // 2x2 converged
        if (++iter > 600) { p -= 2; iter = 0; continue; }   // force deflation of stubborn block

        // Double shift from the trailing 2x2: trace s, determinant t.
        double s = h(p - 1, p - 1) + h(p, p);
        double t = h(p - 1, p - 1) * h(p, p) - h(p - 1, p) * h(p, p - 1);
        if (iter % 10 == 0) {           // LAPACK dlahqr exceptional shift to break symmetry cycles
            const double e = std::fabs(h(p, p - 1)) + (p >= 2 ? std::fabs(h(p - 1, p - 2)) : 0.0);
            const double d1 = h(p, p) + 0.75 * e;
            s = 2.0 * d1;
            t = d1 * d1 - 0.4375 * e * e;
        }
        // First column of (H^2 - sH + tI) on the active block - bulge seed.
        double x = h(l, l) * h(l, l) + h(l, l + 1) * h(l + 1, l) - s * h(l, l) + t;
        double y = h(l + 1, l) * (h(l, l) + h(l + 1, l + 1) - s);
        double w = (l + 2 <= p) ? h(l + 1, l) * h(l + 2, l + 1) : 0.0;

        for (int k = l; k <= p - 1; ++k) {
            const int r = (k <= p - 2) ? 3 : 2;     // reflector size (3 mid-chase, 2 at the end)
            // Householder that zeroes (y[,w]) below x.
            double scale = std::fabs(x) + std::fabs(y) + (r == 3 ? std::fabs(w) : 0.0);
            if (scale == 0.0) continue;
            double xs = x / scale, ys = y / scale, ws = (r == 3 ? w / scale : 0.0);
            double alpha = std::sqrt(xs * xs + ys * ys + ws * ws);
            if (xs < 0.0) alpha = -alpha;
            double v0 = xs + alpha, v1 = ys, v2 = ws;
            const double vnorm2 = v0 * v0 + v1 * v1 + v2 * v2;
            if (vnorm2 == 0.0) continue;
            const double tau = 2.0 / vnorm2;

            const int col0 = (k > l) ? k - 1 : l;   // first affected column
            // Left-apply Háµ¤ to rows k..k+r-1 of H (cols col0..N-1).
            for (int j = col0; j < N; ++j) {
                double d = v0 * h(k, j) + v1 * h(k + 1, j) + (r == 3 ? v2 * h(k + 2, j) : 0.0);
                d *= tau;
                h(k, j)     -= d * v0;
                h(k + 1, j) -= d * v1;
                if (r == 3) h(k + 2, j) -= d * v2;
            }
            // Right-apply Háµ¤ to cols k..k+r-1 of H (rows 0..min(k+r,p)).
            const int rowEnd = std::min(k + r, p);
            for (int i = 0; i <= rowEnd; ++i) {
                double d = v0 * h(i, k) + v1 * h(i, k + 1) + (r == 3 ? v2 * h(i, k + 2) : 0.0);
                d *= tau;
                h(i, k)     -= d * v0;
                h(i, k + 1) -= d * v1;
                if (r == 3) h(i, k + 2) -= d * v2;
            }
            // Accumulate into Z (cols k..k+r-1, all rows).
            for (int i = 0; i < N; ++i) {
                double d = v0 * z(i, k) + v1 * z(i, k + 1) + (r == 3 ? v2 * z(i, k + 2) : 0.0);
                d *= tau;
                z(i, k)     -= d * v0;
                z(i, k + 1) -= d * v1;
                if (r == 3) z(i, k + 2) -= d * v2;
            }
            // Next bulge element.
            x = h(k + 1, k);
            y = h(k + 2, k);
            w = (k + 3 <= p) ? h(k + 3, k) : 0.0;
        }
    }
    // Zero the numerical dust strictly below the quasi-triangular structure.
    for (int j = 0; j + 2 < N; ++j)
        for (int i = j + 2; i < N; ++i)
            h(static_cast<keel::usize>(i), static_cast<keel::usize>(j)) = 0.0;
}

// Standardize every 2Ã—2 diagonal block of the (quasi-triangular) Schur form:
// a block with REAL eigenvalues is triangularized (subdiagonal â†’ 0) via a Givens
// similarity (accumulated into Z); a complex-conjugate pair is left as a valid
// 2Ã—2 block. LAPACK dlanv2 logic. Also covers the n==2 input that francisSchur
// skips. H/Z column-major, nÃ—n.
void standardizeSchur2x2(double *H, double *Z, keel::usize n)
{
    if (n < 2) return;
    auto h = [&](keel::usize i, keel::usize j) -> double & { return H[i + j * n]; };
    auto z = [&](keel::usize i, keel::usize j) -> double & { return Z[i + j * n]; };
    const int N = static_cast<int>(n);

    for (int k = 0; k + 1 < N; ++k) {
        if (k + 1 < N && h(k + 1, k) == 0.0) continue;      // 1Ã—1 block
        if (k + 2 < N && h(k + 2, k + 1) != 0.0) continue;  // part of a larger active region (shouldn't happen post-QR)

        double a = h(k, k), b = h(k, k + 1), c = h(k + 1, k), d = h(k + 1, k + 1);
        if (c == 0.0) continue;

        double cs = 1.0, sn = 0.0;
        const double p = 0.5 * (a - d);
        const double bcmax = std::max(std::fabs(b), std::fabs(c));
        const double bcmis = std::min(std::fabs(b), std::fabs(c)) *
                             (b >= 0.0 ? 1.0 : -1.0) * (c >= 0.0 ? 1.0 : -1.0);
        const double scale = std::max(std::fabs(p), bcmax);
        double zz = (p / scale) * p + (bcmax / scale) * bcmis;

        if (zz >= 0.0) {
            // Real eigenvalues â†’ triangularize.
            double zr = p + (p >= 0.0 ? 1.0 : -1.0) * std::sqrt(scale) * std::sqrt(zz);
            a = d + zr;
            d -= (bcmax / zr) * bcmis;
            const double tau = std::hypot(c, zr);
            cs = zr / tau;
            sn = c / tau;
            b = b - c;
            c = 0.0;
            h(k, k) = a; h(k, k + 1) = b; h(k + 1, k) = 0.0; h(k + 1, k + 1) = d;
            // Apply the rotation G = [cs sn; -sn cs] as a similarity to the rest.
            for (int j = k + 2; j < N; ++j) {       // rows k,k+1 (cols right of the block)
                const double t1 = h(k, j), t2 = h(k + 1, j);
                h(k, j)     =  cs * t1 + sn * t2;
                h(k + 1, j) = -sn * t1 + cs * t2;
            }
            for (int i = 0; i < k; ++i) {           // cols k,k+1 (rows above the block)
                const double t1 = h(i, k), t2 = h(i, k + 1);
                h(i, k)     =  cs * t1 + sn * t2;
                h(i, k + 1) = -sn * t1 + cs * t2;
            }
            for (int i = 0; i < N; ++i) {           // accumulate into Z
                const double t1 = z(i, k), t2 = z(i, k + 1);
                z(i, k)     =  cs * t1 + sn * t2;
                z(i, k + 1) = -sn * t1 + cs * t2;
            }
        }
        // Complex pair (zz<0): leave the 2Ã—2 block as the real Schur form.
        ++k;   // skip the partner row
    }
}

} // anonymous namespace

// General (nonsymmetric) real/complex Schur: A = UÂ·TÂ·Uáµ€, U unitary, T upper-triangular.
std::tuple<Value, Value>
schur_general(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "schur");
    if (A.dims().ndim() != 2)
        throw Error("schur: input must be a 2D matrix",
                    0, 0, "schur", "", "numkit:schur:notMatrix");
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(0));
    if (n != static_cast<keel::usize>(A.dims().dim(1)))
        throw Error("schur: matrix must be square",
                    0, 0, "schur", "", "numkit:schur:notSquare");
    if (A.isComplex()) {
        auto U = Value::complexMatrix(n, n, mr);
        auto T = Value::complexMatrix(n, n, mr);
        if (n == 0) return std::make_tuple(std::move(U), std::move(T));
        std::copy(A.complexData(), A.complexData() + n * n, T.complexDataMut());
        hessReduceInplace(T.complexDataMut(), n, U.complexDataMut(), mr);
        complexSchurQR(T.complexDataMut(), U.complexDataMut(), n);
        return std::make_tuple(detail::narrow_if_real(U, mr), detail::narrow_if_real(T, mr));
    }

    const bool wantSingle = (A.type() == ValueType::SINGLE);
    if (n == 0) {
        return std::make_tuple(
            Value::matrix(0, 0, A.type(), mr),
            Value::matrix(0, 0, A.type(), mr));
    }
    Value Aw = detail::widenSingleToDouble(A, mr);
    auto U = Value::matrix(n, n, ValueType::DOUBLE, mr);
    auto T = Value::matrix(n, n, ValueType::DOUBLE, mr);
    std::copy(Aw.doubleData(), Aw.doubleData() + n * n, T.doubleDataMut());
    hessReduceInplace(T.doubleDataMut(), n, U.doubleDataMut(), mr);
    francisSchur(T.doubleDataMut(), U.doubleDataMut(), n);
    standardizeSchur2x2(T.doubleDataMut(), U.doubleDataMut(), n);
    return std::make_tuple(
        detail::narrowDoubleToSingleIf(U, wantSingle, mr),
        detail::narrowDoubleToSingleIf(T, wantSingle, mr));
}

Value sylvester_sym(const Value &A, const Value &B, const Value &C,
                    keel::Allocator *mr)
{
    if (A.dims().ndim() != 2 || B.dims().ndim() != 2 || C.dims().ndim() != 2)
        throw Error("sylvester: A, B, C must be 2D matrices",
                    0, 0, "sylvester", "", "numkit:sylvester:notMatrix");
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(0));
    const keel::usize m = static_cast<keel::usize>(B.dims().dim(0));
    if (A.dims().dim(0) != A.dims().dim(1))
        throw Error("sylvester: A must be square",
                    0, 0, "sylvester", "", "numkit:sylvester:badA");
    if (B.dims().dim(0) != B.dims().dim(1))
        throw Error("sylvester: B must be square",
                    0, 0, "sylvester", "", "numkit:sylvester:badB");
    if (C.dims().dim(0) != static_cast<int>(n) ||
        C.dims().dim(1) != static_cast<int>(m))
        throw Error("sylvester: C must be n Ã— m where A is nÃ—n, B is mÃ—m",
                    0, 0, "sylvester", "", "numkit:sylvester:badC");

    const bool wantSingle = (A.type() == ValueType::SINGLE ||
                             B.type() == ValueType::SINGLE ||
                             C.type() == ValueType::SINGLE);
    Value Aw = detail::widenSingleToDouble(A, mr);
    Value Bw = detail::widenSingleToDouble(B, mr);
    Value Cw = detail::widenSingleToDouble(C, mr);

    auto [Va, Da] = eig_symmetric(Aw, mr);   // throws if non-sym
    auto [Vb, Db] = eig_symmetric(Bw, mr);   // throws if non-sym

    const double *Vad = Va.doubleData();
    const double *Dad = Da.doubleData();
    const double *Vbd = Vb.doubleData();
    const double *Dbd = Db.doubleData();
    const double *Cd  = Cw.doubleData();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> Y(n * m, scratch.allocator());
    keel::Array<double> tmp(n * m, scratch.allocator());
    // tmp = Va' * C
    for (keel::usize i = 0; i < n; ++i)
        for (keel::usize j = 0; j < m; ++j) {
            double s = 0.0;
            for (keel::usize k = 0; k < n; ++k)
                s += Vad[k + i * n] * Cd[k + j * n];
            tmp[i + j * n] = s;
        }
    // Y = tmp * Vb
    for (keel::usize i = 0; i < n; ++i)
        for (keel::usize j = 0; j < m; ++j) {
            double s = 0.0;
            for (keel::usize k = 0; k < m; ++k)
                s += tmp[i + k * n] * Vbd[k + j * m];
            Y[i + j * n] = s;
        }

    // y_ij /= (d_a_i + d_b_j)
    for (keel::usize i = 0; i < n; ++i) {
        const double dai = Dad[i + i * n];
        for (keel::usize j = 0; j < m; ++j) {
            const double dbj = Dbd[j + j * m];
            const double denom = dai + dbj;
            if (std::fabs(denom) < 1e-300)
                throw Error("sylvester: A and -B share an eigenvalue (no unique solution)",
                            0, 0, "sylvester", "", "numkit:sylvester:singular");
            Y[i + j * n] /= denom;
        }
    }

    auto out = Value::matrix(n, m, ValueType::DOUBLE, mr);
    double *X = out.doubleDataMut();
    // tmp = Va * Y
    for (keel::usize i = 0; i < n; ++i)
        for (keel::usize j = 0; j < m; ++j) {
            double s = 0.0;
            for (keel::usize k = 0; k < n; ++k)
                s += Vad[i + k * n] * Y[k + j * n];
            tmp[i + j * n] = s;
        }
    // X = tmp * Vb'
    for (keel::usize i = 0; i < n; ++i)
        for (keel::usize j = 0; j < m; ++j) {
            double s = 0.0;
            for (keel::usize k = 0; k < m; ++k)
                s += tmp[i + k * n] * Vbd[j + k * m];
            X[i + j * n] = s;
        }
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// polyeig â€” polynomial eigenvalue problem via companion linearisation
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

namespace {

// Build the (kÂ·n Ã— kÂ·n) companion matrix L for the polynomial
// p(Î») = A0 + Î»Â·A1 + â€¦ + Î»^kÂ·Ak. L has the block form
//
//   L = | 0      I        0      â€¦  0      |
//       | 0      0        I      â€¦  0      |
//       | â€¦                                 |
//       | 0      0        0      â€¦  I      |
//       | -B0    -B1      -B2    â€¦  -B_{k-1} |
//
// where Bi = inv(Ak) * Ai. Eigenvalues of L equal the polynomial
// eigenvalues; right eigenvectors have Kronecker structure
// [x; Î» x; Î»Â² x; â€¦; Î»^{k-1} x], so the first n rows yield x.
Value buildCompanion(keel::ArrayRef<Value> coeffs, keel::Allocator *mr)
{
    const keel::usize k = coeffs.size() - 1;          // degree
    const keel::usize n = static_cast<keel::usize>(coeffs[0].dims().dim(0));
    const keel::usize N = k * n;

    // B = inv(Ak) * Ai  for i = 0..k-1.
    // Compute via la_solve once per i â€” cheaper than forming inv(Ak)
    // explicitly when n is moderate. Here we do form inv() for clarity.
    Value Ak_w = detail::widenSingleToDouble(coeffs[k], mr);
    Value Ak_inv = inv(Ak_w, mr);

    auto L = Value::matrix(N, N, ValueType::DOUBLE, mr);
    double *Ld = L.doubleDataMut();
    std::fill(Ld, Ld + N * N, 0.0);

    // Upper-diagonal blocks: L[i*n .. (i+1)*n - 1, (i+1)*n .. (i+2)*n - 1] = I.
    for (keel::usize i = 0; i + 1 < k; ++i) {
        for (keel::usize r = 0; r < n; ++r) {
            const keel::usize row = i * n + r;
            const keel::usize col = (i + 1) * n + r;
            Ld[row + col * N] = 1.0;
        }
    }

    // Bottom row of blocks: -Ak_inv * Ai for i = 0..k-1.
    // Compute block product directly into L.
    const double *Akid = Ak_inv.doubleData();
    for (keel::usize i = 0; i < k; ++i) {
        Value Ai_w = detail::widenSingleToDouble(coeffs[i], mr);
        const double *Aid = Ai_w.doubleData();
        for (keel::usize r = 0; r < n; ++r)
            for (keel::usize c = 0; c < n; ++c) {
                double s = 0.0;
                for (keel::usize p = 0; p < n; ++p)
                    s += Akid[r + p * n] * Aid[p + c * n];
                const keel::usize row = (k - 1) * n + r;
                const keel::usize col = i * n + c;
                Ld[row + col * N] = -s;
            }
    }
    return L;
}

bool anySingleCoeffs(keel::ArrayRef<Value> coeffs)
{
    for (const auto &c : coeffs)
        if (c.type() == ValueType::SINGLE) return true;
    return false;
}

void validatePolyeigCoeffs(keel::ArrayRef<Value> coeffs)
{
    if (coeffs.size() < 2)
        throw Error("polyeig: requires at least 2 coefficient matrices (A0, A1, ...)",
                    0, 0, "polyeig", "", "numkit:polyeig:nargin");
    const keel::usize n = static_cast<keel::usize>(coeffs[0].dims().dim(0));
    if (n != static_cast<keel::usize>(coeffs[0].dims().dim(1)))
        throw Error("polyeig: coefficient matrices must be square",
                    0, 0, "polyeig", "", "numkit:polyeig:notSquare");
    for (keel::usize i = 1; i < coeffs.size(); ++i) {
        if (coeffs[i].dims().ndim() != 2
            || static_cast<keel::usize>(coeffs[i].dims().dim(0)) != n
            || static_cast<keel::usize>(coeffs[i].dims().dim(1)) != n)
            throw Error("polyeig: all coefficient matrices must be n Ã— n with matching n",
                        0, 0, "polyeig", "", "numkit:polyeig:shapeMismatch");
    }
}

} // anonymous namespace

Value polyeig_values(keel::ArrayRef<Value> coeffs, keel::Allocator *mr)
{
    validatePolyeigCoeffs(coeffs);
    Value L = buildCompanion(coeffs, mr);
    // Eigenvalues of L are the polynomial eigenvalues.
    // The companion is generally non-symmetric â†’ use char-poly path
    // (returns complex column when needed).
    return detail::narrowDoubleToSingleIf(eig_general_values(L, mr), anySingleCoeffs(coeffs), mr);
}

std::tuple<Value, Value>
polyeig_VE(keel::ArrayRef<Value> coeffs, keel::Allocator *mr)
{
    validatePolyeigCoeffs(coeffs);
    const keel::usize k = coeffs.size() - 1;
    const keel::usize n = static_cast<keel::usize>(coeffs[0].dims().dim(0));
    const keel::usize N = k * n;

    Value L = buildCompanion(coeffs, mr);
    auto [V_big, D_big] = eig_general_VD(L, mr);   // throws on complex eigvals

    auto V = Value::matrix(n, N, ValueType::DOUBLE, mr);
    auto e = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    double *Vd = V.doubleDataMut();
    double *ed = e.doubleDataMut();
    const double *Vbd = V_big.doubleData();
    const double *Dbd = D_big.doubleData();

    for (keel::usize j = 0; j < N; ++j) {
        ed[j] = Dbd[j + j * N];
        for (keel::usize r = 0; r < n; ++r)
            Vd[r + j * n] = Vbd[r + j * N];
    }
    const bool wantSingle = anySingleCoeffs(coeffs);
    return std::make_tuple(
        detail::narrowDoubleToSingleIf(V, wantSingle, mr),
        detail::narrowDoubleToSingleIf(e, wantSingle, mr));
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// ordeig â€” read eigenvalues off a (quasi-)triangular Schur factor
// in their stored order (no sorting)
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

Value ordeig(const Value &T, keel::Allocator *mr)
{
    detail::requireDoubleInput(T, "ordeig", "T");
    if (T.dims().ndim() != 2)
        throw Error("ordeig: T must be a 2D matrix",
                    0, 0, "ordeig", "", "numkit:ordeig:notMatrix");
    const keel::usize n = static_cast<keel::usize>(T.dims().dim(0));
    if (n != static_cast<keel::usize>(T.dims().dim(1)))
        throw Error("ordeig: T must be square",
                    0, 0, "ordeig", "", "numkit:ordeig:notSquare");
    if (n == 0)
        return Value::matrix(0, 1, T.type(), mr);

    if (T.isComplex()) {
        const keel::c64 *Tc = T.complexData();
        bool any_complex = false;
        for (keel::usize i = 0; i < n; ++i)
            if (Tc[i + i * n].imag() != 0.0) { any_complex = true; break; }
        if (any_complex) {
            auto out = Value::matrix(n, 1, ValueType::COMPLEX, mr);
            keel::c64 *od = out.complexDataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = Tc[i + i * n];
            return out;
        }
        auto out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (keel::usize i = 0; i < n; ++i) od[i] = Tc[i + i * n].real();
        return out;
    }

    const bool wantSingle = (T.type() == ValueType::SINGLE);
    Value Tw = detail::widenSingleToDouble(T, mr);
    const double *Td = Tw.doubleData();

    // First pass: detect 2Ã—2 blocks (non-zero sub-diagonal entries
    // T(i+1, i)). Eigenvalues from them are complex conjugate pairs.
    // Real diagonal entries pass through.
    bool any_complex = false;
    std::vector<keel::c64> out_complex;
    out_complex.reserve(n);

    keel::usize i = 0;
    while (i < n) {
        const double sub = (i + 1 < n) ? Td[(i + 1) + i * n] : 0.0;
        if (std::fabs(sub) <= 1e-14 * (1.0 + std::fabs(Td[i + i * n]))) {
            // Real eigenvalue (diagonal entry).
            out_complex.emplace_back(Td[i + i * n], 0.0);
            ++i;
        } else {
            // 2Ã—2 block at (i, i+1):
            //   [[a c]; [d a']] â†’ eigvals = ((a + a') Â± âˆš((a âˆ’ a')Â² + 4cd)) / 2
            const double a  = Td[i + i * n];
            const double ap = Td[(i + 1) + (i + 1) * n];
            const double c  = Td[i + (i + 1) * n];
            const double d  = sub;
            const double disc_re = (a - ap) * (a - ap) + 4.0 * c * d;
            const double mean = 0.5 * (a + ap);
            if (disc_re >= 0.0) {
                const double half_sqrt = 0.5 * std::sqrt(disc_re);
                out_complex.emplace_back(mean + half_sqrt, 0.0);
                out_complex.emplace_back(mean - half_sqrt, 0.0);
            } else {
                const double half_imag = 0.5 * std::sqrt(-disc_re);
                out_complex.emplace_back(mean,  half_imag);
                out_complex.emplace_back(mean, -half_imag);
                any_complex = true;
            }
            i += 2;
        }
    }

    if (any_complex) {
        auto out = Value::matrix(n, 1, ValueType::COMPLEX, mr);
        keel::c64 *od = out.complexDataMut();
        for (keel::usize k = 0; k < n; ++k) od[k] = out_complex[k];
        return out;
    }
    auto out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (keel::usize k = 0; k < n; ++k) od[k] = out_complex[k].real();
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

} // namespace numkit::linalg
