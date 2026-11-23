// toolboxes/linalg/src/eig.cpp
//
// Eigenvalue / Hessenberg / Schur / Sylvester / polyeig / ordeig —
// Value adapters over keel::linalg kernels.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/linalg.h>
#include <keel/math.h>
#include <numkit/linalg/eig.hpp>
#include "linalg_detail.hpp"

#include <numkit/linalg/decompositions.hpp>
#include <numkit/linalg/properties.hpp>
#include <numkit/linalg/qz.hpp>
#include <numkit/builtin/polyfun.hpp>

// Compute-only TU: Value substrate + Error, no engine. The eig / hess /
// schur / sylvester / polyeig / ordeig builtins (CallContext wrappers)
// live in eig_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <keel/tuple.h>

namespace numkit::linalg {

// ────────────────────────────────────────────────────────────────────────
// Characteristic polynomial + general eig
// ────────────────────────────────────────────────────────────────────────

Value poly_of_matrix(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "poly");
    if (A.dims().ndim() != 2)
        throw Error("poly: input must be a 2D matrix",
                    0, 0, "poly", "", "numkit:poly:notMatrix");
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
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
        keel::f64 *c = out.doubleDataMut();
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

static bool isHermitianComplexApprox(const Value &A, keel::f64 tol = 1e-10)
{
    if (A.dims().ndim() != 2 || !A.isComplex()) return false;
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    if (m != n) return false;
    const keel::c64 *ad = A.complexData();
    keel::f64 aMax = 1.0;
    for (keel::usize idx = 0; idx < n * n; ++idx) {
        const keel::f64 v = keel::abs(ad[idx]);
        if (v > aMax) aMax = v;
    }
    const keel::f64 thresh = tol * aMax;
    for (keel::usize i = 0; i < n; ++i) {
        if (keel::abs(ad[i + i * n].imag()) > thresh) return false;
        for (keel::usize j = i + 1; j < n; ++j) {
            if (keel::abs(ad[i + j * n] - keel::conj(ad[j + i * n])) > thresh)
                return false;
        }
    }
    return true;
}

namespace {

// Clean tiny complex phases on real eigenvectors/eigenvalues for real input matrices.
bool cleanRealSpectrum(keel::c64 *Vd, keel::c64 *Dd, keel::usize n, keel::f64 imagTol, bool allowNonFinite)
{
    bool realSpectrum = true;
    for (keel::usize k = 0; k < n; ++k) {
        const keel::c64 dkk = Dd[k + k * n];
        const bool isRealEig =
            (allowNonFinite && !keel::is_finite(dkk.real())) ||
            (keel::abs(dkk.imag()) <= imagTol * (allowNonFinite ? keel::max(1.0, keel::abs(dkk.real())) : 1.0));
        if (isRealEig) {
            Dd[k + k * n] = keel::c64(dkk.real(), 0.0);
            keel::usize iMax = 0;
            keel::f64 maxAbs = 0.0;
            for (keel::usize i = 0; i < n; ++i) {
                const keel::f64 a = keel::abs(Vd[i + k * n]);
                if (a > maxAbs) { maxAbs = a; iMax = i; }
            }
            if (maxAbs > 0.0) {
                const keel::c64 phase = keel::conj(Vd[iMax + k * n]) / maxAbs;
                keel::f64 colImax = 0.0;
                for (keel::usize i = 0; i < n; ++i) {
                    const keel::c64 rotated = Vd[i + k * n] * phase;
                    Vd[i + k * n] = rotated;
                    colImax = keel::max(colImax, keel::abs(rotated.imag()));
                }
                if (colImax <= 1e3 * imagTol) {
                    keel::f64 rNormSq = 0.0;
                    for (keel::usize i = 0; i < n; ++i) {
                        const keel::f64 re = Vd[i + k * n].real();
                        Vd[i + k * n] = keel::c64(re, 0.0);
                        rNormSq += re * re;
                    }
                    const keel::f64 rNorm = keel::sqrt(rNormSq);
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
    keel::f64 vImax = 0.0;
    for (keel::usize i = 0; i < n * n; ++i)
        vImax = keel::max(vImax, keel::abs(Vd[i].imag()));
    return realSpectrum && vImax <= 1e3 * imagTol;
}

} // anonymous namespace

Value eig_general_values(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "eig");
    if (A.dims().ndim() != 2)
        throw Error("eig: input must be a 2D matrix",
                    0, 0, "eig", "", "numkit:eig:notMatrix");
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(0));
    if (n != keel::as<keel::usize>(A.dims().dim(1)))
        throw Error("eig: matrix must be square",
                    0, 0, "eig", "", "numkit:eig:notSquare");
    if (n == 0) return Value::matrix(0, 1, A.type(), mr);

    const bool wantSingle = (A.type() == ValueType::SINGLE);
    Value Aw = detail::widenSingleToDouble(A, mr);

    auto [U, T] = schur_general(Aw, mr);

    if (isHermitianComplexApprox(Aw)) {
        auto out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        if (T.isComplex()) {
            const keel::c64 *td = T.complexData();
            for (keel::usize i = 0; i < n; ++i) od[i] = td[i + i * n].real();
        } else {
            const keel::f64 *td = T.doubleData();
            for (keel::usize i = 0; i < n; ++i) od[i] = td[i + i * n];
        }
        keel::sort(od, od + n);
        return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
    }

    auto out = Value::complexMatrix(n, 1, mr);
    keel::c64 *o = out.complexDataMut();

    if (T.isComplex()) {
        const keel::c64 *td = T.complexData();
        for (keel::usize i = 0; i < n; ++i) o[i] = td[i + i * n];
    } else {
        const keel::f64 *td = T.doubleData();
        keel::usize i = 0;
        while (i < n) {
            if (i + 1 < n && td[(i + 1) + i * n] != 0.0) {
                keel::f64 a = td[i + i * n];
                keel::f64 b = td[i + (i + 1) * n];
                keel::f64 c = td[(i + 1) + i * n];
                keel::f64 d = td[(i + 1) + (i + 1) * n];
                keel::f64 tr = a + d;
                keel::f64 det = a * d - b * c;
                keel::f64 disc = tr * tr - 4.0 * det;
                keel::f64 re = 0.5 * tr;
                keel::f64 im = 0.5 * keel::sqrt(keel::max(0.0, -disc));
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

keel::Tuple<Value, Value>
eig_general_VD(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "eig");
    if (A.dims().ndim() != 2)
        throw Error("eig: input must be a 2D matrix",
                    0, 0, "eig", "", "numkit:eig:notMatrix");
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(0));
    if (n != keel::as<keel::usize>(A.dims().dim(1)))
        throw Error("eig: matrix must be square",
                    0, 0, "eig", "", "numkit:eig:notSquare");
    if (n == 0) {
        return keel::make_tuple(
            Value::matrix(0, 0, A.type(), mr),
            Value::matrix(0, 0, A.type(), mr));
    }

    const bool wantSingle = (A.type() == ValueType::SINGLE);
    Value Aw = detail::widenSingleToDouble(A, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> Ac(n * n, scratch.allocator());
    if (Aw.isComplex()) {
        keel::copy(Aw.complexData(), Aw.complexData() + n * n, Ac.begin());
    } else {
        const keel::f64 *ad = Aw.doubleData();
        for (keel::usize i = 0; i < n * n; ++i) Ac[i] = keel::c64(ad[i], 0.0);
    }

    keel::Array<keel::c64> Uc(n * n, scratch.allocator());
    keel::Array<keel::c64> Tc(n * n, scratch.allocator());
    if (!keel::linalg::schur_complex(Ac.data(), n, Uc.data(), Tc.data(), mr)) {
        throw Error("eig: QR iteration failed to converge",
                    0, 0, "eig", "", "numkit:eig:qrNoConverge");
    }

    if (isHermitianComplexApprox(Aw)) {
        keel::Array<keel::usize> order(n, scratch.allocator());
        for (keel::usize i = 0; i < n; ++i) order[i] = i;
        keel::sort(order.begin(), order.end(), [&](keel::usize a, keel::usize b) {
            return Tc[a + a * n].real() < Tc[b + b * n].real();
        });
        auto Vout = Value::complexMatrix(n, n, mr);
        auto Dout = Value::matrix(n, n, ValueType::DOUBLE, mr);
        keel::c64 *Vd = Vout.complexDataMut();
        keel::f64 *Dd = Dout.doubleDataMut();
        keel::fill(Dd, Dd + n * n, 0.0);
        for (keel::usize k = 0; k < n; ++k) {
            const keel::usize src = order[k];
            Dd[k + k * n] = Tc[src + src * n].real();
            for (keel::usize i = 0; i < n; ++i)
                Vd[i + k * n] = Uc[i + src * n];
        }
        return keel::make_tuple(
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Vout, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(Dout, wantSingle, mr));
    }

    auto Vout = Value::complexMatrix(n, n, mr);
    auto Dout = Value::complexMatrix(n, n, mr);
    keel::c64 *Vd = Vout.complexDataMut();
    keel::c64 *Dd = Dout.complexDataMut();
    keel::fill(Dd, Dd + n * n, keel::c64(0.0, 0.0));

    keel::Array<keel::c64> evals(n, scratch.allocator());
    keel::linalg::schur_eigenvectors(Uc.data(), Tc.data(), n, Vd, evals.data(), mr);
    for (keel::usize k = 0; k < n; ++k) Dd[k + k * n] = evals[k];

    if (!Aw.isComplex()) {
        keel::f64 aMax = 0.0;
        for (keel::usize i = 0; i < n * n; ++i)
            aMax = keel::max(aMax, keel::abs(Ac[i].real()));
        const keel::f64 tol = 64.0 * keel::as<keel::f64>(n) *
                           keel::Limits<keel::f64>::kEpsilon * (1.0 + aMax);
        if (cleanRealSpectrum(Vd, Dd, n, tol, false)) {
            auto Vr = Value::matrix(n, n, ValueType::DOUBLE, mr);
            auto Dr = Value::matrix(n, n, ValueType::DOUBLE, mr);
            for (keel::usize i = 0; i < n * n; ++i) {
                Vr.doubleDataMut()[i] = Vd[i].real();
                Dr.doubleDataMut()[i] = Dd[i].real();
            }
            return keel::make_tuple(
                detail::narrowDoubleToSingleIf(Vr, wantSingle, mr),
                detail::narrowDoubleToSingleIf(Dr, wantSingle, mr));
        }
    }

    return keel::make_tuple(
        detail::narrowDoubleToSingleIf(detail::narrow_if_real(Vout, mr), wantSingle, mr),
        detail::narrowDoubleToSingleIf(detail::narrow_if_real(Dout, mr), wantSingle, mr));
}

// ────────────────────────────────────────────────────────────────────────
// Symmetric eig (via keel::linalg::eig_sym)
// ────────────────────────────────────────────────────────────────────────

bool isSymmetricApprox(const Value &A, keel::f64 tol)
{
    if (A.dims().ndim() != 2) return false;
    if (A.isComplex()) return false;
    if (!A.isNumeric() && !A.isLogical() && !A.isChar()) return false;
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    if (m != n) return false;
    keel::f64 aMax = 1.0;
    for (keel::usize idx = 0; idx < n * n; ++idx) {
        const keel::f64 v = keel::abs(A.elemAsDouble(idx));
        if (v > aMax) aMax = v;
    }
    const keel::f64 thresh = tol * aMax;
    for (keel::usize i = 0; i < n; ++i)
        for (keel::usize j = i + 1; j < n; ++j) {
            const keel::f64 aij = A.elemAsDouble(i + j * m);
            const keel::f64 aji = A.elemAsDouble(j + i * m);
            const keel::f64 d = keel::abs(aij - aji);
            if (d > thresh) return false;
        }
    return true;
}

keel::Tuple<Value, Value>
eig_symmetric(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "eig");
    if (A.dims().ndim() != 2)
        throw Error("eig: input must be a 2D matrix",
                    0, 0, "eig", "", "numkit:eig:notMatrix");
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    if (m != n)
        throw Error("eig: matrix must be square",
                    0, 0, "eig", "", "numkit:eig:notSquare");
    if (!isSymmetricApprox(A, 1e-10))
        throw Error("eig: only symmetric matrices supported in this revision "
                    "(general eig via Hessenberg + Francis QR is deferred to Phase 2b)",
                    0, 0, "eig", "", "numkit:eig:notSymmetric");
    if (n == 0) {
        return keel::make_tuple(
            Value::matrix(0, 0, A.type(), mr),
            Value::matrix(0, 0, A.type(), mr));
    }

    const bool wantSingle = (A.type() == ValueType::SINGLE);
    Value Aw = detail::widenSingleToDouble(A, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> evals(n, scratch.allocator());

    auto Vout = Value::matrix(n, n, ValueType::DOUBLE, mr);
    auto Dout = Value::matrix(n, n, ValueType::DOUBLE, mr);
    keel::linalg::eig_sym(Aw.doubleData(), n, Vout.doubleDataMut(), evals.data(), mr);

    keel::f64 *D = Dout.doubleDataMut();
    keel::fill(D, D + n * n, 0.0);
    for (keel::usize k = 0; k < n; ++k) D[k + k * n] = evals[k];

    return keel::make_tuple(
        detail::narrowDoubleToSingleIf(Vout, wantSingle, mr),
        detail::narrowDoubleToSingleIf(Dout, wantSingle, mr));
}

Value eig_values(const Value &A, keel::Allocator *mr)
{
    const bool wantSingle = (A.type() == ValueType::SINGLE);
    Value Aw = detail::widenSingleToDouble(A, mr);
    auto [V, D] = eig_symmetric(Aw, mr);
    const keel::usize n = keel::as<keel::usize>(D.dims().dim(0));
    auto out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    const keel::f64 *Ddata = D.doubleData();
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) o[i] = Ddata[i + i * n];
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

// ────────────────────────────────────────────────────────────────────────
// Generalized eig (symmetric Cholesky fast path + QZ general path)
// ────────────────────────────────────────────────────────────────────────

keel::Tuple<Value, Value>
eig_generalized_VD(const Value &A, const Value &B, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "eig");
    detail::requireDoubleInput(B, "eig", "B");
    if (A.dims().ndim() != 2 || B.dims().ndim() != 2)
        throw Error("eig: inputs must be 2D matrices", 0, 0, "eig", "", "numkit:eig:notMatrix");
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(0));
    if (n != keel::as<keel::usize>(A.dims().dim(1)) ||
        n != keel::as<keel::usize>(B.dims().dim(0)) ||
        n != keel::as<keel::usize>(B.dims().dim(1)))
        throw Error("eig: A and B must be square matrices of the same size",
                    0, 0, "eig", "", "numkit:eig:notSquare");

    const bool wantSingle = (A.type() == ValueType::SINGLE || B.type() == ValueType::SINGLE);
    const ValueType rtype = wantSingle ? ValueType::SINGLE : ValueType::DOUBLE;
    if (n == 0) {
        return keel::make_tuple(Value::matrix(0, 0, rtype, mr), Value::matrix(0, 0, rtype, mr));
    }

    Value Aw = detail::widenSingleToDouble(A, mr);
    Value Bw = detail::widenSingleToDouble(B, mr);

    if (!Aw.isComplex() && !Bw.isComplex() &&
        isSymmetricApprox(Aw, 1e-8) && isSymmetricApprox(Bw, 1e-8)) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<keel::f64> evals(n, scratch.allocator());
        auto V = Value::matrix(n, n, ValueType::DOUBLE, mr);
        if (keel::linalg::eig_sym_generalized(Aw.doubleData(), Bw.doubleData(), n,
                                      V.doubleDataMut(), evals.data(), mr)) {
            auto D = Value::matrix(n, n, ValueType::DOUBLE, mr);
            keel::f64 *dd = D.doubleDataMut();
            keel::fill(dd, dd + n * n, 0.0);
            for (keel::usize k = 0; k < n; ++k) dd[k + k * n] = evals[k];
            return keel::make_tuple(
                detail::narrowDoubleToSingleIf(V, wantSingle, mr),
                detail::narrowDoubleToSingleIf(D, wantSingle, mr));
        }
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> Ac(n * n, scratch.allocator());
    keel::Array<keel::c64> Bc(n * n, scratch.allocator());
    if (Aw.isComplex()) {
        keel::copy(Aw.complexData(), Aw.complexData() + n * n, Ac.begin());
    } else {
        for (keel::usize i = 0; i < n * n; ++i) Ac[i] = keel::c64(Aw.doubleData()[i], 0.0);
    }
    if (Bw.isComplex()) {
        keel::copy(Bw.complexData(), Bw.complexData() + n * n, Bc.begin());
    } else {
        for (keel::usize i = 0; i < n * n; ++i) Bc[i] = keel::c64(Bw.doubleData()[i], 0.0);
    }

    keel::Array<keel::c64> AAc(n * n, scratch.allocator());
    keel::Array<keel::c64> BBc(n * n, scratch.allocator());
    keel::Array<keel::c64> Qc(n * n, scratch.allocator());
    keel::Array<keel::c64> Zc(n * n, scratch.allocator());
    if (!keel::linalg::qz_factor(Ac.data(), Bc.data(), n, AAc.data(), BBc.data(), Qc.data(), Zc.data())) {
        throw Error("qz: QZ iteration failed to converge", 0, 0, "qz", "", "numkit:qz:noConverge");
    }

    auto Vout = Value::complexMatrix(n, n, mr);
    auto Dout = Value::complexMatrix(n, n, mr);
    keel::c64 *Vd = Vout.complexDataMut();
    keel::c64 *Dd = Dout.complexDataMut();
    keel::fill(Dd, Dd + n * n, keel::c64(0.0, 0.0));

    keel::Array<keel::c64> evals(n, scratch.allocator());
    keel::linalg::qz_eigenvectors(AAc.data(), BBc.data(), Zc.data(), n, Vd, evals.data(), mr);
    for (keel::usize k = 0; k < n; ++k) Dd[k + k * n] = evals[k];

    if (!Aw.isComplex() && !Bw.isComplex()) {
        keel::f64 aMax = 0.0, bMax = 0.0;
        for (keel::usize i = 0; i < n * n; ++i) {
            aMax = keel::max(aMax, keel::abs(AAc[i]));
            bMax = keel::max(bMax, keel::abs(BBc[i]));
        }
        const keel::f64 abScale = keel::max(1.0, keel::max(aMax, bMax));
        const keel::f64 imagTol = 64.0 * keel::as<keel::f64>(n) *
                               keel::Limits<keel::f64>::kEpsilon * (1.0 + abScale);
        if (cleanRealSpectrum(Vd, Dd, n, imagTol, true)) {
            auto Vr = Value::matrix(n, n, ValueType::DOUBLE, mr);
            auto Dr = Value::matrix(n, n, ValueType::DOUBLE, mr);
            for (keel::usize i = 0; i < n * n; ++i) {
                Vr.doubleDataMut()[i] = Vd[i].real();
                Dr.doubleDataMut()[i] = Dd[i].real();
            }
            return keel::make_tuple(
                detail::narrowDoubleToSingleIf(Vr, wantSingle, mr),
                detail::narrowDoubleToSingleIf(Dr, wantSingle, mr));
        }
    }

    return keel::make_tuple(
        detail::narrowDoubleToSingleIf(detail::narrow_if_real(Vout, mr), wantSingle, mr),
        detail::narrowDoubleToSingleIf(detail::narrow_if_real(Dout, mr), wantSingle, mr));
}

Value eig_generalized_values(const Value &A, const Value &B, keel::Allocator *mr)
{
    auto [V, D] = eig_generalized_VD(A, B, mr);
    (void)V;
    const keel::usize n = keel::as<keel::usize>(D.dims().dim(0));
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
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(0));
    if (n == 0 || k == 0 || k > n)
        return false;
    if (n != keel::as<keel::usize>(A.dims().dim(1)) ||
        n != keel::as<keel::usize>(B.dims().dim(0)) ||
        n != keel::as<keel::usize>(B.dims().dim(1)))
        return false;

    const bool wantSingle = (A.type() == ValueType::SINGLE || B.type() == ValueType::SINGLE);
    Value Aw = detail::widenSingleToDouble(A, mr);
    Value Bw = detail::widenSingleToDouble(B, mr);
    if (!isSymmetricApprox(Aw, 1e-8) || !isSymmetricApprox(Bw, 1e-8))
        return false;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> evals(k, scratch.allocator());
    auto V = Value::matrix(n, k, ValueType::DOUBLE, mr);
    if (!keel::linalg::eigs_sym_generalized_smallest_abs(Aw.doubleData(), Bw.doubleData(), n, k,
                                                 V.doubleDataMut(), evals.data(), mr)) {
        return false;
    }

    auto D = Value::matrix(k, k, ValueType::DOUBLE, mr);
    keel::f64 *dd = D.doubleDataMut();
    keel::fill(dd, dd + k * k, 0.0);
    for (keel::usize col = 0; col < k; ++col) dd[col + col * k] = evals[col];

    Vout = detail::narrowDoubleToSingleIf(V, wantSingle, mr);
    Dout = detail::narrowDoubleToSingleIf(D, wantSingle, mr);
    return true;
}

// ────────────────────────────────────────────────────────────────────────
// Hessenberg reduction (via keel::linalg::hess_reduce)
// ────────────────────────────────────────────────────────────────────────

keel::Tuple<Value, Value>
hess(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "hess");
    if (A.dims().ndim() != 2)
        throw Error("hess: input must be a 2D matrix",
                    0, 0, "hess", "", "numkit:hess:notMatrix");
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    if (m != n)
        throw Error("hess: matrix must be square",
                    0, 0, "hess", "", "numkit:hess:notSquare");
    if (A.isComplex()) {
        auto Hout = Value::complexMatrix(n, n, mr);
        auto Pout = Value::complexMatrix(n, n, mr);
        if (n == 0) return keel::make_tuple(keel::move(Pout), keel::move(Hout));
        keel::copy(A.complexData(), A.complexData() + n * n, Hout.complexDataMut());
        keel::linalg::hess_reduce(Hout.complexDataMut(), n, Pout.complexDataMut(), mr);
        return keel::make_tuple(detail::narrow_if_real(Pout, mr), detail::narrow_if_real(Hout, mr));
    }
    const bool wantSingle = (A.type() == ValueType::SINGLE);
    if (n == 0) {
        return keel::make_tuple(
            Value::matrix(0, 0, A.type(), mr),
            Value::matrix(0, 0, A.type(), mr));
    }
    Value Aw = detail::widenSingleToDouble(A, mr);
    auto Hout = Value::matrix(n, n, ValueType::DOUBLE, mr);
    auto Pout = Value::matrix(n, n, ValueType::DOUBLE, mr);
    keel::copy(Aw.doubleData(), Aw.doubleData() + n * n, Hout.doubleDataMut());
    keel::linalg::hess_reduce(Hout.doubleDataMut(), n, Pout.doubleDataMut(), mr);
    return keel::make_tuple(
        detail::narrowDoubleToSingleIf(Pout, wantSingle, mr),
        detail::narrowDoubleToSingleIf(Hout, wantSingle, mr));
}

Value hess_H_only(const Value &A, keel::Allocator *mr)
{
    auto [P, H] = hess(A, mr);
    return H;
}

// ────────────────────────────────────────────────────────────────────────
// Schur & Sylvester (symmetric A, B)
// ────────────────────────────────────────────────────────────────────────

keel::Tuple<Value, Value>
schur_sym(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "schur");
    if (A.dims().ndim() != 2)
        throw Error("schur: input must be a 2D matrix",
                    0, 0, "schur", "", "numkit:schur:notMatrix");
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(0));
    if (n != keel::as<keel::usize>(A.dims().dim(1)))
        throw Error("schur: matrix must be square",
                    0, 0, "schur", "", "numkit:schur:notSquare");
    return eig_symmetric(A, mr);
}

keel::Tuple<Value, Value>
schur_general(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "schur");
    if (A.dims().ndim() != 2)
        throw Error("schur: input must be a 2D matrix",
                    0, 0, "schur", "", "numkit:schur:notMatrix");
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(0));
    if (n != keel::as<keel::usize>(A.dims().dim(1)))
        throw Error("schur: matrix must be square",
                    0, 0, "schur", "", "numkit:schur:notSquare");
    if (A.isComplex()) {
        auto U = Value::complexMatrix(n, n, mr);
        auto T = Value::complexMatrix(n, n, mr);
        if (n == 0) return keel::make_tuple(keel::move(U), keel::move(T));
        if (!keel::linalg::schur_complex(A.complexData(), n, U.complexDataMut(), T.complexDataMut(), mr)) {
            throw Error("eig: QR iteration failed to converge",
                        0, 0, "eig", "", "numkit:eig:qrNoConverge");
        }
        return keel::make_tuple(detail::narrow_if_real(U, mr), detail::narrow_if_real(T, mr));
    }

    const bool wantSingle = (A.type() == ValueType::SINGLE);
    if (n == 0) {
        return keel::make_tuple(
            Value::matrix(0, 0, A.type(), mr),
            Value::matrix(0, 0, A.type(), mr));
    }
    Value Aw = detail::widenSingleToDouble(A, mr);
    auto U = Value::matrix(n, n, ValueType::DOUBLE, mr);
    auto T = Value::matrix(n, n, ValueType::DOUBLE, mr);
    keel::linalg::schur_real(Aw.doubleData(), n, U.doubleDataMut(), T.doubleDataMut(), mr);
    return keel::make_tuple(
        detail::narrowDoubleToSingleIf(U, wantSingle, mr),
        detail::narrowDoubleToSingleIf(T, wantSingle, mr));
}

Value sylvester_sym(const Value &A, const Value &B, const Value &C,
                    keel::Allocator *mr)
{
    if (A.dims().ndim() != 2 || B.dims().ndim() != 2 || C.dims().ndim() != 2)
        throw Error("sylvester: A, B, C must be 2D matrices",
                    0, 0, "sylvester", "", "numkit:sylvester:notMatrix");
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize m = keel::as<keel::usize>(B.dims().dim(0));
    if (A.dims().dim(0) != A.dims().dim(1))
        throw Error("sylvester: A must be square",
                    0, 0, "sylvester", "", "numkit:sylvester:badA");
    if (B.dims().dim(0) != B.dims().dim(1))
        throw Error("sylvester: B must be square",
                    0, 0, "sylvester", "", "numkit:sylvester:badB");
    if (C.dims().dim(0) != n || C.dims().dim(1) != m)
        throw Error("sylvester: C must be n x m where A is n x n, B is m x m",
                    0, 0, "sylvester", "", "numkit:sylvester:badC");

    const bool wantSingle = (A.type() == ValueType::SINGLE ||
                             B.type() == ValueType::SINGLE ||
                             C.type() == ValueType::SINGLE);
    Value Aw = detail::widenSingleToDouble(A, mr);
    Value Bw = detail::widenSingleToDouble(B, mr);
    Value Cw = detail::widenSingleToDouble(C, mr);

    auto [Va, Da] = eig_symmetric(Aw, mr);
    auto [Vb, Db] = eig_symmetric(Bw, mr);

    const keel::f64 *Vad = Va.doubleData();
    const keel::f64 *Dad = Da.doubleData();
    const keel::f64 *Vbd = Vb.doubleData();
    const keel::f64 *Dbd = Db.doubleData();
    const keel::f64 *Cd  = Cw.doubleData();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> Y(n * m, 0.0, scratch.allocator());
    keel::Array<keel::f64> tmp(n * m, 0.0, scratch.allocator());
    // tmp = Va^T * C; Y = tmp * Vb
    keel::linalg::gemm(keel::linalg::MatrixTranspose::Trans, keel::linalg::MatrixTranspose::NoTrans,
               n, m, n, 1.0, Vad, n, Cd, n, 0.0, tmp.data(), n);
    keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans,
               n, m, m, 1.0, tmp.data(), n, Vbd, m, 0.0, Y.data(), n);

    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 dai = Dad[i + i * n];
        for (keel::usize j = 0; j < m; ++j) {
            const keel::f64 denom = dai + Dbd[j + j * m];
            if (keel::abs(denom) < 1e-300)
                throw Error("sylvester: A and -B share an eigenvalue (no unique solution)",
                            0, 0, "sylvester", "", "numkit:sylvester:singular");
            Y[i + j * n] /= denom;
        }
    }

    auto out = Value::matrix(n, m, ValueType::DOUBLE, mr);
    keel::f64 *X = out.doubleDataMut();
    // tmp = Va * Y; X = tmp * Vb^T
    keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans,
               n, m, n, 1.0, Vad, n, Y.data(), n, 0.0, tmp.data(), n);
    keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::Trans,
               n, m, m, 1.0, tmp.data(), n, Vbd, m, 0.0, X, n);
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

// ────────────────────────────────────────────────────────────────────────
// polyeig — polynomial eigenvalue problem via companion linearisation
// ────────────────────────────────────────────────────────────────────────

namespace {

Value buildCompanion(keel::ArrayRef<Value> coeffs, keel::Allocator *mr)
{
    const keel::usize k = coeffs.size() - 1;
    const keel::usize n = keel::as<keel::usize>(coeffs[0].dims().dim(0));
    const keel::usize N = k * n;

    Value Ak_w = detail::widenSingleToDouble(coeffs[k], mr);
    Value Ak_inv = inv(Ak_w, mr);

    auto L = Value::matrix(N, N, ValueType::DOUBLE, mr);
    keel::f64 *Ld = L.doubleDataMut();
    keel::fill(Ld, Ld + N * N, 0.0);

    for (keel::usize i = 0; i + 1 < k; ++i) {
        for (keel::usize r = 0; r < n; ++r) {
            Ld[(i * n + r) + ((i + 1) * n + r) * N] = 1.0;
        }
    }

    const keel::f64 *Akid = Ak_inv.doubleData();
    for (keel::usize i = 0; i < k; ++i) {
        Value Ai_w = detail::widenSingleToDouble(coeffs[i], mr);
        const keel::f64 *Aid = Ai_w.doubleData();
        keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::NoTrans,
                   n, n, n, -1.0, Akid, n, Aid, n, 0.0, Ld + (k - 1) * n + (i * n) * N, N);
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
    const keel::usize n = keel::as<keel::usize>(coeffs[0].dims().dim(0));
    if (n != keel::as<keel::usize>(coeffs[0].dims().dim(1)))
        throw Error("polyeig: coefficient matrices must be square",
                    0, 0, "polyeig", "", "numkit:polyeig:notSquare");
    for (keel::usize i = 1; i < coeffs.size(); ++i) {
        if (coeffs[i].dims().ndim() != 2
            || keel::as<keel::usize>(coeffs[i].dims().dim(0)) != n
            || keel::as<keel::usize>(coeffs[i].dims().dim(1)) != n)
            throw Error("polyeig: all coefficient matrices must be n x n with matching n",
                        0, 0, "polyeig", "", "numkit:polyeig:shapeMismatch");
    }
}

} // anonymous namespace

Value polyeig_values(keel::ArrayRef<Value> coeffs, keel::Allocator *mr)
{
    validatePolyeigCoeffs(coeffs);
    Value L = buildCompanion(coeffs, mr);
    return detail::narrowDoubleToSingleIf(eig_general_values(L, mr), anySingleCoeffs(coeffs), mr);
}

keel::Tuple<Value, Value>
polyeig_VE(keel::ArrayRef<Value> coeffs, keel::Allocator *mr)
{
    validatePolyeigCoeffs(coeffs);
    const keel::usize k = coeffs.size() - 1;
    const keel::usize n = keel::as<keel::usize>(coeffs[0].dims().dim(0));
    const keel::usize N = k * n;

    Value L = buildCompanion(coeffs, mr);
    auto [V_big, D_big] = eig_general_VD(L, mr);

    auto V = Value::matrix(n, N, ValueType::DOUBLE, mr);
    auto e = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::f64 *Vd = V.doubleDataMut();
    keel::f64 *ed = e.doubleDataMut();
    const keel::f64 *Vbd = V_big.doubleData();
    const keel::f64 *Dbd = D_big.doubleData();

    for (keel::usize j = 0; j < N; ++j) {
        ed[j] = Dbd[j + j * N];
        for (keel::usize r = 0; r < n; ++r)
            Vd[r + j * n] = Vbd[r + j * N];
    }
    const bool wantSingle = anySingleCoeffs(coeffs);
    return keel::make_tuple(
        detail::narrowDoubleToSingleIf(V, wantSingle, mr),
        detail::narrowDoubleToSingleIf(e, wantSingle, mr));
}

// ────────────────────────────────────────────────────────────────────────
// ordeig — read eigenvalues off a (quasi-)triangular Schur factor
// ────────────────────────────────────────────────────────────────────────

Value ordeig(const Value &T, keel::Allocator *mr)
{
    detail::requireDoubleInput(T, "ordeig", "T");
    if (T.dims().ndim() != 2)
        throw Error("ordeig: T must be a 2D matrix",
                    0, 0, "ordeig", "", "numkit:ordeig:notMatrix");
    const keel::usize n = keel::as<keel::usize>(T.dims().dim(0));
    if (n != keel::as<keel::usize>(T.dims().dim(1)))
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
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize i = 0; i < n; ++i) od[i] = Tc[i + i * n].real();
        return out;
    }

    const bool wantSingle = (T.type() == ValueType::SINGLE);
    Value Tw = detail::widenSingleToDouble(T, mr);
    const keel::f64 *Td = Tw.doubleData();

    bool any_complex = false;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> out_complex(scratch.allocator());
    out_complex.reserve(n);

    keel::usize i = 0;
    while (i < n) {
        const keel::f64 sub = (i + 1 < n) ? Td[(i + 1) + i * n] : 0.0;
        if (keel::abs(sub) <= 1e-14 * (1.0 + keel::abs(Td[i + i * n]))) {
            out_complex.push_back(keel::c64(Td[i + i * n], 0.0));
            ++i;
        } else {
            const keel::f64 a  = Td[i + i * n];
            const keel::f64 ap = Td[(i + 1) + (i + 1) * n];
            const keel::f64 c  = Td[i + (i + 1) * n];
            const keel::f64 d  = sub;
            const keel::f64 disc_re = (a - ap) * (a - ap) + 4.0 * c * d;
            const keel::f64 mean = 0.5 * (a + ap);
            if (disc_re >= 0.0) {
                const keel::f64 half_sqrt = 0.5 * keel::sqrt(disc_re);
                out_complex.push_back(keel::c64(mean + half_sqrt, 0.0));
                out_complex.push_back(keel::c64(mean - half_sqrt, 0.0));
            } else {
                const keel::f64 half_imag = 0.5 * keel::sqrt(-disc_re);
                out_complex.push_back(keel::c64(mean,  half_imag));
                out_complex.push_back(keel::c64(mean, -half_imag));
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
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize k = 0; k < n; ++k) od[k] = out_complex[k].real();
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

} // namespace numkit::linalg
