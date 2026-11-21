// toolboxes/linalg/src/pseudo_subspace.cpp
//
// pinv / orth / null / subspace â€” implementations and engine adapters.
// All four are SVD-based; migrated alongside the SVD kernel.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/linalg/pseudo_subspace.hpp>
#include "linalg_detail.hpp"

#include <numkit/linalg/decompositions.hpp>   // svd_decompose / svd_values
// Compute-only TU: Value substrate + Error, no engine. The pinv/orth/null/
// subspace builtins (CallContext wrappers) live in pseudo_subspace_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace numkit::linalg {

namespace {

// Default tolerance for rank-cutoff: max(m,n) * eps(sigma_max).
keel::f64 defaultRankTol(keel::usize m, keel::usize n, keel::f64 sigma_max, bool isSingle = false)
{
    const keel::f64 eps = isSingle
        ? keel::as<keel::f64>(std::numeric_limits<keel::f32>::epsilon())
        : std::numeric_limits<keel::f64>::epsilon();
    return keel::as<keel::f64>(std::max(m, n)) * sigma_max * eps;
}

} // anonymous namespace

Value pinv(const Value &A, keel::f64 tol, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "pinv");
    detail::require2DMatrix(A, "pinv", "A", "pagepinv");
    const bool wantSingle = (A.type() == ValueType::SINGLE);
    const Value Aw = detail::widenSingleToDouble(A, mr);
    auto [U, S, V] = svd_decompose(Aw, mr);
    const keel::usize m = keel::as<keel::usize>(Aw.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(Aw.dims().dim(1));
    const keel::usize k = std::min(m, n);

    if (Aw.isComplex() || U.isComplex() || V.isComplex()) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<keel::f64> Splus(k, 0.0, scratch.allocator());
        keel::f64 sigma_max = 0.0;
        const keel::c64 *sd = S.isComplex() ? S.complexData() : nullptr;
        const keel::f64 *sdd = S.isComplex() ? nullptr : S.doubleData();
        const keel::usize Srows = keel::as<keel::usize>(S.dims().dim(0));

        for (keel::usize i = 0; i < k; ++i) {
            keel::f64 sig = sd ? keel::abs(sd[i + i * Srows]) : sdd[i + i * Srows];
            sigma_max = std::max(sigma_max, sig);
        }
        const keel::f64 cutoff = (tol < 0.0) ? defaultRankTol(m, n, sigma_max, wantSingle) : tol;

        for (keel::usize i = 0; i < k; ++i) {
            keel::f64 sig = sd ? keel::abs(sd[i + i * Srows]) : sdd[i + i * Srows];
            if (sig > cutoff) Splus[i] = 1.0 / sig;
        }

        auto out = Value::complexMatrix(n, m, mr);
        keel::c64 *P = out.complexDataMut();
        std::fill(P, P + n * m, keel::c64(0.0, 0.0));

        const keel::c64 *Vd = V.isComplex() ? V.complexData() : nullptr;
        const keel::f64 *Vdd = V.isComplex() ? nullptr : V.doubleData();
        const keel::c64 *Ud = U.isComplex() ? U.complexData() : nullptr;
        const keel::f64 *Udd = U.isComplex() ? nullptr : U.doubleData();
        const keel::usize Vrows = keel::as<keel::usize>(V.dims().dim(0));
        const keel::usize Urows = keel::as<keel::usize>(U.dims().dim(0));

        for (keel::usize i = 0; i < n; ++i) {
            for (keel::usize j = 0; j < m; ++j) {
                keel::c64 s(0.0, 0.0);
                for (keel::usize a = 0; a < k; ++a) {
                    if (Splus[a] == 0.0) continue;
                    keel::c64 v_val = Vd ? Vd[i + a * Vrows] : keel::c64(Vdd[i + a * Vrows], 0.0);
                    keel::c64 u_val = Ud ? Ud[j + a * Urows] : keel::c64(Udd[j + a * Urows], 0.0);
                    s += v_val * Splus[a] * keel::conj(u_val);
                }
                P[i + j * n] = s;
            }
        }
        return detail::narrowDoubleToSingleIf(detail::narrow_if_real(out, mr), wantSingle, mr);
    }

    const keel::f64 *S_data = S.doubleData();
    const keel::usize Srows = keel::as<keel::usize>(S.dims().dim(0));

    keel::f64 sigma_max = 0.0;
    for (keel::usize i = 0; i < k; ++i)
        sigma_max = std::max(sigma_max, S_data[i + i * Srows]);
    const keel::f64 cutoff = (tol < 0.0) ? defaultRankTol(m, n, sigma_max, wantSingle) : tol;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> Splus(n * m, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < k; ++i) {
        const keel::f64 sig = S_data[i + i * Srows];
        if (sig > cutoff)
            Splus[i + i * n] = 1.0 / sig;
    }

    // pinv(A) = V * S^+ * U' -- output is n × m.
    auto out = Value::matrix(n, m, ValueType::DOUBLE, mr);
    keel::f64 *P = out.doubleDataMut();
    std::fill(P, P + n * m, 0.0);

    const keel::f64 *Vdata = V.doubleData();
    const keel::f64 *Udata = U.doubleData();
    const keel::usize Vrows = keel::as<keel::usize>(V.dims().dim(0));
    const keel::usize Urows = keel::as<keel::usize>(U.dims().dim(0));

    for (keel::usize i = 0; i < n; ++i) {
        for (keel::usize j = 0; j < m; ++j) {
            keel::f64 s = 0.0;
            for (keel::usize a = 0; a < k; ++a) {
                const keel::f64 sp = Splus[a + a * n];
                if (sp == 0.0) continue;
                s += Vdata[i + a * Vrows] * sp * Udata[j + a * Urows];
            }
            P[i + j * n] = s;
        }
    }
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

Value orth(const Value &A, keel::f64 tol, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "orth");
    detail::require2DMatrix(A, "orth", "A", "pagesvd");
    const bool wantSingle = (A.type() == ValueType::SINGLE);
    const Value Aw = detail::widenSingleToDouble(A, mr);
    auto [U, S, V] = svd_decompose(Aw, mr);
    const keel::usize m = keel::as<keel::usize>(Aw.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(Aw.dims().dim(1));
    const keel::usize k = std::min(m, n);

    const keel::f64 *S_data = S.doubleData();
    const keel::usize Srows = keel::as<keel::usize>(S.dims().dim(0));
    const keel::f64 sigma_max = (k > 0) ? S_data[0] : 0.0;
    const keel::f64 cutoff = (tol < 0.0) ? defaultRankTol(m, n, sigma_max, wantSingle) : tol;

    int r = 0;
    for (keel::usize i = 0; i < k; ++i)
        if (S_data[i + i * Srows] > cutoff) ++r;

    auto out = Value::matrix(m, keel::as<keel::usize>(r), ValueType::DOUBLE, mr);
    if (r == 0) return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
    keel::f64 *Q = out.doubleDataMut();
    const keel::f64 *Udata = U.doubleData();
    const keel::usize Urows = keel::as<keel::usize>(U.dims().dim(0));
    for (keel::usize j = 0; j < keel::as<keel::usize>(r); ++j)
        for (keel::usize i = 0; i < m; ++i)
            Q[i + j * m] = Udata[i + j * Urows];
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

Value null_basis(const Value &A, keel::f64 tol, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "null");
    detail::require2DMatrix(A, "null");
    const bool wantSingle = (A.type() == ValueType::SINGLE);
    const Value Aw = detail::widenSingleToDouble(A, mr);
    auto [U, S, V] = svd_decompose(Aw, mr);
    const keel::usize m = keel::as<keel::usize>(Aw.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(Aw.dims().dim(1));
    const keel::usize k = std::min(m, n);

    const keel::f64 *S_data = S.doubleData();
    const keel::usize Srows = keel::as<keel::usize>(S.dims().dim(0));
    const keel::f64 sigma_max = (k > 0) ? S_data[0] : 0.0;
    const keel::f64 cutoff = (tol < 0.0) ? defaultRankTol(m, n, sigma_max, wantSingle) : tol;

    int r = 0;
    for (keel::usize i = 0; i < k; ++i)
        if (S_data[i + i * Srows] > cutoff) ++r;

    const keel::usize null_dim = n - keel::as<keel::usize>(r);
    auto out = Value::matrix(n, null_dim, ValueType::DOUBLE, mr);
    if (null_dim == 0) return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
    keel::f64 *N = out.doubleDataMut();
    const keel::f64 *Vdata = V.doubleData();
    const keel::usize Vrows = keel::as<keel::usize>(V.dims().dim(0));
    for (keel::usize j = 0; j < null_dim; ++j) {
        const keel::usize src = keel::as<keel::usize>(r) + j;
        for (keel::usize i = 0; i < n; ++i)
            N[i + j * n] = Vdata[i + src * Vrows];
    }
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

Value subspace(const Value &A, const Value &B, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "subspace", "A", 1);
    detail::requireDoubleInput(B, "subspace", "B", 2);
    if (A.dims().ndim() != 2 || B.dims().ndim() != 2)
        throw Error("subspace: inputs must be 2D matrices",
                    0, 0, "subspace", "", "numkit:subspace:notMatrix");
    const keel::usize mA = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize mB = keel::as<keel::usize>(B.dims().dim(0));
    if (mA != mB)
        throw Error("subspace: inputs must have the same number of rows",
                    0, 0, "subspace", "", "numkit:subspace:dimMismatch");

    const bool wantSingle = (A.type() == ValueType::SINGLE || B.type() == ValueType::SINGLE);
    auto Qa = detail::widenSingleToDouble(orth(A, -1.0, mr), mr);
    auto Qb = detail::widenSingleToDouble(orth(B, -1.0, mr), mr);
    const keel::usize na = keel::as<keel::usize>(Qa.dims().dim(1));
    const keel::usize nb = keel::as<keel::usize>(Qb.dims().dim(1));
    if (na == 0 || nb == 0) return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);

    // M = Qa' * Qb (na × nb).
    auto Mout = Value::matrix(na, nb, ValueType::DOUBLE, mr);
    keel::f64 *M = Mout.doubleDataMut();
    const keel::f64 *Qad = Qa.doubleData();
    const keel::f64 *Qbd = Qb.doubleData();
    for (keel::usize i = 0; i < na; ++i)
        for (keel::usize j = 0; j < nb; ++j) {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < mA; ++k)
                s += Qad[k + i * mA] * Qbd[k + j * mB];
            M[i + j * na] = s;
        }

    // SVD of M -- singular values are cosines of principal angles.
    auto s = svd_values(Mout, mr);
    const keel::usize k = s.numel();
    if (k == 0) return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);
    const keel::f64 *sd = s.doubleData();
    keel::f64 smin = sd[0];
    for (keel::usize i = 1; i < k; ++i)
        if (sd[i] < smin) smin = sd[i];
    if (smin > 1.0) smin = 1.0;
    if (smin < 0.0) smin = 0.0;
    return detail::narrowDoubleToSingleIf(Value::scalar(std::acos(smin), mr), wantSingle, mr);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Engine adapters â€” registered in LinalgLibrary::install
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

} // namespace numkit::linalg
