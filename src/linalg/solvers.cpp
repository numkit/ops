// toolboxes/linalg/src/solvers.cpp
//
// linsolve / lsqminnorm / lsqnonneg â€” and engine adapters.
// Migrated 2026-05-25 from toolboxes/builtin/src/language/arrays/{matrix,lsq}.cpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/string.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/linalg/solvers.hpp>
#include <numkit/linalg/decompositions.hpp>
#include "linalg_detail.hpp"

#include <numkit/linalg/pseudo_subspace.hpp>      // pinv
#include <numkit/ops/la_solve.hpp>   // numkit::ops::la_solve
#include <numkit/builtin/ops.hpp>  // mtimes

// Compute-only TU: Value substrate + Error, no engine. The linsolve /
// lsqminnorm / lsqnonneg builtins (CallContext wrappers) live in
// solvers_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace numkit::linalg {

Value linsolve(const Value &A, const Value &B, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "linsolve", "A", 1);
    detail::requireDoubleInput(B, "linsolve", "B", 2);
    if (A.dims().ndim() != 2 || B.dims().ndim() != 2)
        throw Error("linsolve: A and B must be 2D matrices",
                    0, 0, "linsolve", "", "numkit:linsolve:notMatrix");
    const keel::usize m = static_cast<keel::usize>(A.dims().dim(0));
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(1));
    const keel::usize mb = static_cast<keel::usize>(B.dims().dim(0));
    const keel::usize nrhs = static_cast<keel::usize>(B.dims().dim(1));
    if (m != mb)
        throw Error("linsolve: A and B must have the same number of rows",
                    0, 0, "linsolve", "", "numkit:linsolve:badDims");

    const bool wantSingle = A.type() == ValueType::SINGLE || B.type() == ValueType::SINGLE;
    Value Aw = detail::widenSingleToDouble(A, mr);
    Value Bw = detail::widenSingleToDouble(B, mr);

    if (Aw.isComplex() || Bw.isComplex()) {
        if (m == n) {
            keel::ArenaTemp scratch = keel::get_scratch(mr);
            keel::Array<keel::c64> A_buf(m * n, scratch.allocator());
            keel::Array<keel::c64> B_buf(m * nrhs, scratch.allocator());

            if (Aw.isComplex()) {
                std::copy(Aw.complexData(), Aw.complexData() + m * n, A_buf.begin());
            } else {
                const double *ad = Aw.doubleData();
                for (keel::usize i = 0; i < m * n; ++i) A_buf[i] = ad[i];
            }

            if (Bw.isComplex()) {
                std::copy(Bw.complexData(), Bw.complexData() + m * nrhs, B_buf.begin());
            } else {
                const double *bd = Bw.doubleData();
                for (keel::usize i = 0; i < m * nrhs; ++i) B_buf[i] = bd[i];
            }

            Value out = Value::complexMatrix(n, nrhs, mr);
            if (!numkit::ops::la_solve(A_buf.data(), n, n, B_buf.data(), nrhs, out.complexDataMut(), mr)) {
                throw Error("linsolve: A is singular or rank-deficient",
                            0, 0, "linsolve", "", "numkit:linsolve:singular");
            }
            return detail::narrowDoubleToSingleIf(detail::narrow_if_real(out, mr), wantSingle, mr);
        } else if (m > n) {
            Value Ac = Aw;
            Value Bc = Bw;
            if (!Ac.isComplex()) {
                Value tmp = Value::complexMatrix(m, n, mr);
                keel::c64 *cd = tmp.complexDataMut();
                const double *dd = Ac.doubleData();
                for (keel::usize i = 0; i < m * n; ++i) cd[i] = dd[i];
                Ac = tmp;
            }
            if (!Bc.isComplex()) {
                Value tmp = Value::complexMatrix(m, nrhs, mr);
                keel::c64 *cd = tmp.complexDataMut();
                const double *dd = Bc.doubleData();
                for (keel::usize i = 0; i < m * nrhs; ++i) cd[i] = dd[i];
                Bc = tmp;
            }
            auto [Q, R] = qr_decompose(Ac, mr);
            keel::ArenaTemp scratch = keel::get_scratch(mr);
            keel::Array<keel::c64> Y(m * nrhs, scratch.allocator());
            const keel::c64 *Qd = Q.complexData();
            const keel::c64 *Bcd = Bc.complexData();

            for (keel::usize i = 0; i < m; ++i) {
                for (keel::usize j = 0; j < nrhs; ++j) {
                    keel::c64 s{0.0, 0.0};
                    for (keel::usize k = 0; k < m; ++k) {
                        s += keel::conj(Qd[k + i * m]) * Bcd[k + j * m];
                    }
                    Y[i + j * m] = s;
                }
            }

            Value X = Value::complexMatrix(n, nrhs, mr);
            keel::c64 *Xd = X.complexDataMut();
            const keel::c64 *Rd = R.complexData();

            for (keel::usize col = 0; col < nrhs; ++col) {
                for (std::intptr_t i = static_cast<std::intptr_t>(n) - 1; i >= 0; --i) {
                    keel::c64 s = Y[static_cast<keel::usize>(i) + col * m];
                    for (keel::usize k = static_cast<keel::usize>(i) + 1; k < n; ++k) {
                        s -= Rd[static_cast<keel::usize>(i) + k * m] * Xd[k + col * n];
                    }
                    keel::c64 diag = Rd[static_cast<keel::usize>(i) + static_cast<keel::usize>(i) * m];
                    if (detail::abs_sq(diag) == 0.0) {
                        throw Error("linsolve: A is singular or rank-deficient",
                                    0, 0, "linsolve", "", "numkit:linsolve:singular");
                    }
                    Xd[static_cast<keel::usize>(i) + col * n] = s / diag;
                }
            }
            return detail::narrowDoubleToSingleIf(detail::narrow_if_real(X, mr), wantSingle, mr);
        } else {
            throw Error("linsolve: complex wide matrices (m < n) deferred",
                        0, 0, "linsolve", "", "numkit:linsolve:wide");
        }
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto out = Value::matrix(n, nrhs, ValueType::DOUBLE, mr);
    if (!numkit::ops::la_solve(Aw.doubleData(), m, n, Bw.doubleData(), nrhs,
                               out.doubleDataMut(), scratch.allocator()))
        throw Error("linsolve: A is singular or rank-deficient",
                    0, 0, "linsolve", "", "numkit:linsolve:singular");
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

Value lsqminnorm(const Value &A, const Value &B, bool have_tol, double tol_user,
                 keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "lsqminnorm", "A", 1);
    detail::requireDoubleInput(B, "lsqminnorm", "B", 2);
    if (A.dims().is3D() || B.dims().is3D())
        throw Error("lsqminnorm: inputs must be 2D",
                    0, 0, "lsqminnorm", "", "numkit:lsqminnorm:Not2D");

    const keel::usize M = A.dims().rows();
    if (B.dims().rows() != M)
        throw Error("lsqminnorm: A and B must have same number of rows",
                    0, 0, "lsqminnorm", "", "numkit:lsqminnorm:DimMismatch");

    const bool wantSingle = A.type() == ValueType::SINGLE || B.type() == ValueType::SINGLE;
    Value Ap = detail::widenSingleToDouble(pinv(A, have_tol ? tol_user : -1.0, mr), mr);
    Value Bw = detail::widenSingleToDouble(B, mr);
    return detail::narrowDoubleToSingleIf(numkit::builtin::mtimes(Ap, Bw, mr), wantSingle, mr);
}

NnlsResult
lsqnonneg_impl(const Value &C, const Value &d, keel::Allocator *mr)
{
    if (C.dims().is3D() || d.dims().is3D())
        throw Error("lsqnonneg: inputs must be 2D",
                    0, 0, "lsqnonneg", "", "numkit:lsqnonneg:Not2D");
    if (C.isComplex() || d.isComplex())
        throw Error("lsqnonneg: real-only inputs in v1",
                    0, 0, "lsqnonneg", "", "numkit:lsqnonneg:NoComplex");
    const keel::usize M = C.dims().rows();
    const keel::usize N = C.dims().cols();
    if (d.numel() != M)
        throw Error("lsqnonneg: C and d must have compatible sizes",
                    0, 0, "lsqnonneg", "", "numkit:lsqnonneg:DimMismatch");

    const bool wantSingle = C.type() == ValueType::SINGLE || d.type() == ValueType::SINGLE;
    Value Cw = detail::widenSingleToDouble(C, mr);

    NnlsResult R;
    R.exitflag = 1;
    R.algorithm = keel::String::from("active-set", mr);
    R.message = keel::String::from("Optimization terminated.", mr);
    R.iterations = 0;

    R.x = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    double *xd = R.x.doubleDataMut();
    std::fill(xd, xd + N, 0.0);

    R.residual = Value::matrix(M, 1, ValueType::DOUBLE, mr);
    double *rd = R.residual.doubleDataMut();
    for (keel::usize i = 0; i < M; ++i) rd[i] = d.elemAsDouble(i);
    R.resnorm = 0.0;
    for (keel::usize i = 0; i < M; ++i) R.resnorm += rd[i] * rd[i];

    if (N == 0) {
        R.x = detail::narrowDoubleToSingleIf(R.x, wantSingle, mr);
        R.residual = detail::narrowDoubleToSingleIf(R.residual, wantSingle, mr);
        return R;
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const double *Cd = Cw.doubleData();

    double cnorm = 0.0;
    for (keel::usize k = 0; k < M * N; ++k) cnorm = std::max(cnorm, keel::abs(Cd[k]));
    const double tol = 10.0 * std::numeric_limits<double>::epsilon()
                            * std::max(cnorm, 1.0)
                            * static_cast<double>(std::max(M, N));

    keel::Array<keel::u8> in_Z(N, scratch.allocator());
    std::fill(in_Z.begin(), in_Z.end(), keel::u8{1});

    keel::Array<double> w(N, scratch.allocator());
    keel::Array<double> s(N, scratch.allocator());
    keel::Array<keel::usize> Pidx(0, scratch.allocator());
    keel::Array<double> CP_buf(0, scratch.allocator());
    keel::Array<double> CtC(0, scratch.allocator());
    keel::Array<double> Ctd(0, scratch.allocator());
    keel::Array<double> sP(0, scratch.allocator());

    const int max_outer = 3 * static_cast<int>(N);
    int outer_iter = 0;

    auto compute_w = [&]() {
        for (keel::usize j = 0; j < N; ++j) {
            double sum = 0.0;
            for (keel::usize i = 0; i < M; ++i) sum += Cd[i + j * M] * rd[i];
            w[j] = sum;
        }
    };

    auto recompute_residual = [&]() {
        for (keel::usize i = 0; i < M; ++i) {
            double sum = 0.0;
            for (keel::usize j = 0; j < N; ++j) sum += Cd[i + j * M] * xd[j];
            rd[i] = d.elemAsDouble(i) - sum;
        }
        R.resnorm = 0.0;
        for (keel::usize i = 0; i < M; ++i) R.resnorm += rd[i] * rd[i];
    };

    while (outer_iter < max_outer) {
        compute_w();

        keel::usize t = N;
        double max_w = tol;
        for (keel::usize j = 0; j < N; ++j) {
            if (in_Z[j] && w[j] > max_w) { max_w = w[j]; t = j; }
        }
        if (t == N) break;
        ++outer_iter;

        in_Z[t] = 0;

        for (int inner = 0; inner < max_outer; ++inner) {
            Pidx.clear();
            for (keel::usize j = 0; j < N; ++j) if (!in_Z[j]) Pidx.push_back(j);
            const keel::usize p = Pidx.size();
            CP_buf.resize(M * p);
            for (keel::usize k = 0; k < p; ++k) {
                const double *src = Cd + Pidx[k] * M;
                std::copy(src, src + M, CP_buf.data() + k * M);
            }
            CtC.assign(p * p, 0.0);
            Ctd.assign(p, 0.0);
            for (keel::usize a = 0; a < p; ++a) {
                for (keel::usize b = 0; b < p; ++b) {
                    double sum = 0.0;
                    for (keel::usize i = 0; i < M; ++i)
                        sum += CP_buf[i + a * M] * CP_buf[i + b * M];
                    CtC[a + b * p] = sum;
                }
                double sum = 0.0;
                for (keel::usize i = 0; i < M; ++i)
                    sum += CP_buf[i + a * M] * d.elemAsDouble(i);
                Ctd[a] = sum;
            }
            sP.assign(p, 0.0);
            if (!numkit::ops::la_solve(CtC.data(), p, p, Ctd.data(), 1,
                                                    sP.data(), scratch.allocator())) {
                R.exitflag = 1;
                break;
            }

            std::fill(s.begin(), s.end(), 0.0);
            for (keel::usize k = 0; k < p; ++k) s[Pidx[k]] = sP[k];

            bool all_pos = true;
            for (keel::usize k = 0; k < p; ++k) if (sP[k] <= 0.0) { all_pos = false; break; }
            if (all_pos) {
                std::copy(s.begin(), s.end(), xd);
                break;
            }

            double alpha = std::numeric_limits<double>::infinity();
            for (keel::usize k = 0; k < p; ++k) {
                keel::usize j = Pidx[k];
                if (sP[k] <= 0.0) {
                    double denom = xd[j] - sP[k];
                    if (denom > 0.0) {
                        double a = xd[j] / denom;
                        if (a < alpha) alpha = a;
                    }
                }
            }
            if (!std::isfinite(alpha)) { R.exitflag = 1; break; }
            for (keel::usize j = 0; j < N; ++j) xd[j] += alpha * (s[j] - xd[j]);
            for (keel::usize k = 0; k < p; ++k) {
                keel::usize j = Pidx[k];
                if (keel::abs(xd[j]) < tol) {
                    xd[j] = 0.0;
                    in_Z[j] = 1;
                }
            }
        }
        recompute_residual();
    }

    R.iterations = outer_iter;
    R.x = detail::narrowDoubleToSingleIf(R.x, wantSingle, mr);
    R.residual = detail::narrowDoubleToSingleIf(R.residual, wantSingle, mr);
    return R;
}

} // namespace numkit::linalg
