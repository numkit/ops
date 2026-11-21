// toolboxes/linalg/src/solvers.cpp
//
// linsolve / lsqminnorm / lsqnonneg — Value adapters over keel::linalg kernels.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/string.h>
#include <keel/complex.h>
#include <keel/linalg.h>
#include <keel/math.h>
#include <numkit/linalg/solvers.hpp>
#include <numkit/linalg/decompositions.hpp>
#include "linalg_detail.hpp"

#include <numkit/linalg/pseudo_subspace.hpp>
#include <numkit/ops/la_solve.hpp>
#include <numkit/builtin/ops.hpp>

// Compute-only TU: Value substrate + Error, no engine. The linsolve /
// lsqminnorm / lsqnonneg builtins (CallContext wrappers) live in
// solvers_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>

namespace numkit::linalg {

Value linsolve(const Value &A, const Value &B, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "linsolve", "A", 1);
    detail::requireDoubleInput(B, "linsolve", "B", 2);
    if (A.dims().ndim() != 2 || B.dims().ndim() != 2)
        throw Error("linsolve: A and B must be 2D matrices",
                    0, 0, "linsolve", "", "numkit:linsolve:notMatrix");
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    const keel::usize mb = keel::as<keel::usize>(B.dims().dim(0));
    const keel::usize nrhs = keel::as<keel::usize>(B.dims().dim(1));
    if (m != mb)
        throw Error("linsolve: A and B must have the same number of rows",
                    0, 0, "linsolve", "", "numkit:linsolve:badDims");

    const bool wantSingle = A.type() == ValueType::SINGLE || B.type() == ValueType::SINGLE;
    Value Aw = detail::widenSingleToDouble(A, mr);
    Value Bw = detail::widenSingleToDouble(B, mr);

    if (Aw.isComplex() || Bw.isComplex()) {
        if (m < n) {
            throw Error("linsolve: complex wide matrices (m < n) deferred",
                        0, 0, "linsolve", "", "numkit:linsolve:wide");
        }
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<keel::c64> A_buf(m * n, scratch.allocator());
        keel::Array<keel::c64> B_buf(m * nrhs, scratch.allocator());

        if (Aw.isComplex()) {
            std::copy(Aw.complexData(), Aw.complexData() + m * n, A_buf.begin());
        } else {
            const keel::f64 *ad = Aw.doubleData();
            for (keel::usize i = 0; i < m * n; ++i) A_buf[i] = ad[i];
        }
        if (Bw.isComplex()) {
            std::copy(Bw.complexData(), Bw.complexData() + m * nrhs, B_buf.begin());
        } else {
            const keel::f64 *bd = Bw.doubleData();
            for (keel::usize i = 0; i < m * nrhs; ++i) B_buf[i] = bd[i];
        }

        Value out = Value::complexMatrix(n, nrhs, mr);
        if (!keel::linalg::la_solve(A_buf.data(), m, n, B_buf.data(), nrhs, out.complexDataMut(), mr)) {
            throw Error("linsolve: A is singular or rank-deficient",
                        0, 0, "linsolve", "", "numkit:linsolve:singular");
        }
        return detail::narrowDoubleToSingleIf(detail::narrow_if_real(out, mr), wantSingle, mr);
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto out = Value::matrix(n, nrhs, ValueType::DOUBLE, mr);
    if (!keel::linalg::la_solve(Aw.doubleData(), m, n, Bw.doubleData(), nrhs,
                        out.doubleDataMut(), scratch.allocator()))
        throw Error("linsolve: A is singular or rank-deficient",
                    0, 0, "linsolve", "", "numkit:linsolve:singular");
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

Value lsqminnorm(const Value &A, const Value &B, bool have_tol, keel::f64 tol_user,
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

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> db(M, scratch.allocator());
    for (keel::usize i = 0; i < M; ++i) db[i] = d.elemAsDouble(i);

    NnlsResult R;
    R.exitflag = 1;
    R.algorithm = keel::String::from("active-set", mr);
    R.message = keel::String::from("Optimization terminated.", mr);
    R.x = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    R.residual = Value::matrix(M, 1, ValueType::DOUBLE, mr);

    const keel::linalg::NnlsStats st = keel::linalg::lsqnonneg(Cw.doubleData(), M, N, db.data(),
                                               R.x.doubleDataMut(), R.residual.doubleDataMut(), mr);
    R.resnorm = st.resnorm;
    R.iterations = keel::as<keel::i32>(st.iterations);
    R.x = detail::narrowDoubleToSingleIf(R.x, wantSingle, mr);
    R.residual = detail::narrowDoubleToSingleIf(R.residual, wantSingle, mr);
    return R;
}

} // namespace numkit::linalg
