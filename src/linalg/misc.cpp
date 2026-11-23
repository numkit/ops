// toolboxes/linalg/src/misc.cpp
//
// Miscellaneous linalg utilities: rref (via keel::linalg::rref) + planerot.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/linalg.h>
#include <keel/math.h>
#include <numkit/linalg/misc.hpp>

#include "linalg_detail.hpp"

// Compute-only TU: Value substrate + Error, no engine. The rref/planerot
// builtins (CallContext wrappers) live in misc_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::linalg {

namespace {

template <typename T>
keel::f64 single_rref_default_tol(const T *A, keel::usize M, keel::usize N)
{
    keel::f64 anorm = 0.0;
    for (keel::usize i = 0; i < M; ++i) {
        keel::f64 s = 0.0;
        for (keel::usize j = 0; j < N; ++j) s += keel::abs(A[i + j * M]);
        if (s > anorm) anorm = s;
    }
    const keel::f32 anorm_f = keel::as<keel::f32>(anorm);
    const keel::f32 eps_f = keel::Limits<keel::f32>::kEpsilon;
    const keel::f64 eps_anorm = (anorm_f == 0.0f)
        ? keel::as<keel::f64>(eps_f)
        : keel::as<keel::f64>(keel::nextafter(anorm_f, keel::Limits<keel::f32>::infinity()) - anorm_f);
    return keel::as<keel::f64>(keel::max(M, N)) * eps_anorm;
}

} // namespace

// rref: Gauss-Jordan elimination with partial pivoting and tolerance gate.
keel::Pair<Value, Value>
rref(const Value &A, bool have_tol, keel::f64 tol_user, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "rref");
    if (A.dims().is3D())
        throw Error("rref: input must be 2D",
                    0, 0, "rref", "", "numkit:rref:Not2D");

    const bool wantSingle = (A.type() == ValueType::SINGLE);
    const keel::usize M = A.dims().rows();
    const keel::usize N = A.dims().cols();
    if (M == 0 || N == 0) {
        Value R = Value::matrix(M, N, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
        Value jb = Value::matrix(1, 0, ValueType::DOUBLE, mr);
        return {R, jb};
    }

    const Value Aw = detail::widenSingleToDouble(A, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::usize> piv_cols(keel::min(M, N), scratch.allocator());

    if (Aw.isComplex()) {
        Value R = Value::complexMatrix(M, N, mr);
        keel::copy(Aw.complexData(), Aw.complexData() + M * N, R.complexDataMut());
        const keel::f64 tol = have_tol ? tol_user
            : (wantSingle ? single_rref_default_tol(R.complexData(), M, N) : -1.0);
        const keel::usize rank = keel::linalg::rref(R.complexDataMut(), M, N, tol, piv_cols.data());
        R = detail::narrow_if_real(R, mr);

        Value jb = Value::matrix(rank == 0 ? 0 : 1, rank, ValueType::DOUBLE, mr);
        keel::f64 *jbd = jb.doubleDataMut();
        for (keel::usize k = 0; k < rank; ++k) jbd[k] = keel::as<keel::f64>(piv_cols[k] + 1);
        return {detail::narrowDoubleToSingleIf(R, wantSingle, mr), jb};
    }

    Value R = Value::matrix(M, N, ValueType::DOUBLE, mr);
    keel::copy(Aw.doubleData(), Aw.doubleData() + M * N, R.doubleDataMut());
    const keel::f64 tol = have_tol ? tol_user
        : (wantSingle ? single_rref_default_tol(R.doubleData(), M, N) : -1.0);
    const keel::usize rank = keel::linalg::rref(R.doubleDataMut(), M, N, tol, piv_cols.data());

    Value jb = Value::matrix(rank == 0 ? 0 : 1, rank, ValueType::DOUBLE, mr);
    keel::f64 *jbd = jb.doubleDataMut();
    for (keel::usize k = 0; k < rank; ++k) jbd[k] = keel::as<keel::f64>(piv_cols[k] + 1);
    return {detail::narrowDoubleToSingleIf(R, wantSingle, mr), jb};
}

// planerot: G * [x; y] = [r; 0] where r = hypot(x, y).
keel::Pair<Value, Value>
planerot(const Value &xy, keel::Allocator *mr)
{
    if (xy.dims().is3D() || xy.numel() != 2)
        throw Error("planerot: input must be a 2-element vector",
                    0, 0, "planerot", "", "numkit:planerot:BadShape");
    if (xy.isComplex())
        throw Error("planerot: complex input not supported",
                    0, 0, "planerot", "", "numkit:planerot:NoComplex");

    const bool wantSingle = (xy.type() == ValueType::SINGLE);
    const keel::f64 x = xy.elemAsDouble(0);
    const keel::f64 y = xy.elemAsDouble(1);

    Value G = Value::matrix(2, 2, ValueType::DOUBLE, mr);
    Value yo = Value::matrix(2, 1, ValueType::DOUBLE, mr);
    keel::f64 *gd = G.doubleDataMut();
    keel::f64 *yd = yo.doubleDataMut();

    if (x == 0.0 && y == 0.0) {
        gd[0] = 1.0; gd[1] = 0.0; gd[2] = 0.0; gd[3] = 1.0;
        yd[0] = 0.0; yd[1] = 0.0;
        return {detail::narrowDoubleToSingleIf(G, wantSingle, mr),
                detail::narrowDoubleToSingleIf(yo, wantSingle, mr)};
    }

    const keel::f64 r = keel::hypot(x, y);
    const keel::f64 c = x / r;
    const keel::f64 s = y / r;
    gd[0] = c;
    gd[1] = -s;
    gd[2] = s;
    gd[3] = c;
    yd[0] = r;
    yd[1] = 0.0;
    return {detail::narrowDoubleToSingleIf(G, wantSingle, mr),
            detail::narrowDoubleToSingleIf(yo, wantSingle, mr)};
}

} // namespace numkit::linalg
