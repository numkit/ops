// toolboxes/linalg/src/qz.cpp
//
// qz — Generalized Schur decomposition [AA, BB, Q, Z] = qz(A, B) via keel::linalg::qz_factor.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/tuple.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/linalg.h>
#include <numkit/linalg/qz.hpp>
#include "linalg_detail.hpp"

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::linalg {

keel::Tuple<Value, Value, Value, Value> qz(const Value &A, const Value &B,
                                           keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "qz", "A", 1);
    detail::requireDoubleInput(B, "qz", "B", 2);
    if (A.dims().ndim() != 2 || B.dims().ndim() != 2)
        throw Error("qz: A and B must be 2D matrices", 0, 0, "qz", "", "numkit:qz:notMatrix");
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(0));
    if (n != keel::as<keel::usize>(A.dims().dim(1)) ||
        keel::as<keel::usize>(B.dims().dim(0)) != n ||
        keel::as<keel::usize>(B.dims().dim(1)) != n)
        throw Error("qz: A and B must be square matrices of the same size", 0, 0, "qz", "", "numkit:qz:badDims");

    const bool wantSingle = A.type() == ValueType::SINGLE || B.type() == ValueType::SINGLE;
    const ValueType rtype = wantSingle ? ValueType::SINGLE : ValueType::DOUBLE;
    if (n == 0) {
        return {Value::matrix(0, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr)};
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> Ac(n * n, scratch.allocator());
    keel::Array<keel::c64> Bc(n * n, scratch.allocator());

    if (A.isComplex()) {
        keel::copy(A.complexData(), A.complexData() + n * n, Ac.begin());
    } else {
        for (keel::usize i = 0; i < n * n; ++i) Ac[i] = keel::c64(A.elemAsDouble(i), 0.0);
    }
    if (B.isComplex()) {
        keel::copy(B.complexData(), B.complexData() + n * n, Bc.begin());
    } else {
        for (keel::usize i = 0; i < n * n; ++i) Bc[i] = keel::c64(B.elemAsDouble(i), 0.0);
    }

    auto AA_out = Value::complexMatrix(n, n, mr);
    auto BB_out = Value::complexMatrix(n, n, mr);
    auto Q_out  = Value::complexMatrix(n, n, mr);
    auto Z_out  = Value::complexMatrix(n, n, mr);

    if (!keel::linalg::qz_factor(Ac.data(), Bc.data(), n,
                         AA_out.complexDataMut(), BB_out.complexDataMut(),
                         Q_out.complexDataMut(), Z_out.complexDataMut())) {
        throw Error("qz: QZ iteration failed to converge", 0, 0, "qz", "", "numkit:qz:noConverge");
    }

    return {detail::narrowDoubleToSingleIf(detail::narrow_if_real(AA_out, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(BB_out, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Q_out, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Z_out, mr), wantSingle, mr)};
}

} // namespace numkit::linalg
