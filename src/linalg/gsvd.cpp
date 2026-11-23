// toolboxes/linalg/src/gsvd.cpp
//
// gsvd — Generalized Singular Value Decomposition via keel::gsvd_*.

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/linalg.h>
#include <numkit/linalg/gsvd.hpp>
#include "linalg_detail.hpp"

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::linalg {

Value gsvd_values(const Value &A, const Value &B, keel::Allocator *mr)
{
    Value Ac = (A.type() == ValueType::LOGICAL)
                   ? detail::coerceToDoubleMatrix(A, mr) : A;
    Value Bc = (B.type() == ValueType::LOGICAL)
                   ? detail::coerceToDoubleMatrix(B, mr) : B;
    detail::requireDoubleInput(Ac, "gsvd", "A", 1);
    detail::requireDoubleInput(Bc, "gsvd", "B", 2);
    if (Ac.dims().ndim() != 2 || Bc.dims().ndim() != 2)
        throw Error("gsvd: A and B must be 2D matrices", 0, 0, "gsvd", "", "numkit:gsvd:notMatrix");
    const keel::usize m = keel::as<keel::usize>(Ac.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(Ac.dims().dim(1));
    const keel::usize p = keel::as<keel::usize>(Bc.dims().dim(0));
    if (keel::as<keel::usize>(Bc.dims().dim(1)) != n)
        throw Error("gsvd: A and B must have the same number of columns", 0, 0, "gsvd", "", "numkit:gsvd:badDims");

    const bool wantSingle = Ac.type() == ValueType::SINGLE || Bc.type() == ValueType::SINGLE;
    if (m == 0 || n == 0 || p == 0)
        return Value::matrix(0, 1, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    Ac = detail::widenSingleToDouble(Ac, mr);
    Bc = detail::widenSingleToDouble(Bc, mr);

    const keel::usize num_vals = keel::min(m, n);
    Value sigmas = Value::matrix(num_vals, 1, ValueType::DOUBLE, mr);

    if (Ac.isComplex() || Bc.isComplex()) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<keel::c64> Abuf(m * n, scratch.allocator());
        keel::Array<keel::c64> Bbuf(p * n, scratch.allocator());
        for (keel::usize i = 0; i < m * n; ++i)
            Abuf[i] = Ac.isComplex() ? Ac.complexData()[i] : keel::c64(Ac.doubleData()[i], 0.0);
        for (keel::usize i = 0; i < p * n; ++i)
            Bbuf[i] = Bc.isComplex() ? Bc.complexData()[i] : keel::c64(Bc.doubleData()[i], 0.0);
        keel::linalg::gsvd_values(Abuf.data(), m, Bbuf.data(), p, n, sigmas.doubleDataMut(), mr);
    } else {
        keel::linalg::gsvd_values(Ac.doubleData(), m, Bc.doubleData(), p, n, sigmas.doubleDataMut(), mr);
    }

    return detail::narrowDoubleToSingleIf(sigmas, wantSingle, mr);
}

keel::Tuple<Value, Value, Value, Value, Value> gsvd(const Value &A, const Value &B,
                                                   keel::Allocator *mr)
{
    Value Ac = (A.type() == ValueType::LOGICAL)
                   ? detail::coerceToDoubleMatrix(A, mr) : A;
    Value Bc = (B.type() == ValueType::LOGICAL)
                   ? detail::coerceToDoubleMatrix(B, mr) : B;
    detail::requireDoubleInput(Ac, "gsvd", "A", 1);
    detail::requireDoubleInput(Bc, "gsvd", "B", 2);
    if (Ac.dims().ndim() != 2 || Bc.dims().ndim() != 2)
        throw Error("gsvd: A and B must be 2D matrices", 0, 0, "gsvd", "", "numkit:gsvd:notMatrix");
    const keel::usize m = keel::as<keel::usize>(Ac.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(Ac.dims().dim(1));
    const keel::usize p = keel::as<keel::usize>(Bc.dims().dim(0));
    if (keel::as<keel::usize>(Bc.dims().dim(1)) != n)
        throw Error("gsvd: A and B must have the same number of columns", 0, 0, "gsvd", "", "numkit:gsvd:badDims");

    const bool wantSingle = Ac.type() == ValueType::SINGLE || Bc.type() == ValueType::SINGLE;
    const ValueType rtype = wantSingle ? ValueType::SINGLE : ValueType::DOUBLE;
    if (m == 0 || n == 0 || p == 0) {
        return {Value::matrix(m, m, rtype, mr),
                Value::matrix(p, p, rtype, mr),
                Value::matrix(n, n, rtype, mr),
                Value::matrix(m, n, rtype, mr),
                Value::matrix(p, n, rtype, mr)};
    }

    Ac = detail::widenSingleToDouble(Ac, mr);
    Bc = detail::widenSingleToDouble(Bc, mr);

    auto C = Value::matrix(m, n, ValueType::DOUBLE, mr);
    auto S_out = Value::matrix(p, n, ValueType::DOUBLE, mr);

    if (Ac.isComplex() || Bc.isComplex()) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<keel::c64> Abuf(m * n, scratch.allocator());
        keel::Array<keel::c64> Bbuf(p * n, scratch.allocator());
        for (keel::usize i = 0; i < m * n; ++i)
            Abuf[i] = Ac.isComplex() ? Ac.complexData()[i] : keel::c64(Ac.doubleData()[i], 0.0);
        for (keel::usize i = 0; i < p * n; ++i)
            Bbuf[i] = Bc.isComplex() ? Bc.complexData()[i] : keel::c64(Bc.doubleData()[i], 0.0);

        auto U = Value::complexMatrix(m, m, mr);
        auto V = Value::complexMatrix(p, n, mr);
        auto X = Value::complexMatrix(n, n, mr);
        keel::linalg::gsvd_full(Abuf.data(), m, Bbuf.data(), p, n,
                        U.complexDataMut(), V.complexDataMut(), X.complexDataMut(),
                        C.doubleDataMut(), S_out.doubleDataMut(), mr);
        return {detail::narrowDoubleToSingleIf(detail::narrow_if_real(U, mr), wantSingle, mr),
                detail::narrowDoubleToSingleIf(detail::narrow_if_real(V, mr), wantSingle, mr),
                detail::narrowDoubleToSingleIf(detail::narrow_if_real(X, mr), wantSingle, mr),
                detail::narrowDoubleToSingleIf(C, wantSingle, mr),
                detail::narrowDoubleToSingleIf(S_out, wantSingle, mr)};
    }

    auto U = Value::matrix(m, m, ValueType::DOUBLE, mr);
    auto V = Value::matrix(p, n, ValueType::DOUBLE, mr);
    auto X = Value::matrix(n, n, ValueType::DOUBLE, mr);
    keel::linalg::gsvd_full(Ac.doubleData(), m, Bc.doubleData(), p, n,
                    U.doubleDataMut(), V.doubleDataMut(), X.doubleDataMut(),
                    C.doubleDataMut(), S_out.doubleDataMut(), mr);
    return {detail::narrowDoubleToSingleIf(U, wantSingle, mr),
            detail::narrowDoubleToSingleIf(V, wantSingle, mr),
            detail::narrowDoubleToSingleIf(X, wantSingle, mr),
            detail::narrowDoubleToSingleIf(C, wantSingle, mr),
            detail::narrowDoubleToSingleIf(S_out, wantSingle, mr)};
}

} // namespace numkit::linalg
