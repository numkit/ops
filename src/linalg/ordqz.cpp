// toolboxes/linalg/src/ordqz.cpp
//
// ordqz — Reorder Generalized Schur decomposition via keel::linalg::ordqz_complex.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/linalg.h>
#include <numkit/linalg/ordqz.hpp>
#include "linalg_detail.hpp"

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <tuple>

namespace numkit::linalg {

std::tuple<Value, Value, Value, Value> ordqz(const Value &AA, const Value &BB,
                                              const Value &Q, const Value &Z,
                                              const Value &select,
                                              keel::Allocator *mr)
{
    if (AA.dims().ndim() != 2 || BB.dims().ndim() != 2 || Q.dims().ndim() != 2 || Z.dims().ndim() != 2)
        throw Error("ordqz: inputs must be 2D matrices", 0, 0, "ordqz", "", "numkit:ordqz:notMatrix");

    const keel::usize n = keel::as<keel::usize>(AA.dims().dim(0));
    if (n != keel::as<keel::usize>(AA.dims().dim(1)) ||
        keel::as<keel::usize>(BB.dims().dim(0)) != n || keel::as<keel::usize>(BB.dims().dim(1)) != n ||
        keel::as<keel::usize>(Q.dims().dim(0)) != n || keel::as<keel::usize>(Q.dims().dim(1)) != n ||
        keel::as<keel::usize>(Z.dims().dim(0)) != n || keel::as<keel::usize>(Z.dims().dim(1)) != n)
        throw Error("ordqz: AA, BB, Q, Z must be n x n square matrices", 0, 0, "ordqz", "", "numkit:ordqz:badDims");

    if (select.numel() != n)
        throw Error("ordqz: select must have length n", 0, 0, "ordqz", "", "numkit:ordqz:badSelect");

    const bool wantSingle = AA.type() == ValueType::SINGLE || BB.type() == ValueType::SINGLE
        || Q.type() == ValueType::SINGLE || Z.type() == ValueType::SINGLE;
    const ValueType rtype = wantSingle ? ValueType::SINGLE : ValueType::DOUBLE;
    if (n == 0) {
        return {Value::matrix(0, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr)};
    }

    Value AAout = Value::complexMatrix(n, n, mr);
    Value BBout = Value::complexMatrix(n, n, mr);
    Value Qout  = Value::complexMatrix(n, n, mr);
    Value Zout  = Value::complexMatrix(n, n, mr);

    auto copyToComplex = [&](const Value &src, keel::c64 *dst) {
        if (src.isComplex()) {
            std::copy(src.complexData(), src.complexData() + n * n, dst);
        } else {
            for (keel::usize i = 0; i < n * n; ++i) dst[i] = keel::c64(src.elemAsDouble(i), 0.0);
        }
    };
    copyToComplex(AA, AAout.complexDataMut());
    copyToComplex(BB, BBout.complexDataMut());
    copyToComplex(Q, Qout.complexDataMut());
    copyToComplex(Z, Zout.complexDataMut());

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u8> sel(n, keel::u8(0), scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) {
        sel[i] = select.isLogical() ? select.logicalData()[i] : (select.elemAsDouble(i) != 0.0 ? keel::u8(1) : keel::u8(0));
    }

    keel::linalg::ordqz_complex(AAout.complexDataMut(), BBout.complexDataMut(),
                        Qout.complexDataMut(), Zout.complexDataMut(),
                        n, sel.data(), mr);

    return {detail::narrowDoubleToSingleIf(detail::narrow_if_real(AAout, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(BBout, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Qout, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Zout, mr), wantSingle, mr)};
}

std::tuple<Value, Value, Value, Value> ordqz(const Value &AA, const Value &BB,
                                              const Value &Q, const Value &Z,
                                              keel::StringRef domain,
                                              keel::Allocator *mr)
{
    const keel::usize n = keel::as<keel::usize>(AA.dims().dim(0));
    auto select = Value::matrix(1, n, ValueType::LOGICAL, mr);
    keel::u8 *sd = select.logicalDataMut();

    for (keel::usize i = 0; i < n; ++i) {
        keel::c64 a = AA.isComplex() ? AA.complexData()[i + i * n] : keel::c64(AA.elemAsDouble(i + i * n), 0.0);
        keel::c64 b = BB.isComplex() ? BB.complexData()[i + i * n] : keel::c64(BB.elemAsDouble(i + i * n), 0.0);
        keel::c64 lambda = (b != keel::c64(0.0, 0.0)) ? (a / b) : keel::c64(1e15, 0.0);

        if (domain == "lhp") {
            sd[i] = (lambda.real() < 0.0) ? 1 : 0;
        } else if (domain == "rhp") {
            sd[i] = (lambda.real() > 0.0) ? 1 : 0;
        } else if (domain == "uip") {
            sd[i] = (keel::abs(lambda) < 1.0) ? 1 : 0;
        } else if (domain == "uop") {
            sd[i] = (keel::abs(lambda) > 1.0) ? 1 : 0;
        } else {
            throw Error(keel::format("ordqz: unknown domain '{}'", domain), 0, 0, "ordqz", "", "numkit:ordqz:badDomain");
        }
    }

    return ordqz(AA, BB, Q, Z, select, mr);
}

} // namespace numkit::linalg
