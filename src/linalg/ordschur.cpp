// toolboxes/linalg/src/ordschur.cpp
//
// ordschur — Reorder Schur decomposition A = U * T * U^H via keel::linalg::ordschur_complex.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/tuple.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/linalg.h>
#include <numkit/linalg/ordschur.hpp>
#include "linalg_detail.hpp"

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::linalg {

keel::Tuple<Value, Value> ordschur(const Value &U, const Value &T,
                                  const Value &select,
                                  keel::Allocator *mr)
{
    detail::requireDoubleInput(U, "ordschur", "U", 1);
    detail::requireDoubleInput(T, "ordschur", "T", 2);
    {
        const ValueType st = select.type();
        if (st != ValueType::DOUBLE && st != ValueType::COMPLEX
            && st != ValueType::SINGLE && st != ValueType::LOGICAL
            && !isIntegerType(st))
            throw Error(keel::format("ordschur: input 3 (select) must be a numeric or"
                                     " logical vector (got a {} {} array).",
                                     detail::shapeOf(select), mtypeName(st)),
                        0, 0, "ordschur", "", "numkit:ordschur:badSelect");
    }
    if (U.dims().ndim() != 2 || T.dims().ndim() != 2)
        throw Error("ordschur: U and T must be 2D matrices", 0, 0, "ordschur", "", "numkit:ordschur:notMatrix");
    const keel::usize n = keel::as<keel::usize>(T.dims().dim(0));
    if (n != keel::as<keel::usize>(T.dims().dim(1)) ||
        keel::as<keel::usize>(U.dims().dim(0)) != n ||
        keel::as<keel::usize>(U.dims().dim(1)) != n)
        throw Error("ordschur: U and T must be n x n square matrices", 0, 0, "ordschur", "", "numkit:ordschur:badDims");

    if (select.numel() != n)
        throw Error("ordschur: select must have length n", 0, 0, "ordschur", "", "numkit:ordschur:badSelect");

    const bool wantSingle = U.type() == ValueType::SINGLE || T.type() == ValueType::SINGLE;
    const ValueType rtype = wantSingle ? ValueType::SINGLE : ValueType::DOUBLE;
    if (n == 0) {
        return {Value::matrix(0, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr)};
    }

    auto U_out = Value::complexMatrix(n, n, mr);
    auto T_out = Value::complexMatrix(n, n, mr);
    keel::c64 *Uc = U_out.complexDataMut();
    keel::c64 *Tc = T_out.complexDataMut();

    if (T.isComplex()) {
        keel::copy(T.complexData(), T.complexData() + n * n, Tc);
    } else {
        for (keel::usize i = 0; i < n * n; ++i) Tc[i] = keel::c64(T.elemAsDouble(i), 0.0);
    }
    if (U.isComplex()) {
        keel::copy(U.complexData(), U.complexData() + n * n, Uc);
    } else {
        for (keel::usize i = 0; i < n * n; ++i) Uc[i] = keel::c64(U.elemAsDouble(i), 0.0);
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u8> sel(n, scratch.allocator());
    if (select.type() == ValueType::LOGICAL) {
        const keel::u8 *ld = select.logicalData();
        for (keel::usize i = 0; i < n; ++i) sel[i] = (ld[i] != 0) ? 1 : 0;
    } else {
        for (keel::usize i = 0; i < n; ++i) sel[i] = (select.elemAsDouble(i) != 0.0) ? 1 : 0;
    }

    keel::linalg::ordschur_complex(Uc, Tc, n, sel.data(), mr);

    return {detail::narrowDoubleToSingleIf(detail::narrow_if_real(U_out, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(T_out, mr), wantSingle, mr)};
}

keel::Tuple<Value, Value> ordschur(const Value &U, const Value &T,
                                  keel::StringRef domain,
                                  keel::Allocator *mr)
{
    detail::requireDoubleInput(U, "ordschur", "U", 1);
    detail::requireDoubleInput(T, "ordschur", "T", 2);
    const keel::usize n = keel::as<keel::usize>(T.dims().dim(0));
    Value select = Value::matrix(1, n, ValueType::LOGICAL, mr);
    keel::u8 *sd = select.logicalDataMut();

    for (keel::usize i = 0; i < n; ++i) {
        keel::c64 val = T.isComplex() ? T.complexData()[i + i * n] : keel::c64(T.elemAsDouble(i + i * n), 0.0);
        bool s = false;
        if (domain == "lhp") {
            s = (val.real() < 0.0);
        } else if (domain == "rhp") {
            s = (val.real() > 0.0);
        } else if (domain == "uip") {
            s = (keel::abs(val) < 1.0);
        } else if (domain == "uop") {
            s = (keel::abs(val) > 1.0);
        } else {
            throw Error(keel::format("ordschur: unknown domain keyword '{}'", domain), 0, 0, "ordschur", "", "numkit:ordschur:badDomain");
        }
        sd[i] = s ? 1 : 0;
    }

    return ordschur(U, T, select, mr);
}

} // namespace numkit::linalg
