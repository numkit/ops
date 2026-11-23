// toolboxes/linalg/src/ldl.cpp
//
// Block LDL' factorization Value adapter over keel::linalg::ldl_factor.
//
//   L * D * L' = A
//
// Migrated 2026-05-25 from toolboxes/builtin/src/language/arrays/ldl.cpp.

#include <keel/core.h>
#include <keel/tuple.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/linalg.h>
#include <keel/math.h>
#include <numkit/linalg/ldl.hpp>

#include "linalg_detail.hpp"

// Compute-only TU: Value substrate + Error, no engine. The ldl builtin
// (CallContext wrapper) lives in ldl_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>

namespace numkit::linalg {

keel::Tuple<Value, Value, Value>
ldl(const Value &A, bool upper_form, bool p_as_vector, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "ldl");
    if (A.dims().is3D())
        throw Error("ldl: input must be 2D",
                    0, 0, "ldl", "", "numkit:ldl:Not2D");
    const keel::usize n = A.dims().rows();
    if (A.dims().cols() != n)
        throw Error("ldl: matrix must be square",
                    0, 0, "ldl", "", "numkit:ldl:NotSquare");

    const bool wantSingle = (A.type() == ValueType::SINGLE);
    if (n == 0) {
        Value L0 = Value::matrix(0, 0, A.type(), mr);
        Value D0 = Value::matrix(0, 0, A.type(), mr);
        Value P0 = p_as_vector ? Value::matrix(1, 0, ValueType::DOUBLE, mr)
                               : Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
        return {L0, D0, P0};
    }

    const Value Aw = detail::widenSingleToDouble(A, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (Aw.isComplex()) {
        const keel::c64 *ad = Aw.complexData();
        for (keel::usize j = 0; j < n; ++j) {
            if (keel::abs(ad[j + j * n].imag()) > 0.0)
                throw Error("ldl: matrix must have real diagonal",
                            0, 0, "ldl", "", "numkit:ldl:ComplexDiag");
        }

        keel::Array<keel::c64> L_buf(n * n, scratch.allocator());
        keel::Array<keel::f64> D_buf(n, scratch.allocator());
        if (keel::linalg::ldl_factor(ad, n, L_buf.data(), D_buf.data(), mr) != 0)
            throw Error("ldl: zero pivot encountered (matrix needs "
                        "Bunch-Kaufman pivoting; not supported in v1)",
                        0, 0, "ldl", "", "numkit:ldl:NeedsBKPivoting");

        Value L = Value::complexMatrix(n, n, mr);
        keel::c64 *Ld = L.complexDataMut();
        if (upper_form) {
            std::fill(Ld, Ld + n * n, keel::c64(0.0, 0.0));
            for (keel::usize j = 0; j < n; ++j)
                for (keel::usize i = 0; i <= j; ++i)
                    Ld[i + j * n] = keel::conj(L_buf[j + i * n]);
        } else {
            std::copy(L_buf.begin(), L_buf.end(), Ld);
        }

        Value D = Value::complexMatrix(n, n, mr);
        keel::c64 *Dd = D.complexDataMut();
        std::fill(Dd, Dd + n * n, keel::c64(0.0, 0.0));
        for (keel::usize k = 0; k < n; ++k) Dd[k + k * n] = keel::c64(D_buf[k], 0.0);

        Value P;
        if (p_as_vector) {
            P = Value::matrix(1, n, ValueType::DOUBLE, mr);
            keel::f64 *pd = P.doubleDataMut();
            for (keel::usize k = 0; k < n; ++k) pd[k] = keel::as<keel::f64>(k + 1);
        } else {
            P = Value::matrix(n, n, ValueType::DOUBLE, mr);
            keel::f64 *pd = P.doubleDataMut();
            std::fill(pd, pd + n * n, 0.0);
            for (keel::usize k = 0; k < n; ++k) pd[k + k * n] = 1.0;
        }
        return {L, D, P};
    }

    keel::Array<keel::f64> L_buf(n * n, scratch.allocator());
    keel::Array<keel::f64> D_buf(n, scratch.allocator());
    if (keel::linalg::ldl_factor(Aw.doubleData(), n, L_buf.data(), D_buf.data(), mr) != 0)
        throw Error("ldl: zero pivot encountered (matrix needs "
                    "Bunch-Kaufman pivoting; not supported in v1)",
                    0, 0, "ldl", "", "numkit:ldl:NeedsBKPivoting");

    Value L = Value::matrix(n, n, ValueType::DOUBLE, mr);
    keel::f64 *Ld = L.doubleDataMut();
    if (upper_form) {
        std::fill(Ld, Ld + n * n, 0.0);
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i <= j; ++i)
                Ld[i + j * n] = L_buf[j + i * n];
    } else {
        std::copy(L_buf.begin(), L_buf.end(), Ld);
    }

    Value D = Value::matrix(n, n, ValueType::DOUBLE, mr);
    keel::f64 *Dd = D.doubleDataMut();
    std::fill(Dd, Dd + n * n, 0.0);
    for (keel::usize k = 0; k < n; ++k) Dd[k + k * n] = D_buf[k];

    Value P;
    if (p_as_vector) {
        P = Value::matrix(1, n, ValueType::DOUBLE, mr);
        keel::f64 *pd = P.doubleDataMut();
        for (keel::usize k = 0; k < n; ++k) pd[k] = keel::as<keel::f64>(k + 1);
    } else {
        P = Value::matrix(n, n, ValueType::DOUBLE, mr);
        keel::f64 *pd = P.doubleDataMut();
        std::fill(pd, pd + n * n, 0.0);
        for (keel::usize k = 0; k < n; ++k) pd[k + k * n] = 1.0;
        P = detail::narrowDoubleToSingleIf(P, wantSingle, mr);
    }
    return {detail::narrowDoubleToSingleIf(L, wantSingle, mr),
            detail::narrowDoubleToSingleIf(D, wantSingle, mr),
            P};
}

} // namespace numkit::linalg
