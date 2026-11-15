// toolboxes/linalg/src/ordschur.cpp
//
// ordschur â€” Reorder Schur decomposition A = U * T * U^H.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/linalg/ordschur.hpp>
#include "linalg_detail.hpp"

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace numkit::linalg {

namespace {

void swapAdjacent(keel::c64 *T, keel::c64 *U, keel::usize n, keel::usize j)
{
    // Diagonal entries: d1 = T[j,j], d2 = T[j+1,j+1], t12 = T[j,j+1]
    keel::c64 d1 = T[j + j * n];
    keel::c64 d2 = T[j + 1 + (j + 1) * n];
    keel::c64 t12 = T[j + (j + 1) * n];

    keel::c64 x = d1 - d2;
    keel::c64 y = -t12;

    double c = 1.0;
    keel::c64 s = keel::c64(0.0, 0.0);

    double norm_y = keel::abs(y);
    if (norm_y > 1e-15) {
        double norm_x = keel::abs(x);
        double r = std::hypot(norm_x, norm_y);
        c = norm_x / r;
        keel::c64 phase = (norm_x > 0.0) ? (x / norm_x) : keel::c64(1.0, 0.0);
        s = phase * keel::conj(y) / r;
    }

    // G = [c, s; -conj(s), c]
    keel::c64 G00 = c;
    keel::c64 G01 = s;
    keel::c64 G10 = -keel::conj(s);
    keel::c64 G11 = c;

    // Apply G to rows j and j+1 of T: T(j:j+1, :) = G * T(j:j+1, :)
    for (keel::usize k = j; k < n; ++k) {
        keel::c64 r0 = T[j + k * n];
        keel::c64 r1 = T[j + 1 + k * n];
        T[j + k * n]     = G00 * r0 + G01 * r1;
        T[j + 1 + k * n] = G10 * r0 + G11 * r1;
    }

    // Apply G^H to cols j and j+1 of T: T(:, j:j+1) = T(:, j:j+1) * G^H
    // G^H = [conj(G00), conj(G10); conj(G01), conj(G11)]
    keel::c64 GH00 = keel::conj(G00);
    keel::c64 GH01 = keel::conj(G10);
    keel::c64 GH10 = keel::conj(G01);
    keel::c64 GH11 = keel::conj(G11);

    for (keel::usize i = 0; i <= j + 1; ++i) {
        keel::c64 c0 = T[i + j * n];
        keel::c64 c1 = T[i + (j + 1) * n];
        T[i + j * n]       = c0 * GH00 + c1 * GH10;
        T[i + (j + 1) * n] = c0 * GH01 + c1 * GH11;
    }

    // Apply G^H to cols j and j+1 of U: U(:, j:j+1) = U(:, j:j+1) * G^H
    for (keel::usize i = 0; i < n; ++i) {
        keel::c64 c0 = U[i + j * n];
        keel::c64 c1 = U[i + (j + 1) * n];
        U[i + j * n]       = c0 * GH00 + c1 * GH10;
        U[i + (j + 1) * n] = c0 * GH01 + c1 * GH11;
    }
}

} // anonymous namespace

std::tuple<Value, Value> ordschur(const Value &U, const Value &T,
                                  const Value &select,
                                  keel::Allocator *mr)
{
    detail::requireDoubleInput(U, "ordschur", "U", 1);
    detail::requireDoubleInput(T, "ordschur", "T", 2);
    // select: numeric or logical â€” the reference coerces any numeric
    // class to truthiness (ordschur(U,T,int8([1 0])) works); cell/string
    // are rejected (R2025b probes).
    {
        const ValueType st = select.type();
        if (st != ValueType::DOUBLE && st != ValueType::COMPLEX
            && st != ValueType::SINGLE && st != ValueType::LOGICAL
            && !isIntegerType(st))
            throw Error("ordschur: input 3 (select) must be a numeric or"
                        " logical vector (got a " + detail::shapeOf(select)
                            + " " + mtypeName(st) + " array).",
                        0, 0, "ordschur", "", "numkit:ordschur:badSelect");
    }
    if (U.dims().ndim() != 2 || T.dims().ndim() != 2)
        throw Error("ordschur: U and T must be 2D matrices", 0, 0, "ordschur", "", "numkit:ordschur:notMatrix");
    const keel::usize n = static_cast<keel::usize>(T.dims().dim(0));
    if (n != static_cast<keel::usize>(T.dims().dim(1)) ||
        static_cast<keel::usize>(U.dims().dim(0)) != n ||
        static_cast<keel::usize>(U.dims().dim(1)) != n)
        throw Error("ordschur: U and T must be n x n square matrices", 0, 0, "ordschur", "", "numkit:ordschur:badDims");

    if (select.numel() != n)
        throw Error("ordschur: select must have length n", 0, 0, "ordschur", "", "numkit:ordschur:badSelect");

    const bool wantSingle = U.type() == ValueType::SINGLE || T.type() == ValueType::SINGLE;
    const ValueType rtype = wantSingle ? ValueType::SINGLE : ValueType::DOUBLE;
    if (n == 0) {
        return {Value::matrix(0, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr)};
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> Tc(n * n, scratch.allocator());
    keel::Array<keel::c64> Uc(n * n, scratch.allocator());
    keel::Array<keel::u8> sel(n, scratch.allocator());

    if (T.isComplex()) {
        std::copy(T.complexData(), T.complexData() + n * n, Tc.begin());
    } else {
        for (keel::usize i = 0; i < n * n; ++i) Tc[i] = keel::c64(T.elemAsDouble(i), 0.0);
    }

    if (U.isComplex()) {
        std::copy(U.complexData(), U.complexData() + n * n, Uc.begin());
    } else {
        for (keel::usize i = 0; i < n * n; ++i) Uc[i] = keel::c64(U.elemAsDouble(i), 0.0);
    }

    if (select.type() == ValueType::LOGICAL) {
        const keel::u8 *ld = select.logicalData();
        for (keel::usize i = 0; i < n; ++i) sel[i] = (ld[i] != 0) ? 1 : 0;
    } else {
        for (keel::usize i = 0; i < n; ++i) sel[i] = (select.elemAsDouble(i) != 0.0) ? 1 : 0;
    }

    // Bubble-sort selected eigenvalues to the top-left
    for (keel::usize i = 0; i < n; ++i) {
        for (std::intptr_t j = static_cast<std::intptr_t>(n) - 2; j >= static_cast<std::intptr_t>(i); --j) {
            keel::usize uj = static_cast<keel::usize>(j);
            if (sel[uj + 1] && !sel[uj]) {
                swapAdjacent(Tc.data(), Uc.data(), n, uj);
                std::swap(sel[uj], sel[uj + 1]);
            }
        }
    }

    auto U_out = Value::complexMatrix(n, n, mr);
    auto T_out = Value::complexMatrix(n, n, mr);
    std::copy(Uc.begin(), Uc.end(), U_out.complexDataMut());
    std::copy(Tc.begin(), Tc.end(), T_out.complexDataMut());

    return {detail::narrowDoubleToSingleIf(detail::narrow_if_real(U_out, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(T_out, mr), wantSingle, mr)};
}

std::tuple<Value, Value> ordschur(const Value &U, const Value &T,
                                  keel::StringRef domain,
                                  keel::Allocator *mr)
{
    detail::requireDoubleInput(U, "ordschur", "U", 1);
    detail::requireDoubleInput(T, "ordschur", "T", 2);
    const keel::usize n = static_cast<keel::usize>(T.dims().dim(0));
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
