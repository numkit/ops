// toolboxes/linalg/src/ordqz.cpp
//
// ordqz â€” Reorder Generalized Schur decomposition Q * AA * Z = AA, Q * BB * Z = BB.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/linalg/ordqz.hpp>
#include "linalg_detail.hpp"

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>

namespace numkit::linalg {

namespace {

void swapAdjacentPencil(keel::c64 *AA, keel::c64 *BB, keel::c64 *Q, keel::c64 *Z, keel::usize n, keel::usize j)
{
    keel::c64 a11 = AA[j + j * n];
    keel::c64 a22 = AA[j + 1 + (j + 1) * n];
    keel::c64 a12 = AA[j + (j + 1) * n];

    keel::c64 b11 = BB[j + j * n];
    keel::c64 b22 = BB[j + 1 + (j + 1) * n];
    keel::c64 b12 = BB[j + (j + 1) * n];

    // Compute right rotation (z1, z2) to zero out bottom-left of (A*b22 - B*a22)
    keel::c64 v1 = a11 * b22 - a22 * b11;
    keel::c64 v2 = a12 * b22 - b12 * a22;

    double c_z = 1.0;
    keel::c64 s_z = keel::c64(0.0, 0.0);
    double norm_v2 = keel::abs(v2);

    if (norm_v2 > 1e-15) {
        double norm_v1 = keel::abs(v1);
        double r = std::hypot(norm_v1, norm_v2);
        c_z = norm_v1 / r;
        keel::c64 phase = (norm_v1 > 0.0) ? (v1 / norm_v1) : keel::c64(1.0, 0.0);
        s_z = phase * keel::conj(v2) / r;
    }

    keel::c64 Z00 = c_z;
    keel::c64 Z01 = s_z;
    keel::c64 Z10 = -keel::conj(s_z);
    keel::c64 Z11 = c_z;

    // Apply Z rotation from right on cols j and j+1 of AA, BB, Z
    for (keel::usize i = 0; i < n; ++i) {
        keel::c64 c0a = AA[i + j * n];
        keel::c64 c1a = AA[i + (j + 1) * n];
        AA[i + j * n]       = c0a * Z00 + c1a * Z10;
        AA[i + (j + 1) * n] = c0a * Z01 + c1a * Z11;

        keel::c64 c0b = BB[i + j * n];
        keel::c64 c1b = BB[i + (j + 1) * n];
        BB[i + j * n]       = c0b * Z00 + c1b * Z10;
        BB[i + (j + 1) * n] = c0b * Z01 + c1b * Z11;

        keel::c64 c0z = Z[i + j * n];
        keel::c64 c1z = Z[i + (j + 1) * n];
        Z[i + j * n]       = c0z * Z00 + c1z * Z10;
        Z[i + (j + 1) * n] = c0z * Z01 + c1z * Z11;
    }

    // Left rotation to eliminate subdiagonal BB(j+1, j) created by Z
    keel::c64 b11_new = BB[j + j * n];
    keel::c64 b21_new = BB[j + 1 + j * n];

    double c_q = 1.0;
    keel::c64 s_q = keel::c64(0.0, 0.0);
    double norm_b21 = keel::abs(b21_new);

    if (norm_b21 > 1e-15) {
        double norm_b11 = keel::abs(b11_new);
        double r = std::hypot(norm_b11, norm_b21);
        c_q = norm_b11 / r;
        keel::c64 phase = (norm_b11 > 0.0) ? (b11_new / norm_b11) : keel::c64(1.0, 0.0);
        s_q = phase * keel::conj(b21_new) / r;
    }

    keel::c64 Q00 = c_q;
    keel::c64 Q01 = s_q;
    keel::c64 Q10 = -keel::conj(s_q);
    keel::c64 Q11 = c_q;

    // Apply Q to rows j and j+1 of AA, BB, Q
    for (keel::usize k = 0; k < n; ++k) {
        keel::c64 r0a = AA[j + k * n];
        keel::c64 r1a = AA[j + 1 + k * n];
        AA[j + k * n]     = Q00 * r0a + Q01 * r1a;
        AA[j + 1 + k * n] = Q10 * r0a + Q11 * r1a;

        keel::c64 r0b = BB[j + k * n];
        keel::c64 r1b = BB[j + 1 + k * n];
        BB[j + k * n]     = Q00 * r0b + Q01 * r1b;
        BB[j + 1 + k * n] = Q10 * r0b + Q11 * r1b;

        keel::c64 r0q = Q[j + k * n];
        keel::c64 r1q = Q[j + 1 + k * n];
        Q[j + k * n]     = Q00 * r0q + Q01 * r1q;
        Q[j + 1 + k * n] = Q10 * r0q + Q11 * r1q;
    }
}

} // anonymous namespace

std::tuple<Value, Value, Value, Value> ordqz(const Value &AA, const Value &BB,
                                              const Value &Q, const Value &Z,
                                              const Value &select,
                                              keel::Allocator *mr)
{
    if (AA.dims().ndim() != 2 || BB.dims().ndim() != 2 || Q.dims().ndim() != 2 || Z.dims().ndim() != 2)
        throw Error("ordqz: inputs must be 2D matrices", 0, 0, "ordqz", "", "numkit:ordqz:notMatrix");

    const keel::usize n = static_cast<keel::usize>(AA.dims().dim(0));
    if (n != static_cast<keel::usize>(AA.dims().dim(1)) ||
        static_cast<keel::usize>(BB.dims().dim(0)) != n || static_cast<keel::usize>(BB.dims().dim(1)) != n ||
        static_cast<keel::usize>(Q.dims().dim(0)) != n || static_cast<keel::usize>(Q.dims().dim(1)) != n ||
        static_cast<keel::usize>(Z.dims().dim(0)) != n || static_cast<keel::usize>(Z.dims().dim(1)) != n)
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

    keel::c64 *aad = AAout.complexDataMut();
    keel::c64 *bbd = BBout.complexDataMut();
    keel::c64 *qd  = Qout.complexDataMut();
    keel::c64 *zd  = Zout.complexDataMut();

    // Copy initial matrices as complex
    if (AA.isComplex()) {
        std::copy(AA.complexData(), AA.complexData() + n * n, aad);
    } else {
        for (keel::usize i = 0; i < n * n; ++i) aad[i] = keel::c64(AA.elemAsDouble(i), 0.0);
    }
    if (BB.isComplex()) {
        std::copy(BB.complexData(), BB.complexData() + n * n, bbd);
    } else {
        for (keel::usize i = 0; i < n * n; ++i) bbd[i] = keel::c64(BB.elemAsDouble(i), 0.0);
    }
    if (Q.isComplex()) {
        std::copy(Q.complexData(), Q.complexData() + n * n, qd);
    } else {
        for (keel::usize i = 0; i < n * n; ++i) qd[i] = keel::c64(Q.elemAsDouble(i), 0.0);
    }
    if (Z.isComplex()) {
        std::copy(Z.complexData(), Z.complexData() + n * n, zd);
    } else {
        for (keel::usize i = 0; i < n * n; ++i) zd[i] = keel::c64(Z.elemAsDouble(i), 0.0);
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u8> sel(n, keel::u8(0), scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) {
        sel[i] = select.isLogical() ? select.logicalData()[i] : (select.elemAsDouble(i) != 0.0);
    }

    // Bubble selected eigenvalues to the top left
    keel::usize target = 0;
    for (keel::usize i = 0; i < n; ++i) {
        if (sel[i]) {
            for (keel::usize j = i; j > target; --j) {
                swapAdjacentPencil(aad, bbd, qd, zd, n, j - 1);
                std::swap(sel[j], sel[j - 1]);
            }
            target++;
        }
    }

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
    const keel::usize n = static_cast<keel::usize>(AA.dims().dim(0));
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
