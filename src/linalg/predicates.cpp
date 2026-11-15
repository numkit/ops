// toolboxes/linalg/src/predicates.cpp
//
// MATLAB matrix-structure predicates and bandwidth queries.
//
//   issymmetric(A [, 'skew'])    bool â€” A == A.'  (transpose, no conj)
//   ishermitian(A [, 'skew'])    bool â€” A == A'   (conjugate transpose)
//   isbanded(A, lower, upper)    bool â€” outside-band entries are zero
//   isdiag(A)                    bool â€” same as isbanded(A, 0, 0)
//   istril(A)                    bool â€” same as isbanded(A, n-1, 0)
//   istriu(A)                    bool â€” same as isbanded(A, 0, n-1)
//   bandwidth(A)                 [lower, upper] (or just lower if 1-out)
//   bandwidth(A, 'lower'|'upper')  scalar
//
// All entries comparisons are exact (== 0).
// Migrated 2026-05-25 from toolboxes/builtin/src/language/arrays/predicates.cpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/complex.h>
#include <numkit/linalg/predicates.hpp>

// Compute-only TU: Value substrate + Error, no engine. The is*/bandwidth
// builtins (CallContext wrappers) live in predicates_reg.cpp.
#include "linalg_detail.hpp"           // shared argcheck cores
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <string>

namespace numkit::linalg {

namespace {

inline keel::usize lin(keel::usize i, keel::usize j, keel::usize R) { return i + j * R; }

// 2-D guard for the bandwidth queries only: the reference REJECTS 3-D
// bandwidth inputs (inputMustBe2D), while every is* predicate answers
// logical 0 for N-D arrays instead (reference-probed 2026-09-18 â€” even
// an all-zero 3-D array is not "triangular").
inline void requireMatrix(const Value &A, const char *who)
{
    if (A.dims().is3D() || A.dims().ndim() > 3)
        throw Error(std::string(who) + ": input must be 2D",
                    0, 0, who, "", std::string("numkit:") + who + ":Not2D");
}

bool isbandedImpl(const Value &A, long lower, long upper)
{
    const long R = static_cast<long>(A.dims().rows());
    const long C = static_cast<long>(A.dims().cols());
    if (R == 0 || C == 0) return true;

    const keel::usize Rs = static_cast<keel::usize>(R);
    if (A.isComplex()) {
        for (long j = 0; j < C; ++j)
            for (long i = 0; i < R; ++i) {
                if (j - i > upper || i - j > lower) {
                    keel::c64 z = A.complexElem(static_cast<keel::usize>(i), static_cast<keel::usize>(j));
                    if (z.real() != 0.0 || z.imag() != 0.0) return false;
                }
            }
        return true;
    }
    for (long j = 0; j < C; ++j)
        for (long i = 0; i < R; ++i) {
            if (j - i > upper || i - j > lower) {
                if (A.elemAsDouble(lin(static_cast<keel::usize>(i),
                                      static_cast<keel::usize>(j), Rs)) != 0.0)
                    return false;
            }
        }
    return true;
}

bool issymmetricImpl(const Value &A, bool skew)
{
    const keel::usize R = A.dims().rows();
    const keel::usize C = A.dims().cols();
    if (R != C) return false;
    if (A.isComplex()) {
        for (keel::usize j = 0; j < C; ++j)
            for (keel::usize i = 0; i <= j; ++i) {
                keel::c64 aij = A.complexElem(i, j);
                keel::c64 aji = A.complexElem(j, i);
                keel::c64 want = skew ? -aji : aji;
                if (aij != want) return false;
            }
        return true;
    }
    for (keel::usize j = 0; j < C; ++j)
        for (keel::usize i = 0; i <= j; ++i) {
            double aij = A.elemAsDouble(lin(i, j, R));
            double aji = A.elemAsDouble(lin(j, i, R));
            double want = skew ? -aji : aji;
            if (aij != want) return false;
        }
    return true;
}

bool ishermitianImpl(const Value &A, bool skew)
{
    const keel::usize R = A.dims().rows();
    const keel::usize C = A.dims().cols();
    if (R != C) return false;
    if (A.isComplex()) {
        for (keel::usize j = 0; j < C; ++j)
            for (keel::usize i = 0; i <= j; ++i) {
                keel::c64 aij = A.complexElem(i, j);
                keel::c64 aji = A.complexElem(j, i);
                keel::c64 conj_aji = keel::conj(aji);
                keel::c64 want = skew ? -conj_aji : conj_aji;
                if (aij != want) return false;
            }
        return true;
    }
    return issymmetricImpl(A, skew);
}

std::pair<long, long> bandwidthImpl(const Value &A)
{
    const long R = static_cast<long>(A.dims().rows());
    const long C = static_cast<long>(A.dims().cols());
    long lower = 0, upper = 0;
    if (R == 0 || C == 0) return {0, 0};

    const keel::usize Rs = static_cast<keel::usize>(R);
    if (A.isComplex()) {
        for (long j = 0; j < C; ++j)
            for (long i = 0; i < R; ++i) {
                keel::c64 z = A.complexElem(static_cast<keel::usize>(i), static_cast<keel::usize>(j));
                if (z.real() == 0.0 && z.imag() == 0.0) continue;
                long d = i - j;
                if (d > 0) lower = std::max(lower, d);
                else        upper = std::max(upper, -d);
            }
    } else {
        for (long j = 0; j < C; ++j)
            for (long i = 0; i < R; ++i) {
                if (A.elemAsDouble(lin(static_cast<keel::usize>(i),
                                       static_cast<keel::usize>(j), Rs)) == 0.0)
                    continue;
                long d = i - j;
                if (d > 0) lower = std::max(lower, d);
                else        upper = std::max(upper, -d);
            }
    }
    return {lower, upper};
}

} // anonymous namespace

Value isbanded(const Value &A, long lower, long upper, keel::Allocator *mr)
{
    detail::requireFloatOrLogicalInput(A, "isbanded");
    if (A.dims().ndim() > 2)
        return Value::logicalScalar(false, mr);
    if (lower < 0 || upper < 0)
        throw Error("isbanded: bandwidths must be non-negative",
                    0, 0, "isbanded", "", "numkit:isbanded:NegBand");
    return Value::logicalScalar(isbandedImpl(A, lower, upper), mr);
}

Value isdiag(const Value &A, keel::Allocator *mr)
{
    detail::requireFloatOrLogicalInput(A, "isdiag");
    if (A.dims().ndim() > 2)
        return Value::logicalScalar(false, mr);
    return Value::logicalScalar(isbandedImpl(A, 0, 0), mr);
}

Value istril(const Value &A, keel::Allocator *mr)
{
    detail::requireFloatOrLogicalInput(A, "istril");
    if (A.dims().ndim() > 2)
        return Value::logicalScalar(false, mr);
    const long R = static_cast<long>(A.dims().rows());
    return Value::logicalScalar(isbandedImpl(A, R, 0), mr);
}

Value istriu(const Value &A, keel::Allocator *mr)
{
    detail::requireFloatOrLogicalInput(A, "istriu");
    if (A.dims().ndim() > 2)
        return Value::logicalScalar(false, mr);
    const long C = static_cast<long>(A.dims().cols());
    return Value::logicalScalar(isbandedImpl(A, 0, C), mr);
}

Value issymmetric(const Value &A, bool skew, keel::Allocator *mr)
{
    detail::requireFloatOrLogicalInput(A, "issymmetric");
    if (A.dims().ndim() > 2)
        return Value::logicalScalar(false, mr);
    return Value::logicalScalar(issymmetricImpl(A, skew), mr);
}

Value ishermitian(const Value &A, bool skew, keel::Allocator *mr)
{
    detail::requireFloatOrLogicalInput(A, "ishermitian");
    if (A.dims().ndim() > 2)
        return Value::logicalScalar(false, mr);
    return Value::logicalScalar(ishermitianImpl(A, skew), mr);
}

std::pair<Value, Value>
bandwidth(const Value &A, keel::Allocator *mr)
{
    detail::requireFloatOrLogicalInput(A, "bandwidth");
    requireMatrix(A, "bandwidth");
    auto [lo, up] = bandwidthImpl(A);
    return {Value::scalar(static_cast<double>(lo), mr),
            Value::scalar(static_cast<double>(up), mr)};
}

Value bandwidthOpt(const Value &A, keel::StringRef which, keel::Allocator *mr)
{
    detail::requireFloatOrLogicalInput(A, "bandwidth");
    requireMatrix(A, "bandwidth");
    auto [lo, up] = bandwidthImpl(A);
    if (which == "lower") return Value::scalar(static_cast<double>(lo), mr);
    if (which == "upper") return Value::scalar(static_cast<double>(up), mr);
    throw Error("bandwidth: option must be 'lower' or 'upper'",
                0, 0, "bandwidth", "", "numkit:bandwidth:BadOpt");
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Engine adapters
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

} // namespace numkit::linalg
