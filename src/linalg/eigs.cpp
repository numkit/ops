// toolboxes/linalg/src/eigs.cpp
//
// eigs & svds: Subset of eigenvalues / singular values.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/linalg/eigs.hpp>
#include <numkit/linalg/eig.hpp>
#include <numkit/linalg/decompositions.hpp>
#include "linalg_detail.hpp"

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace numkit::linalg {

namespace {

struct EigenPair {
    keel::c64 val;
    keel::usize original_index;
};

bool isFiniteComplex(const keel::c64 &c) {
    return std::isfinite(c.real()) && std::isfinite(c.imag());
}

void sortEigenPairsBySigma(keel::Array<EigenPair> &pairs, keel::usize k, const Value &sigma)
{
    enum class Mode { LargestAbs, SmallestAbs, LargestReal, SmallestReal, BothEndsReal, Shift };
    Mode mode = Mode::LargestAbs;
    keel::c64 shiftVal(0.0, 0.0);

    if (!sigma.isEmpty()) {
        if (sigma.isChar() || sigma.isString()) {
            std::string s = sigma.toString();
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (s == "largestabs" || s == "lm") mode = Mode::LargestAbs;
            else if (s == "smallestabs" || s == "sm") mode = Mode::SmallestAbs;
            else if (s == "largestreal" || s == "lr" || s == "la") mode = Mode::LargestReal;
            else if (s == "smallestreal" || s == "sr" || s == "sa") mode = Mode::SmallestReal;
            else if (s == "bothendsreal" || s == "be") mode = Mode::BothEndsReal;
            else
                throw Error("eigs: unknown sigma option '" + s + "'",
                            0, 0, "eigs", "", "numkit:eigs:badSigma");
        } else if (sigma.isNumeric() && sigma.isScalar()) {
            mode = Mode::Shift;
            shiftVal = sigma.isComplex() ? sigma.complexData()[0] : keel::c64(sigma.toScalar(), 0.0);
        }
    }

    std::stable_sort(pairs.begin(), pairs.end(), [&](const EigenPair &a, const EigenPair &b) {
        const bool fa = isFiniteComplex(a.val);
        const bool fb = isFiniteComplex(b.val);
        if (fa != fb) return fa; // finite always before Inf/NaN
        if (!fa) return false;
        switch (mode) {
        case Mode::LargestAbs:
            return keel::abs(a.val) > keel::abs(b.val);
        case Mode::SmallestAbs:
            return keel::abs(a.val) < keel::abs(b.val);
        case Mode::LargestReal:
        case Mode::BothEndsReal:
            return a.val.real() > b.val.real();
        case Mode::SmallestReal:
            return a.val.real() < b.val.real();
        case Mode::Shift:
            return keel::abs(a.val - shiftVal) < keel::abs(b.val - shiftVal);
        }
        return keel::abs(a.val) > keel::abs(b.val);
    });

    if (mode == Mode::BothEndsReal && k > 1 && pairs.size() > 1) {
        keel::usize numFinite = 0;
        while (numFinite < pairs.size() && isFiniteComplex(pairs[numFinite].val))
            ++numFinite;
        if (numFinite > 1) {
            std::vector<EigenPair> reordered;
            reordered.reserve(pairs.size());
            keel::usize lo = 0, hi = numFinite - 1;
            while (lo <= hi && reordered.size() < k) {
                reordered.push_back(pairs[lo++]);
                if (lo <= hi && reordered.size() < k)
                    reordered.push_back(pairs[hi--]);
            }
            for (keel::usize i = 0; i < reordered.size(); ++i)
                pairs[i] = reordered[i];
        }
    }
}

std::tuple<Value, Value> eig_auto(const Value &A, keel::Allocator *mr) {
    if (!A.isComplex() && isSymmetricApprox(A, 1e-10)) {
        return eig_symmetric(A, mr);
    }
    return eig_general_VD(A, mr);
}

std::tuple<Value, Value> eigs_from_VD(const Value &V, const Value &D, keel::usize n,
                                      keel::usize k, const Value &sigma, bool wantSingle,
                                      keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<EigenPair> pairs(n, scratch.allocator());

    for (keel::usize i = 0; i < n; ++i) {
        pairs[i].original_index = i;
        pairs[i].val = D.isComplex() ? D.complexData()[i + i * n]
                                     : keel::c64(D.elemAsDouble(i + i * n), 0.0);
    }

    sortEigenPairsBySigma(pairs, k, sigma);

    auto Vk = Value::complexMatrix(n, k, mr);
    auto Dk = Value::complexMatrix(k, k, mr);
    keel::c64 *vkd = Vk.complexDataMut();
    keel::c64 *dkd = Dk.complexDataMut();
    std::fill(dkd, dkd + k * k, keel::c64(0.0, 0.0));

    for (keel::usize col = 0; col < k; ++col) {
        keel::usize orig_col = pairs[col].original_index;
        dkd[col + col * k] = pairs[col].val;
        for (keel::usize row = 0; row < n; ++row) {
            vkd[row + col * n] = V.isComplex()
                ? V.complexData()[row + orig_col * n]
                : keel::c64(V.elemAsDouble(row + orig_col * n), 0.0);
        }
    }

    return {detail::narrowDoubleToSingleIf(detail::narrow_if_real(Vk, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Dk, mr), wantSingle, mr)};
}

} // anonymous namespace

Value eigs_values(const Value &A, keel::usize k, const Value &sigma, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "eigs");
    if (A.dims().ndim() != 2)
        throw Error("eigs: input must be a 2D matrix", 0, 0, "eigs", "", "numkit:eigs:notMatrix");
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(0));
    if (n != static_cast<keel::usize>(A.dims().dim(1)))
        throw Error("eigs: matrix must be square", 0, 0, "eigs", "", "numkit:eigs:notSquare");

    const bool wantSingle = A.type() == ValueType::SINGLE;
    if (n == 0 || k == 0)
        return Value::matrix(0, 1, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
    if (k > n) k = n;

    Value Aw = detail::widenSingleToDouble(A, mr);
    Value ev = (Aw.isComplex() || !isSymmetricApprox(Aw, 1e-10))
                   ? eig_general_values(Aw, mr)
                   : eig_values(Aw, mr);
    const keel::usize num_eig = ev.numel();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<EigenPair> pairs(num_eig, scratch.allocator());

    for (keel::usize i = 0; i < num_eig; ++i) {
        pairs[i].original_index = i;
        pairs[i].val = ev.isComplex() ? ev.complexData()[i] : keel::c64(ev.doubleData()[i], 0.0);
    }

    sortEigenPairsBySigma(pairs, k, sigma);

    Value out = Value::complexMatrix(k, 1, mr);
    keel::c64 *od = out.complexDataMut();
    for (keel::usize i = 0; i < k; ++i) od[i] = pairs[i].val;

    return detail::narrowDoubleToSingleIf(detail::narrow_if_real(out, mr), wantSingle, mr);
}

Value eigs_values(const Value &A, keel::usize k, keel::Allocator *mr)
{
    return eigs_values(A, k, Value(), mr);
}

std::tuple<Value, Value> eigs(const Value &A, keel::usize k, const Value &sigma, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "eigs");
    if (A.dims().ndim() != 2)
        throw Error("eigs: input must be a 2D matrix", 0, 0, "eigs", "", "numkit:eigs:notMatrix");
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(0));
    if (n != static_cast<keel::usize>(A.dims().dim(1)))
        throw Error("eigs: matrix must be square", 0, 0, "eigs", "", "numkit:eigs:notSquare");

    const bool wantSingle = A.type() == ValueType::SINGLE;
    const ValueType rtype = wantSingle ? ValueType::SINGLE : ValueType::DOUBLE;
    if (n == 0 || k == 0) {
        return {Value::matrix(0, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr)};
    }
    if (k > n) k = n;

    Value Aw = detail::widenSingleToDouble(A, mr);
    auto [V, D] = eig_auto(Aw, mr);
    return eigs_from_VD(V, D, n, k, sigma, wantSingle, mr);
}

std::tuple<Value, Value> eigs(const Value &A, keel::usize k, keel::Allocator *mr)
{
    return eigs(A, k, Value(), mr);
}

std::tuple<Value, Value> eigs_generalized(const Value &A, const Value &B, keel::usize k,
                                          const Value &sigma, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "eigs");
    detail::requireDoubleInput(B, "eigs", "B");
    if (A.dims().ndim() != 2 || B.dims().ndim() != 2)
        throw Error("eigs: inputs must be 2D matrices", 0, 0, "eigs", "", "numkit:eigs:notMatrix");
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(0));
    if (n != static_cast<keel::usize>(A.dims().dim(1)) ||
        n != static_cast<keel::usize>(B.dims().dim(0)) ||
        n != static_cast<keel::usize>(B.dims().dim(1)))
        throw Error("eigs: A and B must be square matrices of the same size",
                    0, 0, "eigs", "", "numkit:eigs:notSquare");

    const bool wantSingle = (A.type() == ValueType::SINGLE || B.type() == ValueType::SINGLE);
    const ValueType rtype = wantSingle ? ValueType::SINGLE : ValueType::DOUBLE;
    if (n == 0 || k == 0) {
        return {Value::matrix(0, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr)};
    }
    if (k > n) k = n;

    if (n > 96 && 2 * k < n && (sigma.isChar() || sigma.isString())) {
        std::string s = sigma.toString();
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (s == "smallestabs" || s == "sm") {
            Value Vk, Dk;
            if (try_eigs_generalized_smallestabs(A, B, k, Vk, Dk, mr)) {
                return {std::move(Vk), std::move(Dk)};
            }
        }
    }

    auto [V, D] = eig_generalized_VD(A, B, mr);
    return eigs_from_VD(V, D, n, k, sigma, wantSingle, mr);
}

Value eigs_generalized_values(const Value &A, const Value &B, keel::usize k,
                              const Value &sigma, keel::Allocator *mr)
{
    auto [V, D] = eigs_generalized(A, B, k, sigma, mr);
    (void)V;
    const keel::usize m = static_cast<keel::usize>(D.dims().dim(0));
    if (D.isComplex()) {
        auto out = Value::complexMatrix(m, 1, mr);
        const keel::c64 *dd = D.complexData();
        keel::c64 *od = out.complexDataMut();
        for (keel::usize i = 0; i < m; ++i) od[i] = dd[i + i * m];
        return out;
    }
    const bool wantSingle = (D.type() == ValueType::SINGLE);
    auto out = Value::matrix(m, 1, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < m; ++i)
        out.doubleDataMut()[i] = D.elemAsDouble(i + i * m);
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

Value svds_values(const Value &A, keel::usize k, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "svds");
    if (A.dims().ndim() != 2)
        throw Error("svds: input must be a 2D matrix", 0, 0, "svds", "", "numkit:svds:notMatrix");
    const keel::usize m = static_cast<keel::usize>(A.dims().dim(0));
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(1));
    const keel::usize min_mn = std::min(m, n);

    const bool wantSingle = A.type() == ValueType::SINGLE;
    if (min_mn == 0 || k == 0)
        return Value::matrix(0, 1, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
    if (k > min_mn) k = min_mn;

    Value Aw = detail::widenSingleToDouble(A, mr);
    Value sv = svd_values(Aw, mr);
    Value out = Value::matrix(k, 1, ValueType::DOUBLE, mr);
    std::copy(sv.doubleData(), sv.doubleData() + k, out.doubleDataMut());

    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

std::tuple<Value, Value, Value> svds(const Value &A, keel::usize k, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "svds");
    if (A.dims().ndim() != 2)
        throw Error("svds: input must be a 2D matrix", 0, 0, "svds", "", "numkit:svds:notMatrix");
    const keel::usize m = static_cast<keel::usize>(A.dims().dim(0));
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(1));
    const keel::usize min_mn = std::min(m, n);

    const bool wantSingle = A.type() == ValueType::SINGLE;
    const ValueType rtype = wantSingle ? ValueType::SINGLE : ValueType::DOUBLE;
    if (min_mn == 0 || k == 0) {
        return {Value::matrix(m, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr),
                Value::matrix(n, 0, rtype, mr)};
    }
    if (k > min_mn) k = min_mn;

    Value Aw = detail::widenSingleToDouble(A, mr);
    auto [U, S, V] = svd_decompose(Aw, mr);

    auto Uk = Value::matrix(m, k, U.type(), mr);
    auto Sk = Value::matrix(k, k, S.type(), mr);
    auto Vk = Value::matrix(n, k, V.type(), mr);

    if (U.isComplex()) {
        const keel::c64 *ud = U.complexData();
        keel::c64 *ukd = Uk.complexDataMut();
        for (keel::usize j = 0; j < k; ++j) {
            for (keel::usize i = 0; i < m; ++i) ukd[i + j * m] = ud[i + j * m];
        }
    } else {
        const double *ud = U.doubleData();
        double *ukd = Uk.doubleDataMut();
        for (keel::usize j = 0; j < k; ++j) {
            for (keel::usize i = 0; i < m; ++i) ukd[i + j * m] = ud[i + j * m];
        }
    }

    if (S.isComplex()) {
        const keel::c64 *sd = S.complexData();
        keel::c64 *skd = Sk.complexDataMut();
        std::fill(skd, skd + k * k, keel::c64(0.0, 0.0));
        for (keel::usize i = 0; i < k; ++i) skd[i + i * k] = sd[i + i * m];
    } else {
        const double *sd = S.doubleData();
        double *skd = Sk.doubleDataMut();
        std::fill(skd, skd + k * k, 0.0);
        for (keel::usize i = 0; i < k; ++i) skd[i + i * k] = sd[i + i * m];
    }

    if (V.isComplex()) {
        const keel::c64 *vd = V.complexData();
        keel::c64 *vkd = Vk.complexDataMut();
        for (keel::usize j = 0; j < k; ++j) {
            for (keel::usize i = 0; i < n; ++i) vkd[i + j * n] = vd[i + j * n];
        }
    } else {
        const double *vd = V.doubleData();
        double *vkd = Vk.doubleDataMut();
        for (keel::usize j = 0; j < k; ++j) {
            for (keel::usize i = 0; i < n; ++i) vkd[i + j * n] = vd[i + j * n];
        }
    }

    return {detail::narrowDoubleToSingleIf(detail::narrow_if_real(Uk, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Sk, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Vk, mr), wantSingle, mr)};
}

} // namespace numkit::linalg
