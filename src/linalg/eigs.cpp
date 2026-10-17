// toolboxes/linalg/src/eigs.cpp
//
// eigs & svds: Subset of eigenvalues / singular values.

#include <numkit/linalg/eigs.hpp>
#include <numkit/linalg/eig.hpp>
#include <numkit/linalg/decompositions.hpp>
#include "linalg_detail.hpp"

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <vector>

namespace numkit::linalg {

namespace {

using Complex = std::complex<double>;

struct EigenPair {
    Complex val;
    std::size_t original_index;
};

std::tuple<Value, Value> eig_auto(const Value &A, std::pmr::memory_resource *mr) {
    if (isSymmetricApprox(A, 1e-10)) {
        return eig_symmetric(A, mr);
    }
    auto [U, T] = schur_general(A, mr);
    Value e = ordeig(T, mr);
    const std::size_t n = T.dims().rows();
    Value D = Value::complexMatrix(n, n, mr);
    Complex *dd = D.complexDataMut();
    std::fill(dd, dd + n * n, Complex(0.0, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        dd[i + i * n] = e.isComplex() ? e.complexData()[i] : Complex(e.elemAsDouble(i), 0.0);
    }
    return {detail::narrow_if_real(U, mr), detail::narrow_if_real(D, mr)};
}

} // anonymous namespace

Value eigs_values(const Value &A, std::size_t k, std::pmr::memory_resource *mr)
{
    detail::requireDoubleInput(A, "eigs");
    if (A.dims().ndim() != 2)
        throw Error("eigs: input must be a 2D matrix", 0, 0, "eigs", "", "numkit:eigs:notMatrix");
    const std::size_t n = static_cast<std::size_t>(A.dims().dim(0));
    if (n != static_cast<std::size_t>(A.dims().dim(1)))
        throw Error("eigs: matrix must be square", 0, 0, "eigs", "", "numkit:eigs:notSquare");

    const bool wantSingle = A.type() == ValueType::SINGLE;
    if (n == 0 || k == 0)
        return Value::matrix(0, 1, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
    if (k > n) k = n;

    Value Aw = detail::widenSingleToDouble(A, mr);

    // Full eig. Symmetric real A takes the precise Jacobi path; complex
    // or nonsymmetric A takes the Schur route (eig_values itself is the
    // symmetric-only entry).
    Value ev = (Aw.isComplex() || !isSymmetricApprox(Aw, 1e-10))
                   ? eig_general_values(Aw, mr)
                   : eig_values(Aw, mr);
    const std::size_t num_eig = ev.numel();

    ScratchArena scratch(mr);
    ScratchVec<EigenPair> pairs(num_eig, &scratch);

    for (std::size_t i = 0; i < num_eig; ++i) {
        pairs[i].original_index = i;
        pairs[i].val = ev.isComplex() ? ev.complexData()[i] : Complex(ev.doubleData()[i], 0.0);
    }

    // Sort descending by magnitude
    std::sort(pairs.begin(), pairs.end(), [](const EigenPair &a, const EigenPair &b) {
        return std::abs(a.val) > std::abs(b.val);
    });

    Value out = Value::complexMatrix(k, 1, mr);
    Complex *od = out.complexDataMut();
    for (std::size_t i = 0; i < k; ++i) od[i] = pairs[i].val;

    return detail::narrowDoubleToSingleIf(detail::narrow_if_real(out, mr), wantSingle, mr);
}

std::tuple<Value, Value> eigs(const Value &A, std::size_t k, std::pmr::memory_resource *mr)
{
    detail::requireDoubleInput(A, "eigs");
    if (A.dims().ndim() != 2)
        throw Error("eigs: input must be a 2D matrix", 0, 0, "eigs", "", "numkit:eigs:notMatrix");
    const std::size_t n = static_cast<std::size_t>(A.dims().dim(0));
    if (n != static_cast<std::size_t>(A.dims().dim(1)))
        throw Error("eigs: matrix must be square", 0, 0, "eigs", "", "numkit:eigs:notSquare");

    const bool wantSingle = A.type() == ValueType::SINGLE;
    const ValueType rtype = wantSingle ? ValueType::SINGLE : ValueType::DOUBLE;
    if (n == 0 || k == 0) {
        return {Value::matrix(0, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr)};
    }
    if (k > n) k = n;

    Value Aw = detail::widenSingleToDouble(A, mr);

    // Full eig decomposition A * V = V * D
    auto [V, D] = eig_auto(Aw, mr);

    ScratchArena scratch(mr);
    ScratchVec<EigenPair> pairs(n, &scratch);

    for (std::size_t i = 0; i < n; ++i) {
        pairs[i].original_index = i;
        pairs[i].val = D.isComplex() ? D.complexData()[i + i * n] : Complex(D.doubleData()[i + i * n], 0.0);
    }

    std::sort(pairs.begin(), pairs.end(), [](const EigenPair &a, const EigenPair &b) {
        return std::abs(a.val) > std::abs(b.val);
    });

    auto Vk = Value::complexMatrix(n, k, mr);
    auto Dk = Value::complexMatrix(k, k, mr);
    Complex *vkd = Vk.complexDataMut();
    Complex *dkd = Dk.complexDataMut();
    std::fill(dkd, dkd + k * k, Complex(0.0, 0.0));

    for (std::size_t col = 0; col < k; ++col) {
        std::size_t orig_col = pairs[col].original_index;
        dkd[col + col * k] = pairs[col].val;

        for (std::size_t row = 0; row < n; ++row) {
            vkd[row + col * n] = V.isComplex() ? V.complexData()[row + orig_col * n] : Complex(V.doubleData()[row + orig_col * n], 0.0);
        }
    }

    return {detail::narrowDoubleToSingleIf(detail::narrow_if_real(Vk, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Dk, mr), wantSingle, mr)};
}

Value svds_values(const Value &A, std::size_t k, std::pmr::memory_resource *mr)
{
    detail::requireDoubleInput(A, "svds");
    if (A.dims().ndim() != 2)
        throw Error("svds: input must be a 2D matrix", 0, 0, "svds", "", "numkit:svds:notMatrix");
    const std::size_t m = static_cast<std::size_t>(A.dims().dim(0));
    const std::size_t n = static_cast<std::size_t>(A.dims().dim(1));
    const std::size_t min_mn = std::min(m, n);

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

std::tuple<Value, Value, Value> svds(const Value &A, std::size_t k, std::pmr::memory_resource *mr)
{
    detail::requireDoubleInput(A, "svds");
    if (A.dims().ndim() != 2)
        throw Error("svds: input must be a 2D matrix", 0, 0, "svds", "", "numkit:svds:notMatrix");
    const std::size_t m = static_cast<std::size_t>(A.dims().dim(0));
    const std::size_t n = static_cast<std::size_t>(A.dims().dim(1));
    const std::size_t min_mn = std::min(m, n);

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
        const Complex *ud = U.complexData();
        Complex *ukd = Uk.complexDataMut();
        for (std::size_t j = 0; j < k; ++j) {
            for (std::size_t i = 0; i < m; ++i) ukd[i + j * m] = ud[i + j * m];
        }
    } else {
        const double *ud = U.doubleData();
        double *ukd = Uk.doubleDataMut();
        for (std::size_t j = 0; j < k; ++j) {
            for (std::size_t i = 0; i < m; ++i) ukd[i + j * m] = ud[i + j * m];
        }
    }

    if (S.isComplex()) {
        const Complex *sd = S.complexData();
        Complex *skd = Sk.complexDataMut();
        std::fill(skd, skd + k * k, Complex(0.0, 0.0));
        for (std::size_t i = 0; i < k; ++i) skd[i + i * k] = sd[i + i * m];
    } else {
        const double *sd = S.doubleData();
        double *skd = Sk.doubleDataMut();
        std::fill(skd, skd + k * k, 0.0);
        for (std::size_t i = 0; i < k; ++i) skd[i + i * k] = sd[i + i * m];
    }

    if (V.isComplex()) {
        const Complex *vd = V.complexData();
        Complex *vkd = Vk.complexDataMut();
        for (std::size_t j = 0; j < k; ++j) {
            for (std::size_t i = 0; i < n; ++i) vkd[i + j * n] = vd[i + j * n];
        }
    } else {
        const double *vd = V.doubleData();
        double *vkd = Vk.doubleDataMut();
        for (std::size_t j = 0; j < k; ++j) {
            for (std::size_t i = 0; i < n; ++i) vkd[i + j * n] = vd[i + j * n];
        }
    }

    return {detail::narrowDoubleToSingleIf(detail::narrow_if_real(Uk, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Sk, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Vk, mr), wantSingle, mr)};
}

} // namespace numkit::linalg
