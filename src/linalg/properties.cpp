// toolboxes/linalg/src/properties.cpp
//
// inv / det / trace / rank / cond / normest / rcond â€” and engine adapters.
// Migrated from toolboxes/builtin/src/language/arrays/{matrix,linalg_extras}.cpp.
//
// rank_of / cond_2norm / normest call the SVD kernel from
// nk/linalg/decompositions.hpp. inv / rcond use the la_solve
// kernel still living in toolboxes/builtin.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/linalg/properties.hpp>
#include "linalg_detail.hpp"

#include <numkit/ops/la_solve.hpp>   // numkit::ops::la_solve
#include <numkit/linalg/decompositions.hpp>       // svd_values
#include <numkit/linalg/eig.hpp>                  // condeig uses eig_symmetric / eig_general_VD
#include <numkit/linalg/norms.hpp>                // cond_pnorm uses norm_*

// Compute-only TU: Value substrate + Error, no engine. The inv/trace/det/
// rank/cond/normest/rcond/condest/condeig builtins (CallContext wrappers)
// live in properties_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace numkit::linalg {

namespace {

// Solve A_buf (mÃ—n column-major) against B_buf (mÃ—nrhs col-major) and
// write the result (nÃ—nrhs) into outX. Returns false on singular /
// rank-deficient / wide A.
[[maybe_unused]] bool laSolveWrap(const keel::f64 *A_buf, keel::usize m, keel::usize n,
                 const keel::f64 *B_buf, keel::usize nrhs, keel::f64 *outX,
                 keel::Allocator *mr)
{
    return numkit::ops::la_solve(A_buf, m, n, B_buf, nrhs, outX, mr);
}

// Default tolerance for rank-cutoff: max(m,n) * eps(sigma_max).
keel::f64 defaultRankTol(keel::usize m, keel::usize n, keel::f64 sigma_max, bool isSingle = false)
{
    const keel::f64 eps = isSingle
        ? keel::as<keel::f64>(std::numeric_limits<keel::f32>::epsilon())
        : std::numeric_limits<keel::f64>::epsilon();
    return keel::as<keel::f64>(std::max(m, n)) * sigma_max * eps;
}

// 1-norm of a column-major M×N matrix: max column sum of |a_ij|.
// Used by rcond and condest.
template <typename T>
keel::f64 matrix_one_norm(const T *A, keel::usize M, keel::usize N)
{
    if (M == 0 || N == 0) return 0.0;
    keel::f64 maxv = 0.0;
    for (keel::usize j = 0; j < N; ++j) {
        keel::f64 s = 0.0;
        for (keel::usize i = 0; i < M; ++i) s += keel::abs(A[i + j * M]);
        if (s > maxv) maxv = s;
    }
    return maxv;
}

keel::f64 value_one_norm(const Value &A, keel::usize M, keel::usize N)
{
    if (A.isComplex())
        return matrix_one_norm(A.complexData(), M, N);
    if (A.type() == ValueType::SINGLE)
        return matrix_one_norm(A.singleData(), M, N);
    return matrix_one_norm(A.doubleData(), M, N);
}

} // anonymous namespace

Value inv(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "inv");
    if (A.dims().ndim() != 2)
        throw Error("inv: input must be a 2D matrix",
                    0, 0, "inv", "", "numkit:inv:notMatrix");
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    if (m != n)
        throw Error("inv: matrix must be square",
                    0, 0, "inv", "", "numkit:inv:notSquare");
    if (m == 0)
        return Value::matrix(0, 0, A.type(), mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (A.isComplex()) {
        keel::Array<keel::c64> I_buf(n * n, 0.0, scratch.allocator());
        for (keel::usize i = 0; i < n; ++i) I_buf[i + i * n] = keel::c64(1.0, 0.0);
        Value out = Value::complexMatrix(n, n, mr);
        if (!numkit::ops::la_solve(A.complexData(), n, n, I_buf.data(), n, out.complexDataMut(), mr)) {
            const keel::f64 inf = std::numeric_limits<keel::f64>::infinity();
            for (keel::usize i = 0; i < n * n; ++i) out.complexDataMut()[i] = keel::c64(inf, 0.0);
        }
        return detail::narrow_if_real(out, mr);
    }

    const bool wantSingle = A.type() == ValueType::SINGLE;
    Value Aw = detail::widenSingleToDouble(A, mr);

    keel::Array<keel::f64> I_buf(n * n, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) I_buf[i + i * n] = 1.0;

    Value out = Value::matrix(n, n, ValueType::DOUBLE, mr);
    if (!numkit::ops::la_solve(Aw.doubleData(), n, n, I_buf.data(), n, out.doubleDataMut(), mr)) {
        const keel::f64 inf = std::numeric_limits<keel::f64>::infinity();
        std::fill(out.doubleDataMut(), out.doubleDataMut() + n * n, inf);
    }
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

Value trace(const Value &A, keel::Allocator *mr)
{
    const ValueType t = A.type();
    if (t != ValueType::DOUBLE && t != ValueType::COMPLEX
        && t != ValueType::SINGLE && t != ValueType::LOGICAL
        && !isIntegerType(t))
        throw Error(keel::format("trace: input 1 (A) must be a numeric or logical matrix"
                                 " (got a {} {} array).",
                                 detail::shapeOf(A), mtypeName(t)),
                    0, 0, "trace", "", "numkit:trace:inputType");
    if (A.dims().ndim() != 2)
        throw Error("trace: input must be a 2D matrix",
                    0, 0, "trace", "", "numkit:trace:notMatrix");
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    const keel::usize k = std::min(m, n);
    if (A.isComplex()) {
        keel::c64 s{0.0, 0.0};
        const keel::c64 *p = A.complexData();
        for (keel::usize i = 0; i < k; ++i)
            s += p[i + i * m];
        return numkit::narrowComplex(Value::complexScalar(s, mr), mr);
    }
    keel::f64 s = 0.0;
    if (t == ValueType::DOUBLE) {
        const keel::f64 *p = A.doubleData();
        for (keel::usize i = 0; i < k; ++i)
            s += p[i + i * m];
    } else {
        for (keel::usize i = 0; i < k; ++i)
            s += A.elemAsDouble(i + i * m);
    }
    return detail::narrowDoubleToSingleIf(Value::scalar(s, mr), t == ValueType::SINGLE, mr);
}

Value det(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "det");
    if (A.dims().ndim() != 2)
        throw Error("det: input must be a 2D matrix",
                    0, 0, "det", "", "numkit:det:notMatrix");
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    if (m != n)
        throw Error("det: matrix must be square",
                    0, 0, "det", "", "numkit:det:notSquare");
    const bool wantSingle = A.type() == ValueType::SINGLE;
    if (m == 0)
        return detail::narrowDoubleToSingleIf(Value::scalar(1.0, mr), wantSingle, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (A.isComplex()) {
        keel::Array<keel::c64> A_buf(m * n, scratch.allocator());
        std::copy(A.complexData(), A.complexData() + m * n, A_buf.begin());
        keel::Array<keel::i32> piv(n, scratch.allocator());
        if (!numkit::ops::lu_factor_inplace(A_buf.data(), n, piv.data(), n, n))
            return Value::scalar(0.0, mr);

        int sign = 1;
        for (keel::usize k = 0; k < n; ++k) {
            if (piv[k] != keel::as<keel::i32>(k)) {
                sign = -sign;
            }
        }
        keel::c64 prod = keel::as<keel::f64>(sign);
        for (keel::usize i = 0; i < n; ++i)
            prod *= A_buf[i + i * n];

        return detail::narrow_if_real(Value::complexScalar(prod, mr), mr);
    }

    Value Aw = detail::widenSingleToDouble(A, mr);
    keel::Array<keel::f64> A_buf(m * n, scratch.allocator());
    std::copy(Aw.doubleData(), Aw.doubleData() + m * n, A_buf.begin());
    keel::Array<keel::i32> piv(n, scratch.allocator());
    if (!numkit::ops::lu_factor_inplace(A_buf.data(), n, piv.data(), n, n))
        return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);

    int sign = 1;
    for (keel::usize k = 0; k < n; ++k) {
        if (piv[k] != keel::as<keel::i32>(k)) {
            sign = -sign;
        }
    }

    long double prod = keel::as<long double>(sign);
    for (keel::usize i = 0; i < n; ++i)
        prod *= keel::as<long double>(A_buf[i + i * n]);
    return detail::narrowDoubleToSingleIf(Value::scalar(keel::as<keel::f64>(prod), mr), wantSingle, mr);
}

Value rank_of(const Value &A, keel::f64 tol, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "rank");
    detail::require2DMatrix(A, "rank", "A", "pagesvd");
    const bool isSingle = A.type() == ValueType::SINGLE;
    Value Aw = detail::widenSingleToDouble(A, mr);
    auto sv = svd_values(Aw, mr);
    const keel::usize k = sv.numel();
    const keel::f64 *s = sv.doubleData();
    if (k == 0) return Value::scalar(0.0, mr);
    const keel::f64 sigma_max = s[0];
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    const keel::f64 cutoff = (tol < 0.0) ? defaultRankTol(m, n, sigma_max, isSingle) : tol;
    int r = 0;
    for (keel::usize i = 0; i < k; ++i)
        if (s[i] > cutoff) ++r;
    return Value::scalar(keel::as<keel::f64>(r), mr);
}

Value cond_2norm(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "cond");
    const bool wantSingle = A.type() == ValueType::SINGLE;
    Value Aw = detail::widenSingleToDouble(A, mr);
    auto sv = svd_values(Aw, mr);
    const keel::usize k = sv.numel();
    if (k == 0)
        return detail::narrowDoubleToSingleIf(Value::scalar(std::numeric_limits<keel::f64>::quiet_NaN(), mr), wantSingle, mr);
    const keel::f64 *s = sv.doubleData();
    const keel::f64 sigma_max = s[0];
    const keel::f64 sigma_min = s[k - 1];
    if (sigma_min <= 0.0)
        return detail::narrowDoubleToSingleIf(Value::scalar(std::numeric_limits<keel::f64>::infinity(), mr), wantSingle, mr);
    return detail::narrowDoubleToSingleIf(Value::scalar(sigma_max / sigma_min, mr), wantSingle, mr);
}

Value cond_pnorm(const Value &A, int p_kind, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "cond");
    if (p_kind == 2) return cond_2norm(A, mr);

    if (A.dims().ndim() != 2)
        throw Error("cond: input must be a 2D matrix",
                    0, 0, "cond", "", "numkit:cond:notMatrix");
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    if (m != n)
        throw Error("cond: matrix must be square for non-2-norm forms",
                    0, 0, "cond", "", "numkit:cond:notSquare");
    const bool wantSingle = A.type() == ValueType::SINGLE;
    if (m == 0)
        return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);

    auto norm_for_kind = [&](const Value &X) -> keel::f64 {
        switch (p_kind) {
        case 1: return norm_value(X, 1.0, mr).toScalar();
        case 3: return norm_inf(X, mr).toScalar();
        case 4: return norm_fro(X, mr).toScalar();
        default:
            throw Error("cond: p must be 1, 2, Inf, or 'fro'",
                        0, 0, "cond", "", "numkit:cond:badP");
        }
    };

    const keel::f64 an = norm_for_kind(A);
    if (an == 0.0)
        return detail::narrowDoubleToSingleIf(Value::scalar(std::numeric_limits<keel::f64>::infinity(), mr), wantSingle, mr);

    Value Ainv;
    try {
        Ainv = inv(A, mr);
    } catch (...) {
        return detail::narrowDoubleToSingleIf(Value::scalar(std::numeric_limits<keel::f64>::infinity(), mr), wantSingle, mr);
    }
    const keel::f64 in = norm_for_kind(Ainv);
    if (!std::isfinite(in))
        return detail::narrowDoubleToSingleIf(Value::scalar(std::numeric_limits<keel::f64>::infinity(), mr), wantSingle, mr);
    return detail::narrowDoubleToSingleIf(Value::scalar(an * in, mr), wantSingle, mr);
}

Value normest(const Value &A, keel::Allocator *mr)
{
    const ValueType t = A.type();
    if (t != ValueType::DOUBLE && t != ValueType::COMPLEX
        && t != ValueType::SINGLE && t != ValueType::LOGICAL
        && t != ValueType::CHAR)
        throw Error(keel::format("normest: input 1 (A) must be a double, single, logical"
                                 " or char matrix (got a {} {} array).",
                                 detail::shapeOf(A), mtypeName(t)),
                    0, 0, "normest", "", "numkit:normest:inputType");
    const bool wantSingle = t == ValueType::SINGLE;
    auto sv = svd_values(detail::coerceToDoubleMatrix(A, mr), mr);
    if (sv.numel() == 0)
        return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);
    return detail::narrowDoubleToSingleIf(Value::scalar(sv.doubleData()[0], mr), wantSingle, mr);
}

Value rcond(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "rcond");
    if (A.dims().is3D())
        throw Error("rcond: input must be 2D",
                    0, 0, "rcond", "", "numkit:rcond:Not2D");
    const keel::usize M = A.dims().rows();
    const keel::usize N = A.dims().cols();
    if (M != N)
        throw Error("rcond: matrix must be square",
                    0, 0, "rcond", "", "numkit:rcond:NotSquare");
    const bool wantSingle = A.type() == ValueType::SINGLE;
    if (M == 0)
        return detail::narrowDoubleToSingleIf(Value::scalar(std::numeric_limits<keel::f64>::infinity(), mr), wantSingle, mr);

    const keel::f64 anorm = value_one_norm(A, M, N);
    if (anorm == 0.0)
        return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);

    Value Ainv;
    try {
        Ainv = inv(A, mr);
    } catch (...) {
        return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);
    }
    const keel::f64 inv_norm = value_one_norm(Ainv, M, N);
    if (!std::isfinite(inv_norm) || inv_norm == 0.0)
        return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);
    return detail::narrowDoubleToSingleIf(Value::scalar(1.0 / (anorm * inv_norm), mr), wantSingle, mr);
}

Value condest(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "condest");
    if (A.dims().is3D())
        throw Error("condest: input must be 2D",
                    0, 0, "condest", "", "numkit:condest:Not2D");
    const keel::usize M = A.dims().rows();
    const keel::usize N = A.dims().cols();
    if (M != N)
        throw Error("condest: matrix must be square",
                    0, 0, "condest", "", "numkit:condest:NotSquare");
    const bool wantSingle = A.type() == ValueType::SINGLE;
    if (M == 0)
        return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);

    const keel::f64 anorm = value_one_norm(A, M, N);
    if (anorm == 0.0)
        return detail::narrowDoubleToSingleIf(Value::scalar(std::numeric_limits<keel::f64>::infinity(), mr), wantSingle, mr);

    Value Ainv;
    try {
        Ainv = inv(A, mr);
    } catch (...) {
        return detail::narrowDoubleToSingleIf(Value::scalar(std::numeric_limits<keel::f64>::infinity(), mr), wantSingle, mr);
    }
    const keel::f64 inv_norm = value_one_norm(Ainv, M, N);
    if (!std::isfinite(inv_norm))
        return detail::narrowDoubleToSingleIf(Value::scalar(std::numeric_limits<keel::f64>::infinity(), mr), wantSingle, mr);
    return detail::narrowDoubleToSingleIf(Value::scalar(anorm * inv_norm, mr), wantSingle, mr);
}

Value condeig(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "condeig");
    if (A.dims().ndim() != 2)
        throw Error("condeig: input must be a 2D matrix",
                    0, 0, "condeig", "", "numkit:condeig:notMatrix");
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    if (m != n)
        throw Error("condeig: matrix must be square",
                    0, 0, "condeig", "", "numkit:condeig:notSquare");
    const bool wantSingle = A.type() == ValueType::SINGLE;
    if (n == 0)
        return Value::matrix(0, 1, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    Value Aw = detail::widenSingleToDouble(A, mr);

    // Real symmetric short-circuit: every eigenvalue is perfectly conditioned.
    if (!Aw.isComplex()) {
        auto isSymmetric = [&]() -> bool {
            const keel::f64 *p = Aw.doubleData();
            const keel::f64 tol = 1e-10;
            for (keel::usize i = 0; i < n; ++i)
                for (keel::usize j = i + 1; j < n; ++j) {
                    const keel::f64 d = std::fabs(p[i + j * n] - p[j + i * n]);
                    const keel::f64 s = std::max(std::fabs(p[i + j * n]),
                                               std::fabs(p[j + i * n]));
                    if (d > tol * (1.0 + s)) return false;
                }
            return true;
        };
        if (isSymmetric()) {
            auto out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
            keel::f64 *o = out.doubleDataMut();
            for (keel::usize i = 0; i < n; ++i) o[i] = 1.0;
            return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
        }
    }

    auto [V, D] = eig_general_VD(Aw, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    keel::f64 *s = out.doubleDataMut();

    if (V.isComplex()) {
        keel::Array<keel::c64> V_buf(n * n, scratch.allocator());
        keel::Array<keel::c64> I_buf(n * n, keel::c64(0.0, 0.0), scratch.allocator());
        std::copy(V.complexData(), V.complexData() + n * n, V_buf.begin());
        for (keel::usize i = 0; i < n; ++i) I_buf[i + i * n] = keel::c64(1.0, 0.0);

        keel::Array<keel::c64> Vinv(n * n, scratch.allocator());
        if (!numkit::ops::la_solve(V_buf.data(), n, n, I_buf.data(), n,
                                   Vinv.data(), scratch.allocator()))
            throw Error("condeig: right eigenvector matrix is singular",
                        0, 0, "condeig", "", "numkit:condeig:singular");

        const keel::c64 *vd = V.complexData();
        for (keel::usize i = 0; i < n; ++i) {
            keel::f64 v_norm = 0.0;
            for (keel::usize k = 0; k < n; ++k)
                v_norm += keel::norm(vd[k + i * n]);
            v_norm = std::sqrt(v_norm);

            keel::f64 w_norm = 0.0;
            for (keel::usize k = 0; k < n; ++k)
                w_norm += keel::norm(Vinv[i + k * n]);
            w_norm = std::sqrt(w_norm);

            keel::c64 dot(0.0, 0.0);
            for (keel::usize k = 0; k < n; ++k)
                dot += Vinv[i + k * n] * vd[k + i * n];

            const keel::f64 abs_dot = keel::abs(dot);
            if (abs_dot < 1e-300) {
                s[i] = std::numeric_limits<keel::f64>::infinity();
            } else {
                s[i] = v_norm * w_norm / abs_dot;
            }
        }
        return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
    }

    keel::Array<keel::f64> V_buf(n * n, scratch.allocator());
    keel::Array<keel::f64> I_buf(n * n, 0.0, scratch.allocator());
    std::copy(V.doubleData(), V.doubleData() + n * n, V_buf.begin());
    for (keel::usize i = 0; i < n; ++i) I_buf[i + i * n] = 1.0;

    keel::Array<keel::f64> Vinv(n * n, scratch.allocator());
    if (!numkit::ops::la_solve(V_buf.data(), n, n, I_buf.data(), n,
                                            Vinv.data(), scratch.allocator()))
        throw Error("condeig: right eigenvector matrix is singular",
                    0, 0, "condeig", "", "numkit:condeig:singular");

    const keel::f64 *vd = V.doubleData();
    for (keel::usize i = 0; i < n; ++i) {
        keel::f64 v_norm = 0.0;
        for (keel::usize k = 0; k < n; ++k)
            v_norm += vd[k + i * n] * vd[k + i * n];
        v_norm = std::sqrt(v_norm);

        keel::f64 w_norm = 0.0;
        for (keel::usize k = 0; k < n; ++k) {
            const keel::f64 w_k = Vinv[i + k * n];
            w_norm += w_k * w_k;
        }
        w_norm = std::sqrt(w_norm);

        keel::f64 dot = 0.0;
        for (keel::usize k = 0; k < n; ++k)
            dot += Vinv[i + k * n] * vd[k + i * n];

        if (std::fabs(dot) < 1e-300) {
            s[i] = std::numeric_limits<keel::f64>::infinity();
        } else {
            s[i] = v_norm * w_norm / std::fabs(dot);
        }
    }
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Engine adapters â€” registered in LinalgLibrary::install
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

} // namespace numkit::linalg
