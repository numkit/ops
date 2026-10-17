// toolboxes/linalg/src/linalg_detail.hpp
//
// PRIVATE (src-level) header â€” NOT part of the public linalg API.
// Templated kernel access layer helpers for double & std::complex<double>.

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <type_traits>
#include <vector>

#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>   // mtypeName (argcheck got-clauses)
#include <numkit/ops/blas.hpp>

namespace numkit::linalg::detail {

using Complex = std::complex<double>;

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Shared <fn>_argcheck cores (argcheck separation,
// dev-docs/todo/argcheck_separation.md â€” boundary rule: raw arguments in,
// all message digestion here; call sites are one pass-through line).
// Rejection classes reference-probed against R2025b 2026-09-18.
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// Render "2x2x2" for got-clauses.
inline std::string shapeOf(const Value &v)
{
    const Dims &d = v.dims();
    std::string s = std::to_string(d.dim(0));
    for (int k = 1; k < d.ndim(); ++k) {
        s += "x";
        s += std::to_string(d.dim(k));
    }
    return s;
}

// Factorization / norm family (svd, pinv, orth, null, rank, subspace,
// norm, eig, pagesvd, pagepinv, pagenorm): the reference accepts single
// or double, real or complex, and rejects logical, integer classes,
// char, string and cell as one inputType class.
inline void requireDoubleInput(const Value &A, std::string_view fn,
                               std::string_view argName = "A",
                               int argNum = 1)
{
    const ValueType t = A.type();
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX || t == ValueType::SINGLE)
        return;
    const std::string who = std::string(fn) + ": input "
                            + std::to_string(argNum) + " ("
                            + std::string(argName) + ")";
    throw Error(who + " must be a single, double or complex matrix "
                    "(got a " + shapeOf(A) + " " + mtypeName(t)
                        + " array).",
                0, 0, std::string(fn), "",
                "numkit:" + std::string(fn) + ":inputType");
}

inline Value widenSingleToDouble(const Value &v, std::pmr::memory_resource *mr = nullptr)
{
    if (v.type() != ValueType::SINGLE)
        return v;
    Value out = (v.dims().ndim() == 3)
        ? Value::matrix3d(v.dims().dim(0), v.dims().dim(1), v.dims().dim(2), ValueType::DOUBLE, mr)
        : Value::matrix(v.dims().rows(), v.dims().cols(), ValueType::DOUBLE, mr);
    const float *sd = v.singleData();
    double *dd = out.doubleDataMut();
    const std::size_t n = v.numel();
    for (std::size_t i = 0; i < n; ++i)
        dd[i] = static_cast<double>(sd[i]);
    return out;
}

inline Value narrowDoubleToSingle(const Value &v, std::pmr::memory_resource *mr = nullptr)
{
    if (v.type() != ValueType::DOUBLE)
        return v;
    Value out = (v.dims().ndim() == 3)
        ? Value::matrix3d(v.dims().dim(0), v.dims().dim(1), v.dims().dim(2), ValueType::SINGLE, mr)
        : Value::matrix(v.dims().rows(), v.dims().cols(), ValueType::SINGLE, mr);
    const double *dd = v.doubleData();
    float *sd = out.singleDataMut();
    const std::size_t n = v.numel();
    for (std::size_t i = 0; i < n; ++i)
        sd[i] = static_cast<float>(dd[i]);
    return out;
}

inline Value narrowDoubleToSingleIf(const Value &v, bool wantSingle, std::pmr::memory_resource *mr = nullptr)
{
    return wantSingle ? narrowDoubleToSingle(v, mr) : v;
}

// Matrix predicate family (isbanded, isdiag, istril, istriu,
// issymmetric, ishermitian, bandwidth): double, single, complex or
// logical are accepted; integer classes, char, string and cell are
// rejected (the element reads coerce, so single/logical answers stay
// class-correct logical).
inline void requireFloatOrLogicalInput(const Value &A, std::string_view fn,
                                       std::string_view argName = "A",
                                       int argNum = 1)
{
    const ValueType t = A.type();
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX
        || t == ValueType::SINGLE || t == ValueType::LOGICAL)
        return;
    throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                    + " (" + std::string(argName) + ") must be a double, "
                    "single or logical matrix (got a " + shapeOf(A) + " "
                    + mtypeName(t) + " array).",
                0, 0, std::string(fn), "",
                "numkit:" + std::string(fn) + ":inputType");
}

// Coerce a coercible operand (single / logical / char / integer) to a
// double matrix of the same shape; every other class (including
// cell/string/struct) passes through UNCHANGED so the caller's
// requireDoubleInput gate voices the rejection. Callers gate the
// accepted classes per-function first â€” the reference's coercion set
// differs per entry (gsvd coerces logical, normest logical+char, trace
// numeric+logical; probed R2025b 2026-09-18).
inline Value coerceToDoubleMatrix(const Value &A, std::pmr::memory_resource *mr)
{
    const ValueType t = A.type();
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX
        || t == ValueType::SINGLE || t == ValueType::LOGICAL
        || t == ValueType::CHAR || isIntegerType(t)) {
        if (t == ValueType::DOUBLE || t == ValueType::COMPLEX)
            return A;
        Value out = Value::matrix(A.dims().rows(), A.dims().cols(),
                                  ValueType::DOUBLE, mr);
        const std::size_t n = A.numel();
        double *od = out.doubleDataMut();
        if (t == ValueType::LOGICAL) {
            const std::uint8_t *ld = A.logicalData();
            for (std::size_t i = 0; i < n; ++i)
                od[i] = ld[i] ? 1.0 : 0.0;
        } else {
            for (std::size_t i = 0; i < n; ++i)
                od[i] = A.elemAsDouble(i);
        }
        return out;
    }
    return A;
}

// 2-D requirement voiced per entry (pinv/orth/null/rank previously
// answered 3-D inputs with the internal svd diagnostic). pageHint names
// the page function the entry's family offers.
inline void require2DMatrix(const Value &A, std::string_view fn,
                            std::string_view argName = "A",
                            std::string_view pageHint = {})
{
    if (A.dims().ndim() == 2)
        return;
    std::string msg = std::string(fn) + ": input 1 (" + std::string(argName)
                    + ") must be a 2-D matrix (argument is "
                    + shapeOf(A) + ")";
    if (!pageHint.empty())
        msg += "; use " + std::string(pageHint)
             + " to compute each N-D page separately";
    msg += ".";
    throw Error(std::move(msg), 0, 0, std::string(fn), "",
                "numkit:" + std::string(fn) + ":notMatrix");
}

// Helper to determine if type is std::complex<T>
template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

// Conjugate helper: conj for complex, identity for real
template <typename T>
inline T conj_if_complex(const T &val) {
    if constexpr (is_complex_v<T>) {
        return std::conj(val);
    } else {
        return val;
    }
}

// Absolute value squared: |x|^2
template <typename T>
inline double abs_sq(const T &val) {
    if constexpr (is_complex_v<T>) {
        return std::norm(val);
    } else {
        return val * val;
    }
}

// Magnitude: |x|
template <typename T>
inline double abs_val(const T &val) {
    if constexpr (is_complex_v<T>) {
        return std::abs(val);
    } else {
        return std::abs(val);
    }
}

// Real part helper
template <typename T>
inline double real_part(const T &val) {
    if constexpr (is_complex_v<T>) {
        return val.real();
    } else {
        return val;
    }
}

// Data pointer accessors for Value
template <typename T>
inline const T *get_data(const Value &v) {
    if constexpr (is_complex_v<T>) {
        return v.complexData();
    } else {
        return v.doubleData();
    }
}

template <typename T>
inline T *get_data_mut(Value &v) {
    if constexpr (is_complex_v<T>) {
        return v.complexDataMut();
    } else {
        return v.doubleDataMut();
    }
}

// Value matrix factory for double / complex
template <typename T>
inline Value make_matrix(std::size_t rows, std::size_t cols,
                         std::pmr::memory_resource *mr = nullptr) {
    if constexpr (is_complex_v<T>) {
        return Value::complexMatrix(rows, cols, mr);
    } else {
        return Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    }
}

// Narrowing helper: converts a complex Value to a real Value if all imaginary parts are zero (+/- 0)
inline Value narrow_if_real(const Value &v, std::pmr::memory_resource *mr = nullptr) {
    if (!v.isComplex()) {
        return v;
    }
    const Complex *cd = v.complexData();
    const std::size_t n = v.numel();
    for (std::size_t i = 0; i < n; ++i) {
        if (cd[i].imag() != 0.0) {
            return v;
        }
    }
    // All imaginary parts are zero, narrow to double matrix
    Value out = Value::matrix(v.dims().rows(), v.dims().cols(), ValueType::DOUBLE, mr);
    double *dd = out.doubleDataMut();
    for (std::size_t i = 0; i < n; ++i) {
        dd[i] = cd[i].real();
    }
    return out;
}

// In-place LU with partial pivoting on a column-major nÃ—n matrix (blocked dgetrf pattern).

} // namespace numkit::linalg::detail
