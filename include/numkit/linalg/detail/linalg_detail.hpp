// toolboxes/linalg/src/linalg_detail.hpp
//
// PRIVATE (src-level) header â€” NOT part of the public linalg API.
// Templated kernel access layer helpers for double & std::complex<double>.

#pragma once

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <algorithm>
#include <utility>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>   // mtypeName (argcheck got-clauses)
#include <numkit/ops/blas.hpp>

namespace numkit::linalg::detail {

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Shared <fn>_argcheck cores (argcheck separation,
// dev-docs/todo/argcheck_separation.md â€” boundary rule: raw arguments in,
// all message digestion here; call sites are one pass-through line).
// Rejection classes reference-probed against R2025b 2026-09-18.
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// Render "2x2x2" for got-clauses.
inline keel::String shapeOf(const Value &v)
{
    const Dims &d = v.dims();
    keel::String s = keel::format("{}", d.dim(0));
    for (keel::i32 k = 1; k < d.ndim(); ++k) {
        keel::format_into(s, "x{}", d.dim(k));
    }
    return s;
}

// Factorization / norm family (svd, pinv, orth, null, rank, subspace,
// norm, eig, pagesvd, pagepinv, pagenorm): the reference accepts single
// or double, real or complex, and rejects logical, integer classes,
// char, string and cell as one inputType class.
inline void requireDoubleInput(const Value &A, keel::StringRef fn,
                               keel::StringRef argName = "A",
                               keel::i32 argNum = 1)
{
    const ValueType t = A.type();
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX || t == ValueType::SINGLE)
        return;
    throw Error(keel::format("{}: input {} ({}) must be a single, double or complex matrix "
                             "(got a {} {} array).",
                             fn, argNum, argName, shapeOf(A), mtypeName(t)),
                0, 0, fn, "",
                keel::format("numkit:{}:inputType", fn));
}

inline Value widenSingleToDouble(const Value &v, keel::Allocator *mr = nullptr)
{
    if (v.type() != ValueType::SINGLE)
        return v;
    Value out = (v.dims().ndim() == 3)
        ? Value::matrix3d(v.dims().dim(0), v.dims().dim(1), v.dims().dim(2), ValueType::DOUBLE, mr)
        : Value::matrix(v.dims().rows(), v.dims().cols(), ValueType::DOUBLE, mr);
    const keel::f32 *sd = v.singleData();
    keel::f64 *dd = out.doubleDataMut();
    const keel::usize n = v.numel();
    for (keel::usize i = 0; i < n; ++i)
        dd[i] = keel::as<keel::f64>(sd[i]);
    return out;
}

inline Value narrowDoubleToSingle(const Value &v, keel::Allocator *mr = nullptr)
{
    if (v.type() != ValueType::DOUBLE)
        return v;
    Value out = (v.dims().ndim() == 3)
        ? Value::matrix3d(v.dims().dim(0), v.dims().dim(1), v.dims().dim(2), ValueType::SINGLE, mr)
        : Value::matrix(v.dims().rows(), v.dims().cols(), ValueType::SINGLE, mr);
    const keel::f64 *dd = v.doubleData();
    keel::f32 *sd = out.singleDataMut();
    const keel::usize n = v.numel();
    for (keel::usize i = 0; i < n; ++i)
        sd[i] = keel::as<keel::f32>(dd[i]);
    return out;
}

inline Value narrowDoubleToSingleIf(const Value &v, bool wantSingle, keel::Allocator *mr = nullptr)
{
    return wantSingle ? narrowDoubleToSingle(v, mr) : v;
}

// Matrix predicate family (isbanded, isdiag, istril, istriu,
// issymmetric, ishermitian, bandwidth): double, single, complex or
// logical are accepted; integer classes, char, string and cell are
// rejected (the element reads coerce, so single/logical answers stay
// class-correct logical).
inline void requireFloatOrLogicalInput(const Value &A, keel::StringRef fn,
                                       keel::StringRef argName = "A",
                                       keel::i32 argNum = 1)
{
    const ValueType t = A.type();
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX
        || t == ValueType::SINGLE || t == ValueType::LOGICAL)
        return;
    throw Error(keel::format("{}: input {} ({}) must be a double, single or logical matrix "
                             "(got a {} {} array).",
                             fn, argNum, argName, shapeOf(A), mtypeName(t)),
                0, 0, fn, "",
                keel::format("numkit:{}:inputType", fn));
}

// Coerce a coercible operand (single / logical / char / integer) to a
// double matrix of the same shape; every other class (including
// cell/string/struct) passes through UNCHANGED so the caller's
// requireDoubleInput gate voices the rejection. Callers gate the
// accepted classes per-function first â€” the reference's coercion set
// differs per entry (gsvd coerces logical, normest logical+char, trace
// numeric+logical; probed R2025b 2026-09-18).
inline Value coerceToDoubleMatrix(const Value &A, keel::Allocator *mr)
{
    const ValueType t = A.type();
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX
        || t == ValueType::SINGLE || t == ValueType::LOGICAL
        || t == ValueType::CHAR || isIntegerType(t)) {
        if (t == ValueType::DOUBLE || t == ValueType::COMPLEX)
            return A;
        Value out = Value::matrix(A.dims().rows(), A.dims().cols(),
                                  ValueType::DOUBLE, mr);
        const keel::usize n = A.numel();
        keel::f64 *od = out.doubleDataMut();
        if (t == ValueType::LOGICAL) {
            const keel::u8 *ld = A.logicalData();
            for (keel::usize i = 0; i < n; ++i)
                od[i] = ld[i] ? 1.0 : 0.0;
        } else {
            for (keel::usize i = 0; i < n; ++i)
                od[i] = A.elemAsDouble(i);
        }
        return out;
    }
    return A;
}

// 2-D requirement voiced per entry (pinv/orth/null/rank previously
// answered 3-D inputs with the internal svd diagnostic). pageHint names
// the page function the entry's family offers.
inline void require2DMatrix(const Value &A, keel::StringRef fn,
                            keel::StringRef argName = "A",
                            keel::StringRef pageHint = {})
{
    if (A.dims().ndim() == 2)
        return;
    keel::String msg = keel::format("{}: input 1 ({}) must be a 2-D matrix (argument is {})",
                                    fn, argName, shapeOf(A));
    if (!pageHint.empty())
        keel::format_into(msg, "; use {} to compute each N-D page separately", pageHint);
    msg.push_back('.');
    throw Error(std::move(msg), 0, 0, fn, "",
                keel::format("numkit:{}:notMatrix", fn));
}

// Helper to determine if type is std::complex<T>
template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<keel::Complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

// Conjugate helper: conj for complex, identity for real
template <typename T>
inline T conj_if_complex(const T &val) {
    if constexpr (is_complex_v<T>) {
        return keel::conj(val);
    } else {
        return val;
    }
}

// Absolute value squared: |x|^2
template <typename T>
inline keel::f64 abs_sq(const T &val) {
    if constexpr (is_complex_v<T>) {
        return keel::norm(val);
    } else {
        return val * val;
    }
}

// Magnitude: |x|
template <typename T>
inline keel::f64 abs_val(const T &val) {
    if constexpr (is_complex_v<T>) {
        return keel::abs(val);
    } else {
        return keel::abs(val);
    }
}

// Real part helper
template <typename T>
inline keel::f64 real_part(const T &val) {
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
inline Value make_matrix(keel::usize rows, keel::usize cols,
                         keel::Allocator *mr = nullptr) {
    if constexpr (is_complex_v<T>) {
        return Value::complexMatrix(rows, cols, mr);
    } else {
        return Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    }
}

// Narrowing helper: converts a complex Value to a real Value if all imaginary parts are zero (+/- 0)
inline Value narrow_if_real(const Value &v, keel::Allocator *mr = nullptr) {
    if (!v.isComplex()) {
        return v;
    }
    const keel::c64 *cd = v.complexData();
    const keel::usize n = v.numel();
    for (keel::usize i = 0; i < n; ++i) {
        if (cd[i].imag() != 0.0) {
            return v;
        }
    }
    // All imaginary parts are zero, narrow to double matrix
    Value out = Value::matrix(v.dims().rows(), v.dims().cols(), ValueType::DOUBLE, mr);
    keel::f64 *dd = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) {
        dd[i] = cd[i].real();
    }
    return out;
}

// In-place LU with partial pivoting on a column-major nÃ—n matrix (blocked dgetrf pattern).

} // namespace numkit::linalg::detail
