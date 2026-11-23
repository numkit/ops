#include <keel/core.h>
#include <keel/algo.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/builtin/datatypes.hpp>
#include <numkit/builtin/strfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>


// Forward-declare the SIMD/portable cast backend (defined in
// language/datatypes/casts_{highway,portable}.cpp).
#include "types_detail.hpp"

namespace numkit::builtin::detail {
void doubleToInt8 (const keel::f64 *in, keel::i8   *out, keel::usize n);
void doubleToInt16(const keel::f64 *in, keel::i16  *out, keel::usize n);
void doubleToInt32(const keel::f64 *in, keel::i32  *out, keel::usize n);
void doubleToInt64(const keel::f64 *in, keel::i64  *out, keel::usize n);
void doubleToUInt8 (const keel::f64 *in, keel::u8  *out, keel::usize n);
void doubleToUInt16(const keel::f64 *in, keel::u16 *out, keel::usize n);
void doubleToUInt32(const keel::f64 *in, keel::u32 *out, keel::usize n);
void doubleToUInt64(const keel::f64 *in, keel::u64 *out, keel::usize n);
} // namespace numkit::builtin::detail

// Forward-declare the isnan/isinf/isfinite backend (defined in
// math/arithmetic/isfinite_{highway,portable}.cpp — numkit::builtin::detail after C4).
namespace numkit::builtin::detail {
void doubleIsNaNLoop(const keel::f64 *in, keel::u8 *out, keel::usize n);
void doubleIsInfLoop(const keel::f64 *in, keel::u8 *out, keel::usize n);
void doubleIsFiniteLoop(const keel::f64 *in, keel::u8 *out, keel::usize n);
} // namespace numkit::builtin::detail

namespace numkit::builtin {

// ════════════════════════════════════════════════════════════════════════
// Implementation helpers
// ════════════════════════════════════════════════════════════════════════


// ════════════════════════════════════════════════════════════════════════
// Public API — numeric constructors
// ════════════════════════════════════════════════════════════════════════

Value toDouble(const Value &x, keel::Allocator *mr)
{
    return numericConstructor<keel::f64>(ValueType::DOUBLE, x, mr);
}

Value single(const Value &x, keel::Allocator *mr)
{
    return numericConstructor<keel::f32>(ValueType::SINGLE, x, mr);
}

Value int8(const Value &x, keel::Allocator *mr)   { return numericConstructor<keel::i8>(ValueType::INT8, x, mr); }
Value int16(const Value &x, keel::Allocator *mr)  { return numericConstructor<keel::i16>(ValueType::INT16, x, mr); }
Value int32(const Value &x, keel::Allocator *mr)  { return numericConstructor<keel::i32>(ValueType::INT32, x, mr); }
Value int64(const Value &x, keel::Allocator *mr)  { return numericConstructor<keel::i64>(ValueType::INT64, x, mr); }
Value uint8(const Value &x, keel::Allocator *mr)  { return numericConstructor<keel::u8>(ValueType::UINT8, x, mr); }
Value uint16(const Value &x, keel::Allocator *mr) { return numericConstructor<keel::u16>(ValueType::UINT16, x, mr); }
Value uint32(const Value &x, keel::Allocator *mr) { return numericConstructor<keel::u32>(ValueType::UINT32, x, mr); }
Value uint64(const Value &x, keel::Allocator *mr) { return numericConstructor<keel::u64>(ValueType::UINT64, x, mr); }

Value logical(const Value &x, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (x.isLogical())
        return x;
    if (x.isScalar())
        return Value::logicalScalar(x.toScalar() != 0, p);
    Value r = createLike(x, ValueType::LOGICAL, p);
    for (keel::usize i = 0; i < x.numel(); ++i)
        r.logicalDataMut()[i] = x.elemAsDouble(i) != 0 ? 1 : 0;
    return r;
}

// ════════════════════════════════════════════════════════════════════════
// Public API — type predicates
// ════════════════════════════════════════════════════════════════════════

Value isnumeric(const Value &x, keel::Allocator *mr) { return Value::logicalScalar(x.isNumeric(), mr); }
Value islogical(const Value &x, keel::Allocator *mr) { return Value::logicalScalar(x.isLogical(), mr); }
Value ischar(const Value &x, keel::Allocator *mr)    { return Value::logicalScalar(x.isChar(), mr); }
Value isstring(const Value &x, keel::Allocator *mr)  { return Value::logicalScalar(x.isString(), mr); }
Value iscell(const Value &x, keel::Allocator *mr)    { return Value::logicalScalar(x.isCell(), mr); }
Value isstruct(const Value &x, keel::Allocator *mr)  { return Value::logicalScalar(x.isStruct(), mr); }
Value isempty(const Value &x, keel::Allocator *mr)   { return Value::logicalScalar(x.isEmpty(), mr); }
Value isscalar(const Value &x, keel::Allocator *mr)  { return Value::logicalScalar(x.isScalar(), mr); }
Value isinteger(const Value &x, keel::Allocator *mr) { return Value::logicalScalar(isIntegerType(x.type()), mr); }
Value isfloat(const Value &x, keel::Allocator *mr)   { return Value::logicalScalar(isFloatType(x.type()), mr); }
Value issingle(const Value &x, keel::Allocator *mr)  { return Value::logicalScalar(x.type() == ValueType::SINGLE || x.isSingleComplex(), mr); }
// nk has no sparse-matrix storage class -- issparse always returns
// false for any input. Matches MATLAB on dense inputs (which is all
// nk can produce). Covers the FAIL row in PROGRESS for issparse.
Value issparse(const Value & /*x*/, keel::Allocator *mr) { return Value::logicalScalar(false, mr); }

Value isnan(const Value &x, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (x.isComplex()) {
        auto r = createLike(x, ValueType::LOGICAL, p);
        const auto *cd = x.complexData();
        keel::u8 *ld = r.logicalDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            ld[i] = (keel::is_nan(cd[i].real()) || keel::is_nan(cd[i].imag())) ? 1 : 0;
        return r;
    }
    if (x.isScalar() && x.type() == ValueType::DOUBLE)
        return Value::logicalScalar(keel::is_nan(x.toScalar()), p);
    auto r = createLike(x, ValueType::LOGICAL, p);
    if (x.numel() == 0) return r;
    if (x.type() == ValueType::DOUBLE) {
        ::numkit::builtin::detail::doubleIsNaNLoop(x.doubleData(), r.logicalDataMut(), x.numel());
    } else if (x.type() == ValueType::SINGLE) {
        const keel::f32 *sd = x.singleData();
        keel::u8 *ld = r.logicalDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            ld[i] = keel::is_nan(sd[i]) ? 1 : 0;
    } else {
        keel::mem_zero(r.logicalDataMut(), x.numel());
    }
    return r;
}

Value isinf(const Value &x, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (x.isComplex()) {
        auto r = createLike(x, ValueType::LOGICAL, p);
        const auto *cd = x.complexData();
        keel::u8 *ld = r.logicalDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            ld[i] = (keel::is_inf(cd[i].real()) || keel::is_inf(cd[i].imag())) ? 1 : 0;
        return r;
    }
    if (x.isScalar() && x.type() == ValueType::DOUBLE)
        return Value::logicalScalar(keel::is_inf(x.toScalar()), p);
    auto r = createLike(x, ValueType::LOGICAL, p);
    if (x.numel() == 0) return r;
    if (x.type() == ValueType::DOUBLE) {
        ::numkit::builtin::detail::doubleIsInfLoop(x.doubleData(), r.logicalDataMut(), x.numel());
    } else if (x.type() == ValueType::SINGLE) {
        const keel::f32 *sd = x.singleData();
        keel::u8 *ld = r.logicalDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            ld[i] = keel::is_inf(sd[i]) ? 1 : 0;
    } else {
        keel::mem_zero(r.logicalDataMut(), x.numel());
    }
    return r;
}

Value isfinite(const Value &x, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (x.isComplex()) {
        auto r = createLike(x, ValueType::LOGICAL, p);
        const auto *cd = x.complexData();
        keel::u8 *ld = r.logicalDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            ld[i] = (keel::is_finite(cd[i].real()) && keel::is_finite(cd[i].imag())) ? 1 : 0;
        return r;
    }
    if (x.isScalar() && x.type() == ValueType::DOUBLE)
        return Value::logicalScalar(keel::is_finite(x.toScalar()), p);
    auto r = createLike(x, ValueType::LOGICAL, p);
    if (x.numel() == 0) return r;
    if (x.type() == ValueType::DOUBLE) {
        ::numkit::builtin::detail::doubleIsFiniteLoop(x.doubleData(), r.logicalDataMut(), x.numel());
    } else if (x.type() == ValueType::SINGLE) {
        const keel::f32 *sd = x.singleData();
        keel::u8 *ld = r.logicalDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            ld[i] = keel::is_finite(sd[i]) ? 1 : 0;
    } else {
        keel::mem_set(r.logicalDataMut(), 1, x.numel());
    }
    return r;
}

// ── missing-value predicates ─────────────────────────────────────────


Value ismissing(const Value &x, const Value &indicator,
                keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    auto r = createLike(x, ValueType::LOGICAL, p);
    const keel::usize n = x.numel();
    if (n == 0) return r;

    keel::u8 *out = r.logicalDataMut();

    // Build the indicator value list (empty → standard missing only).
    keel::ArenaTemp scratch = keel::get_scratch(p);
    const bool have_ind = (!indicator.isEmpty());
    keel::Array<keel::f64> ind_vals(scratch.allocator());
    if (have_ind) {
        ind_vals.reserve(indicator.numel());
        for (keel::usize k = 0; k < indicator.numel(); ++k)
            ind_vals.push_back(indicator.elemAsDouble(k));
    }
    const bool float_class = typeHasMissing(x.type());

    if (!have_ind) {
        // Standard missing only: NaN for float, never for ints/logical.
        if (float_class) {
            if (x.type() == ValueType::DOUBLE) {
                ::numkit::builtin::detail::doubleIsNaNLoop(x.doubleData(), out, n);
            } else {  // SINGLE
                const keel::f32 *src = x.singleData();
                for (keel::usize i = 0; i < n; ++i)
                    out[i] = keel::is_nan(src[i]) ? 1u : 0u;
            }
        } else {
            keel::fill(out, out + n, keel::u8{0});
        }
        return r;
    }

    // With indicator: ONLY values listed in `indicator` are missing
    // (NaN is NOT auto-flagged when an indicator is given — MATLAB
    // behaviour, probed). NaN entries in the indicator itself match
    // NaN values in `x` (special-cased because NaN != NaN).
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 xi = elemD(x, i);
        const bool xi_nan = keel::is_nan(xi);
        bool tf = false;
        for (keel::f64 v : ind_vals) {
            if (keel::is_nan(v)) {
                if (xi_nan) { tf = true; break; }
            } else if (xi == v) {
                tf = true; break;
            }
        }
        out[i] = tf ? 1u : 0u;
    }
    return r;
}

Value standardizeMissing(const Value &x, const Value &indicator,
                         keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    const keel::usize n = x.numel();
    const ValueType T = x.type();

    // Indicator-compatibility contract (probed R2025b,
    // ismissingKernel semantics): double indicators match every
    // numeric/logical array; logical indicators match double +
    // logical; single indicators match single/double; integer
    // indicators match double + the SAME integer class; a char
    // row vector matches char arrays only.
    const ValueType IT = indicator.type();
    const bool indNum = indicator.isNumeric() || indicator.isLogical();
    if (T == ValueType::CHAR) {
        const bool charRow = (IT == ValueType::CHAR) &&
                             (indicator.numel() == 0 ||
                              indicator.dims().dim(0) <= 1);
        if (!charRow)
            throw Error("standardizeMissing: second argument must be a "
                        "character row vector for char inputs",
                        0, 0, "standardizeMissing", "",
                        "numkit:standardizeMissing:indicatorsChar");
        return x;
    }
    if (!indNum)
        throw Error(keel::format("standardizeMissing: second argument must be numeric or "
                                 "logical (got {})", mtypeName(IT)),
                    0, 0, "standardizeMissing", "",
                    "numkit:standardizeMissing:indicatorsDouble");

    // Integer / logical: no missing concept → return a copy, but the
    // indicator must still be a compatible class.
    if (!typeHasMissing(T)) {
        const bool intLike = T != ValueType::SINGLE && T != ValueType::DOUBLE;
        const bool compat = (IT == ValueType::DOUBLE) || indicator.isLogical() ||
                            (intLike && IT == T);
        if (!compat)
            throw Error(keel::format("standardizeMissing: second argument must be double for {} inputs",
                                     mtypeName(T)),
                        0, 0, "standardizeMissing", "",
                        "numkit:standardizeMissing:indicatorsIntegerLogical");
        return x;
    }
    if (T == ValueType::SINGLE &&
        !(IT == ValueType::DOUBLE || IT == ValueType::SINGLE ||
          indicator.isLogical()))
        throw Error("standardizeMissing: second argument must be single or "
                    "double for single inputs",
                    0, 0, "standardizeMissing", "",
                    "numkit:standardizeMissing:indicatorsDouble");

    // Build indicator list (skip NaN — won't match anything via ==).
    keel::ArenaTemp scratch = keel::get_scratch(p);
    keel::Array<keel::f64> ind_vals(scratch.allocator());
    if (!indicator.isEmpty()) {
        ind_vals.reserve(indicator.numel());
        for (keel::usize k = 0; k < indicator.numel(); ++k) {
            const keel::f64 v = indicator.elemAsDouble(k);
            if (!keel::is_nan(v)) ind_vals.push_back(v);
        }
    }

    Value out = createLike(x, T, p);
    if (n == 0) return out;

    if (T == ValueType::DOUBLE) {
        const keel::f64 *src = x.doubleData();
        keel::f64 *dst       = out.doubleDataMut();
        const keel::f64 nan  = keel::Limits<keel::f64>::quiet_nan();
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 v = src[i];
            bool hit = false;
            for (keel::f64 ind : ind_vals) {
                if (v == ind) { hit = true; break; }
            }
            dst[i] = hit ? nan : v;
        }
    } else {  // SINGLE
        const keel::f32 *src = x.singleData();
        keel::f32 *dst       = out.singleDataMut();
        const keel::f32 nan  = keel::Limits<keel::f32>::quiet_nan();
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f32 v = src[i];
            bool hit = false;
            for (keel::f64 ind : ind_vals) {
                if (v == keel::as<keel::f32>(ind)) { hit = true; break; }
            }
            dst[i] = hit ? nan : v;
        }
    }
    return out;
}

Value anymissing(const Value &x, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    const keel::usize n = x.numel();
    if (n == 0) return Value::logicalScalar(false, p);
    if (!typeHasMissing(x.type()))
        return Value::logicalScalar(false, p);
    if (x.type() == ValueType::DOUBLE) {
        const keel::f64 *src = x.doubleData();
        for (keel::usize i = 0; i < n; ++i)
            if (keel::is_nan(src[i])) return Value::logicalScalar(true, p);
    } else {
        const keel::f32 *src = x.singleData();
        for (keel::usize i = 0; i < n; ++i)
            if (keel::is_nan(src[i])) return Value::logicalScalar(true, p);
    }
    return Value::logicalScalar(false, p);
}

// ── Shape predicates ─────────────────────────────────────────────────

Value isvector(const Value &x, keel::Allocator *mr)
{
    const auto &d = x.dims();
    bool tf = d.ndim() <= 2 && (d.rows() == 1 || d.cols() == 1) && x.numel() > 0;
    return Value::logicalScalar(tf, mr);
}

Value isrow(const Value &x, keel::Allocator *mr)
{
    const auto &d = x.dims();
    bool tf = d.ndim() <= 2 && d.rows() == 1;
    return Value::logicalScalar(tf, mr);
}

Value iscolumn(const Value &x, keel::Allocator *mr)
{
    const auto &d = x.dims();
    bool tf = d.ndim() <= 2 && d.cols() == 1;
    return Value::logicalScalar(tf, mr);
}

Value ismatrix(const Value &x, keel::Allocator *mr)
{
    return Value::logicalScalar(x.dims().ndim() <= 2, mr);
}

// ── Order predicates ─────────────────────────────────────────────────


Value issorted(const Value &x, const Value &mode, keel::Allocator *mr)
{
    SortMode m = SortMode::Ascend;
    if (!readSortMode(mode, m))
        throw numkit::Error("issorted: unrecognized sort mode");
    if (x.isEmpty() || x.isScalar())
        return Value::logicalScalar(true, mr);

    const auto &d = x.dims();
    const keel::f64 *p = x.doubleData();
    if (d.ndim() <= 2 && (d.rows() == 1 || d.cols() == 1)) {
        return Value::logicalScalar(runSorted(p, x.numel(), m), mr);
    }
    // Matrix / 3-D / N-D: every column (along dim 1) must satisfy `mode`.
    const keel::usize r = d.rows();
    const keel::usize restCount = x.numel() / r;
    for (keel::usize k = 0; k < restCount; ++k) {
        if (!runSorted(p + k * r, r, m))
            return Value::logicalScalar(false, mr);
    }
    return Value::logicalScalar(true, mr);
}

Value issortedrows(const Value &x, keel::Allocator *mr)
{
    if (x.isEmpty() || x.isScalar())
        return Value::logicalScalar(true, mr);
    const auto &d = x.dims();
    if (d.ndim() > 2)
        throw numkit::Error("issortedrows: input must be 2-D");
    const keel::usize R = d.rows(), C = d.cols();
    if (R < 2) return Value::logicalScalar(true, mr);
    const keel::f64 *p = x.doubleData();
    auto getElem = [&](keel::usize row, keel::usize col) { return p[col * R + row]; };
    for (keel::usize i = 1; i < R; ++i) {
        for (keel::usize c = 0; c < C; ++c) {
            const keel::f64 a = getElem(i - 1, c);
            const keel::f64 b = getElem(i, c);
            if (keel::is_nan(a) || keel::is_nan(b))
                return Value::logicalScalar(false, mr);
            if (a < b) break;          // strictly less → ordered, next row.
            if (a > b)                 // strictly greater → out of order.
                return Value::logicalScalar(false, mr);
            // equal → look at next column.
        }
    }
    return Value::logicalScalar(true, mr);
}

Value isuniform(const Value &x, keel::Allocator *mr)
{
    if (x.isEmpty()) return Value::logicalScalar(true, mr);
    if (x.isScalar()) return Value::logicalScalar(true, mr);
    const keel::usize n = x.numel();
    const auto &d = x.dims();
    // Treat as 1-D: only meaningful for row/column vectors.
    if (d.ndim() > 2 || (d.rows() != 1 && d.cols() != 1))
        throw numkit::Error("isuniform: input must be a vector");
    if (n < 2) return Value::logicalScalar(true, mr);
    const keel::f64 *p = x.doubleData();
    for (keel::usize i = 0; i < n; ++i)
        if (!keel::is_finite(p[i])) return Value::logicalScalar(false, mr);
    const keel::f64 step = p[1] - p[0];
    // Tolerance: 4 * eps(max-magnitude in vector or step), follows MATLAB
    // isuniform's "approximately uniform" semantics.
    keel::f64 scale = keel::abs(step);
    for (keel::usize i = 0; i < n; ++i) scale = keel::max(scale, keel::abs(p[i]));
    const keel::f64 tol = 4 * keel::Limits<keel::f64>::kEpsilon * scale;
    for (keel::usize i = 1; i < n - 1; ++i) {
        const keel::f64 s = p[i + 1] - p[i];
        if (keel::abs(s - step) > tol)
            return Value::logicalScalar(false, mr);
    }
    return Value::logicalScalar(true, mr);
}

// ── Numeric limits ───────────────────────────────────────────────────


Value flintmax(const Value &t, keel::Allocator *mr)
{
    auto name = readTypeName(t, "double");
    if (name == "single")
        return typedScalar<keel::f32>(ValueType::SINGLE, keel::as<keel::f32>(1u << 24), mr);
    if (name == "double")
        return Value::scalar(9007199254740992.0, mr);  // 2^53 exactly
    throw numkit::Error("flintmax: type must be 'double' or 'single'");
}

Value intmax(const Value &t, keel::Allocator *mr)
{
    auto name = readTypeName(t, "int32");
    if (name == "int8")   return typedScalar<keel::i8>(ValueType::INT8, keel::Limits<keel::i8>::kMax, mr);
    if (name == "int16")  return typedScalar<keel::i16>(ValueType::INT16, keel::Limits<keel::i16>::kMax, mr);
    if (name == "int32")  return typedScalar<keel::i32>(ValueType::INT32, keel::Limits<keel::i32>::kMax, mr);
    if (name == "int64")  return typedScalar<keel::i64>(ValueType::INT64, keel::Limits<keel::i64>::kMax, mr);
    if (name == "uint8")  return typedScalar<keel::u8>(ValueType::UINT8, keel::Limits<keel::u8>::kMax, mr);
    if (name == "uint16") return typedScalar<keel::u16>(ValueType::UINT16, keel::Limits<keel::u16>::kMax, mr);
    if (name == "uint32") return typedScalar<keel::u32>(ValueType::UINT32, keel::Limits<keel::u32>::kMax, mr);
    if (name == "uint64") return typedScalar<keel::u64>(ValueType::UINT64, keel::Limits<keel::u64>::kMax, mr);
    throw numkit::Error("intmax: unsupported integer class");
}

Value intmin(const Value &t, keel::Allocator *mr)
{
    auto name = readTypeName(t, "int32");
    if (name == "int8")   return typedScalar<keel::i8>(ValueType::INT8, keel::Limits<keel::i8>::kMin, mr);
    if (name == "int16")  return typedScalar<keel::i16>(ValueType::INT16, keel::Limits<keel::i16>::kMin, mr);
    if (name == "int32")  return typedScalar<keel::i32>(ValueType::INT32, keel::Limits<keel::i32>::kMin, mr);
    if (name == "int64")  return typedScalar<keel::i64>(ValueType::INT64, keel::Limits<keel::i64>::kMin, mr);
    if (name == "uint8")  return typedScalar<keel::u8>(ValueType::UINT8, keel::as<keel::u8>(0), mr);
    if (name == "uint16") return typedScalar<keel::u16>(ValueType::UINT16, keel::as<keel::u16>(0), mr);
    if (name == "uint32") return typedScalar<keel::u32>(ValueType::UINT32, keel::as<keel::u32>(0), mr);
    if (name == "uint64") return typedScalar<keel::u64>(ValueType::UINT64, keel::as<keel::u64>(0), mr);
    throw numkit::Error("intmin: unsupported integer class");
}

Value realmax(const Value &t, keel::Allocator *mr)
{
    auto name = readTypeName(t, "double");
    if (name == "single")
        return typedScalar<keel::f32>(ValueType::SINGLE, keel::Limits<keel::f32>::kMax, mr);
    if (name == "double")
        return Value::scalar(keel::Limits<keel::f64>::kMax, mr);
    throw numkit::Error("realmax: type must be 'double' or 'single'");
}

Value realmin(const Value &t, keel::Allocator *mr)
{
    auto name = readTypeName(t, "double");
    if (name == "single")
        return typedScalar<keel::f32>(ValueType::SINGLE, keel::Limits<keel::f32>::kMin, mr);
    if (name == "double")
        return Value::scalar(keel::Limits<keel::f64>::kMin, mr);
    throw numkit::Error("realmin: type must be 'double' or 'single'");
}

Value flintmax(keel::StringRef className, keel::Allocator *mr)
{
    return flintmax(Value::fromString(className, mr), mr);
}

Value intmax(keel::StringRef className, keel::Allocator *mr)
{
    return intmax(Value::fromString(className, mr), mr);
}

Value intmin(keel::StringRef className, keel::Allocator *mr)
{
    return intmin(Value::fromString(className, mr), mr);
}

Value realmax(keel::StringRef className, keel::Allocator *mr)
{
    return realmax(Value::fromString(className, mr), mr);
}

Value realmin(keel::StringRef className, keel::Allocator *mr)
{
    return realmin(Value::fromString(className, mr), mr);
}

// ── Whole-array float predicates ────────────────────────────────────

Value allfinite(const Value &x, keel::Allocator *mr)
{
    if (x.isEmpty()) return Value::logicalScalar(true, mr);
    if (!x.isNumeric()) return Value::logicalScalar(false, mr);
    if (x.isScalar()) {
        if (x.isComplex()) {
            const auto c = x.toComplex();
            return Value::logicalScalar(keel::is_finite(c.real()) && keel::is_finite(c.imag()), mr);
        }
        return Value::logicalScalar(keel::is_finite(x.toScalar()), mr);
    }
    const keel::usize n = x.numel();
    if (x.isComplex()) {
        const keel::c64 *p = x.complexData();
        for (keel::usize i = 0; i < n; ++i)
            if (!keel::is_finite(p[i].real()) || !keel::is_finite(p[i].imag()))
                return Value::logicalScalar(false, mr);
        return Value::logicalScalar(true, mr);
    }
    if (isFloatType(x.type())) {
        const keel::f64 *p = x.doubleData();
        for (keel::usize i = 0; i < n; ++i)
            if (!keel::is_finite(p[i])) return Value::logicalScalar(false, mr);
        return Value::logicalScalar(true, mr);
    }
    // Integer / logical types are always finite.
    return Value::logicalScalar(true, mr);
}

Value anynan(const Value &x, keel::Allocator *mr)
{
    if (x.isEmpty()) return Value::logicalScalar(false, mr);
    if (!x.isNumeric()) return Value::logicalScalar(false, mr);
    if (x.isScalar()) {
        if (x.isComplex()) {
            const auto c = x.toComplex();
            return Value::logicalScalar(keel::is_nan(c.real()) || keel::is_nan(c.imag()), mr);
        }
        return Value::logicalScalar(keel::is_nan(x.toScalar()), mr);
    }
    const keel::usize n = x.numel();
    if (x.isComplex()) {
        const keel::c64 *p = x.complexData();
        for (keel::usize i = 0; i < n; ++i)
            if (keel::is_nan(p[i].real()) || keel::is_nan(p[i].imag()))
                return Value::logicalScalar(true, mr);
        return Value::logicalScalar(false, mr);
    }
    if (isFloatType(x.type())) {
        const keel::f64 *p = x.doubleData();
        for (keel::usize i = 0; i < n; ++i)
            if (keel::is_nan(p[i])) return Value::logicalScalar(true, mr);
        return Value::logicalScalar(false, mr);
    }
    return Value::logicalScalar(false, mr);
}

// ════════════════════════════════════════════════════════════════════════
// Public API — equality + introspection
// ════════════════════════════════════════════════════════════════════════

Value isequal(const Value &a, const Value &b, keel::Allocator *mr)
{
    return Value::logicalScalar(valuesEqual(a, b, false), mr);
}

Value isequaln(const Value &a, const Value &b, keel::Allocator *mr)
{
    return Value::logicalScalar(valuesEqual(a, b, true), mr);
}

Value classOf(const Value &x, keel::Allocator *mr)
{
    // OBJECT instances report their registered class name, not "object".
    if (x.isObject())
        return Value::fromString(x.objectClassNameRef(), mr);
    if (x.isSingleComplex())
        return Value::fromString("single", mr);
    return Value::fromString(mtypeName(x.type()), mr);
}

// ── Pack 36: cast + swapbytes ────────────────────────────────────────
Value cast(const Value &x, keel::StringRef classname, keel::Allocator *mr)
{
    if (classname == "double")  return toDouble(x, mr);
    if (classname == "single")  return single(x, mr);
    if (classname == "int8")    return int8(x, mr);
    if (classname == "int16")   return int16(x, mr);
    if (classname == "int32")   return int32(x, mr);
    if (classname == "int64")   return int64(x, mr);
    if (classname == "uint8")   return uint8(x, mr);
    if (classname == "uint16")  return uint16(x, mr);
    if (classname == "uint32")  return uint32(x, mr);
    if (classname == "uint64")  return uint64(x, mr);
    if (classname == "logical") return logical(x, mr);
    if (classname == "char")    return toChar(x, mr);
    if (classname == "string")  return toString(x, mr);
    throw Error(keel::format("cast: unsupported class '{}'", classname),
                 0, 0, "cast", "", "numkit:cast:badClass");
}


// Map a MATLAB classname string → (ValueType, element-size-in-bytes).

Value typecast(const Value &x, keel::StringRef classname, keel::Allocator *mr)
{
    TypeInfo info = typeInfoFor(classname);
    if (info.elemSize == 0)
        throw Error(keel::format("typecast: unsupported class '{}'", classname),
                     0, 0, "typecast", "", "numkit:typecast:badClass");

    const keel::usize srcElemSize = elemSizeOf(x.type());
    if (srcElemSize == 0)
        throw Error("typecast: input type does not have a contiguous byte buffer",
                     0, 0, "typecast", "", "numkit:typecast:badInputType");

    const keel::usize totalBytes = x.numel() * srcElemSize;
    if (totalBytes % info.elemSize != 0)
        throw Error("typecast: input byte count must be a multiple of the "
                    "destination element size",
                     0, 0, "typecast", "", "numkit:typecast:badSize");
    const keel::usize newCount = totalBytes / info.elemSize;

    // Output is always a row vector (matches MATLAB).
    Value out = Value::matrix(1, newCount, info.vt, mr);
    if (totalBytes > 0)
        keel::mem_copy(out.rawDataMut(), x.rawData(), totalBytes);
    return out;
}

Value swapbytes(const Value &x, keel::Allocator *mr)
{
    switch (x.type()) {
    case ValueType::INT8:    return int8(x, mr);     // copy through (1 byte = identity)
    case ValueType::UINT8:   return uint8(x, mr);
    case ValueType::LOGICAL: return logical(x, mr);
    case ValueType::INT16:   return swapBytesArray<keel::i16>(x, mr);
    case ValueType::UINT16:  return swapBytesArray<keel::u16>(x, mr);
    case ValueType::INT32:   return swapBytesArray<keel::i32>(x, mr);
    case ValueType::UINT32:  return swapBytesArray<keel::u32>(x, mr);
    case ValueType::INT64:   return swapBytesArray<keel::i64>(x, mr);
    case ValueType::UINT64:  return swapBytesArray<keel::u64>(x, mr);
    case ValueType::SINGLE:  return swapBytesArray<keel::f32>(x, mr);
    case ValueType::DOUBLE:  return swapBytesArray<keel::f64>(x, mr);
    default:
        throw Error("swapbytes: input must be a numeric or logical array",
                     0, 0, "swapbytes", "", "numkit:swapbytes:badType");
    }
}

// ════════════════════════════════════════════════════════════════════════
// Adapters
// ════════════════════════════════════════════════════════════════════════


} // namespace numkit::builtin
