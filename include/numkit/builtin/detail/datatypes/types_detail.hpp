// toolboxes/.../types_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by types.cpp + types_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/complex.h>
#include <keel/utf8.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <utility>



namespace numkit::builtin::detail {
// SIMD/portable cast + isfinite backend (defs in casts_{highway,portable}.cpp /
// isfinite_{highway,portable}.cpp) — needed by the numericConstructor template below.
void doubleToInt8 (const keel::f64 *in, keel::i8   *out, keel::usize n);
void doubleToInt16(const keel::f64 *in, keel::i16  *out, keel::usize n);
void doubleToInt32(const keel::f64 *in, keel::i32  *out, keel::usize n);
void doubleToInt64(const keel::f64 *in, keel::i64  *out, keel::usize n);
void doubleToUInt8 (const keel::f64 *in, keel::u8  *out, keel::usize n);
void doubleToUInt16(const keel::f64 *in, keel::u16 *out, keel::usize n);
void doubleToUInt32(const keel::f64 *in, keel::u32 *out, keel::usize n);
void doubleToUInt64(const keel::f64 *in, keel::u64 *out, keel::usize n);
} // namespace numkit::builtin::detail

// isnan/isinf/isfinite backend lives in arithmetic → numkit::builtin::detail (C4).
namespace numkit::builtin::detail {
void doubleIsNaNLoop(const keel::f64 *in, keel::u8 *out, keel::usize n);
void doubleIsInfLoop(const keel::f64 *in, keel::u8 *out, keel::usize n);
void doubleIsFiniteLoop(const keel::f64 *in, keel::u8 *out, keel::usize n);
} // namespace numkit::builtin::detail

namespace numkit::builtin {

namespace {

template <typename T>
T saturateCast(keel::f64 v)
{
    if (keel::is_nan(v)) return 0;
    v = keel::round(v);
    constexpr keel::f64 lo = keel::as<keel::f64>(keel::Limits<T>::kMin);
    constexpr keel::f64 hi = keel::as<keel::f64>(keel::Limits<T>::kMax);
    if (v < lo) return keel::Limits<T>::kMin;
    if (v > hi) return keel::Limits<T>::kMax;
    return static_cast<T>(v);
}

template <typename T>
Value numericConstructor(ValueType targetType, const Value &x, keel::Allocator *mr)
{
    if (x.isComplex()) {
        if (targetType == ValueType::SINGLE) {
            if (x.isSingleComplex())
                return x;
            const keel::usize n = x.numel();
            Value r = createLike(x, ValueType::COMPLEX, mr);
            r.setSingleComplex(true);
            const keel::c64 *src = x.complexData();
            keel::c64 *dstC = r.complexDataMut();
            for (keel::usize i = 0; i < n; ++i) {
                dstC[i] = keel::c64(keel::as<keel::f64>(keel::as<keel::f32>(src[i].real())),
                                  keel::as<keel::f64>(keel::as<keel::f32>(src[i].imag())));
            }
            return r;
        }
        if (targetType == ValueType::DOUBLE) {
            if (!x.isSingleComplex())
                return x;
            const keel::usize n = x.numel();
            Value r = createLike(x, ValueType::COMPLEX, mr);
            r.setSingleComplex(false);
            const keel::c64 *src = x.complexData();
            keel::c64 *dstC = r.complexDataMut();
            for (keel::usize i = 0; i < n; ++i)
                dstC[i] = src[i];
            return r;
        }
    }
    if (x.type() == targetType)
        return x;
    const keel::usize n = x.numel();
    Value r = createLike(x, targetType, mr);
    T *dst = static_cast<T *>(r.rawDataMut());

    // SIMD fast path: double → integer target. Skips the per-element
    // virtual dispatch through elemAsDouble. Only kicks in when source
    // is DOUBLE and there's actual data; falls through for other source
    // types (single, int*, logical, char) where the bulk of the cost
    // is type-conversion not iteration.
    if constexpr (std::is_integral_v<T>) {
        if (x.type() == ValueType::DOUBLE && n > 0) {
            const keel::f64 *src = x.doubleData();
            if constexpr (std::is_same_v<T, keel::i8>)        ::numkit::builtin::detail::doubleToInt8 (src, dst, n);
            else if constexpr (std::is_same_v<T, keel::i16>)  ::numkit::builtin::detail::doubleToInt16(src, dst, n);
            else if constexpr (std::is_same_v<T, keel::i32>)  ::numkit::builtin::detail::doubleToInt32(src, dst, n);
            else if constexpr (std::is_same_v<T, keel::i64>)  ::numkit::builtin::detail::doubleToInt64(src, dst, n);
            else if constexpr (std::is_same_v<T, keel::u8>)  ::numkit::builtin::detail::doubleToUInt8 (src, dst, n);
            else if constexpr (std::is_same_v<T, keel::u16>) ::numkit::builtin::detail::doubleToUInt16(src, dst, n);
            else if constexpr (std::is_same_v<T, keel::u32>) ::numkit::builtin::detail::doubleToUInt32(src, dst, n);
            else if constexpr (std::is_same_v<T, keel::u64>) ::numkit::builtin::detail::doubleToUInt64(src, dst, n);
            return r;
        }
    }

    for (keel::usize i = 0; i < n; ++i) {
        keel::f64 v = x.elemAsDouble(i);
        if constexpr (std::is_integral_v<T>)
            dst[i] = saturateCast<T>(v);
        else
            dst[i] = static_cast<T>(v);
    }
    return r;
}

inline bool isNumericComparable(ValueType t)
{
    return t == ValueType::DOUBLE || t == ValueType::SINGLE || t == ValueType::COMPLEX
        || t == ValueType::LOGICAL || t == ValueType::CHAR || isIntegerType(t);
}

inline keel::f64 elemComparableDouble(const Value &v, keel::usize i)
{
    if (v.type() == ValueType::CHAR)
        return keel::as<keel::f64>(keel::as<keel::u8>(v.charData()[i]));
    return v.elemAsDouble(i);
}

inline bool int64EqualScalar(const Value &a, keel::usize ia, const Value &b, keel::usize ib, bool nanEqual)
{
    // Exact comparison when at least one side is INT64 or UINT64 (avoids double
    // precision loss above 2^53).
    auto cmpU64WithF64 = [](keel::u64 u, keel::f64 d) -> bool {
        if (!keel::is_finite(d) || d < 0.0 || d >= 18446744073709551616.0) return false;
        if (keel::floor(d) != d) return false;
        return static_cast<keel::u64>(d) == u;
    };
    auto cmpI64WithF64 = [](keel::i64 s, keel::f64 d) -> bool {
        if (!keel::is_finite(d) || d < -9223372036854775808.0 || d >= 9223372036854775808.0) return false;
        if (keel::floor(d) != d) return false;
        return static_cast<keel::i64>(d) == s;
    };
    const ValueType ta = a.type(), tb = b.type();
    if (ta == ValueType::INT64 && tb == ValueType::UINT64) {
        const keel::i64 sa = a.int64Data()[ia];
        const keel::u64 ub = b.uint64Data()[ib];
        return sa >= 0 && static_cast<keel::u64>(sa) == ub;
    }
    if (ta == ValueType::UINT64 && tb == ValueType::INT64) {
        const keel::u64 ua = a.uint64Data()[ia];
        const keel::i64 sb = b.int64Data()[ib];
        return sb >= 0 && ua == static_cast<keel::u64>(sb);
    }
    if (ta == ValueType::INT64) {
        const keel::i64 sa = a.int64Data()[ia];
        if (tb == ValueType::COMPLEX) {
            const keel::c64 cb = b.complexData()[ib];
            return cb.imag() == 0.0 && cmpI64WithF64(sa, cb.real());
        }
        return cmpI64WithF64(sa, elemComparableDouble(b, ib));
    }
    if (ta == ValueType::UINT64) {
        const keel::u64 ua = a.uint64Data()[ia];
        if (tb == ValueType::COMPLEX) {
            const keel::c64 cb = b.complexData()[ib];
            return cb.imag() == 0.0 && cmpU64WithF64(ua, cb.real());
        }
        return cmpU64WithF64(ua, elemComparableDouble(b, ib));
    }
    if (tb == ValueType::INT64) {
        const keel::i64 sb = b.int64Data()[ib];
        if (ta == ValueType::COMPLEX) {
            const keel::c64 ca = a.complexData()[ia];
            return ca.imag() == 0.0 && cmpI64WithF64(sb, ca.real());
        }
        return cmpI64WithF64(sb, elemComparableDouble(a, ia));
    }
    if (tb == ValueType::UINT64) {
        const keel::u64 ub = b.uint64Data()[ib];
        if (ta == ValueType::COMPLEX) {
            const keel::c64 ca = a.complexData()[ia];
            return ca.imag() == 0.0 && cmpU64WithF64(ub, ca.real());
        }
        return cmpU64WithF64(ub, elemComparableDouble(a, ia));
    }
    (void)nanEqual;
    return false;
}

bool valuesEqual(const Value &a, const Value &b, bool nanEqual)
{
    const keel::i32 ndA = a.dims().ndims();
    const keel::i32 ndB = b.dims().ndims();
    const keel::i32 ndMax = (ndA > ndB) ? ndA : ndB;
    for (keel::i32 d = 0; d < ndMax; ++d) {
        const keel::usize da = (d < ndA) ? a.dims().dim(d) : 1u;
        const keel::usize db = (d < ndB) ? b.dims().dim(d) : 1u;
        if (da != db) return false;
    }

    const ValueType ta = a.type(), tb = b.type();
    const keel::usize n = a.numel();

    if (ta != tb) {
        if (!isNumericComparable(ta) || !isNumericComparable(tb))
            return false;
        if (n == 0) return true;
        const bool has64 = (ta == ValueType::INT64 || ta == ValueType::UINT64
                         || tb == ValueType::INT64 || tb == ValueType::UINT64);
        if (has64) {
            for (keel::usize i = 0; i < n; ++i) {
                if (!int64EqualScalar(a, i, b, i, nanEqual)) return false;
            }
            return true;
        }
        if (ta == ValueType::COMPLEX || tb == ValueType::COMPLEX) {
            for (keel::usize i = 0; i < n; ++i) {
                const keel::f64 ra = (ta == ValueType::COMPLEX) ? a.complexData()[i].real() : elemComparableDouble(a, i);
                const keel::f64 ia = (ta == ValueType::COMPLEX) ? a.complexData()[i].imag() : 0.0;
                const keel::f64 rb = (tb == ValueType::COMPLEX) ? b.complexData()[i].real() : elemComparableDouble(b, i);
                const keel::f64 ib = (tb == ValueType::COMPLEX) ? b.complexData()[i].imag() : 0.0;
                const bool rEq = (ra == rb) || (nanEqual && keel::is_nan(ra) && keel::is_nan(rb));
                const bool iEq = (ia == ib) || (nanEqual && keel::is_nan(ia) && keel::is_nan(ib));
                if (!rEq || !iEq) return false;
            }
            return true;
        }
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 da = elemComparableDouble(a, i);
            const keel::f64 db = elemComparableDouble(b, i);
            if (da == db) continue;
            if (nanEqual && keel::is_nan(da) && keel::is_nan(db)) continue;
            return false;
        }
        return true;
    }

    ValueType t = ta;

    if (t == ValueType::DOUBLE) {
        const keel::f64 *da = a.doubleData(), *db = b.doubleData();
        for (keel::usize i = 0; i < n; ++i) {
            if (da[i] == db[i]) continue;
            if (nanEqual && keel::is_nan(da[i]) && keel::is_nan(db[i])) continue;
            return false;
        }
        return true;
    }
    if (t == ValueType::SINGLE) {
        const keel::f32 *fa = a.singleData(), *fb = b.singleData();
        for (keel::usize i = 0; i < n; ++i) {
            if (fa[i] == fb[i]) continue;
            if (nanEqual && keel::is_nan(fa[i]) && keel::is_nan(fb[i])) continue;
            return false;
        }
        return true;
    }
    if (t == ValueType::COMPLEX) {
        const keel::c64 *ca = a.complexData(), *cb = b.complexData();
        for (keel::usize i = 0; i < n; ++i) {
            if (ca[i] == cb[i]) continue;
            if (nanEqual) {
                bool rEq = (ca[i].real() == cb[i].real())
                           || (keel::is_nan(ca[i].real()) && keel::is_nan(cb[i].real()));
                bool iEq = (ca[i].imag() == cb[i].imag())
                           || (keel::is_nan(ca[i].imag()) && keel::is_nan(cb[i].imag()));
                if (rEq && iEq) continue;
            }
            return false;
        }
        return true;
    }
    if (t == ValueType::CHAR)
        return keel::mem_cmp(a.charData(), b.charData(), n) == 0;
    if (t == ValueType::LOGICAL)
        return keel::mem_cmp(a.logicalData(), b.logicalData(), n) == 0;
    if (isIntegerType(t))
        return keel::mem_cmp(a.rawData(), b.rawData(), n * elementSize(t)) == 0;
    if (t == ValueType::CELL) {
        for (keel::usize i = 0; i < n; ++i)
            if (!valuesEqual(a.cellAt(i), b.cellAt(i), nanEqual)) return false;
        return true;
    }
    if (t == ValueType::STRUCT) {
        auto &fa = a.structFields(), &fb = b.structFields();
        if (fa.size() != fb.size()) return false;
        for (auto entry : fa) {
            const Value *bv = fb.get(entry.key);
            if (!bv) return false;
            if (!valuesEqual(entry.value, *bv, nanEqual)) return false;
        }
        return true;
    }
    if (t == ValueType::STRING)
        return a.toString() == b.toString();
    return false;
}

} // namespace
namespace {

// True iff this element-type can hold "standard" missing (i.e. NaN).
inline bool typeHasMissing(ValueType t)
{
    return t == ValueType::DOUBLE || t == ValueType::SINGLE;
}

// Read element i as double for the comparison path; works for any
// numeric class (integer types convert losslessly; LOGICAL → 0/1).
inline keel::f64 elemD(const Value &x, keel::usize i) { return x.elemAsDouble(i); }

} // anonymous
namespace {
enum class SortMode { Ascend, Descend, Monotonic, StrictAscend, StrictDescend };

inline bool readSortMode(const Value &m, SortMode &out)
{
    out = SortMode::Ascend;
    if (m.isEmpty()) return true;
    if (!m.isChar() && !m.isString()) return false;
    auto s = m.toString();
    s.to_ascii_lower_in_place();
    if (s == "ascend")        { out = SortMode::Ascend;        return true; }
    if (s == "descend")       { out = SortMode::Descend;       return true; }
    if (s == "monotonic")     { out = SortMode::Monotonic;     return true; }
    if (s == "strictascend")  { out = SortMode::StrictAscend;  return true; }
    if (s == "strictdescend") { out = SortMode::StrictDescend; return true; }
    return false;
}

// Returns true if values [first, first+n) are sorted under `mode`.
inline bool runSorted(const keel::f64 *first, keel::usize n, SortMode mode)
{
    if (n < 2) return true;
    // NaN anywhere → not sorted (matches MATLAB issorted behaviour for
    // simple ascend/descend modes).
    for (keel::usize i = 0; i < n; ++i)
        if (keel::is_nan(first[i])) return false;

    switch (mode) {
    case SortMode::Ascend:
        for (keel::usize i = 1; i < n; ++i) if (first[i] < first[i-1]) return false;
        return true;
    case SortMode::Descend:
        for (keel::usize i = 1; i < n; ++i) if (first[i] > first[i-1]) return false;
        return true;
    case SortMode::StrictAscend:
        for (keel::usize i = 1; i < n; ++i) if (first[i] <= first[i-1]) return false;
        return true;
    case SortMode::StrictDescend:
        for (keel::usize i = 1; i < n; ++i) if (first[i] >= first[i-1]) return false;
        return true;
    case SortMode::Monotonic: {
        bool asc = true, desc = true;
        for (keel::usize i = 1; i < n; ++i) {
            if (first[i] < first[i-1]) asc = false;
            if (first[i] > first[i-1]) desc = false;
        }
        return asc || desc;
    }
    }
    return true; // unreachable
}
} // anon
namespace {
inline keel::String readTypeName(const Value &t, const char *def)
{
    if (t.isEmpty()) return keel::String::from(def);
    if (!t.isChar() && !t.isString())
        throw std::runtime_error("numeric-limit: type argument must be a string");
    keel::String s = t.toString();
    s.to_ascii_lower_in_place();
    return s;
}

template <typename T>
inline Value typedScalar(ValueType vt, T v, keel::Allocator *mr)
{
    auto r = Value::matrix(1, 1, vt, mr);
    *static_cast<T *>(r.rawDataMut()) = v;
    return r;
}
} // anon
namespace {

// Byte-swap helper. 1-byte types pass through; 2/4/8-byte types
// re-interpret as the matching unsigned int and reverse.
template <typename T>
T swapBytesScalar(T v)
{
    static_assert(std::is_trivially_copyable_v<T>);
    if constexpr (sizeof(T) == 1) {
        return v;
    } else if constexpr (sizeof(T) == 2) {
        keel::u16 u; keel::mem_copy(&u, &v, 2);
        u = keel::as<keel::u16>((u << 8) | (u >> 8));
        T r; keel::mem_copy(&r, &u, 2); return r;
    } else if constexpr (sizeof(T) == 4) {
        keel::u32 u; keel::mem_copy(&u, &v, 4);
        u = ((u & 0xFF000000U) >> 24) | ((u & 0x00FF0000U) >> 8)
          | ((u & 0x0000FF00U) << 8)  | ((u & 0x000000FFU) << 24);
        T r; keel::mem_copy(&r, &u, 4); return r;
    } else {  // 8
        keel::u64 u; keel::mem_copy(&u, &v, 8);
        u = ((u & 0xFF00000000000000ULL) >> 56) | ((u & 0x00FF000000000000ULL) >> 40)
          | ((u & 0x0000FF0000000000ULL) >> 24) | ((u & 0x000000FF00000000ULL) >> 8)
          | ((u & 0x00000000FF000000ULL) << 8)  | ((u & 0x0000000000FF0000ULL) << 24)
          | ((u & 0x000000000000FF00ULL) << 40) | ((u & 0x00000000000000FFULL) << 56);
        T r; keel::mem_copy(&r, &u, 8); return r;
    }
}

template <typename T>
Value swapBytesArray(const Value &x, keel::Allocator *mr)
{
    Value r = createLike(x, x.type(), mr);
    const T *src = static_cast<const T *>(x.rawData());
    T *dst = static_cast<T *>(r.rawDataMut());
    const keel::usize n = x.numel();
    for (keel::usize i = 0; i < n; ++i)
        dst[i] = swapBytesScalar(src[i]);
    return r;
}

} // namespace
namespace {

struct TypeInfo { ValueType vt; keel::usize elemSize; };

[[maybe_unused]] TypeInfo typeInfoFor(keel::StringRef classname)
{
    if (classname == "double")  return { ValueType::DOUBLE,  sizeof(keel::f64)   };
    if (classname == "single")  return { ValueType::SINGLE,  sizeof(keel::f32)    };
    if (classname == "int8")    return { ValueType::INT8,    sizeof(keel::i8)   };
    if (classname == "int16")   return { ValueType::INT16,   sizeof(keel::i16)  };
    if (classname == "int32")   return { ValueType::INT32,   sizeof(keel::i32)  };
    if (classname == "int64")   return { ValueType::INT64,   sizeof(keel::i64)  };
    if (classname == "uint8")   return { ValueType::UINT8,   sizeof(keel::u8)  };
    if (classname == "uint16")  return { ValueType::UINT16,  sizeof(keel::u16) };
    if (classname == "uint32")  return { ValueType::UINT32,  sizeof(keel::u32) };
    if (classname == "uint64")  return { ValueType::UINT64,  sizeof(keel::u64) };
    if (classname == "logical") return { ValueType::LOGICAL, sizeof(keel::u8)  };
    if (classname == "char")    return { ValueType::CHAR,    sizeof(char)     };
    return { ValueType::DOUBLE, 0u };  // sentinel: 0 size = unknown
}

[[maybe_unused]] keel::usize elemSizeOf(ValueType t)
{
    switch (t) {
    case ValueType::DOUBLE:  return sizeof(keel::f64);
    case ValueType::SINGLE:  return sizeof(keel::f32);
    case ValueType::COMPLEX: return sizeof(keel::f64) * 2;
    case ValueType::INT8:
    case ValueType::UINT8:
    case ValueType::LOGICAL:
    case ValueType::CHAR:    return 1;
    case ValueType::INT16:
    case ValueType::UINT16:  return 2;
    case ValueType::INT32:
    case ValueType::UINT32:  return 4;
    case ValueType::INT64:
    case ValueType::UINT64:  return 8;
    default:                 return 0;
    }
}

} // namespace

} // namespace numkit::builtin
