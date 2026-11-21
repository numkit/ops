// src/builtin/src/ops/unary_ops.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/builtin/ops.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "unary_ops_detail.hpp"

namespace numkit::builtin {

// ════════════════════════════════════════════════════════════════════════
// Public API — unary operators
// ════════════════════════════════════════════════════════════════════════

Value uminus(const Value &x, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (x.isEmpty()) {
        ValueType outType = x.isComplex()                   ? ValueType::COMPLEX
                        : (x.isChar() || x.isLogical()) ? ValueType::DOUBLE
                                                        : x.type();
        auto r = createLike(x, outType, p);
        if (x.isSingleComplex()) r.setSingleComplex(true);
        return r;
    }
    if (x.isComplex())
        return unaryComplex(x, [](keel::c64 v) { return -v; }, p);
    if (x.type() == ValueType::DOUBLE)
        return unaryDouble(x, [](keel::f64 v) { return -v; }, p);
    if (x.type() == ValueType::SINGLE)
        return unaryTyped<keel::f32>(x, ValueType::SINGLE, [](keel::f32 v) { return -v; }, p);
    if (isIntegerType(x.type())) {
        switch (x.type()) {
        case ValueType::INT8:   return unaryTyped<keel::i8>(x, x.type(),  [](keel::i8 v)   { return saturateNeg(v); }, p);
        case ValueType::INT16:  return unaryTyped<keel::i16>(x, x.type(), [](keel::i16 v)  { return saturateNeg(v); }, p);
        case ValueType::INT32:  return unaryTyped<keel::i32>(x, x.type(), [](keel::i32 v)  { return saturateNeg(v); }, p);
        case ValueType::INT64:  return unaryTyped<keel::i64>(x, x.type(), [](keel::i64 v)  { return saturateNeg(v); }, p);
        case ValueType::UINT8:  return unaryTyped<keel::u8>(x, x.type(), [](keel::u8)    { return keel::u8(0); },    p);
        case ValueType::UINT16: return unaryTyped<keel::u16>(x, x.type(),[](keel::u16)   { return keel::u16(0); },   p);
        case ValueType::UINT32: return unaryTyped<keel::u32>(x, x.type(),[](keel::u32)   { return keel::u32(0); },   p);
        case ValueType::UINT64: return unaryTyped<keel::u64>(x, x.type(),[](keel::u64)   { return keel::u64(0); },   p);
        default: break;
        }
    }
    throw Error("Unsupported unary -", 0, 0, "uminus", "", "numkit:uminus:unsupportedTypes");
}

Value uplus(const Value &x, keel::Allocator *)
{
    return x;
}

Value logical_not(const Value &x, keel::Allocator *mr)
{
    auto nanErr = [] {
        throw std::runtime_error("NaN values cannot be converted to logicals.");
    };
    keel::Allocator *p = mr;
    if (x.isLogical()) {
        if (x.isScalar())
            return Value::logicalScalar(!x.toBool(), p);
        auto r = createLike(x, ValueType::LOGICAL, p);
        const keel::u8 *src = x.logicalData();
        keel::u8 *dst = r.logicalDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            dst[i] = src[i] ? 0 : 1;
        return r;
    }
    if (x.type() == ValueType::DOUBLE) {
        if (x.isScalar()) {
            keel::f64 v = x.toScalar();
            if (std::isnan(v))
                nanErr();
            return Value::logicalScalar(v == 0.0, p);
        }
        auto r = createLike(x, ValueType::LOGICAL, p);
        const keel::f64 *src = x.doubleData();
        keel::u8 *dst = r.logicalDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i) {
            if (std::isnan(src[i]))
                nanErr();
            dst[i] = (src[i] == 0.0) ? 1 : 0;
        }
        return r;
    }
    if (x.isComplex()) {
        auto r = createLike(x, ValueType::LOGICAL, p);
        const keel::c64 *src = x.complexData();
        keel::u8 *dst = r.logicalDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i) {
            if (std::isnan(src[i].real()) || std::isnan(src[i].imag()))
                nanErr();
            dst[i] = (src[i].real() == 0.0 && src[i].imag() == 0.0) ? 1 : 0;
        }
        return r;
    }
    if (x.isScalar()) {
        keel::f64 v = x.elemAsDouble(0);
        if (std::isnan(v))
            nanErr();
        return Value::logicalScalar(v == 0.0, p);
    }
    auto r = createLike(x, ValueType::LOGICAL, p);
    keel::u8 *dst = r.logicalDataMut();
    for (keel::usize i = 0; i < x.numel(); ++i) {
        keel::f64 v = x.elemAsDouble(i);
        if (std::isnan(v))
            nanErr();
        dst[i] = (v == 0.0) ? 1 : 0;
    }
    return r;
}

Value ctranspose(const Value &x, keel::Allocator *mr)
{
    return transpose2D(x, /*conjugate=*/true, "ctranspose", mr);
}

Value transpose(const Value &x, keel::Allocator *mr)
{
    return transpose2D(x, /*conjugate=*/false, "transpose", mr);
}

Value any(const Value &x, int /*dim*/, keel::Allocator *mr)
{
    if (x.isEmpty())
        return Value::logicalScalar(false, mr);
    const keel::usize n = x.numel();
    if (x.isLogical()) {
        const keel::u8 *d = x.logicalData();
        for (keel::usize i = 0; i < n; ++i) if (d[i]) return Value::logicalScalar(true, mr);
        return Value::logicalScalar(false, mr);
    }
    for (keel::usize i = 0; i < n; ++i) {
        if (x.elemAsDouble(i) != 0.0) return Value::logicalScalar(true, mr);
    }
    return Value::logicalScalar(false, mr);
}

Value all(const Value &x, int /*dim*/, keel::Allocator *mr)
{
    if (x.isEmpty())
        return Value::logicalScalar(true, mr);
    const keel::usize n = x.numel();
    if (x.isLogical()) {
        const keel::u8 *d = x.logicalData();
        for (keel::usize i = 0; i < n; ++i) if (!d[i]) return Value::logicalScalar(false, mr);
        return Value::logicalScalar(true, mr);
    }
    for (keel::usize i = 0; i < n; ++i) {
        if (x.elemAsDouble(i) == 0.0) return Value::logicalScalar(false, mr);
    }
    return Value::logicalScalar(true, mr);
}

} // namespace numkit::builtin
