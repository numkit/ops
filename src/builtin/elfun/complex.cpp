// math/src/complex/complex.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/builtin/elfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>


namespace numkit::builtin {

// ════════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════════

Value real(const Value &x, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (!x.isComplex())
        return x;
    if (x.isSingleComplex()) {
        auto r = createLike(x, ValueType::SINGLE, p);
        keel::f32 *dst = r.singleDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            dst[i] = keel::as<keel::f32>(x.complexData()[i].real());
        return r;
    }
    if (x.isScalar())
        return Value::scalar(x.toComplex().real(), p);
    auto r = createLike(x, ValueType::DOUBLE, p);
    for (keel::usize i = 0; i < x.numel(); ++i)
        r.doubleDataMut()[i] = x.complexData()[i].real();
    return r;
}

Value imag(const Value &x, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (!x.isComplex()) {
        if (x.type() == ValueType::SINGLE)
            return createLike(x, ValueType::SINGLE, p);
        if (x.isScalar())
            return Value::scalar(0.0, p);
        return createLike(x, ValueType::DOUBLE, p);
    }
    if (x.isSingleComplex()) {
        auto r = createLike(x, ValueType::SINGLE, p);
        keel::f32 *dst = r.singleDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            dst[i] = keel::as<keel::f32>(x.complexData()[i].imag());
        return r;
    }
    if (x.isScalar())
        return Value::scalar(x.toComplex().imag(), p);
    auto r = createLike(x, ValueType::DOUBLE, p);
    for (keel::usize i = 0; i < x.numel(); ++i)
        r.doubleDataMut()[i] = x.complexData()[i].imag();
    return r;
}

Value conj(const Value &x, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (!x.isComplex())
        return x;
    return unaryComplex(x, [](const keel::c64 &c) { return keel::conj(c); }, p);
}

Value complex(const Value &re, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (re.isComplex())
        return re;
    const bool outSingle = (re.type() == ValueType::SINGLE);
    if (re.isScalar() && !outSingle)
        return Value::complexScalar(re.toScalar(), 0.0, p);
    auto r = createLike(re, ValueType::COMPLEX, p);
    if (outSingle) r.setSingleComplex(true);
    keel::c64 *dst = r.complexDataMut();
    for (keel::usize i = 0; i < re.numel(); ++i) {
        keel::f64 v = re.elemAsDouble(i);
        if (outSingle) v = keel::as<keel::f64>(keel::as<keel::f32>(v));
        dst[i] = keel::c64(v, 0.0);
    }
    return r;
}

Value complex(const Value &re, const Value &im, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    const bool outSingle = (re.type() == ValueType::SINGLE || re.isSingleComplex() ||
                            im.type() == ValueType::SINGLE || im.isSingleComplex());
    if (re.isScalar() && im.isScalar() && !outSingle)
        return Value::complexScalar(re.toScalar(), im.toScalar(), p);
    const Value &shape = re.isScalar() ? im : re;
    if (!re.isScalar() && !im.isScalar() && re.dims() != im.dims())
        throw Error("complex: real and imaginary parts must have matching dimensions",
                     0, 0, "complex", "", "numkit:complex:dimagree");
    auto r = createLike(shape, ValueType::COMPLEX, p);
    if (outSingle) r.setSingleComplex(true);
    keel::c64 *dst = r.complexDataMut();
    const keel::usize n = shape.numel();
    const keel::f64 reScalar = re.isScalar() ? re.toScalar() : 0.0;
    const keel::f64 imScalar = im.isScalar() ? im.toScalar() : 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        keel::f64 r0 = re.isScalar() ? reScalar : re.elemAsDouble(i);
        keel::f64 i0 = im.isScalar() ? imScalar : im.elemAsDouble(i);
        if (outSingle) {
            r0 = keel::as<keel::f64>(keel::as<keel::f32>(r0));
            i0 = keel::as<keel::f64>(keel::as<keel::f32>(i0));
        }
        dst[i] = keel::c64(r0, i0);
    }
    return r;
}

Value angle(const Value &x, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (x.isComplex()) {
        if (x.isSingleComplex()) {
            auto r = createLike(x, ValueType::SINGLE, p);
            keel::f32 *dst = r.singleDataMut();
            for (keel::usize i = 0; i < x.numel(); ++i)
                dst[i] = keel::as<keel::f32>(keel::arg(x.complexData()[i]));
            return r;
        }
        if (x.isScalar())
            return Value::scalar(keel::arg(x.toComplex()), p);
        auto r = createLike(x, ValueType::DOUBLE, p);
        for (keel::usize i = 0; i < x.numel(); ++i)
            r.doubleDataMut()[i] = keel::arg(x.complexData()[i]);
        return r;
    }
    if (x.type() == ValueType::SINGLE) {
        auto r = createLike(x, ValueType::SINGLE, p);
        const keel::f32 *src = x.singleData();
        keel::f32 *dst = r.singleDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            dst[i] = keel::atan2(0.0f, src[i]);
        return r;
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::atan2(0.0, v); }, p);
}

Value isreal(const Value &x, keel::Allocator *mr)
{
    return Value::scalar(!x.isComplex() ? 1.0 : 0.0, mr);
}

Value unwrap(const Value &p, keel::f64 tol, keel::i32 /*dim*/, keel::Allocator *mr)
{
    if (p.isEmpty())
        return p;

    Value out = p;
    if (out.type() != ValueType::DOUBLE)
        out = toDoubleValue(p, mr);
    else
        out = Value(p); // copy

    keel::f64 *data = out.doubleDataMut();
    const keel::usize n = out.numel();
    if (n <= 1) return out;

    constexpr keel::f64 kTwoPi = keel::kTau;
    keel::f64 dp_corr = 0.0;

    for (keel::usize i = 1; i < n; ++i) {
        keel::f64 d = data[i] - data[i - 1];
        if (d > tol) {
            keel::f64 jumps = keel::floor((d + tol) / kTwoPi);
            dp_corr -= jumps * kTwoPi;
        } else if (d < -tol) {
            keel::f64 jumps = keel::floor((-d + tol) / kTwoPi);
            dp_corr += jumps * kTwoPi;
        }
        data[i] += dp_corr;
    }

    return out;
}

} // namespace numkit::builtin
