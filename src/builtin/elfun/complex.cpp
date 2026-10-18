// math/src/complex/complex.cpp

#include <numkit/builtin/elfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>

#include <cmath>
#include <complex>

namespace numkit::builtin {

// ════════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════════

Value real(const Value &x, std::pmr::memory_resource *mr)
{
    std::pmr::memory_resource *p = mr;
    if (!x.isComplex())
        return x;
    if (x.isSingleComplex()) {
        auto r = createLike(x, ValueType::SINGLE, p);
        float *dst = r.singleDataMut();
        for (size_t i = 0; i < x.numel(); ++i)
            dst[i] = static_cast<float>(x.complexData()[i].real());
        return r;
    }
    if (x.isScalar())
        return Value::scalar(x.toComplex().real(), p);
    auto r = createLike(x, ValueType::DOUBLE, p);
    for (size_t i = 0; i < x.numel(); ++i)
        r.doubleDataMut()[i] = x.complexData()[i].real();
    return r;
}

Value imag(const Value &x, std::pmr::memory_resource *mr)
{
    std::pmr::memory_resource *p = mr;
    if (!x.isComplex()) {
        if (x.type() == ValueType::SINGLE)
            return createLike(x, ValueType::SINGLE, p);
        if (x.isScalar())
            return Value::scalar(0.0, p);
        return createLike(x, ValueType::DOUBLE, p);
    }
    if (x.isSingleComplex()) {
        auto r = createLike(x, ValueType::SINGLE, p);
        float *dst = r.singleDataMut();
        for (size_t i = 0; i < x.numel(); ++i)
            dst[i] = static_cast<float>(x.complexData()[i].imag());
        return r;
    }
    if (x.isScalar())
        return Value::scalar(x.toComplex().imag(), p);
    auto r = createLike(x, ValueType::DOUBLE, p);
    for (size_t i = 0; i < x.numel(); ++i)
        r.doubleDataMut()[i] = x.complexData()[i].imag();
    return r;
}

Value conj(const Value &x, std::pmr::memory_resource *mr)
{
    std::pmr::memory_resource *p = mr;
    if (!x.isComplex())
        return x;
    return unaryComplex(x, [](const Complex &c) { return std::conj(c); }, p);
}

Value complex(const Value &re, std::pmr::memory_resource *mr)
{
    std::pmr::memory_resource *p = mr;
    if (re.isComplex())
        return re;
    const bool outSingle = (re.type() == ValueType::SINGLE);
    if (re.isScalar() && !outSingle)
        return Value::complexScalar(re.toScalar(), 0.0, p);
    auto r = createLike(re, ValueType::COMPLEX, p);
    if (outSingle) r.setSingleComplex(true);
    Complex *dst = r.complexDataMut();
    for (size_t i = 0; i < re.numel(); ++i) {
        double v = re.elemAsDouble(i);
        if (outSingle) v = static_cast<double>(static_cast<float>(v));
        dst[i] = Complex(v, 0.0);
    }
    return r;
}

Value complex(const Value &re, const Value &im, std::pmr::memory_resource *mr)
{
    std::pmr::memory_resource *p = mr;
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
    Complex *dst = r.complexDataMut();
    const size_t n = shape.numel();
    const double reScalar = re.isScalar() ? re.toScalar() : 0.0;
    const double imScalar = im.isScalar() ? im.toScalar() : 0.0;
    for (size_t i = 0; i < n; ++i) {
        double r0 = re.isScalar() ? reScalar : re.elemAsDouble(i);
        double i0 = im.isScalar() ? imScalar : im.elemAsDouble(i);
        if (outSingle) {
            r0 = static_cast<double>(static_cast<float>(r0));
            i0 = static_cast<double>(static_cast<float>(i0));
        }
        dst[i] = Complex(r0, i0);
    }
    return r;
}

Value angle(const Value &x, std::pmr::memory_resource *mr)
{
    std::pmr::memory_resource *p = mr;
    if (x.isComplex()) {
        if (x.isSingleComplex()) {
            auto r = createLike(x, ValueType::SINGLE, p);
            float *dst = r.singleDataMut();
            for (size_t i = 0; i < x.numel(); ++i)
                dst[i] = static_cast<float>(std::arg(x.complexData()[i]));
            return r;
        }
        if (x.isScalar())
            return Value::scalar(std::arg(x.toComplex()), p);
        auto r = createLike(x, ValueType::DOUBLE, p);
        for (size_t i = 0; i < x.numel(); ++i)
            r.doubleDataMut()[i] = std::arg(x.complexData()[i]);
        return r;
    }
    if (x.type() == ValueType::SINGLE) {
        auto r = createLike(x, ValueType::SINGLE, p);
        const float *src = x.singleData();
        float *dst = r.singleDataMut();
        for (size_t i = 0; i < x.numel(); ++i)
            dst[i] = std::atan2(0.0f, src[i]);
        return r;
    }
    return unaryDouble(x, [](double v) { return std::atan2(0.0, v); }, p);
}

Value isreal(const Value &x, std::pmr::memory_resource *mr)
{
    return Value::scalar(!x.isComplex() ? 1.0 : 0.0, mr);
}

Value unwrap(const Value &p, double tol, int dim, std::pmr::memory_resource *mr)
{
    if (p.isEmpty())
        return p;

    Value out = p;
    if (out.type() != ValueType::DOUBLE)
        out = toDoubleValue(p, mr);
    else
        out = Value(p); // copy

    double *data = out.doubleDataMut();
    const size_t n = out.numel();
    if (n <= 1) return out;

    constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
    double dp_corr = 0.0;

    for (size_t i = 1; i < n; ++i) {
        double d = data[i] - data[i - 1];
        if (d > tol) {
            double jumps = std::floor((d + tol) / kTwoPi);
            dp_corr -= jumps * kTwoPi;
        } else if (d < -tol) {
            double jumps = std::floor((-d + tol) / kTwoPi);
            dp_corr += jumps * kTwoPi;
        }
        data[i] += dp_corr;
    }

    return out;
}

} // namespace numkit::builtin
