// toolboxes/builtin/src/math/elementary/backends/abs_portable.cpp
//
// Reference scalar implementation of abs(). Compiled when
// NUMKIT_WITH_SIMD=OFF (see toolboxes/builtin/src/CMakeLists.txt). The
// Highway-dispatched variant lives in abs_highway.cpp — both share
// exactly this behaviour for small / complex / scalar inputs; the
// SIMD backend only diverges on the real-vector fast path.

#include <numkit/builtin/elfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>

#include <cmath>
#include <complex>

namespace numkit::builtin {

Value abs(const Value &x, Value *hint, std::pmr::memory_resource *mr)
{
    // Integer types keep their class and saturate (abs(int8(-128))=127 int8).
    if (isIntegerType(x.type()))
        return absIntegerSaturate(x, mr);
    if (x.isComplex()) {
        if (x.isScalar())
            return Value::scalar(std::abs(x.toComplex()), mr);
        auto r = createLike(x, ValueType::DOUBLE, mr);
        for (size_t i = 0; i < x.numel(); ++i)
            r.doubleDataMut()[i] = std::abs(x.complexData()[i]);
        return r;
    }
    // Single stays single; logical / char widen to double — see
    // abs_highway.cpp (bugs: runtime/abs-single-matrix.md).
    if (x.type() == ValueType::SINGLE) {
        Value r = createLike(x, ValueType::SINGLE, mr);
        const float *in = x.singleData();
        float *out = r.singleDataMut();
        for (size_t i = 0; i < x.numel(); ++i)
            out[i] = std::fabs(in[i]);
        return r;
    }
    if (x.type() == ValueType::LOGICAL || x.type() == ValueType::CHAR) {
        Value r = createLike(x, ValueType::DOUBLE, mr);
        double *out = r.doubleDataMut();
        if (x.type() == ValueType::LOGICAL) {
            const auto *in = x.logicalData();
            for (size_t i = 0; i < x.numel(); ++i)
                out[i] = in[i] ? 1.0 : 0.0;
        } else {
            const auto *in = x.charData();
            for (size_t i = 0; i < x.numel(); ++i)
                out[i] = static_cast<double>(
                    static_cast<unsigned char>(in[i]));
        }
        return r;
    }
    if (x.isScalar())
        return Value::scalar(std::fabs(x.toScalar()), mr);
    // Output-reuse fast path: caller-provided hint is a heap double
    // of matching shape with unique ownership — write straight into
    // its buffer instead of allocating a fresh one.
    if (hint && hint->isHeapDouble() && hint->heapRefCount() == 1
        && hint->dims() == x.dims()) {
        Value r = std::move(*hint);
        const double *in  = x.doubleData();
        double       *out = r.doubleDataMut();
        for (size_t i = 0; i < x.numel(); ++i)
            out[i] = std::fabs(in[i]);
        return r;
    }
    return unaryDouble(x, [](double v) { return std::abs(v); }, mr);
}

} // namespace numkit::builtin
