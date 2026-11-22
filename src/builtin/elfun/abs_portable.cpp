// toolboxes/builtin/src/math/elementary/backends/abs_portable.cpp
//
// Reference scalar implementation of abs(). Compiled when
// NUMKIT_WITH_SIMD=OFF (see toolboxes/builtin/src/CMakeLists.txt). The
// Highway-dispatched variant lives in abs_highway.cpp — both share
// exactly this behaviour for small / complex / scalar inputs; the
// SIMD backend only diverges on the real-vector fast path.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <numkit/builtin/elfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>


namespace numkit::builtin {

Value abs(const Value &x, Value *hint, keel::Allocator *mr)
{
    // Integer types keep their class and saturate (abs(int8(-128))=127 int8).
    if (isIntegerType(x.type()))
        return absIntegerSaturate(x, mr);
    if (x.isComplex()) {
        if (x.isSingleComplex()) {
            auto r = createLike(x, ValueType::SINGLE, mr);
            keel::f32 *dst = r.singleDataMut();
            for (keel::usize i = 0; i < x.numel(); ++i)
                dst[i] = keel::as<keel::f32>(keel::abs(x.complexData()[i]));
            return r;
        }
        if (x.isScalar())
            return Value::scalar(keel::abs(x.toComplex()), mr);
        auto r = createLike(x, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < x.numel(); ++i)
            r.doubleDataMut()[i] = keel::abs(x.complexData()[i]);
        return r;
    }
    // Single stays single; logical / char widen to double — see
    // abs_highway.cpp (bugs: runtime/abs-single-matrix.md).
    if (x.type() == ValueType::SINGLE) {
        Value r = createLike(x, ValueType::SINGLE, mr);
        const keel::f32 *in = x.singleData();
        keel::f32 *out = r.singleDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            out[i] = keel::abs(in[i]);
        return r;
    }
    if (x.type() == ValueType::LOGICAL || x.type() == ValueType::CHAR) {
        Value r = createLike(x, ValueType::DOUBLE, mr);
        keel::f64 *out = r.doubleDataMut();
        if (x.type() == ValueType::LOGICAL) {
            const auto *in = x.logicalData();
            for (keel::usize i = 0; i < x.numel(); ++i)
                out[i] = in[i] ? 1.0 : 0.0;
        } else {
            const auto *in = x.charData();
            for (keel::usize i = 0; i < x.numel(); ++i)
                out[i] = keel::as<keel::f64>(
                    keel::as<keel::u8>(in[i]));
        }
        return r;
    }
    if (x.isScalar())
        return Value::scalar(keel::abs(x.toScalar()), mr);
    // Output-reuse fast path: caller-provided hint is a heap double
    // of matching shape with unique ownership — write straight into
    // its buffer instead of allocating a fresh one.
    if (hint && hint->isHeapDouble() && hint->heapRefCount() == 1
        && hint->dims() == x.dims()) {
        Value r = std::move(*hint);
        const keel::f64 *in  = x.doubleData();
        keel::f64       *out = r.doubleDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            out[i] = keel::abs(in[i]);
        return r;
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::abs(v); }, mr);
}

} // namespace numkit::builtin
