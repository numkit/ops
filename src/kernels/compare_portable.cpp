// toolboxes/builtin/src/lang/operators/backends/compare_portable.cpp
//
// Scalar reference for the comparison fast path. Compiled when
// NUMKIT_WITH_SIMD=OFF. Same dispatch contract as the Highway variant —
// returns LOGICAL output for pure DOUBLE × DOUBLE inputs (or DOUBLE
// scalar broadcast), otherwise returns an unset Value so the generic
// `compareImpl` in binary_ops.cpp handles it.

#include <keel/core.h>
#include <numkit/ops/compare.hpp>
#include <numkit/ops/value_factory.hpp>

#include <numkit/value/value.hpp>


#include <cstddef>
#include <cstdint>

namespace numkit::ops {
namespace {

bool isDoubleish(const Value &v)
{
    return v.type() == ValueType::DOUBLE && !v.isComplex();
}

bool compatibleShape(const Value &a, const Value &b)
{
    if (a.isScalar() || b.isScalar()) return true;
    return a.dims() == b.dims();
}

template <typename Op>
Value dispatchFast(const Value &a, const Value &b, Op op)
{
    if (!isDoubleish(a) || !isDoubleish(b)) return Value{};
    if (!compatibleShape(a, b))             return Value{};
    if (a.isScalar() && b.isScalar())       return Value{};

    if (a.isScalar()) {
        const keel::f64 s = a.toScalar();
        const keel::f64 *pb = b.doubleData();
        auto r = createLike(b, ValueType::LOGICAL, nullptr);
        keel::u8 *out = r.logicalDataMut();
        const keel::usize n = b.numel();
        for (keel::usize i = 0; i < n; ++i)
            out[i] = op(s, pb[i]) ? 1 : 0;
        return r;
    }
    if (b.isScalar()) {
        const keel::f64 s = b.toScalar();
        const keel::f64 *pa = a.doubleData();
        auto r = createLike(a, ValueType::LOGICAL, nullptr);
        keel::u8 *out = r.logicalDataMut();
        const keel::usize n = a.numel();
        for (keel::usize i = 0; i < n; ++i)
            out[i] = op(pa[i], s) ? 1 : 0;
        return r;
    }
    const keel::f64 *pa = a.doubleData();
    const keel::f64 *pb = b.doubleData();
    auto r = createLike(a, ValueType::LOGICAL, nullptr);
    keel::u8 *out = r.logicalDataMut();
    const keel::usize n = a.numel();
    for (keel::usize i = 0; i < n; ++i)
        out[i] = op(pa[i], pb[i]) ? 1 : 0;
    return r;
}

} // namespace

Value eqFast(const Value &a, const Value &b) { return dispatchFast(a, b, [](keel::f64 x, keel::f64 y){ return x == y; }); }
Value neFast(const Value &a, const Value &b) { return dispatchFast(a, b, [](keel::f64 x, keel::f64 y){ return x != y; }); }
Value ltFast(const Value &a, const Value &b) { return dispatchFast(a, b, [](keel::f64 x, keel::f64 y){ return x <  y; }); }
Value gtFast(const Value &a, const Value &b) { return dispatchFast(a, b, [](keel::f64 x, keel::f64 y){ return x >  y; }); }
Value leFast(const Value &a, const Value &b) { return dispatchFast(a, b, [](keel::f64 x, keel::f64 y){ return x <= y; }); }
Value geFast(const Value &a, const Value &b) { return dispatchFast(a, b, [](keel::f64 x, keel::f64 y){ return x >= y; }); }

} // namespace numkit::ops
