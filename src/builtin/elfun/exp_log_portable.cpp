// toolboxes/builtin/src/math/exp_log/exp_log_portable.cpp
//
// Reference scalar implementations of exp / log. Compiled when
// NUMKIT_WITH_SIMD=OFF; the Highway-dispatched variant lives in
// exp_log_highway.cpp and matches this file bit-for-bit on complex
// inputs (SIMD only helps the real-vector fast path).

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/builtin/elfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>


namespace numkit::builtin {

namespace {

// Shared scaffolding: complex / scalar shortcuts, then either reuse
// `*hint` (uniquely-owned heap double of matching shape) or allocate
// a fresh result and apply ScalarOp element-wise. See the docblock
// on abs() in math/elementary/misc.hpp for the full hint contract.
template <typename ScalarOp, typename ComplexOp>
Value unaryRealDoubleHint(const Value &x, Value *hint, ScalarOp scalarOp, ComplexOp complexOp, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, complexOp, mr);
    if (x.isScalar())
        return Value::scalar(scalarOp(x.toScalar()), mr);

    if (hint && hint->isHeapDouble() && hint->heapRefCount() == 1
        && hint->dims() == x.dims()) {
        Value r = std::move(*hint);
        const keel::f64 *in  = x.doubleData();
        keel::f64       *out = r.doubleDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            out[i] = scalarOp(in[i]);
        return r;
    }
    return unaryDouble(x, scalarOp, mr);
}

} // namespace

Value exp(const Value &x, Value *hint, keel::Allocator *mr)
{
    return unaryRealDoubleHint(x, hint, [](keel::f64 v) { return keel::exp(v); }, [](const keel::c64 &c) { return keel::exp(c); }, mr);
}

static bool anyNegative_p(const Value &x);  // defined below (shared with sqrt)

Value log(const Value &x, Value *hint, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::log(c); }, mr);
    if (x.isScalar() && x.toScalar() < 0)
        return Value::complexScalar(keel::log(keel::c64(x.toScalar(), 0.0)), mr);
    // Real array with any negative element -> promote the whole array (MATLAB).
    if (anyNegative_p(x)) {
        Value cx = x; cx.promoteToComplex(mr);
        return unaryComplex(cx, [](const keel::c64 &c) { return keel::log(c); }, mr);
    }
    return unaryRealDoubleHint(x, hint, [](keel::f64 v) { return keel::log(v); }, [](const keel::c64 &c) { return keel::log(c); }, mr);
}

// expm1 / log1p / log2 — reference scalar path (the SIMD variants live in
// exp_log_highway.cpp).
Value expm1(const Value &x, keel::Allocator *mr)
{
    // MATLAB: expm1(z) = exp(z) - 1 on complex.
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::exp(c) - 1.0; }, mr);
    return unaryDouble(x, [](keel::f64 v) { return keel::expm1(v); }, mr);
}

// log1p of a real x as a (possibly) complex value, accurate per element:
//   x >= -1 : log1p(x)  (real; -Inf at x == -1)
//   x <  -1 : 1+x < 0 → log|1+x| + i·π   (MATLAB principal branch)
static inline keel::c64 log1pRealToComplex_p(keel::f64 x)
{
    constexpr keel::f64 kPi = keel::kPi;
    if (x >= -1.0) return keel::c64(keel::log1p(x), 0.0);
    return keel::c64(keel::log(-(1.0 + x)), kPi);
}
static inline bool anyLessThanMinusOne_p(const Value &x)
{
    const keel::usize n = x.numel();
    for (keel::usize i = 0; i < n; ++i)
        if (x.elemAsDouble(i) < -1.0) return true;
    return false;
}

// log1p(x) = log(1+x). For x < -1 the argument 1+x is negative → MATLAB returns
// complex (log1p(-2) = i·π); any element < -1 promotes the whole real array.
// Complex input uses log(1+z). The real (x >= -1) path keeps accurate log1p.
Value log1p(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) {
            return keel::log(keel::c64(1.0, 0.0) + c); }, mr);
    if (x.isScalar()) {
        const keel::f64 v = x.toScalar();
        if (v < -1.0) return Value::complexScalar(log1pRealToComplex_p(v), mr);
        return Value::scalar(keel::log1p(v), mr);
    }
    if (anyLessThanMinusOne_p(x)) {
        Value r = createLike(x, ValueType::COMPLEX, mr);
        keel::c64 *out = r.complexDataMut();
        const keel::usize n = x.numel();
        for (keel::usize i = 0; i < n; ++i)
            out[i] = log1pRealToComplex_p(x.elemAsDouble(i));
        return r;
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::log1p(v); }, mr);
}

Value log2(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::log(c) / keel::log(2.0); }, mr);
    if (x.isScalar() && x.toScalar() < 0.0)
        return Value::complexScalar(keel::log(keel::c64(x.toScalar(), 0.0)) / keel::log(2.0), mr);
    if (anyNegative_p(x)) {
        Value cx = x; cx.promoteToComplex(mr);
        return unaryComplex(cx, [](const keel::c64 &c) { return keel::log(c) / keel::log(2.0); }, mr);
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::log2(v); }, mr);
}

Value log10(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::log10(c); }, mr);
    if (x.isScalar() && x.toScalar() < 0.0)
        return Value::complexScalar(keel::log10(keel::c64(x.toScalar(), 0.0)), mr);
    if (anyNegative_p(x)) {
        Value cx = x; cx.promoteToComplex(mr);
        return unaryComplex(cx, [](const keel::c64 &c) { return keel::log10(c); }, mr);
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::log10(v); }, mr);
}

Value reallog(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](keel::f64 v) {
        if (v < 0.0)
            throw numkit::Error("reallog produced complex result — use log(...) instead");
        return keel::log(v);
    }, mr);
}

// True if any real element is < 0 (NaN compares false → stays real). If ANY
// element is negative the whole array is promoted to complex (MATLAB).
static bool anyNegative_p(const Value &x)
{
    const keel::usize n = x.numel();
    for (keel::usize i = 0; i < n; ++i)
        if (x.elemAsDouble(i) < 0.0) return true;
    return false;
}

Value sqrt(const Value &x, keel::Allocator *mr)
{
    if (x.isComplex())
        return unaryComplex(x, [](const keel::c64 &c) { return keel::sqrt(c); }, mr);
    if (anyNegative_p(x)) {
        Value cx = x; cx.promoteToComplex(mr);
        return unaryComplex(cx, [](const keel::c64 &c) { return keel::sqrt(c); }, mr);
    }
    return unaryDouble(x, [](keel::f64 v) { return keel::sqrt(v); }, mr);
}

Value realsqrt(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](keel::f64 v) {
        if (v < 0.0)
            throw numkit::Error("realsqrt produced complex result — use sqrt(...) instead");
        return keel::sqrt(v);
    }, mr);
}

Value pow2(const Value &y, keel::Allocator *mr)
{
    return unaryDouble(y, [](keel::f64 v) { return keel::exp2(v); }, mr);
}

} // namespace numkit::builtin
