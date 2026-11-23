// toolboxes/builtin/src/math/elementary/reductions.cpp
//
// Reductions (sum / prod / mean / max / min — single-return forms,
// NaN-aware variants, complex variants) plus the typed-output dispatcher
// used by their adapters. Also hosts linspace / logspace.
//
// trigonometry / exponents / rounding / misc / special live in sibling
// files under math/elementary/. Random generators (rand/randn) live in
// math/random/rng.cpp.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/tuple.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/builtin/elmat.hpp>       // reshape (for 'all')
#include <numkit/builtin/elfun.hpp>      // exp / log adapters
#include <numkit/builtin/datafun.hpp>
#include <numkit/builtin/elfun.hpp>       // abs adapter
#include <numkit/builtin/elfun.hpp>   // sin / cos adapters

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/ops/reductions.hpp>
#include "var_reduction.hpp"  // for sumScan + addInto
#include "../elfun/_unary_hint.hpp"  // 3-arg sin/cos/exp/log/abs hint overloads


#include "reductions_detail.hpp"

namespace numkit::builtin {

// ════════════════════════════════════════════════════════════════════════
// Reductions (single-return) — sum / prod / mean
// ════════════════════════════════════════════════════════════════════════

Value sum(const Value &x, keel::Allocator *mr)
{
    return reduce(x, [](keel::f64 a, keel::f64 b) { return a + b; }, 0.0, mr);
}

Value sum(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    if (dim <= 0) return sum(x, mr);
    const keel::i32 d = numkit::ops::resolveDim(x, dim, "sum");

    // Phase P6 followup: 2D dim=2 column-pass row reduction. The
    // applyAlongDim path gathers each row into a scratch buffer with a
    // strided per-element copy (R reads at stride R per row, R rows ->
    // O(R^2) accesses with bad cache locality), then scalar-sums each
    // slice. Column-pass reads each input column contiguously and
    // accumulates into a row-totals vector with SIMD addInto. Net cost
    // floors at 1 read of M + 1 write of totals = roughly memory
    // bandwidth.
    if (d == 2 && x.type() == ValueType::DOUBLE && x.dims().ndim() == 2
        && !x.isScalar() && !x.dims().isVector()) {
        const keel::usize R = x.dims().rows(), C = x.dims().cols();
        auto r = Value::matrix(R, 1, ValueType::DOUBLE, mr);
        keel::f64 *totals = r.doubleDataMut();
        keel::fill(totals, totals + R, 0.0);
        const keel::f64 *src = x.doubleData();
        for (keel::usize c = 0; c < C; ++c)
            addInto(totals, src + c * R, R);
        return r;
    }

    // Generic dim path (1D, 3D, dim=1, dim=3) — slice through scratch.
    return numkit::ops::applyAlongDim(x, d,
        [](keel::usize, keel::f64 *slice, keel::usize n) {
            keel::f64 acc = 0.0;
            for (keel::usize i = 0; i < n; ++i) acc += slice[i];
            return acc;
        }, mr);
}

Value prod(const Value &x, keel::Allocator *mr)
{
    return reduce(x, [](keel::f64 a, keel::f64 b) { return a * b; }, 1.0, mr);
}

Value prod(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    if (dim <= 0) return prod(x, mr);
    const keel::i32 d = numkit::ops::resolveDim(x, dim, "prod");
    return numkit::ops::applyAlongDim(x, d,
        [](keel::usize, keel::f64 *slice, keel::usize n) {
            keel::f64 acc = 1.0;
            for (keel::usize i = 0; i < n; ++i) acc *= slice[i];
            return acc;
        }, mr);
}

Value mean(const Value &x, keel::Allocator *mr)
{
    return reduce(x, [](keel::f64 a, keel::f64 b) { return a + b; }, 0.0, mr, /*meanMode=*/true);
}

Value mean(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    if (dim <= 0) return mean(x, mr);
    const keel::i32 d = numkit::ops::resolveDim(x, dim, "mean");
    return numkit::ops::applyAlongDim(x, d,
        [](keel::usize, keel::f64 *slice, keel::usize n) {
            keel::f64 acc = 0.0;
            for (keel::usize i = 0; i < n; ++i) acc += slice[i];
            return acc / keel::as<keel::f64>(n);
        }, mr);
}

// ── max/min with index ───────────────────────────────────────────────
//
// Per MATLAB semantics, min/max preserve the input element type
// (default 'native' mode). The value array is T-typed; the index
// array is always DOUBLE. COMPLEX inputs are rejected (no order
// defined on complex). Dispatch over ValueType picks the right T
// instantiation for DOUBLE, SINGLE, INT8..INT64, UINT8..UINT64,
// LOGICAL (storage = uint8) and CHAR (storage = char).


keel::Tuple<Value, Value> max(const Value &x, keel::Allocator *mr)
{
    return dispatchMinMaxAll<true>(x, [](auto v, auto best) { return v > best; }, mr, "max");
}

keel::Tuple<Value, Value> min(const Value &x, keel::Allocator *mr)
{
    return dispatchMinMaxAll<false>(x, [](auto v, auto best) { return v < best; }, mr, "min");
}

keel::Tuple<Value, Value> max(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    if (dim <= 0) return max(x, mr);
    const keel::i32 d = numkit::ops::resolveDim(x, dim, "max");
    return dispatchMinMaxAlongDim<true>(x, d, [](auto v, auto best) { return v > best; }, mr, "max");
}

keel::Tuple<Value, Value> maxOmitNan(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    if (dim <= 0)
        return dispatchMinMaxNanAll<true>(x, [](auto v, auto best) { return v > best; }, mr, "max");
    const keel::i32 d = numkit::ops::resolveDim(x, dim, "max");
    return dispatchMinMaxNanAlongDim<true>(x, d, [](auto v, auto best) { return v > best; }, mr, "max");
}

keel::Tuple<Value, Value> minOmitNan(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    if (dim <= 0)
        return dispatchMinMaxNanAll<false>(x, [](auto v, auto best) { return v < best; }, mr, "min");
    const keel::i32 d = numkit::ops::resolveDim(x, dim, "min");
    return dispatchMinMaxNanAlongDim<false>(x, d, [](auto v, auto best) { return v < best; }, mr, "min");
}

keel::Tuple<Value, Value> min(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    if (dim <= 0) return min(x, mr);
    const keel::i32 d = numkit::ops::resolveDim(x, dim, "min");
    return dispatchMinMaxAlongDim<false>(x, d, [](auto v, auto best) { return v < best; }, mr, "min");
}

// Elementwise complex binary min/max, shared by max/min and their omitnan
// variants: compare by |z|, ties by angle(z); a NaN-component operand is omitted
// (omitNan) or propagates (see complexMinMaxPick). Diverted FIRST in each binary
// fn — a complex operand must be intercepted before elementwiseDouble (which
// reads doubleData() and throws on complex). bugs/math/maxmin-complex.md.
template <bool IsMax>
static Value complexBinaryMinMax(const Value &a, const Value &b, bool omitNan,
                                 keel::Allocator *p)
{
    return elementwiseComplex(a, b,
        [omitNan](keel::c64 x, keel::c64 y) { return complexMinMaxPick<IsMax>(x, y, omitNan); }, p);
}

Value max(const Value &a, const Value &b, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (a.isComplex() || b.isComplex()) return complexBinaryMinMax<true>(a, b, /*omitNan=*/false, p);
    // Integer / single binary form: result follows MATLAB type promotion
    // (integer wins over double; single wins over double; same-class
    // integers stay; mixed-class integers throw).
    {
        auto r = dispatchIntegerBinaryOp(a, b,
            [](auto x, auto y) { return x > y ? x : y; }, p);
        if (!r.isUnset()) return r;
    }
    // fmax (not std::max) ignores NaN like MATLAB: max([NaN 6],[5 7]) = [5 7]
    // not [NaN 7]. std::max is order-dependent for NaN and returned NaN here.
    return elementwiseDouble(a, b, [](keel::f64 aa, keel::f64 bb) { return keel::fmax(aa, bb); }, p);
}

Value min(const Value &a, const Value &b, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (a.isComplex() || b.isComplex()) return complexBinaryMinMax<false>(a, b, /*omitNan=*/false, p);
    {
        auto r = dispatchIntegerBinaryOp(a, b,
            [](auto x, auto y) { return x < y ? x : y; }, p);
        if (!r.isUnset()) return r;
    }
    return elementwiseDouble(a, b, [](keel::f64 aa, keel::f64 bb) { return keel::fmin(aa, bb); }, p);
}

// Binary nan-aware variants. For floating types, NaN propagates as
// "missing": when one arg is NaN, take the other; both NaN → NaN.
// For integer types, NaN can't occur so omitnan is a no-op (same as
// the regular max/min).
Value maxOmitNanBinary(const Value &a, const Value &b, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (a.isComplex() || b.isComplex()) return complexBinaryMinMax<true>(a, b, /*omitNan=*/true, p);
    {
        auto r = dispatchIntegerBinaryOp(a, b,
            [](auto x, auto y) {
                using T = decltype(x);
                if constexpr (keel::IsFloatingPoint<T>) {
                    if (keel::is_nan(x)) return y;
                    if (keel::is_nan(y)) return x;
                }
                return x > y ? x : y;
            }, p);
        if (!r.isUnset()) return r;
    }
    return elementwiseDouble(a, b, [](keel::f64 aa, keel::f64 bb) {
        if (keel::is_nan(aa)) return bb;
        if (keel::is_nan(bb)) return aa;
        return keel::max(aa, bb);
    }, p);
}

Value minOmitNanBinary(const Value &a, const Value &b, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (a.isComplex() || b.isComplex()) return complexBinaryMinMax<false>(a, b, /*omitNan=*/true, p);
    {
        auto r = dispatchIntegerBinaryOp(a, b,
            [](auto x, auto y) {
                using T = decltype(x);
                if constexpr (keel::IsFloatingPoint<T>) {
                    if (keel::is_nan(x)) return y;
                    if (keel::is_nan(y)) return x;
                }
                return x < y ? x : y;
            }, p);
        if (!r.isUnset()) return r;
    }
    return elementwiseDouble(a, b, [](keel::f64 aa, keel::f64 bb) {
        if (keel::is_nan(aa)) return bb;
        if (keel::is_nan(bb)) return aa;
        return keel::min(aa, bb);
    }, p);
}

// ── Generators ───────────────────────────────────────────────────────
Value linspace(keel::f64 a, keel::f64 b, keel::usize n, keel::Allocator *mr)
{
    auto r = Value::matrix(1, n, ValueType::DOUBLE, mr);
    if (n == 0)
        return r;
    keel::f64 *out = r.doubleDataMut();
    if (n == 1) {
        out[0] = b;
        return r;
    }
    const keel::f64 n1 = keel::as<keel::f64>(n - 1);
    if (a == -b) {
        const keel::f64 step = b / n1;
        for (keel::usize i = 0; i < n; ++i)
            out[i] = (2.0 * keel::as<keel::f64>(i) - n1) * step;
        out[0] = a;
        out[n - 1] = b;
        if (n % 2 == 1) out[n / 2] = 0.0;
        return r;
    }
    for (keel::usize i = 0; i < n; ++i)
        out[i] = a + (b - a) * keel::as<keel::f64>(i) / n1;
    out[0] = a;
    out[n - 1] = b;
    return r;
}

Value logspace(keel::f64 a, keel::f64 b, keel::usize n, keel::Allocator *mr)
{
    auto r = Value::matrix(1, n, ValueType::DOUBLE, mr);
    if (n == 0)
        return r;
    if (n == 1) {
        r.doubleDataMut()[0] = keel::pow(10.0, b);
        return r;
    }
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 exponent = a + (b - a) * keel::as<keel::f64>(i) / keel::as<keel::f64>(n - 1);
        r.doubleDataMut()[i] = keel::pow(10.0, exponent);
    }
    return r;
}

} // namespace numkit::builtin
