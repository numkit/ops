// toolboxes/signal/src/transforms/transform_helpers.cpp
//
// nextpow2 / fftshift / ifftshift. Split from library.cpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/transforms/transform_helpers.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include "../dsp_helpers.hpp"  // Complex typedef
#include <numkit/ops/helpers.hpp>         // createLike

#include <utility>

namespace numkit::signal {

namespace {

// Cyclic shift along a single dim (1=rows, 2=cols, 3=pages). Element-major
// flat layout is column-major; we walk dst by destination index and pull
// from src using a per-axis offset.
//
// `mode = +1` -> fftshift (shift = ceil(extent/2))
// `mode = -1` -> ifftshift (shift = floor(extent/2))
template <typename T>
void shiftAlongDim(const T *src, T *dst, keel::usize R, keel::usize C, keel::usize P,
                   keel::i32 dim, keel::i32 mode)
{
    auto pickShift = [mode](keel::usize extent) -> keel::usize {
        if (extent <= 1) return 0;
        return (mode > 0) ? (extent + 1) / 2 : extent / 2;  // ceil vs floor
    };
    const keel::usize sR = (dim == 1) ? pickShift(R) : 0;
    const keel::usize sC = (dim == 2) ? pickShift(C) : 0;
    const keel::usize sP = (dim == 3) ? pickShift(P) : 0;
    for (keel::usize p = 0; p < P; ++p) {
        const keel::usize ps = (sP ? (p + sP) % P : p);
        for (keel::usize c = 0; c < C; ++c) {
            const keel::usize cs = (sC ? (c + sC) % C : c);
            for (keel::usize r = 0; r < R; ++r) {
                const keel::usize rs = (sR ? (r + sR) % R : r);
                dst[r + R * (c + C * p)]   = src[rs + R * (cs + C * ps)];
            }
        }
    }
}

// Apply fftshift/ifftshift across all non-singleton dims (mode=+1/-1).
Value cyclicShiftAll(const Value &x, keel::i32 mode, keel::Allocator *mr)
{
    const keel::usize R = x.dims().rows();
    const keel::usize C = x.dims().cols();
    const keel::usize P = x.dims().is3D() ? x.dims().pages() : 1;
    const bool cplx = x.isComplex();
    auto r = createLike(x, cplx ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    if (x.numel() == 0) return r;
    // Determine which dims to shift: every non-singleton dim. (For a row
    // vector R=1, only cols shift; for a column vector C=1, only rows.)
    const bool dR = R > 1, dC = C > 1, dP = P > 1;
    if (cplx) {
        const keel::c64 *src = x.complexData();
        keel::c64 *dst = r.complexDataMut();
        // Shift one dim at a time using two scratch buffers.
        keel::Array<keel::c64> buf(R * C * P, mr);
        keel::mem_copy(buf.data(), src, sizeof(keel::c64) * R * C * P);
        keel::c64 *cur = buf.data(), *nxt = dst;
        if (dR) { shiftAlongDim<keel::c64>(cur, nxt, R, C, P, 1, mode); std::swap(cur, nxt); }
        if (dC) { shiftAlongDim<keel::c64>(cur, nxt, R, C, P, 2, mode); std::swap(cur, nxt); }
        if (dP) { shiftAlongDim<keel::c64>(cur, nxt, R, C, P, 3, mode); std::swap(cur, nxt); }
        if (cur != dst) keel::mem_copy(dst, cur, sizeof(keel::c64) * R * C * P);
    } else {
        const keel::f64 *src = x.doubleData();
        keel::f64 *dst = r.doubleDataMut();
        keel::Array<keel::f64> buf(R * C * P, mr);
        keel::mem_copy(buf.data(), src, sizeof(keel::f64) * R * C * P);
        keel::f64 *cur = buf.data(), *nxt = dst;
        if (dR) { shiftAlongDim<keel::f64>(cur, nxt, R, C, P, 1, mode); std::swap(cur, nxt); }
        if (dC) { shiftAlongDim<keel::f64>(cur, nxt, R, C, P, 2, mode); std::swap(cur, nxt); }
        if (dP) { shiftAlongDim<keel::f64>(cur, nxt, R, C, P, 3, mode); std::swap(cur, nxt); }
        if (cur != dst) keel::mem_copy(dst, cur, sizeof(keel::f64) * R * C * P);
    }
    return r;
}

// Single-dim form (when user passes explicit dim argument).
Value cyclicShiftOneDim(const Value &x, keel::i32 dim, keel::i32 mode,
                        keel::Allocator *mr)
{
    const keel::usize R = x.dims().rows();
    const keel::usize C = x.dims().cols();
    const keel::usize P = x.dims().is3D() ? x.dims().pages() : 1;
    const bool cplx = x.isComplex();
    auto r = createLike(x, cplx ? ValueType::COMPLEX : ValueType::DOUBLE, mr);
    if (x.numel() == 0) return r;
    if (cplx)
        shiftAlongDim<keel::c64>(x.complexData(), r.complexDataMut(), R, C, P, dim, mode);
    else
        shiftAlongDim<keel::f64>(x.doubleData(), r.doubleDataMut(), R, C, P, dim, mode);
    return r;
}

} // anonymous namespace

// Scalar element rule: smallest integer p such that 2^p >= |x|.
// MATLAB R2025b conventions:
//   |x| = 0 -> 0
//   NaN    -> NaN
//   ±Inf   -> +Inf
//   else   -> ceil(log2(|x|))
static keel::f64 nextpow2Element(keel::f64 v_abs)
{
    if (keel::is_nan(v_abs)) return keel::Limits<keel::f64>::quiet_nan();
    if (keel::is_inf(v_abs)) return keel::Limits<keel::f64>::infinity();
    if (v_abs == 0.0) return 0.0;
    return keel::ceil(keel::log2(v_abs));
}

Value nextpow2(keel::f64 n, keel::Allocator *mr)
{
    return Value::scalar(nextpow2Element(keel::abs(n)), mr);
}

// Vectorized form: applies nextpow2 elementwise. For complex input we
// use |z| = sqrt(re² + im²) per MATLAB's documented behavior.
Value nextpow2(const Value &x, keel::Allocator *mr)
{
    if (x.isEmpty()) return x;
    if (x.isComplex()) {
        if (x.isScalar()) {
            const keel::c64 z = x.complexData()[0];
            return Value::scalar(nextpow2Element(keel::abs(z)), mr);
        }
        auto out = createLike(x, ValueType::DOUBLE, mr);
        keel::f64 *dst = out.doubleDataMut();
        const keel::c64 *src = x.complexData();
        const keel::usize n = x.numel();
        for (keel::usize i = 0; i < n; ++i) dst[i] = nextpow2Element(keel::abs(src[i]));
        return out;
    }
    if (x.isScalar()) return Value::scalar(nextpow2Element(keel::abs(x.toScalar())), mr);
    auto out = createLike(x, ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();
    const keel::usize n = x.numel();
    for (keel::usize i = 0; i < n; ++i)
        dst[i] = nextpow2Element(keel::abs(x.elemAsDouble(i)));
    return out;
}

Value fftshift(const Value &x, keel::Allocator *mr)
{
    return cyclicShiftAll(x, +1, mr);
}

Value ifftshift(const Value &x, keel::Allocator *mr)
{
    return cyclicShiftAll(x, -1, mr);
}

Value fftshift(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    return cyclicShiftOneDim(x, dim, +1, mr);
}

Value ifftshift(const Value &x, keel::i32 dim, keel::Allocator *mr)
{
    return cyclicShiftOneDim(x, dim, -1, mr);
}

} // namespace numkit::signal
