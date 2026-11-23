#pragma once

// NOTE: helpers.hpp is engine-free — it uses only the Value substrate + ops
// factories, never Engine/CallContext. Keeping it off <nk/core/engine.hpp>
// is what lets toolbox compute files include it without dragging in the engine.
#include <keel/core.h>
#include <keel/algo.h>
#include <keel/math.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>    // numkit::argcheck (size-input validation)
#include <numkit/ops/value_factory.hpp>   // DimsArg/createMatrix/createForDims/createLike (now in numkit::ops)
#include <numkit/value/shape_ops.hpp>


namespace numkit {

// ============================================================
// Helper: promote pair to complex if needed
// ============================================================
inline keel::Pair<Value, Value> promoteToComplex(const Value &a, const Value &b, keel::Allocator *mr = nullptr)
{
    Value ca = a, cb = b;
    if (a.isComplex() && !b.isComplex())
        cb.promoteToComplex(mr);
    else if (!a.isComplex() && b.isComplex())
        ca.promoteToComplex(mr);
    return {ca, cb};
}

// ============================================================
// Implicit expansion (broadcasting) check
// Returns true if dims are compatible, sets outR/outC to result size.
// Rule: each dim must match or one of them must be 1.
// ============================================================
inline bool broadcastDims(keel::usize ar, keel::usize ac, keel::usize br, keel::usize bc,
                          keel::usize &outR, keel::usize &outC)
{
    if (ar != br && ar != 1 && br != 1) return false;
    if (ac != bc && ac != 1 && bc != 1) return false;
    outR = keel::max(ar, br);
    outC = keel::max(ac, bc);
    return true;
}

// 3D variant — each axis must match or be 1. A 2D operand is treated as
// having pages = 1 (caller passes ap = 1 for 2D inputs).
inline bool broadcastDims3D(keel::usize ar, keel::usize ac, keel::usize ap,
                            keel::usize br, keel::usize bc, keel::usize bp,
                            keel::usize &outR, keel::usize &outC, keel::usize &outP)
{
    if (ar != br && ar != 1 && br != 1) return false;
    if (ac != bc && ac != 1 && bc != 1) return false;
    if (ap != bp && ap != 1 && bp != 1) return false;
    outR = keel::max(ar, br);
    outC = keel::max(ac, bc);
    outP = keel::max(ap, bp);
    return true;
}

// Linearise (r, c, p) into a column-major + page-stride buffer offset,
// applying broadcast (an axis of 1 collapses its index to 0). Used by
// the 3D broadcast loops in elementwise{Double,Complex} and compareImpl.
inline keel::usize broadcastOffset3D(keel::usize r, keel::usize c, keel::usize p,
                                keel::usize aR, keel::usize aC, keel::usize aP)
{
    const keel::usize rr = (aR == 1) ? 0 : r;
    const keel::usize cc = (aC == 1) ? 0 : c;
    const keel::usize pp = (aP == 1) ? 0 : p;
    (void)aP; // strides are R*C per page regardless
    return pp * aR * aC + cc * aR + rr;
}

// ============================================================
// Shape helpers (must precede the elementwise templates below —
// template bodies reference these as non-dependent names and the
// two-phase lookup rule requires them to be visible at template
// definition time, not at instantiation).
// ============================================================

// The Value-shape factories moved to the L0.5 ops layer
// (<nk/ops/value_factory.hpp>). Re-export them into numkit:: so the ~90
// existing unqualified `createMatrix` / `createForDims` / `createLike` / DimsArg
// call sites (here and in the toolboxes) are unchanged. ops owns the one
// definition; toolboxes going pure will include ops/value_factory.hpp directly.
using ops::DimsArg;
using ops::createMatrix;
using ops::createForDims;
using ops::createLike;

// abs() for an integer-typed Value: MATLAB keeps the integer class and
// SATURATES, so abs(intmin) -> intmax (abs(int8(-128))=127 int8); unsigned
// types are returned unchanged. Reading via elemAsDouble + clamp to the
// type range handles every signed/unsigned width uniformly (|intmin| would
// overflow the type, the clamp catches it). Caller guards isIntegerType().
inline Value absIntegerSaturate(const Value &x, keel::Allocator *mr = nullptr)
{
    Value r = createLike(x, x.type(), mr);
    const keel::usize n = x.numel();
    auto fill = [&](auto *dst) {
        using T = keel::RemovePointer<keel::Decay<decltype(dst)>>;
        const keel::f64 lo = keel::as<keel::f64>(keel::Limits<T>::kMin);
        const keel::f64 hi = keel::as<keel::f64>(keel::Limits<T>::kMax);
        for (keel::usize i = 0; i < n; ++i) {
            keel::f64 v = keel::abs(x.elemAsDouble(i));
            if (v < lo) v = lo;
            else if (v > hi) v = hi;
            dst[i] = static_cast<T>(v);
        }
    };
    switch (x.type()) {
    case ValueType::INT8:   fill(r.int8DataMut());   break;
    case ValueType::INT16:  fill(r.int16DataMut());  break;
    case ValueType::INT32:  fill(r.int32DataMut());  break;
    case ValueType::INT64:  fill(r.int64DataMut());  break;
    case ValueType::UINT8:  fill(r.uint8DataMut());  break;
    case ValueType::UINT16: fill(r.uint16DataMut()); break;
    case ValueType::UINT32: fill(r.uint32DataMut()); break;
    case ValueType::UINT64: fill(r.uint64DataMut()); break;
    default: break;
    }
    return r;
}

// Cast a numeric Value to an integer class by per-element static_cast with
// saturation. Used to restore the integer class after computing through the
// double path (e.g. sort of an integer array; mod/rem of integers). Values
// that are already exact in-range integers round-trip exactly. Reads via
// elemAsDouble so a scalar-stored result is handled too. `vt` must be an
// integer ValueType.
inline Value doubleToIntegerExact(const Value &d, ValueType vt,
                                  keel::Allocator *mr = nullptr)
{
    Value r = createForDims(d.dims(), vt, mr);
    const keel::usize n = d.numel();
    auto fill = [&](auto *dst) {
        using T = keel::RemovePointer<keel::Decay<decltype(dst)>>;
        const keel::f64 lo = keel::as<keel::f64>(keel::Limits<T>::kMin);
        const keel::f64 hi = keel::as<keel::f64>(keel::Limits<T>::kMax);
        for (keel::usize i = 0; i < n; ++i) {
            keel::f64 v = d.elemAsDouble(i);
            if (v < lo) v = lo;
            else if (v > hi) v = hi;
            dst[i] = static_cast<T>(v);
        }
    };
    switch (vt) {
    case ValueType::INT8:   fill(r.int8DataMut());   break;
    case ValueType::INT16:  fill(r.int16DataMut());  break;
    case ValueType::INT32:  fill(r.int32DataMut());  break;
    case ValueType::INT64:  fill(r.int64DataMut());  break;
    case ValueType::UINT8:  fill(r.uint8DataMut());  break;
    case ValueType::UINT16: fill(r.uint16DataMut()); break;
    case ValueType::UINT32: fill(r.uint32DataMut()); break;
    case ValueType::UINT64: fill(r.uint64DataMut()); break;
    default: break;
    }
    return r;
}

// Convert any numeric Value to a DOUBLE Value (element-wise via elemAsDouble,
// so integer arrays — which lack a doubleData() buffer — are handled). A
// DOUBLE input is returned as-is (shares storage). Used to route integer
// operands through double-only elementwise helpers, then cast the result back.
inline Value toDoubleValue(const Value &x, keel::Allocator *mr = nullptr)
{
    if (x.type() == ValueType::DOUBLE) return x;
    Value r = createForDims(x.dims(), ValueType::DOUBLE, mr);
    keel::f64 *dst = r.doubleDataMut();
    const keel::usize n = x.numel();
    for (keel::usize i = 0; i < n; ++i) dst[i] = x.elemAsDouble(i);
    return r;
}

// Exact same-class copy of an integer-typed Value (raw typed copy, so int64
// values above 2^53 are preserved). Used for operations that are the
// IDENTITY on integers but keep the class (floor/ceil/round/fix in MATLAB).
// Caller guards isIntegerType().
inline Value copyIntegerSameClass(const Value &x, keel::Allocator *mr = nullptr)
{
    Value r = createLike(x, x.type(), mr);
    const keel::usize n = x.numel();
    switch (x.type()) {
    case ValueType::INT8:   keel::copy(x.int8Data(),   x.int8Data()   + n, r.int8DataMut());   break;
    case ValueType::INT16:  keel::copy(x.int16Data(),  x.int16Data()  + n, r.int16DataMut());  break;
    case ValueType::INT32:  keel::copy(x.int32Data(),  x.int32Data()  + n, r.int32DataMut());  break;
    case ValueType::INT64:  keel::copy(x.int64Data(),  x.int64Data()  + n, r.int64DataMut());  break;
    case ValueType::UINT8:  keel::copy(x.uint8Data(),  x.uint8Data()  + n, r.uint8DataMut());  break;
    case ValueType::UINT16: keel::copy(x.uint16Data(), x.uint16Data() + n, r.uint16DataMut()); break;
    case ValueType::UINT32: keel::copy(x.uint32Data(), x.uint32Data() + n, r.uint32DataMut()); break;
    case ValueType::UINT64: keel::copy(x.uint64Data(), x.uint64Data() + n, r.uint64DataMut()); break;
    default: break;
    }
    return r;
}

// Shape-preserving empty result for a binary op where at least one
// operand is empty. The non-scalar operand contributes the output
// shape; if both are non-scalar the dims must match, otherwise throw.
// `outType` is chosen by the caller based on its promotion rules.
inline Value emptyResultForBinary(const Value &a, const Value &b,
                                   ValueType outType, keel::Allocator *mr = nullptr)
{
    const bool aShaper = !a.isScalar();
    const bool bShaper = !b.isScalar();
    if (aShaper && bShaper && a.dims() != b.dims()) {
        auto msg = keel::format("Matrix dimensions must agree. a=[{}x{}], b=[{}x{}] (line 211)",
                                a.dims().rows(), a.dims().cols(), b.dims().rows(), b.dims().cols());
        throw numkit::Error(msg.c_str());
    }
    const Value &shape = aShaper ? a : (bShaper ? b : a);
    return createLike(shape, outType, mr);
}

// Pick the arithmetic output type for a pair of operands using
// MATLAB-style promotion: complex > single > integer > double.
// Char and logical operands promote to double.
inline ValueType arithOutType(const Value &a, const Value &b)
{
    if (a.isComplex() || b.isComplex()) return ValueType::COMPLEX;
    if (a.type() == ValueType::SINGLE || b.type() == ValueType::SINGLE) return ValueType::SINGLE;
    if (isIntegerType(a.type())) return a.type();
    if (isIntegerType(b.type())) return b.type();
    return ValueType::DOUBLE;
}

// Convenience wrapper: shape-preserving empty arithmetic result with
// type chosen by arithOutType().
inline Value emptyArithResult(const Value &a, const Value &b, keel::Allocator *mr = nullptr)
{
    Value r = emptyResultForBinary(a, b, arithOutType(a, b), mr);
    if (r.isComplex() && (a.isSingleComplex() || a.type() == ValueType::SINGLE ||
                          b.isSingleComplex() || b.type() == ValueType::SINGLE))
        r.setSingleComplex(true);
    return r;
}

template <typename T, typename Op>
Value elementwiseTyped(const Value &a, const Value &b, ValueType targetType, Op op, keel::Allocator *mr = nullptr);

// ============================================================
// Elementwise binary op on double arrays (with implicit expansion)
// ============================================================
template<typename Op>
Value elementwiseDouble(const Value &a, const Value &b, Op op, keel::Allocator *mr = nullptr)
{
    if (a.type() == ValueType::SINGLE || b.type() == ValueType::SINGLE) {
        return elementwiseTyped<keel::f32>(a, b, ValueType::SINGLE,
            [op](keel::f32 x, keel::f32 y) -> keel::f32 {
                return keel::as<keel::f32>(op(keel::as<keel::f64>(x), keel::as<keel::f64>(y)));
            }, mr);
    }
    if (a.isEmpty() || b.isEmpty())
        return emptyResultForBinary(a, b, ValueType::DOUBLE, mr);
    if (a.isScalar() && b.isScalar())
        return Value::scalar(op(a.toScalar(), b.toScalar()), mr);
    if (a.type() != ValueType::DOUBLE || b.type() != ValueType::DOUBLE)
        return elementwiseTyped<keel::f64>(a, b, ValueType::DOUBLE, op, mr);

    // ND fallback — at least one operand has rank ≥ 4, or NumPy-style
    // broadcasting needed past 3D. Handles same-shape (fast path) and
    // arbitrary-rank broadcast via per-operand offset arithmetic.
    if (a.dims().ndim() >= 4 || b.dims().ndim() >= 4) {
        if (a.isScalar()) {
            auto r = createLike(b, ValueType::DOUBLE, mr);
            const keel::f64 s = a.toScalar();
            const keel::f64 *db = b.doubleData();
            keel::f64 *dst = r.doubleDataMut();
            for (keel::usize i = 0; i < b.numel(); ++i) dst[i] = op(s, db[i]);
            return r;
        }
        if (b.isScalar()) {
            auto r = createLike(a, ValueType::DOUBLE, mr);
            const keel::f64 s = b.toScalar();
            const keel::f64 *da = a.doubleData();
            keel::f64 *dst = r.doubleDataMut();
            for (keel::usize i = 0; i < a.numel(); ++i) dst[i] = op(da[i], s);
            return r;
        }
        // ND broadcast (NumPy: align right, dims match or 1)
        Dims outD;
        if (!broadcastDimsND(a.dims(), b.dims(), outD))
            throw numkit::Error("ND dimensions must broadcast: each axis must match or be 1");
        auto r = createForDims(outD, ValueType::DOUBLE, mr);
        const keel::f64 *da = a.doubleData(), *db = b.doubleData();
        keel::f64 *dst = r.doubleDataMut();
        forEachNDPair(a.dims(), b.dims(), outD,
            [&](keel::usize outIdx, keel::usize aOff, keel::usize bOff) {
                dst[outIdx] = op(da[aOff], db[bOff]);
            });
        return r;
    }

    // 3D paths — same-shape, scalar, or general 3D broadcasting (each
    // axis must equal or be 1; 2D operands implicitly have pages = 1).
    if (a.dims().is3D() || b.dims().is3D()) {
        if (a.isScalar()) {
            auto r = createLike(b, ValueType::DOUBLE, mr);
            keel::f64 s = a.toScalar();
            for (keel::usize i = 0; i < b.numel(); ++i)
                r.doubleDataMut()[i] = op(s, b.doubleData()[i]);
            return r;
        }
        if (b.isScalar()) {
            auto r = createLike(a, ValueType::DOUBLE, mr);
            keel::f64 s = b.toScalar();
            for (keel::usize i = 0; i < a.numel(); ++i)
                r.doubleDataMut()[i] = op(a.doubleData()[i], s);
            return r;
        }
        const keel::usize aR = a.dims().rows(), aC = a.dims().cols();
        const keel::usize aP = a.dims().is3D() ? a.dims().pages() : 1;
        const keel::usize bR = b.dims().rows(), bC = b.dims().cols();
        const keel::usize bP = b.dims().is3D() ? b.dims().pages() : 1;
        keel::usize outR, outC, outP;
        if (!broadcastDims3D(aR, aC, aP, bR, bC, bP, outR, outC, outP))
            throw numkit::Error(
                "3D dimensions must broadcast: each axis must match or be 1");

        // Same-shape fast path skips the per-axis broadcast index math.
        if (aR == bR && aC == bC && aP == bP) {
            auto r = (outP > 1) ? Value::matrix3d(outR, outC, outP, ValueType::DOUBLE, mr)
                                : Value::matrix(outR, outC, ValueType::DOUBLE, mr);
            for (keel::usize i = 0; i < a.numel(); ++i)
                r.doubleDataMut()[i] = op(a.doubleData()[i], b.doubleData()[i]);
            return r;
        }
        auto r = (outP > 1) ? Value::matrix3d(outR, outC, outP, ValueType::DOUBLE, mr)
                            : Value::matrix(outR, outC, ValueType::DOUBLE, mr);
        keel::f64 *dst = r.doubleDataMut();
        const keel::f64 *da = a.doubleData(), *db = b.doubleData();
        for (keel::usize pp = 0; pp < outP; ++pp)
            for (keel::usize cc = 0; cc < outC; ++cc)
                for (keel::usize rr = 0; rr < outR; ++rr) {
                    const keel::usize aOff = broadcastOffset3D(rr, cc, pp, aR, aC, aP);
                    const keel::usize bOff = broadcastOffset3D(rr, cc, pp, bR, bC, bP);
                    dst[pp * outR * outC + cc * outR + rr] = op(da[aOff], db[bOff]);
                }
        return r;
    }

    keel::usize ar = a.dims().rows(), ac = a.dims().cols();
    keel::usize br = b.dims().rows(), bc = b.dims().cols();
    keel::usize outR, outC;
    if (!broadcastDims(ar, ac, br, bc, outR, outC)) {
        auto msg = keel::format("Matrix dimensions must agree. a=[{}x{}], b=[{}x{}] (line 332)",
                                ar, ac, br, bc);
        throw numkit::Error(msg.c_str());
    }

    // Fast path: same dimensions, no broadcasting needed
    if (ar == br && ac == bc) {
        auto r = Value::matrix(outR, outC, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < a.numel(); ++i)
            r.doubleDataMut()[i] = op(a.doubleData()[i], b.doubleData()[i]);
        return r;
    }

    // General broadcasting: column-major indexing
    auto r = Value::matrix(outR, outC, ValueType::DOUBLE, mr);
    keel::f64 *dst = r.doubleDataMut();
    const keel::f64 *da = a.doubleData(), *db = b.doubleData();
    for (keel::usize c = 0; c < outC; ++c) {
        keel::usize ca = (ac == 1) ? 0 : c;
        keel::usize cb = (bc == 1) ? 0 : c;
        for (keel::usize row = 0; row < outR; ++row) {
            keel::usize ra = (ar == 1) ? 0 : row;
            keel::usize rb = (br == 1) ? 0 : row;
            dst[c * outR + row] = op(da[ca * ar + ra], db[cb * br + rb]);
        }
    }
    return r;
}

// narrowComplex (all-real complex -> real, MATLAB) now lives in the value layer
// (nk/value/value.hpp) so value-layer indexing/assignment can use it too.

// ============================================================
// Elementwise binary op on complex arrays. The public `elementwiseComplex`
// narrows an all-real result to real (MATLAB complex-arithmetic semantics); the
// *Impl below does the raw complex computation.
// ============================================================
template<typename Op>
Value elementwiseComplexImpl(const Value &a, const Value &b, Op op, keel::Allocator *mr = nullptr)
{
    const bool outSingle = a.isSingleComplex() || a.type() == ValueType::SINGLE ||
                           b.isSingleComplex() || b.type() == ValueType::SINGLE;
    auto finalize = [&](Value &&r) -> Value {
        if (outSingle && r.isComplex()) r.setSingleComplex(true);
        return keel::move(r);
    };
    if (a.isEmpty() || b.isEmpty())
        return finalize(emptyResultForBinary(a, b, ValueType::COMPLEX, mr));
    auto [ca, cb] = promoteToComplex(a, b, mr);
    if (ca.isScalar() && cb.isScalar())
        return finalize(Value::complexScalar(op(ca.toComplex(), cb.toComplex()), mr));

    // ND fallback (rank ≥ 4 or NumPy-style broadcast past 3D)
    if (ca.dims().ndim() >= 4 || cb.dims().ndim() >= 4) {
        if (ca.isScalar()) {
            auto r = createLike(cb, ValueType::COMPLEX, mr);
            keel::c64 s = ca.toComplex();
            const keel::c64 *db = cb.complexData();
            keel::c64 *dst = r.complexDataMut();
            for (keel::usize i = 0; i < cb.numel(); ++i) dst[i] = op(s, db[i]);
            return finalize(keel::move(r));
        }
        if (cb.isScalar()) {
            auto r = createLike(ca, ValueType::COMPLEX, mr);
            keel::c64 s = cb.toComplex();
            const keel::c64 *da = ca.complexData();
            keel::c64 *dst = r.complexDataMut();
            for (keel::usize i = 0; i < ca.numel(); ++i) dst[i] = op(da[i], s);
            return finalize(keel::move(r));
        }
        Dims outD;
        if (!broadcastDimsND(ca.dims(), cb.dims(), outD))
            throw numkit::Error("ND dimensions must broadcast: each axis must match or be 1");
        auto r = createForDims(outD, ValueType::COMPLEX, mr);
        const keel::c64 *da = ca.complexData(), *db = cb.complexData();
        keel::c64 *dst = r.complexDataMut();
        forEachNDPair(ca.dims(), cb.dims(), outD,
            [&](keel::usize outIdx, keel::usize aOff, keel::usize bOff) {
                dst[outIdx] = op(da[aOff], db[bOff]);
            });
        return finalize(keel::move(r));
    }

    if (ca.dims().is3D() || cb.dims().is3D()) {
        if (ca.isScalar()) {
            auto r = createLike(cb, ValueType::COMPLEX, mr);
            keel::c64 s = ca.toComplex();
            for (keel::usize i = 0; i < cb.numel(); ++i)
                r.complexDataMut()[i] = op(s, cb.complexData()[i]);
            return finalize(keel::move(r));
        }
        if (cb.isScalar()) {
            auto r = createLike(ca, ValueType::COMPLEX, mr);
            keel::c64 s = cb.toComplex();
            for (keel::usize i = 0; i < ca.numel(); ++i)
                r.complexDataMut()[i] = op(ca.complexData()[i], s);
            return finalize(keel::move(r));
        }
        const keel::usize aR = ca.dims().rows(), aC = ca.dims().cols();
        const keel::usize aP = ca.dims().is3D() ? ca.dims().pages() : 1;
        const keel::usize bR = cb.dims().rows(), bC = cb.dims().cols();
        const keel::usize bP = cb.dims().is3D() ? cb.dims().pages() : 1;
        keel::usize outR, outC, outP;
        if (!broadcastDims3D(aR, aC, aP, bR, bC, bP, outR, outC, outP))
            throw numkit::Error(
                "3D dimensions must broadcast: each axis must match or be 1");

        if (aR == bR && aC == bC && aP == bP) {
            auto r = createLike(ca, ValueType::COMPLEX, mr);
            for (keel::usize i = 0; i < ca.numel(); ++i)
                r.complexDataMut()[i] = op(ca.complexData()[i], cb.complexData()[i]);
            return finalize(keel::move(r));
        }
        auto r = (outP > 1) ? Value::matrix3d(outR, outC, outP, ValueType::COMPLEX, mr)
                            : Value::matrix(outR, outC, ValueType::COMPLEX, mr);
        keel::c64 *dst = r.complexDataMut();
        const keel::c64 *da = ca.complexData(), *db = cb.complexData();
        for (keel::usize pp = 0; pp < outP; ++pp)
            for (keel::usize cc = 0; cc < outC; ++cc)
                for (keel::usize rr = 0; rr < outR; ++rr) {
                    const keel::usize aOff = broadcastOffset3D(rr, cc, pp, aR, aC, aP);
                    const keel::usize bOff = broadcastOffset3D(rr, cc, pp, bR, bC, bP);
                    dst[pp * outR * outC + cc * outR + rr] = op(da[aOff], db[bOff]);
                }
        return finalize(keel::move(r));
    }

    keel::usize ar = ca.dims().rows(), ac = ca.dims().cols();
    keel::usize br = cb.dims().rows(), bc = cb.dims().cols();
    keel::usize outR, outC;
    if (!broadcastDims(ar, ac, br, bc, outR, outC)) {
        auto msg = keel::format("Matrix dimensions must agree. a=[{}x{}], b=[{}x{}] (line 454)",
                                ar, ac, br, bc);
        throw numkit::Error(msg.c_str());
    }

    if (ar == br && ac == bc) {
        auto r = Value::complexMatrix(outR, outC, mr);
        for (keel::usize i = 0; i < ca.numel(); ++i)
            r.complexDataMut()[i] = op(ca.complexData()[i], cb.complexData()[i]);
        return finalize(keel::move(r));
    }

    auto r = Value::complexMatrix(outR, outC, mr);
    keel::c64 *dst = r.complexDataMut();
    const keel::c64 *da = ca.complexData(), *db = cb.complexData();
    for (keel::usize c = 0; c < outC; ++c) {
        keel::usize cca = (ac == 1) ? 0 : c;
        keel::usize ccb = (bc == 1) ? 0 : c;
        for (keel::usize row = 0; row < outR; ++row) {
            keel::usize ra = (ar == 1) ? 0 : row;
            keel::usize rb = (br == 1) ? 0 : row;
            dst[c * outR + row] = op(da[cca * ar + ra], db[ccb * br + rb]);
        }
    }
    return finalize(keel::move(r));
}

// Public entry: raw complex elementwise op, then narrow an all-real result to
// real (MATLAB). All callers (binary +,-,.*,./,.^, max/min) want this.
template<typename Op>
Value elementwiseComplex(const Value &a, const Value &b, Op op, keel::Allocator *mr = nullptr)
{
    return narrowComplex(elementwiseComplexImpl(a, b, keel::move(op), mr), mr);
}

// ============================================================
// Elementwise unary on double
// ============================================================
template<typename Op>
Value unaryDouble(const Value &a, Op op, keel::Allocator *mr = nullptr)
{
    if (a.type() == ValueType::SINGLE) {
        auto r = createLike(a, ValueType::SINGLE, mr);
        const keel::f32 *src = static_cast<const keel::f32 *>(a.rawData());
        keel::f32 *dst = static_cast<keel::f32 *>(r.rawDataMut());
        for (keel::usize i = 0; i < a.numel(); ++i)
            dst[i] = keel::as<keel::f32>(op(keel::as<keel::f64>(src[i])));
        return r;
    }
    if (a.isScalar())
        return Value::scalar(op(a.toScalar()), mr);
    auto r = createLike(a, ValueType::DOUBLE, mr);
    if (a.type() == ValueType::DOUBLE) {
        const keel::f64 *src = a.doubleData();
        keel::f64 *dst = r.doubleDataMut();
        for (keel::usize i = 0; i < a.numel(); ++i)
            dst[i] = op(src[i]);
    } else {
        keel::f64 *dst = r.doubleDataMut();
        for (keel::usize i = 0; i < a.numel(); ++i)
            dst[i] = op(a.elemAsDouble(i));
    }
    return r;
}

// ============================================================
// Elementwise unary on complex. Public name narrows an all-real result to real
// (MATLAB: uminus/uplus/conj and unary math of an all-real complex give real).
// ============================================================
template<typename Op>
Value unaryComplexImpl(const Value &a, Op op, keel::Allocator *mr = nullptr)
{
    if (a.isScalar()) {
        auto r = Value::complexScalar(op(a.toComplex()), mr);
        if (a.isSingleComplex()) r.setSingleComplex(true);
        return r;
    }
    auto r = createLike(a, ValueType::COMPLEX, mr);
    if (a.isSingleComplex()) r.setSingleComplex(true);
    for (keel::usize i = 0; i < a.numel(); ++i)
        r.complexDataMut()[i] = op(a.complexData()[i]);
    return r;
}
template<typename Op>
Value unaryComplex(const Value &a, Op op, keel::Allocator *mr = nullptr)
{
    return narrowComplex(unaryComplexImpl(a, keel::move(op), mr), mr);
}

// ============================================================
// Parse dimension arguments for array creation functions
// Supports: f(n), f(m,n), f(m,n,p), f([m n]), f([m n p]), f(size(x))
// Returns {rows, cols, pages}. pages=0 means 2D.
//
// Size validation lives in numkit::argcheck (value layer): numkit-voiced
// diagnostics whose rejection behavior is validated against the
// reference engine. `fn` names the calling builtin for error
// attribution.
// ============================================================
inline DimsArg parseDimsArgs(keel::ArrayRef<Value> args, keel::StringRef fn)
{
    if (args.empty())
        return {1, 1, 0};

    // Single vector argument: [m n] or [m n p]. Any numeric class is a
    // legal size vector (int8/single/logical); complex is not.
    if (args.size() == 1 && !args[0].isScalar() && args[0].numel() >= 2) {
        argcheck::arrayDimReal(args[0], fn);
        keel::usize n = args[0].numel();
        keel::usize r = argcheck::arrayDim(args[0].elemAsDouble(0), fn);
        keel::usize c = argcheck::arrayDim(args[0].elemAsDouble(1), fn);
        keel::usize p = (n >= 3) ? argcheck::arrayDim(args[0].elemAsDouble(2), fn) : 0;
        return {r, c, p};
    }

    // Single scalar: f(n) → n×n
    if (args.size() == 1) {
        const keel::usize n = argcheck::arrayDim(args[0], fn);
        return {n, n, 0};
    }

    // Two scalars: f(m, n)
    keel::usize r = argcheck::arrayDim(args[0], fn);
    keel::usize c = argcheck::arrayDim(args[1], fn);

    // Three scalars: f(m, n, p)
    if (args.size() >= 3) {
        keel::usize p = argcheck::arrayDim(args[2], fn);
        return {r, c, p};
    }

    return {r, c, 0};
}

// ND dim parser. Same call shapes as parseDimsArgs but produces a full
// ND dim vector (no 3-cap). Returns at least 2 dims (matrices stay 2D).
//   * f(n)              → {n, n}
//   * f(m, n, p, ...)   → {m, n, p, ...}
//   * f([m n])          → {m, n}
//   * f([m n p q ...])  → {m, n, p, q, ...}
// Caller can decide whether to use the 2D / 3D fast paths or matrixND.
//
// The returned vector is backed by `mr` — typical use is a per-call
// ScratchArena for the array-creation builtin (zeros/ones/rand/...).
inline keel::Array<keel::usize> parseDimsArgsND(keel::ArrayRef<Value> args,
                                          keel::StringRef fn,
                                          keel::Allocator *mr = nullptr)
{
    keel::Array<keel::usize> out(mr);
    if (args.empty()) {
        out.push_back(1);
        out.push_back(1);
        return out;
    }
    // Vector form: f([m n p ...]) — any numeric class, never complex.
    if (args.size() == 1 && !args[0].isScalar() && args[0].numel() >= 2) {
        argcheck::arrayDimReal(args[0], fn);
        const keel::usize n = args[0].numel();
        out.reserve(n);
        for (keel::usize i = 0; i < n; ++i)
            out.push_back(argcheck::arrayDim(args[0].elemAsDouble(i), fn));
        return out;
    }
    // Single scalar: f(n) → n×n
    if (args.size() == 1) {
        const keel::usize n = argcheck::arrayDim(args[0], fn);
        out.push_back(n);
        out.push_back(n);
        return out;
    }
    // Multiple scalars: f(m, n, p, ...)
    out.reserve(args.size());
    for (const auto &a : args)
        out.push_back(argcheck::arrayDim(a, fn));
    return out;
}

// Strip trailing 1s past the 2nd dim (MATLAB convention: ones(3,4,1)
// returns a 2D matrix, not 3D). Keeps at least 2 dims so a vector stays
// {1, n} rather than collapsing to scalar shape.
inline void stripTrailingOnes(keel::Array<keel::usize> &dims)
{
    while (dims.size() > 2 && dims.back() == 1)
        dims.pop_back();
}

// Map a MATLAB class-name string to a ValueType. Returns true on
// recognised names (the standard numeric / logical types accepted by
// zeros/ones/true/false/rand). Unknown names return false (caller can
// throw or fall through to default DOUBLE).
inline bool valueTypeFromName(keel::StringRef name, ValueType &out)
{
    if      (name == "double")  { out = ValueType::DOUBLE;  return true; }
    else if (name == "single")  { out = ValueType::SINGLE;  return true; }
    else if (name == "int8")    { out = ValueType::INT8;    return true; }
    else if (name == "int16")   { out = ValueType::INT16;   return true; }
    else if (name == "int32")   { out = ValueType::INT32;   return true; }
    else if (name == "int64")   { out = ValueType::INT64;   return true; }
    else if (name == "uint8")   { out = ValueType::UINT8;   return true; }
    else if (name == "uint16")  { out = ValueType::UINT16;  return true; }
    else if (name == "uint32")  { out = ValueType::UINT32;  return true; }
    else if (name == "uint64")  { out = ValueType::UINT64;  return true; }
    else if (name == "logical") { out = ValueType::LOGICAL; return true; }
    return false;
}

// Detect a trailing class-name argument in zeros/ones/etc. If args.back()
// is a char/string with a recognised type name, return the matching
// ValueType and trim args by one. Otherwise return DOUBLE (default) and
// leave args unchanged. Also handles the 'like' form -- (..., 'like', X)
// pulls the type from X.
//
// Sets `outType` and modifies `args` (Span is value-copied at callsite).
inline keel::ArrayRef<Value>
extractTypeArg(keel::ArrayRef<Value> args, ValueType &outType)
{
    outType = ValueType::DOUBLE;
    if (args.empty()) return args;

    // 'like' form: zeros(..., 'like', X). Need at least 2 trailing args.
    if (args.size() >= 2) {
        const Value &maybeLike = args[args.size() - 2];
        if ((maybeLike.isChar() || maybeLike.isString())
            && maybeLike.toString() == "like") {
            outType = args[args.size() - 1].type();
            return keel::ArrayRef<Value>(args.data(), args.size() - 2);
        }
    }

    // Trailing class-name string: zeros(M, N, ..., 'uint8').
    const Value &last = args[args.size() - 1];
    if (last.isChar() || last.isString()) {
        ValueType t;
        if (valueTypeFromName(last.toString(), t)) {
            outType = t;
            return keel::ArrayRef<Value>(args.data(), args.size() - 1);
        }
    }
    return args;
}

// Create a zero matrix from a flat ND dim list, picking the matrix /
// matrix3d / matrixND constructor that matches the rank. Pointer + size
// so the same helper composes with std::vector, std::pmr::vector, raw
// arrays, etc.
inline Value createMatrixND(const keel::usize *dims, keel::usize nDims,
                             ValueType type, keel::Allocator *mr = nullptr)
{
    const keel::i32 nd = keel::as<keel::i32>(nDims);
    if (nd <= 1) return Value::matrix(nd == 1 ? dims[0] : 0, 1, type, mr);
    if (nd == 2) return Value::matrix(dims[0], dims[1], type, mr);
    if (nd == 3) return Value::matrix3d(dims[0], dims[1], dims[2], type, mr);
    return Value::matrixND(dims, nd, type, mr);
}

// ============================================================
// Saturating arithmetic for integer types
// ============================================================

template <typename T>
inline T saturateAdd(T a, T b)
{
    if constexpr (keel::IsFloatingPoint<T>) {
        return a + b;
    } else if constexpr (keel::IsUnsigned<T>) {
        T r = a + b;
        return (r < a) ? keel::Limits<T>::kMax : r;
    } else {
        using W = keel::i64;
        W r = static_cast<W>(a) + static_cast<W>(b);
        if (r > keel::Limits<T>::kMax) return keel::Limits<T>::kMax;
        if (r < keel::Limits<T>::kMin) return keel::Limits<T>::kMin;
        return static_cast<T>(r);
    }
}

template <typename T>
inline T saturateSub(T a, T b)
{
    if constexpr (keel::IsFloatingPoint<T>) {
        return a - b;
    } else if constexpr (keel::IsUnsigned<T>) {
        return (b > a) ? T(0) : a - b;
    } else {
        using W = keel::i64;
        W r = static_cast<W>(a) - static_cast<W>(b);
        if (r > keel::Limits<T>::kMax) return keel::Limits<T>::kMax;
        if (r < keel::Limits<T>::kMin) return keel::Limits<T>::kMin;
        return static_cast<T>(r);
    }
}

template <typename T>
inline T saturateMul(T a, T b)
{
    if constexpr (keel::IsFloatingPoint<T>) {
        return a * b;
    } else if constexpr (keel::IsSame<T, keel::i64> || keel::IsSame<T, keel::u64>) {
        keel::f64 r = keel::as<keel::f64>(a) * keel::as<keel::f64>(b);
        if (r > keel::as<keel::f64>(keel::Limits<T>::kMax)) return keel::Limits<T>::kMax;
        if (r < keel::as<keel::f64>(keel::Limits<T>::kMin)) return keel::Limits<T>::kMin;
        return a * b;
    } else {
        using W = keel::i64;
        W r = static_cast<W>(a) * static_cast<W>(b);
        if (r > keel::Limits<T>::kMax) return keel::Limits<T>::kMax;
        if (r < keel::Limits<T>::kMin) return keel::Limits<T>::kMin;
        return static_cast<T>(r);
    }
}

template <typename T>
inline T saturateDiv(T a, T b)
{
    if constexpr (keel::IsFloatingPoint<T>) {
        return a / b;
    } else {
        if (b == 0) {
            if constexpr (keel::IsUnsigned<T>)
                return (a == 0) ? T(0) : keel::Limits<T>::kMax;
            else
                return (a > 0) ? keel::Limits<T>::kMax
                     : (a < 0) ? keel::Limits<T>::kMin
                               : T(0);
        }
        if constexpr (!keel::IsUnsigned<T>) {
            if (a == keel::Limits<T>::kMin && b == T(-1))
                return keel::Limits<T>::kMax;
        }
        // MATLAB integer division: round result
        keel::f64 r = keel::as<keel::f64>(a) / keel::as<keel::f64>(b);
        r = keel::round(r);
        if (r > keel::as<keel::f64>(keel::Limits<T>::kMax)) return keel::Limits<T>::kMax;
        if (r < keel::as<keel::f64>(keel::Limits<T>::kMin)) return keel::Limits<T>::kMin;
        return static_cast<T>(r);
    }
}

template <typename T>
inline T saturateNeg(T a)
{
    if constexpr (keel::IsFloatingPoint<T>) {
        return -a;
    } else if constexpr (keel::IsUnsigned<T>) {
        return T(0);
    } else {
        if (a == keel::Limits<T>::kMin) return keel::Limits<T>::kMax;
        return -a;
    }
}

// ============================================================
// Elementwise binary op on typed arrays (integer/single)
// ============================================================

template <typename T, typename Op>
Value elementwiseTyped(const Value &a, const Value &b, ValueType targetType, Op op, keel::Allocator *mr)
{
    if (a.isEmpty() || b.isEmpty())
        return emptyResultForBinary(a, b, targetType, mr);

    // Read element from source, converting to target type
    auto readAt = [](const Value &v, ValueType tgt, keel::usize r, keel::usize c) -> T {
        keel::usize idx = c * v.dims().rows() + r; // column-major
        if (v.type() == tgt) return static_cast<const T *>(v.rawData())[idx];
        keel::f64 d = v.elemAsDouble(idx);
        if constexpr (keel::IsIntegral<T>)
            return static_cast<T>(keel::clamp(keel::round(d),
                keel::as<keel::f64>(keel::Limits<T>::kMin),
                keel::as<keel::f64>(keel::Limits<T>::kMax)));
        else
            return static_cast<T>(d);
    };
    auto readLinearTyped = [](const Value &v, ValueType tgt, keel::usize i) -> T {
        if (v.type() == tgt) return static_cast<const T *>(v.rawData())[i];
        keel::f64 d = v.elemAsDouble(i);
        if constexpr (keel::IsIntegral<T>)
            return static_cast<T>(keel::clamp(keel::round(d),
                keel::as<keel::f64>(keel::Limits<T>::kMin),
                keel::as<keel::f64>(keel::Limits<T>::kMax)));
        else
            return static_cast<T>(d);
    };

    // ND fallback (rank ≥ 4) — uses ND broadcast helpers.
    if (a.dims().ndim() >= 4 || b.dims().ndim() >= 4) {
        if (a.isScalar()) {
            auto r = createLike(b, targetType, mr);
            T *dst = static_cast<T *>(r.rawDataMut());
            T sa = readLinearTyped(a, targetType, 0);
            for (keel::usize i = 0; i < b.numel(); ++i)
                dst[i] = op(sa, readLinearTyped(b, targetType, i));
            return r;
        }
        if (b.isScalar()) {
            auto r = createLike(a, targetType, mr);
            T *dst = static_cast<T *>(r.rawDataMut());
            T sb = readLinearTyped(b, targetType, 0);
            for (keel::usize i = 0; i < a.numel(); ++i)
                dst[i] = op(readLinearTyped(a, targetType, i), sb);
            return r;
        }
        Dims outD;
        if (!broadcastDimsND(a.dims(), b.dims(), outD))
            throw numkit::Error("ND dimensions must broadcast: each axis must match or be 1");
        auto r = createForDims(outD, targetType, mr);
        T *dst = static_cast<T *>(r.rawDataMut());
        forEachNDPair(a.dims(), b.dims(), outD,
            [&](keel::usize outIdx, keel::usize aOff, keel::usize bOff) {
                dst[outIdx] = op(readLinearTyped(a, targetType, aOff),
                                 readLinearTyped(b, targetType, bOff));
            });
        return r;
    }

    // 3D path — same-shape elementwise and scalar broadcasting only.
    // broadcastDims below is 2D-only; reach here for 3D and we would
    // silently drop the page dim.
    if (a.dims().is3D() || b.dims().is3D()) {
        if (a.isScalar()) {
            auto r = createLike(b, targetType, mr);
            T *dst = static_cast<T *>(r.rawDataMut());
            T sa = readLinearTyped(a, targetType, 0);
            for (keel::usize i = 0; i < b.numel(); ++i)
                dst[i] = op(sa, readLinearTyped(b, targetType, i));
            return r;
        }
        if (b.isScalar()) {
            auto r = createLike(a, targetType, mr);
            T *dst = static_cast<T *>(r.rawDataMut());
            T sb = readLinearTyped(b, targetType, 0);
            for (keel::usize i = 0; i < a.numel(); ++i)
                dst[i] = op(readLinearTyped(a, targetType, i), sb);
            return r;
        }
        const keel::usize aR = a.dims().rows(), aC = a.dims().cols();
        const keel::usize aP = a.dims().is3D() ? a.dims().pages() : 1;
        const keel::usize bR = b.dims().rows(), bC = b.dims().cols();
        const keel::usize bP = b.dims().is3D() ? b.dims().pages() : 1;
        keel::usize outR, outC, outP;
        if (!broadcastDims3D(aR, aC, aP, bR, bC, bP, outR, outC, outP))
            throw numkit::Error(
                "3D dimensions must broadcast: each axis must match or be 1");

        if (aR == bR && aC == bC && aP == bP) {
            auto r = createLike(a, targetType, mr);
            T *dst = static_cast<T *>(r.rawDataMut());
            for (keel::usize i = 0; i < a.numel(); ++i)
                dst[i] = op(readLinearTyped(a, targetType, i),
                            readLinearTyped(b, targetType, i));
            return r;
        }
        auto r = (outP > 1) ? Value::matrix3d(outR, outC, outP, targetType, mr)
                            : Value::matrix(outR, outC, targetType, mr);
        T *dst = static_cast<T *>(r.rawDataMut());
        for (keel::usize pp = 0; pp < outP; ++pp)
            for (keel::usize cc = 0; cc < outC; ++cc)
                for (keel::usize rr = 0; rr < outR; ++rr) {
                    const keel::usize aOff = broadcastOffset3D(rr, cc, pp, aR, aC, aP);
                    const keel::usize bOff = broadcastOffset3D(rr, cc, pp, bR, bC, bP);
                    dst[pp * outR * outC + cc * outR + rr] =
                        op(readLinearTyped(a, targetType, aOff),
                           readLinearTyped(b, targetType, bOff));
                }
        return r;
    }

    keel::usize ar = a.dims().rows(), ac = a.dims().cols();
    keel::usize br = b.dims().rows(), bc = b.dims().cols();
    keel::usize outR, outC;
    if (!broadcastDims(ar, ac, br, bc, outR, outC)) {
        auto msg = keel::format("Matrix dimensions must agree. a=[{}x{}], b=[{}x{}] (line 913)",
                                ar, ac, br, bc);
        throw numkit::Error(msg.c_str());
    }

    auto r = Value::matrix(outR, outC, targetType, mr);
    T *dst = static_cast<T *>(r.rawDataMut());
    for (keel::usize c = 0; c < outC; ++c) {
        keel::usize ca = (ac == 1) ? 0 : c;
        keel::usize cb = (bc == 1) ? 0 : c;
        for (keel::usize row = 0; row < outR; ++row) {
            keel::usize ra = (ar == 1) ? 0 : row;
            keel::usize rb = (br == 1) ? 0 : row;
            dst[c * outR + row] = op(readAt(a, targetType, ra, ca),
                                     readAt(b, targetType, rb, cb));
        }
    }
    return r;
}

// ============================================================
// Dispatch integer/single binary op by ValueType
// ============================================================

// Resolve the target integer type from a pair.
// MATLAB rules: intN + double → intN, intN + intN → intN, intN + intM → error
inline ValueType resolveIntegerPairType(const Value &a, const Value &b)
{
    ValueType ta = a.type(), tb = b.type();
    bool ai = isIntegerType(ta), bi = isIntegerType(tb);
    bool af = isFloatType(ta),  bf = isFloatType(tb);
    bool aS = (ta == ValueType::SINGLE), bS = (tb == ValueType::SINGLE);

    if (ai && bi) {
        if (ta != tb)
            throw numkit::Error("Integers can only be combined with integers of the same class");
        return ta;
    }
    if (ai && (bf || tb == ValueType::LOGICAL)) return ta;
    if (bi && (af || ta == ValueType::LOGICAL)) return tb;
    if (aS || bS) return ValueType::SINGLE;
    return ValueType::EMPTY; // not an integer/single case
}

template <typename Op>
Value dispatchIntegerBinaryOp(const Value &a, const Value &b, Op op, keel::Allocator *mr = nullptr)
{
    ValueType target = resolveIntegerPairType(a, b);
    if (target == ValueType::EMPTY) return Value(); // signal: not handled

    switch (target) {
    case ValueType::INT8:   return elementwiseTyped<keel::i8>(a, b, target, [&](keel::i8 x, keel::i8 y) { return op(x, y); }, mr);
    case ValueType::INT16:  return elementwiseTyped<keel::i16>(a, b, target, [&](keel::i16 x, keel::i16 y) { return op(x, y); }, mr);
    case ValueType::INT32:  return elementwiseTyped<keel::i32>(a, b, target, [&](keel::i32 x, keel::i32 y) { return op(x, y); }, mr);
    case ValueType::INT64:  return elementwiseTyped<keel::i64>(a, b, target, [&](keel::i64 x, keel::i64 y) { return op(x, y); }, mr);
    case ValueType::UINT8:  return elementwiseTyped<keel::u8>(a, b, target, [&](keel::u8 x, keel::u8 y) { return op(x, y); }, mr);
    case ValueType::UINT16: return elementwiseTyped<keel::u16>(a, b, target, [&](keel::u16 x, keel::u16 y) { return op(x, y); }, mr);
    case ValueType::UINT32: return elementwiseTyped<keel::u32>(a, b, target, [&](keel::u32 x, keel::u32 y) { return op(x, y); }, mr);
    case ValueType::UINT64: return elementwiseTyped<keel::u64>(a, b, target, [&](keel::u64 x, keel::u64 y) { return op(x, y); }, mr);
    case ValueType::SINGLE: return elementwiseTyped<keel::f32>(a, b, target, [&](keel::f32 x, keel::f32 y) { return op(x, y); }, mr);
    default: return Value();
    }
}

// ============================================================
// Unary typed op
// ============================================================

template <typename T, typename Op>
Value unaryTyped(const Value &a, ValueType targetType, Op op, keel::Allocator *mr = nullptr)
{
    auto r = createLike(a, targetType, mr);
    const T *src = static_cast<const T *>(a.rawData());
    T *dst = static_cast<T *>(r.rawDataMut());
    for (keel::usize i = 0; i < a.numel(); ++i)
        dst[i] = op(src[i]);
    return r;
}

} // namespace numkit
