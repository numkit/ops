// src/builtin/src/datafun/accum.cpp
//
// accumarray — group-by reduction algorithm.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/math.h>
#include <numkit/builtin/datafun.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>

namespace numkit::builtin {

namespace {

// 1-based subscript validation: must be a positive integer <= outDim.
keel::usize toSubIndex(keel::f64 v, keel::usize maxAllowed, const char *fn)
{
    if (!keel::is_finite(v) || v < 1.0)
        throw Error(keel::format("{}: subscripts must be positive integers", fn),
                     0, 0, fn, "", keel::format("numkit:{}:subRange", fn));
    const keel::f64 rounded = keel::round(v);
    if (keel::abs(v - rounded) > 1e-9)
        throw Error(keel::format("{}: subscripts must be integer-valued", fn),
                     0, 0, fn, "", keel::format("numkit:{}:subInt", fn));
    const keel::usize idx = keel::as<keel::usize>(rounded);
    if (maxAllowed > 0 && idx > maxAllowed)
        throw Error(keel::format("{}: subscript exceeds output dimension", fn),
                     0, 0, fn, "", keel::format("numkit:{}:subOOB", fn));
    return idx;
}

keel::f64 identityFor(AccumReducer op)
{
    switch (op) {
    case AccumReducer::Sum:  return 0.0;
    case AccumReducer::Mean: return 0.0;
    case AccumReducer::Prod: return 1.0;
    case AccumReducer::Max:  return -keel::Limits<keel::f64>::infinity();
    case AccumReducer::Min:  return  keel::Limits<keel::f64>::infinity();
    case AccumReducer::Any:  return 0.0;
    case AccumReducer::All:  return 1.0;
    }
    return 0.0;
}

inline keel::f64 applyReducer(AccumReducer op, keel::f64 acc, keel::f64 v)
{
    switch (op) {
    case AccumReducer::Sum:
    case AccumReducer::Mean: return acc + v;
    case AccumReducer::Prod: return acc * v;
    case AccumReducer::Max:  return (v > acc || keel::is_nan(v)) ? v : acc;
    case AccumReducer::Min:  return (v < acc || keel::is_nan(v)) ? v : acc;
    case AccumReducer::Any:  return (acc != 0.0 || v != 0.0) ? 1.0 : 0.0;
    case AccumReducer::All:  return (acc != 0.0 && v != 0.0) ? 1.0 : 0.0;
    }
    return acc;
}

// Build the output shape. userShape (passed-in) wins if non-zero
// length, else derive from per-column max(subs). subs is N x D.
keel::Array<keel::usize> resolveOutShape(const Value &subs, const keel::usize *userShape,
                                   keel::usize nUserShape, const char *fn,
                                   keel::Allocator *mr)
{
    const auto &d = subs.dims();
    const keel::usize N = d.rows();
    const keel::usize D = (d.ndim() <= 1) ? 1 : d.cols();
    if (nUserShape > 0) {
        if (nUserShape < D)
            throw Error(keel::format("{}: sz length must be at least size(subs, 2)", fn),
                         0, 0, fn, "", keel::format("numkit:{}:sizeRank", fn));
        keel::Array<keel::usize> out(keel::ArrayRef<keel::usize>(userShape, nUserShape), mr);
        return out;
    }
    // Auto-derive from max per column. For 1D subs, 1D output.
    keel::Array<keel::usize> shape(D, mr);
    if (N == 0) {
        // Empty subs + no sz -> 0x0 (matches MATLAB).
        return keel::Array<keel::usize>(std::max<keel::usize>(D, 1), 0, mr);
    }
    const keel::f64 *p = subs.doubleData();
    for (keel::usize c = 0; c < D; ++c) {
        keel::f64 mx = 0.0;
        for (keel::usize r = 0; r < N; ++r) {
            const keel::f64 v = p[c * N + r];
            if (v > mx) mx = v;
        }
        shape[c] = toSubIndex(mx, 0, fn);
    }
    return shape;
}

// Column-major linear index from the row-of-subs r of the N x D matrix.
keel::usize linearIndexFromSubs(const Value &subs, keel::usize r, keel::usize N, keel::usize D,
                           const keel::usize *shape, const char *fn)
{
    const keel::f64 *p = subs.doubleData();
    keel::usize idx = 0;
    keel::usize stride = 1;
    for (keel::usize c = 0; c < D; ++c) {
        const keel::usize s = toSubIndex(p[c * N + r], shape[c], fn);
        idx += (s - 1) * stride;
        stride *= shape[c];
    }
    return idx;
}

// Allocate the output Value for the given shape.
Value allocOutput(const keel::usize *shape, keel::usize nShape, keel::Allocator *mr)
{
    if (nShape == 1)
        return Value::matrix(shape[0], 1, ValueType::DOUBLE, mr);
    if (nShape == 2)
        return Value::matrix(shape[0], shape[1], ValueType::DOUBLE, mr);
    return Value::matrixND(shape, keel::as<keel::i32>(nShape),
                            ValueType::DOUBLE, mr);
}

inline keel::f64 readVal(const Value &vals, keel::usize i, bool valIsScalar)
{
    return valIsScalar ? vals.toScalar() : vals.doubleData()[i];
}

} // namespace

Value accumarray(const Value &subs, const Value &vals, keel::ArrayRef<keel::usize> outShape,
                 AccumReducer op, keel::f64 fillVal, keel::Allocator *mr)
{
    const char *fn = "accumarray";

    if (subs.type() != ValueType::DOUBLE)
        throw Error("accumarray: subs must be DOUBLE",
                     0, 0, fn, "", "numkit:accumarray:subType");
    if (vals.type() != ValueType::DOUBLE)
        throw Error("accumarray: vals must be DOUBLE",
                     0, 0, fn, "", "numkit:accumarray:valType");

    const auto &sd = subs.dims();
    if (sd.ndim() > 2)
        throw Error("accumarray: subs must be a 2D matrix",
                     0, 0, fn, "", "numkit:accumarray:subND");

    const keel::usize N = sd.rows();
    const keel::usize D = (sd.ndim() <= 1 || sd.cols() == 0) ? 1 : sd.cols();

    const bool valIsScalar = vals.isScalar();
    if (!valIsScalar && vals.numel() != N)
        throw Error("accumarray: vals must be a scalar or a length-N vector",
                     0, 0, fn, "", "numkit:accumarray:valSize");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto shape = resolveOutShape(subs, outShape.data(), outShape.size(), fn, scratch.allocator());
    if (shape.size() < D) shape.resize(D, 1);

    Value out = allocOutput(shape.data(), shape.size(), mr);
    const keel::usize total = out.numel();
    keel::f64 *dst = out.doubleDataMut();

    if (total == 0)
        return out;

    auto touched = keel::Array<keel::u8>(total, scratch.allocator());
    const bool needCount = (op == AccumReducer::Mean);
    auto count = keel::Array<keel::usize>(scratch.allocator());
    if (needCount) count.assign(total, 0);

    const keel::f64 init = identityFor(op);

    for (keel::usize r = 0; r < N; ++r) {
        const keel::usize lin = linearIndexFromSubs(subs, r, N, D, shape.data(), fn);
        const keel::f64 v = readVal(vals, r, valIsScalar);
        if (!touched[lin]) {
            dst[lin] = applyReducer(op, init, v);
            touched[lin] = 1;
        } else {
            dst[lin] = applyReducer(op, dst[lin], v);
        }
        if (needCount) ++count[lin];
    }

    if (needCount) {
        for (keel::usize i = 0; i < total; ++i) {
            if (!touched[i])       dst[i] = fillVal;
            else if (count[i] > 0) dst[i] /= keel::as<keel::f64>(count[i]);
        }
    } else if (fillVal != 0.0 ||
               op == AccumReducer::Prod || op == AccumReducer::Max ||
               op == AccumReducer::Min  || op == AccumReducer::All) {
        for (keel::usize i = 0; i < total; ++i)
            if (!touched[i]) dst[i] = fillVal;
    }

    return out;
}

} // namespace numkit::builtin
