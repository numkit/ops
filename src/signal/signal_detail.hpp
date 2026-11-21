// toolboxes/signal/src/signal_detail.hpp
//
// PRIVATE (src-level) header — NOT part of the public signal API.
// Shared <fn>_argcheck cores (argcheck separation,
// dev-docs/todo/argcheck_separation.md — boundary rule: raw arguments in,
// all message digestion here; call sites are one pass-through line).
// Rejection classes reference-probed against R2025b 2026-09-18
// (build/probe_signal_argcheck.m, probe_signal_classes.m).

#pragma once

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>   // mtypeName (argcheck got-clauses)

namespace numkit::signal::detail {

// Render "2x2x2" for got-clauses.
inline keel::String shapeOf(const Value &v)
{
    const Dims &d = v.dims();
    keel::String s = keel::format("{}", d.dim(0));
    for (keel::i32 k = 1; k < d.ndim(); ++k) {
        keel::format_into(s, "x{}", d.dim(k));
    }
    return s;
}

inline void throwInputType(const Value &v, keel::StringRef fn,
                           keel::StringRef argName, keel::i32 argNum,
                           keel::StringRef expectation)
{
    throw Error(keel::format("{}: input {} ({}) must be {} (got a {} {} array).",
                             fn, argNum, argName, expectation, shapeOf(v),
                             mtypeName(v.type())),
                0, 0, fn, "",
                keel::format("numkit:{}:inputType", fn));
}

// Numeric-or-logical family (conv, bitrevorder, ifft 'symmetric'):
// every numeric class plus logical passes; char, string, cell and struct are
// rejected as one inputType class.
inline void requireNumericOrLogicalInput(const Value &v, keel::StringRef fn,
                                         keel::StringRef argName = "x",
                                         keel::i32 argNum = 1)
{
    if (v.isNumeric() || v.isLogical())
        return;
    throwInputType(v, fn, argName, argNum, "numeric or logical");
}

inline void requireNumericSignalInput(const Value &v, keel::StringRef fn,
                                      keel::StringRef argName = "x",
                                      keel::i32 argNum = 1)
{
    requireNumericOrLogicalInput(v, fn, argName, argNum);
}

// Float family (the reference's "must be double or single" class):
// double, single, and complex double pass; logical / integer /
// char / cell are rejected (the reference rejects them as invalidType).
inline void requireFloatInput(const Value &v, keel::StringRef fn,
                              keel::StringRef argName = "x", keel::i32 argNum = 1)
{
    const ValueType t = v.type();
    if (t == ValueType::DOUBLE || t == ValueType::SINGLE
        || t == ValueType::COMPLEX || t == ValueType::EMPTY)
        return;
    throwInputType(v, fn, argName, argNum, "a double or single array");
}

inline Value widenSingleToDouble(const Value &v, keel::Allocator *mr)
{
    if (v.type() != ValueType::SINGLE) return v;
    const auto &d = v.dims();
    Value out;
    if (d.ndim() <= 2) {
        out = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    } else if (d.ndim() == 3) {
        out = Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr);
    } else {
        keel::Array<keel::usize> od(d.ndim(), mr);
        for (keel::i32 k = 0; k < d.ndim(); ++k) od[keel::as<keel::usize>(k)] = d.dim(k);
        out = Value::matrixND(od.data(), d.ndim(), ValueType::DOUBLE, mr);
    }
    const keel::f32 *src = v.singleData();
    keel::f64 *dst = out.doubleDataMut();
    const keel::usize n = v.numel();
    for (keel::usize i = 0; i < n; ++i)
        dst[i] = keel::as<keel::f64>(src[i]);
    return out;
}

inline Value narrowDoubleToSingle(const Value &v, keel::Allocator *mr)
{
    if (v.type() != ValueType::DOUBLE) return v;
    const auto &d = v.dims();
    Value out;
    if (d.ndim() <= 2) {
        out = Value::matrix(d.rows(), d.cols(), ValueType::SINGLE, mr);
    } else if (d.ndim() == 3) {
        out = Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::SINGLE, mr);
    } else {
        keel::Array<keel::usize> od(d.ndim(), mr);
        for (keel::i32 k = 0; k < d.ndim(); ++k) od[keel::as<keel::usize>(k)] = d.dim(k);
        out = Value::matrixND(od.data(), d.ndim(), ValueType::SINGLE, mr);
    }
    const keel::f64 *src = v.doubleData();
    keel::f32 *dst = out.singleDataMut();
    const keel::usize n = v.numel();
    for (keel::usize i = 0; i < n; ++i)
        dst[i] = keel::as<keel::f32>(src[i]);
    return out;
}

inline Value narrowDoubleToSingleIf(const Value &v, bool wantSingle,
                                    keel::Allocator *mr)
{
    return wantSingle ? narrowDoubleToSingle(v, mr) : v;
}

// Real float family (medfilt1): the reference additionally rejects
// complex input with an expectedReal-class error.
inline void requireRealFloatInput(const Value &v, keel::StringRef fn,
                                  keel::StringRef argName = "x",
                                  keel::i32 argNum = 1)
{
    if (v.type() == ValueType::COMPLEX)
        throw Error(keel::format("{}: input {} ({}) must be real (got a {} complex array).",
                                 fn, argNum, argName, shapeOf(v)),
                    0, 0, fn, "",
                    keel::format("numkit:{}:inputReal", fn));
    requireFloatInput(v, fn, argName, argNum);
}

// Complex kernels that do not exist yet voice their own identifier so
// the rejection is attributed and grep-able
// (bugs/opened/signal/complex-kernel-gaps.md).
inline void requireDoubleNotComplexInput(const Value &v, keel::StringRef fn,
                                         keel::StringRef argName = "x",
                                         keel::i32 argNum = 1)
{
    if (v.type() == ValueType::COMPLEX)
        throw Error(keel::format("{}: input {} ({}): complex inputs are not supported yet (got a {} complex array).",
                                 fn, argNum, argName, shapeOf(v)),
                    0, 0, fn, "",
                    keel::format("numkit:{}:complexUnsupported", fn));
    requireFloatInput(v, fn, argName, argNum);
}

// Non-empty vector (conv / convmtx / polystab / extractsigroi x):
// the reference rejects matrices AND empties with one notVector class.
inline void requireVectorInput(const Value &v, keel::StringRef fn,
                               keel::StringRef argName = "x", keel::i32 argNum = 1)
{
    if (!v.dims().isVector() || v.numel() == 0)
        throw Error(keel::format("{}: input {} ({}) must be a non-empty row or column vector (got a {} {} array).",
                                 fn, argNum, argName, shapeOf(v), mtypeName(v.type())),
                    0, 0, fn, "",
                    keel::format("numkit:{}:notVector", fn));
}

// Shape option shared by conv / conv2 / convn / filter2.
inline void validateShapeOption(keel::StringRef shape, keel::StringRef fn)
{
    if (shape == "full" || shape == "same" || shape == "valid")
        return;
    throw Error(keel::format("{}: shape option must be 'full', 'same', or 'valid' (got '{}')",
                             fn, shape),
                0, 0, fn, "",
                keel::format("numkit:{}:badShape", fn));
}

} // namespace numkit::signal::detail
