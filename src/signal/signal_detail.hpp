// toolboxes/signal/src/signal_detail.hpp
//
// PRIVATE (src-level) header — NOT part of the public signal API.
// Shared <fn>_argcheck cores (argcheck separation,
// dev-docs/todo/argcheck_separation.md — boundary rule: raw arguments in,
// all message digestion here; call sites are one pass-through line).
// Rejection classes reference-probed against R2025b 2026-09-18
// (build/probe_signal_argcheck.m, probe_signal_classes.m).

#pragma once

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>   // mtypeName (argcheck got-clauses)

#include <memory_resource>
#include <string>
#include <string_view>
#include <vector>

namespace numkit::signal::detail {

// Render "2x2x2" for got-clauses.
inline std::string shapeOf(const Value &v)
{
    const Dims &d = v.dims();
    std::string s = std::to_string(d.dim(0));
    for (int k = 1; k < d.ndim(); ++k) {
        s += "x";
        s += std::to_string(d.dim(k));
    }
    return s;
}

inline void throwInputType(const Value &v, std::string_view fn,
                           std::string_view argName, int argNum,
                           std::string_view expectation)
{
    throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                    + " (" + std::string(argName) + ") must be "
                    + std::string(expectation) + " (got a " + shapeOf(v)
                    + " " + mtypeName(v.type()) + " array).",
                0, 0, std::string(fn), "",
                "numkit:" + std::string(fn) + ":inputType");
}

// Numeric-or-logical family (conv, bitrevorder, ifft 'symmetric'):
// every numeric class plus logical passes; char, string, cell and struct are
// rejected as one inputType class.
inline void requireNumericOrLogicalInput(const Value &v, std::string_view fn,
                                         std::string_view argName = "x",
                                         int argNum = 1)
{
    if (v.isNumeric() || v.isLogical())
        return;
    throwInputType(v, fn, argName, argNum, "numeric or logical");
}

inline void requireNumericSignalInput(const Value &v, std::string_view fn,
                                      std::string_view argName = "x",
                                      int argNum = 1)
{
    requireNumericOrLogicalInput(v, fn, argName, argNum);
}

// Float family (the reference's "must be double or single" class):
// double, single, and complex double pass; logical / integer /
// char / cell are rejected (the reference rejects them as invalidType).
inline void requireFloatInput(const Value &v, std::string_view fn,
                              std::string_view argName = "x", int argNum = 1)
{
    const ValueType t = v.type();
    if (t == ValueType::DOUBLE || t == ValueType::SINGLE
        || t == ValueType::COMPLEX || t == ValueType::EMPTY)
        return;
    throwInputType(v, fn, argName, argNum, "a double or single array");
}

inline Value widenSingleToDouble(const Value &v, std::pmr::memory_resource *mr)
{
    if (v.type() != ValueType::SINGLE) return v;
    const auto &d = v.dims();
    Value out;
    if (d.ndim() <= 2) {
        out = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    } else if (d.ndim() == 3) {
        out = Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr);
    } else {
        std::vector<std::size_t> od(static_cast<std::size_t>(d.ndim()));
        for (int k = 0; k < d.ndim(); ++k) od[static_cast<std::size_t>(k)] = d.dim(k);
        out = Value::matrixND(od.data(), d.ndim(), ValueType::DOUBLE, mr);
    }
    const float *src = v.singleData();
    double *dst = out.doubleDataMut();
    const std::size_t n = v.numel();
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = static_cast<double>(src[i]);
    return out;
}

inline Value narrowDoubleToSingle(const Value &v, std::pmr::memory_resource *mr)
{
    if (v.type() != ValueType::DOUBLE) return v;
    const auto &d = v.dims();
    Value out;
    if (d.ndim() <= 2) {
        out = Value::matrix(d.rows(), d.cols(), ValueType::SINGLE, mr);
    } else if (d.ndim() == 3) {
        out = Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::SINGLE, mr);
    } else {
        std::vector<std::size_t> od(static_cast<std::size_t>(d.ndim()));
        for (int k = 0; k < d.ndim(); ++k) od[static_cast<std::size_t>(k)] = d.dim(k);
        out = Value::matrixND(od.data(), d.ndim(), ValueType::SINGLE, mr);
    }
    const double *src = v.doubleData();
    float *dst = out.singleDataMut();
    const std::size_t n = v.numel();
    for (std::size_t i = 0; i < n; ++i)
        dst[i] = static_cast<float>(src[i]);
    return out;
}

inline Value narrowDoubleToSingleIf(const Value &v, bool wantSingle,
                                    std::pmr::memory_resource *mr)
{
    return wantSingle ? narrowDoubleToSingle(v, mr) : v;
}

// Real float family (medfilt1): the reference additionally rejects
// complex input with an expectedReal-class error.
inline void requireRealFloatInput(const Value &v, std::string_view fn,
                                  std::string_view argName = "x",
                                  int argNum = 1)
{
    if (v.type() == ValueType::COMPLEX)
        throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                        + " (" + std::string(argName)
                        + ") must be real (got a " + shapeOf(v)
                        + " complex array).",
                    0, 0, std::string(fn), "",
                    "numkit:" + std::string(fn) + ":inputReal");
    requireFloatInput(v, fn, argName, argNum);
}

// Complex kernels that do not exist yet voice their own identifier so
// the rejection is attributed and grep-able
// (bugs/opened/signal/complex-kernel-gaps.md).
inline void requireDoubleNotComplexInput(const Value &v, std::string_view fn,
                                         std::string_view argName = "x",
                                         int argNum = 1)
{
    if (v.type() == ValueType::COMPLEX)
        throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                        + " (" + std::string(argName)
                        + "): complex inputs are not supported yet (got a "
                        + shapeOf(v) + " complex array).",
                    0, 0, std::string(fn), "",
                    "numkit:" + std::string(fn) + ":complexUnsupported");
    requireFloatInput(v, fn, argName, argNum);
}

// Non-empty vector (conv / convmtx / polystab / extractsigroi x):
// the reference rejects matrices AND empties with one notVector class.
inline void requireVectorInput(const Value &v, std::string_view fn,
                               std::string_view argName = "x", int argNum = 1)
{
    if (!v.dims().isVector() || v.numel() == 0)
        throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                        + " (" + std::string(argName)
                        + ") must be a non-empty row or column vector "
                          "(got a " + shapeOf(v) + " "
                        + mtypeName(v.type()) + " array).",
                    0, 0, std::string(fn), "",
                    "numkit:" + std::string(fn) + ":notVector");
}

// Shape option shared by conv / conv2 / convn / filter2.
inline void validateShapeOption(const std::string &shape, std::string_view fn)
{
    if (shape == "full" || shape == "same" || shape == "valid")
        return;
    throw Error(std::string(fn) + ": shape option must be 'full', 'same', "
                    "or 'valid' (got '" + shape + "')",
                0, 0, std::string(fn), "",
                "numkit:" + std::string(fn) + ":badShape");
}

} // namespace numkit::signal::detail
