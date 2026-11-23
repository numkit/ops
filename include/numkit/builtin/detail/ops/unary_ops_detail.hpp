// src/builtin/src/ops/unary_ops_detail.hpp
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>

#include <algorithm>
#include <numeric>


namespace numkit::builtin {

namespace {

inline keel::i8 saturateNeg(keel::i8 v) {
    if (v == keel::Limits<keel::i8>::kMin) return keel::Limits<keel::i8>::kMax;
    return keel::as<keel::i8>(-v);
}
inline keel::i16 saturateNeg(keel::i16 v) {
    if (v == keel::Limits<keel::i16>::kMin) return keel::Limits<keel::i16>::kMax;
    return keel::as<keel::i16>(-v);
}
inline keel::i32 saturateNeg(keel::i32 v) {
    if (v == keel::Limits<keel::i32>::kMin) return keel::Limits<keel::i32>::kMax;
    return keel::as<keel::i32>(-v);
}
inline keel::i64 saturateNeg(keel::i64 v) {
    if (v == keel::Limits<keel::i64>::kMin) return keel::Limits<keel::i64>::kMax;
    return keel::as<keel::i64>(-v);
}

inline Value transpose2D(const Value &x, bool conjugate, const char *fnName,
                         keel::Allocator *p)
{
    if (x.dims().is3D())
        throw Error("transpose is not defined for N-D arrays",
                     0, 0, fnName, "", "numkit:transpose:3DInput");
    const keel::usize rows = x.dims().rows(), cols = x.dims().cols();
    const ValueType t = x.type();

    if (t == ValueType::COMPLEX) {
        if (x.isScalar()) {
            auto r = Value::complexScalar(conjugate ? keel::conj(x.toComplex())
                                                    : x.toComplex(), p);
            if (x.isSingleComplex()) r.setSingleComplex(true);
            return r;
        }
        auto r = Value::complexMatrix(cols, rows, p);
        if (x.isSingleComplex()) r.setSingleComplex(true);
        keel::c64 *dst = r.complexDataMut();
        for (keel::usize i = 0; i < rows; ++i)
            for (keel::usize j = 0; j < cols; ++j) {
                const keel::c64 v = x.complexElem(i, j);
                dst[i * cols + j] = conjugate ? keel::conj(v) : v;
            }
        return r;
    }

    if (t == ValueType::CELL) {
        auto r = Value::cell(cols, rows, p);
        for (keel::usize i = 0; i < rows; ++i)
            for (keel::usize j = 0; j < cols; ++j)
                r.cellAt(i * cols + j) = x.cellAt(j * rows + i);
        return r;
    }

    if (t == ValueType::OBJECT)
        return x.objectTranspose(p);

    if (t == ValueType::STRING) {
        if (x.isScalar())
            return x;
        auto r = Value::stringArray(cols, rows, p);
        for (keel::usize i = 0; i < rows; ++i)
            for (keel::usize j = 0; j < cols; ++j)
                r.stringElemSet(i * cols + j, x.stringElem(j * rows + i));
        return r;
    }

    if (t == ValueType::STRUCT) {
        if (!x.isStructArray())
            return x;
        auto r = Value::structArray(cols, rows, p);
        for (const auto &fn : x.fieldNamesInOrder())
            r.setFieldAll(fn, Value());
        for (keel::usize i = 0; i < rows; ++i)
            for (keel::usize j = 0; j < cols; ++j)
                for (auto entry : x.structArrayElem(j * rows + i))
                    r.setField(i * cols + j, entry.key, entry.value);
        return r;
    }

    if (t == ValueType::FUNC_HANDLE)
        throw Error("Transpose not supported for this type",
                     0, 0, fnName, "", "numkit:transpose:unsupportedType");

    if (t == ValueType::DOUBLE && x.isScalar())
        return Value::scalar(x.toScalar(), p);
    const keel::usize es = elementSize(t);
    auto r = Value::matrix(cols, rows, t, p);
    const char *src = static_cast<const char *>(x.rawData());
    char *dst = static_cast<char *>(r.rawDataMut());
    for (keel::usize i = 0; i < rows; ++i)
        for (keel::usize j = 0; j < cols; ++j)
            keel::mem_copy(dst + (i * cols + j) * es,
                        src + (j * rows + i) * es, es);
    return r;
}

} // namespace

} // namespace numkit::builtin
