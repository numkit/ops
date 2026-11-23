// ops/include/numkit/ops/value_factory.hpp
//
// Value-shape factories shared by the op kernels and the toolboxes: allocate a
// Value of a given shape/type. Built only on the Value substrate (value.hpp) —
// no engine, no toolbox — so they live at the L0.5 ops layer and every layer
// above can use them.
//
// (Historically in toolboxes/builtin/src/helpers.hpp under namespace nk;
// helpers.hpp now re-exports these into numkit:: so existing unqualified
// callers are unaffected.)

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/value/value.hpp>


namespace numkit::ops {

// 2D/3D shape descriptor for createMatrix (pages == 0 → 2D).
struct DimsArg
{
    keel::usize rows = 1, cols = 1, pages = 0;
};

// Create a zero matrix / 3D array with the given dimensions + element type.
inline Value createMatrix(DimsArg d, ValueType type, keel::Allocator *mr = nullptr)
{
    if (d.pages > 0)
        return Value::matrix3d(d.rows, d.cols, d.pages, type, mr);
    return Value::matrix(d.rows, d.cols, type, mr);
}

// Allocate a Value with the given Dims, picking the rank-appropriate ctor
// (matrix / matrix3d / matrixND).
inline Value createForDims(const Dims &d, ValueType type, keel::Allocator *mr = nullptr)
{
    const keel::i32 nd = d.ndim();
    if (nd >= 4) {
        keel::usize dims[Dims::kMaxRank];
        for (keel::i32 i = 0; i < nd; ++i)
            dims[i] = d.dim(i);
        return Value::matrixND(dims, nd, type, mr);
    }
    return createMatrix({d.rows(), d.cols(), d.is3D() ? d.pages() : 0}, type, mr);
}

// Allocate a Value with the same shape as `src`, optionally of a different
// type. Required for any elementwise output (a numel()-into-2D allocation
// corrupts the heap when src is 3D+).
inline Value createLike(const Value &src, ValueType type, keel::Allocator *mr = nullptr)
{
    return createForDims(src.dims(), type, mr);
}

} // namespace numkit::ops
