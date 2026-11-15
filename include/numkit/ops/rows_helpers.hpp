// ops/include/numkit/ops/rows_helpers.hpp
//
// Shared helpers for "treat each row as a tuple" operations
// (unique 'rows', sortrows, future ismember 'rows' etc.).
// All paths assume a column-major DOUBLE matrix.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/value/value.hpp>

#include <cmath>
#include <cstddef>

namespace numkit::ops {

// Lex-compare rows `a` and `b` of a column-major DOUBLE buffer.
// Returns -1 / 0 / +1 in standard cmp convention. NaN is treated as
// strictly greater than any non-NaN (push to end on ascending sort) —
// matches MATLAB sort/sortrows ordering for finite vs NaN entries.
inline int rowLexCmp(const keel::f64 *p, keel::usize cols, keel::usize rows,
                     keel::usize a, keel::usize b)
{
    for (keel::usize c = 0; c < cols; ++c) {
        const keel::f64 av = p[c * rows + a];
        const keel::f64 bv = p[c * rows + b];
        const bool aNan = std::isnan(av);
        const bool bNan = std::isnan(bv);
        if (aNan && bNan) continue;
        if (aNan) return  1;
        if (bNan) return -1;
        if (av < bv) return -1;
        if (av > bv) return  1;
    }
    return 0;
}

// Lex-compare rows by an explicit column priority list. Each entry of
// `cols1based` is the 1-based column index to compare on; a negative
// entry flips the direction for that key (descending). Out-of-range
// entries throw via the caller's pre-validation (this function trusts
// the caller). Pointer + size so the same helper composes with
// std::vector / std::pmr::vector / arrays.
inline int rowLexCmpByCols(const keel::f64 *p, keel::usize totalCols,
                           keel::usize rows, keel::usize a, keel::usize b,
                           const int *cols1based, keel::usize nCols)
{
    for (keel::usize k = 0; k < nCols; ++k) {
        const int rawCol = cols1based[k];
        const bool desc = rawCol < 0;
        const keel::usize cIdx = static_cast<keel::usize>(desc ? -rawCol : rawCol) - 1;
        (void) totalCols;  // caller validated rawCol in range
        const keel::f64 av = p[cIdx * rows + a];
        const keel::f64 bv = p[cIdx * rows + b];
        const bool aNan = std::isnan(av);
        const bool bNan = std::isnan(bv);
        if (aNan && bNan) continue;
        // NaN goes to the end on ascending; flip on descending.
        if (aNan) return desc ? -1 :  1;
        if (bNan) return desc ?  1 : -1;
        if (av < bv) return desc ?  1 : -1;
        if (av > bv) return desc ? -1 :  1;
    }
    return 0;
}

// Gather the rows of `x` indicated by `origRows` into a new DOUBLE matrix
// of shape (nRows, x.dims().cols()), preserving column-major layout.
// `x` must be a 2D DOUBLE matrix. Pointer + size so any contiguous
// row-index container (std::vector / pmr::vector / array) works.
inline Value collectRowsByIndex(const Value &x,
                                 const keel::usize *origRows, keel::usize nRows,
                                 keel::Allocator *mr = nullptr)
{
    const keel::usize rows = x.dims().rows();
    const keel::usize cols = x.dims().cols();
    auto r = Value::matrix(nRows, cols, ValueType::DOUBLE, mr);
    const keel::f64 *src = x.doubleData();
    keel::f64 *dst = r.doubleDataMut();
    for (keel::usize newRow = 0; newRow < nRows; ++newRow) {
        const keel::usize srcRow = origRows[newRow];
        for (keel::usize c = 0; c < cols; ++c)
            dst[c * nRows + newRow] = src[c * rows + srcRow];
    }
    return r;
}

} // namespace numkit::ops
