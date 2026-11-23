// math/include/numkit/math/integration/integration_detail.hpp
//
// Internal (numkit::builtin::detail) entry points of the integration compute,
// exposed only so the Engine-coupled *_reg adapters in bundle can reuse the
// exact same machinery as the public gradient/del2/cumtrapz/trapz/integral.
// Not a stable API — these are the lower-level dispatch helpers + the
// Gauss-Kronrod node tables; lifted out of integration.cpp's anonymous
// namespace in Phase E so the adapters could move to the bundle layer.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/quad.h>
#include <numkit/value/value.hpp>


namespace numkit::builtin::detail {

// 15-point Gauss-Kronrod nodes (symmetric about 0) and weights, plus the
// embedded 7-point Gauss weights — re-exported from keel::quad so the
// __gk15_nodes primitive shares the single source of truth.
using ::keel::quad::kKronrodX;
using ::keel::quad::kKronrodW;
using ::keel::quad::kGaussW;

// Copy any numeric/logical Value to a fresh DOUBLE Value (used by every
// cumtrapz/trapz path that needs a real double buffer).
Value toDoubleCopy(const Value &x, keel::Allocator *mr);

// N-D gradient: one gradient per dimension, up to `nout`, spacings in `hs`.
keel::Array<Value> gradientND(const Value &f, const keel::f64 *hs, keel::usize nh,
                                   keel::usize nout, keel::Allocator *mr);

// cumtrapz down columns / along rows, real and complex variants.
Value cumtrapzMatrixCols(const keel::f64 *src, const keel::f64 *xData, keel::usize rows,
                         keel::usize cols, keel::Allocator *mr);
Value cumtrapzMatrixRows(const keel::f64 *src, const keel::f64 *xData, keel::usize rows,
                         keel::usize cols, keel::Allocator *mr);
Value cumtrapzMatrixColsC(const keel::c64 *src, const keel::f64 *xData, keel::usize rows,
                          keel::usize cols, keel::Allocator *mr);
Value cumtrapzMatrixRowsC(const keel::c64 *src, const keel::f64 *xData, keel::usize rows,
                          keel::usize cols, keel::Allocator *mr);

// trapz along `dim`; `xData` null → unit spacing.
Value trapzImpl(const Value &y, keel::i32 dim, const keel::f64 *xData,
                keel::Allocator *mr);

} // namespace numkit::builtin::detail
