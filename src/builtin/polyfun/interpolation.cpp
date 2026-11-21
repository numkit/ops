// toolboxes/builtin/src/math/interpolation/interp.cpp
//
// 1-D / 2-D / 3-D interpolation. polyfit / polyval moved to
// math/elementary/polynomials.cpp; trapz to math/integration/integration.cpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <tuple>
#include <numkit/builtin/polyfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "interp_detail.hpp"

namespace numkit::builtin {

// ── Internal algorithm helpers ────────────────────────────────────────


// ── interp1 ───────────────────────────────────────────────────────────
Value interp1(const Value &x, const Value &y, const Value &xq, keel::StringRef method, keel::Allocator *mr)
{
    // Public typed entry point: MATLAB's default extrapolation policy
    // (NaN out-of-range except for spline/pchip/makima).
    return interp1Dispatch(x, y, xq, method, Interp1Extrap::Default,
                           std::numeric_limits<double>::quiet_NaN(), mr);
}

// ── interp2 ───────────────────────────────────────────────────────────

Value interp2(const Value &V, const Value &Xq, const Value &Yq, keel::StringRef method, keel::Allocator *mr)
{
    if (V.dims().is3D() || V.dims().ndim() > 2)
        throw Error("interp2: V must be a 2D matrix",
                     0, 0, "interp2", "", "numkit:interp2:rank");
    const keel::usize R = V.dims().rows();
    const keel::usize C = V.dims().cols();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xGrid = keel::Array<double>(C, scratch.allocator());
    auto yGrid = keel::Array<double>(R, scratch.allocator());
    for (keel::usize i = 0; i < C; ++i) xGrid[i] = static_cast<double>(i + 1);
    for (keel::usize i = 0; i < R; ++i) yGrid[i] = static_cast<double>(i + 1);
    return interp2Impl(V, xGrid.data(), xGrid.size(), yGrid.data(), yGrid.size(), Xq, Yq, method, mr);
}

Value interp2(const Value &X, const Value &Y, const Value &V, const Value &Xq, const Value &Yq, keel::StringRef method, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> xGrid(scratch.allocator()), yGrid(scratch.allocator());
    readGridAxis(X, xGrid, "X");
    readGridAxis(Y, yGrid, "Y");
    return interp2Impl(V, xGrid.data(), xGrid.size(), yGrid.data(), yGrid.size(), Xq, Yq, method, mr);
}

// ── interp3 ───────────────────────────────────────────────────────────

Value interp3(const Value &V, const Value &Xq, const Value &Yq, const Value &Zq, keel::StringRef method, keel::Allocator *mr)
{
    if (!V.dims().is3D())
        throw Error("interp3: V must be a 3D array",
                     0, 0, "interp3", "", "numkit:interp3:rank");
    const keel::usize R = V.dims().rows();
    const keel::usize C = V.dims().cols();
    const keel::usize P = V.dims().pages();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xGrid = keel::Array<double>(C, scratch.allocator());
    auto yGrid = keel::Array<double>(R, scratch.allocator());
    auto zGrid = keel::Array<double>(P, scratch.allocator());
    for (keel::usize i = 0; i < C; ++i) xGrid[i] = static_cast<double>(i + 1);
    for (keel::usize i = 0; i < R; ++i) yGrid[i] = static_cast<double>(i + 1);
    for (keel::usize i = 0; i < P; ++i) zGrid[i] = static_cast<double>(i + 1);
    return interp3Impl(V, xGrid.data(), xGrid.size(), yGrid.data(), yGrid.size(), zGrid.data(), zGrid.size(), Xq, Yq, Zq, method, mr);
}

Value interp3(const Value &X, const Value &Y, const Value &Z, const Value &V, const Value &Xq, const Value &Yq, const Value &Zq, keel::StringRef method, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> xGrid(scratch.allocator()), yGrid(scratch.allocator()),
                        zGrid(scratch.allocator());
    readGridAxis(X, xGrid, "X");
    readGridAxis(Y, yGrid, "Y");
    readGridAxis(Z, zGrid, "Z");
    return interp3Impl(V, xGrid.data(), xGrid.size(), yGrid.data(), yGrid.size(), zGrid.data(), zGrid.size(), Xq, Yq, Zq, method, mr);
}

// ── spline ────────────────────────────────────────────────────────────
Value spline(const Value &x, const Value &y, const Value &xq, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    if (n != y.numel())
        throw Error("spline: x and y must have same length",
                     0, 0, "spline", "", "numkit:spline:lengthMismatch");
    if (n < 2)
        throw Error("spline: need at least 2 data points",
                     0, 0, "spline", "", "numkit:spline:tooFewPoints");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto yq = interpSpline(x.doubleData(), y.doubleData(), n, xq.doubleData(), xq.numel(), scratch.allocator());
    return packInterpResult(yq.data(), yq.size(), xq, mr);
}

// ── pchip ─────────────────────────────────────────────────────────────
Value pchip(const Value &x, const Value &y, const Value &xq, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    if (n != y.numel())
        throw Error("pchip: x and y must have same length",
                     0, 0, "pchip", "", "numkit:pchip:lengthMismatch");
    if (n < 2)
        throw Error("pchip: need at least 2 data points",
                     0, 0, "pchip", "", "numkit:pchip:tooFewPoints");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto yq = interpPchip(x.doubleData(), y.doubleData(), n, xq.doubleData(), xq.numel(), scratch.allocator());
    return packInterpResult(yq.data(), yq.size(), xq, mr);
}

// ── makima (modified Akima) ───────────────────────────────────────────
//
// Same call shape as pchip / spline. v1 supports the explicit 3-arg
// form `yi = makima(x, y, xq)`; the 2-arg pp-form is a documented gap.
Value makima(const Value &x, const Value &y, const Value &xq,
             keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    if (n != y.numel())
        throw Error("makima: x and y must have same length",
                     0, 0, "makima", "", "numkit:makima:lengthMismatch");
    if (n < 2)
        throw Error("makima: need at least 2 data points",
                     0, 0, "makima", "", "numkit:makima:tooFewPoints");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto yq = interpMakima(x.doubleData(), y.doubleData(), n,
                            xq.doubleData(), xq.numel(), scratch.allocator());
    return packInterpResult(yq.data(), yq.size(), xq, mr);
}

// polyfit / polyval moved to math/elementary/polynomials.cpp
// trapz moved to math/integration/integration.cpp

// ── Pack 30: mkpp / ppval ────────────────────────────────────────────

Value mkpp(const Value &breaks, const Value &coefs, keel::Allocator *mr)
{
    if (breaks.numel() < 2)
        throw Error("mkpp: breaks must have at least 2 entries",
                     0, 0, "mkpp", "", "numkit:mkpp:breaks");
    const keel::usize L = breaks.numel() - 1;  // pieces
    if (coefs.dims().ndim() > 2)
        throw Error("mkpp: only 2-D coefs (pieces × order) supported",
                     0, 0, "mkpp", "", "numkit:mkpp:rank");
    const keel::usize pieces = coefs.dims().rows();
    const keel::usize order  = coefs.dims().cols();
    if (pieces != L)
        throw Error("mkpp: rows(coefs) must equal numel(breaks) - 1",
                     0, 0, "mkpp", "", "numkit:mkpp:shape");

    auto pp = Value::structure(mr);
    pp.field("form")   = Value::fromString("pp", mr);
    pp.field("breaks") = breaks;
    pp.field("coefs")  = coefs;
    pp.field("pieces") = Value::scalar(static_cast<double>(L), mr);
    pp.field("order")  = Value::scalar(static_cast<double>(order), mr);
    pp.field("dim")    = Value::scalar(1.0, mr);
    return pp;
}

Value ppval(const Value &pp, const Value &x, keel::Allocator *mr)
{
    if (!pp.isStruct() || !pp.hasField("breaks") || !pp.hasField("coefs"))
        throw Error("ppval: first argument must be a pp struct",
                     0, 0, "ppval", "", "numkit:ppval:notPp");
    const Value &breaks = pp.field("breaks");
    const Value &coefs  = pp.field("coefs");
    const keel::usize L      = breaks.numel() - 1;
    const keel::usize order  = coefs.dims().cols();
    const double *bp    = breaks.doubleData();
    const double *cp    = coefs.doubleData();
    const keel::usize cR     = coefs.dims().rows();   // pieces (column-major stride)

    auto out = createLike(x, ValueType::DOUBLE, mr);
    const keel::usize N = x.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> xq(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) xq[i] = x.elemAsDouble(i);
    (void)keel::interp::ppval({bp, L + 1}, {cp, cR * order}, cR, order, xq.as_ref(),
                      {out.doubleDataMut(), N});
    return out;
}

} // namespace numkit::builtin
