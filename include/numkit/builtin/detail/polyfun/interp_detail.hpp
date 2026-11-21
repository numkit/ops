// toolboxes/.../interp_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by interp.cpp + interp_reg.cpp.
// Thin Value adapters delegating to keel::interp (<keel/interp.h>).
#pragma once

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/interp.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/ops/reductions.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <cstring>
#include <limits>
#include <utility>

namespace numkit::builtin {

namespace {

keel::Array<double> interpSpline(const double *x, const double *y, keel::usize n,
                                 const double *xq, keel::usize nq, keel::Allocator *mr)
{
    keel::Array<double> yq(nq, mr);
    (void)keel::interp::interp1({x, n}, {y, n}, {xq, nq}, {yq.data(), nq}, keel::interp::Interp1Method::Spline,
                        keel::interp::ExtrapMode::Default, 0.0, mr);
    return yq;
}

keel::Array<double> interpPchip(const double *x, const double *y, keel::usize n,
                                const double *xq, keel::usize nq, keel::Allocator *mr)
{
    keel::Array<double> yq(nq, mr);
    (void)keel::interp::interp1({x, n}, {y, n}, {xq, nq}, {yq.data(), nq}, keel::interp::Interp1Method::Pchip,
                        keel::interp::ExtrapMode::Default, 0.0, mr);
    return yq;
}

keel::Array<double> interpMakima(const double *x, const double *y, keel::usize n,
                                 const double *xq, keel::usize nq, keel::Allocator *mr)
{
    if (n < 2)
        throw Error("makima: need at least 2 data points",
                     0, 0, "makima", "", "numkit:makima:tooFewPoints");
    keel::Array<double> yq(nq, mr);
    (void)keel::interp::interp1({x, n}, {y, n}, {xq, nq}, {yq.data(), nq}, keel::interp::Interp1Method::Makima,
                        keel::interp::ExtrapMode::Default, 0.0, mr);
    return yq;
}

Value packInterpResult(const double *yq, keel::usize nq, const Value &xq, keel::Allocator *mr)
{
    const bool isRow = xq.dims().rows() == 1;
    auto r = isRow ? Value::matrix(1, nq, ValueType::DOUBLE, mr)
                   : Value::matrix(nq, 1, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < nq; ++i)
        r.doubleDataMut()[i] = yq[i];
    return r;
}

enum class Interp1Extrap { Default, Method, Const };

keel::interp::Interp1Method parseInterp1Method(keel::StringRef method)
{
    if (method == "linear")   return keel::interp::Interp1Method::Linear;
    if (method == "nearest")  return keel::interp::Interp1Method::Nearest;
    if (method == "previous") return keel::interp::Interp1Method::Previous;
    if (method == "next")     return keel::interp::Interp1Method::Next;
    if (method == "spline")   return keel::interp::Interp1Method::Spline;
    if (method == "pchip")    return keel::interp::Interp1Method::Pchip;
    if (method == "makima")   return keel::interp::Interp1Method::Makima;
    if (method == "cubic" || method == "v5cubic") return keel::interp::Interp1Method::Cubic;
    throw Error(keel::format("interp1: unknown method '{}'", method),
                 0, 0, "interp1", "", "numkit:interp1:badMethod");
}

keel::interp::ExtrapMode toKeelExtrap(Interp1Extrap mode) noexcept
{
    switch (mode) {
    case Interp1Extrap::Default: return keel::interp::ExtrapMode::Default;
    case Interp1Extrap::Method:  return keel::interp::ExtrapMode::Method;
    case Interp1Extrap::Const:   return keel::interp::ExtrapMode::Constant;
    }
    return keel::interp::ExtrapMode::Default;
}

Value interp1Dispatch(const Value &x, const Value &y, const Value &xq,
                      keel::StringRef method, Interp1Extrap mode,
                      double fill, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    const keel::usize nq = xq.numel();

    if (n < 2)
        throw Error("interp1: need at least 2 data points",
                     0, 0, "interp1", "", "numkit:interp1:tooFewPoints");

    if (y.type() == ValueType::COMPLEX) {
        const keel::usize m = y.numel();
        Value yr = createLike(y, ValueType::DOUBLE, mr);
        Value yi = createLike(y, ValueType::DOUBLE, mr);
        double *rp = yr.doubleDataMut();
        double *ip = yi.doubleDataMut();
        const keel::c64 *c = y.complexData();
        for (keel::usize i = 0; i < m; ++i) { rp[i] = c[i].real(); ip[i] = c[i].imag(); }
        Value rr = interp1Dispatch(x, yr, xq, method, mode, fill, mr);
        Value ri = interp1Dispatch(x, yi, xq, method, mode, fill, mr);
        Value out = createLike(rr, ValueType::COMPLEX, mr);
        keel::c64 *o = out.complexDataMut();
        const double *re = rr.doubleData(), *im = ri.doubleData();
        const keel::usize no = out.numel();
        for (keel::usize i = 0; i < no; ++i) o[i] = keel::c64(re[i], im[i]);
        return out;
    }

    const keel::interp::Interp1Method km = parseInterp1Method(method);
    const keel::interp::ExtrapMode ke = toKeelExtrap(mode);
    const double *xd = x.doubleData();
    const double *xqd = xq.doubleData();

    keel::ArenaTemp scratch = keel::get_scratch(mr);

    const bool yIsVector = y.dims().isVector() || y.isScalar();
    if (yIsVector) {
        if (n != y.numel())
            throw Error("interp1: x and y must have same length",
                         0, 0, "interp1", "", "numkit:interp1:lengthMismatch");
        keel::Array<double> yq(nq, scratch.allocator());
        (void)keel::interp::interp1({xd, n}, {y.doubleData(), n}, {xqd, nq}, {yq.data(), nq},
                            km, ke, fill, scratch.allocator());
        return packInterpResult(yq.data(), yq.size(), xq, mr);
    }

    if (y.dims().ndim() > 2)
        throw Error("interp1: N-D Y arrays are not supported in this "
                    "revision (vector or 2-D matrix only)",
                     0, 0, "interp1", "", "numkit:interp1:ndY");
    const keel::usize yr = keel::as<keel::usize>(y.dims().dim(0));
    const keel::usize yc = keel::as<keel::usize>(y.dims().dim(1));
    if (yr != n)
        throw Error("interp1: for a matrix Y, size(Y,1) must equal length(X)",
                     0, 0, "interp1", "", "numkit:interp1:lengthMismatch");
    const double *yd = y.doubleData();
    auto out = Value::matrix(nq, yc, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (keel::usize j = 0; j < yc; ++j) {
        (void)keel::interp::interp1({xd, n}, {yd + j * yr, n}, {xqd, nq}, {od + j * nq, nq},
                            km, ke, fill, scratch.allocator());
    }
    return out;
}

keel::interp::Interp2Method parseInterp2Method(keel::StringRef m, bool allowSeparable = false)
{
    if (m.empty() || m.equals_ignore_ascii_case("linear"))  return keel::interp::Interp2Method::Linear;
    if (m.equals_ignore_ascii_case("nearest"))              return keel::interp::Interp2Method::Nearest;
    if (m.equals_ignore_ascii_case("cubic"))                return keel::interp::Interp2Method::Cubic;
    if (allowSeparable && m.equals_ignore_ascii_case("spline")) return keel::interp::Interp2Method::Spline;
    if (m.equals_ignore_ascii_case("spline") || m.equals_ignore_ascii_case("pchip") ||
        m.equals_ignore_ascii_case("makima"))
        throw Error(keel::format("interp2: '{}' method not yet supported (linear / nearest / cubic / spline available)", m),
                     0, 0, "interp2", "", "numkit:interp2:unsupportedMethod");
    throw Error(keel::format("interp2: unknown method '{}'", m),
                 0, 0, "interp2", "", "numkit:interp2:badMethod");
}

void validateMonotonicAscending(const double *g, keel::usize n, const char *axis)
{
    if (!keel::interp::is_strictly_increasing({g, n}))
        throw Error(keel::format("interp2: {} must be strictly increasing", axis),
                     0, 0, "interp2", "", "numkit:interp2:notMonotonic");
}

[[maybe_unused]] void readGridAxis(const Value &g, keel::Array<double> &out, const char *axis)
{
    if (g.dims().isVector() || g.isScalar()) {
        out.resize(g.numel());
        for (keel::usize i = 0; i < g.numel(); ++i) out[i] = g.elemAsDouble(i);
        return;
    }
    const keel::usize r = g.dims().rows();
    const keel::usize c = g.dims().cols();
    const keel::usize p = g.dims().is3D() ? g.dims().pages() : 1;
    const keel::usize pageStride = r * c;

    auto extractAlong = [&](int dim) {
        keel::usize n = (dim == 0) ? r : (dim == 1) ? c : p;
        keel::Array<keel::f64> v(n, out.allocator());
        for (keel::usize k = 0; k < n; ++k) {
            keel::usize flat = (dim == 0) ? k
                              : (dim == 1) ? (k * r)
                              : (k * pageStride);
            v[k] = g.elemAsDouble(flat);
        }
        return v;
    };
    auto isConst = [](const keel::Array<double> &v) {
        if (v.size() < 2) return true;
        for (keel::usize i = 1; i < v.size(); ++i)
            if (v[i] != v[0]) return false;
        return true;
    };

    const keel::StringRef ax(axis);
    int preferred = (ax == "X") ? 1 : (ax == "Y") ? 0 : (ax == "Z") ? 2 : 1;
    for (int tries = 0; tries < 3; ++tries) {
        int dim = (preferred + tries) % 3;
        if (dim == 2 && !g.dims().is3D()) continue;
        auto v = extractAlong(dim);
        if (!isConst(v)) {
            out = std::move(v);
            return;
        }
    }
    out = extractAlong(preferred);
}

[[maybe_unused]] Value interp2Impl(const Value &V, const double *xGrid, keel::usize xN,
                                   const double *yGrid, keel::usize yN, const Value &Xq,
                                   const Value &Yq, keel::StringRef method, keel::Allocator *mr)
{
    if (V.type() == ValueType::COMPLEX)
        throw Error("interp2: complex inputs are not supported",
                     0, 0, "interp2", "", "numkit:interp2:complex");
    if (V.dims().is3D() || V.dims().ndim() > 2)
        throw Error("interp2: V must be a 2D matrix",
                     0, 0, "interp2", "", "numkit:interp2:rank");

    const keel::usize R = V.dims().rows();
    const keel::usize C = V.dims().cols();
    if (xN != C)
        throw Error("interp2: length(X) must equal cols(V)",
                     0, 0, "interp2", "", "numkit:interp2:gridSize");
    if (yN != R)
        throw Error("interp2: length(Y) must equal rows(V)",
                     0, 0, "interp2", "", "numkit:interp2:gridSize");
    validateMonotonicAscending(xGrid, C, "X");
    validateMonotonicAscending(yGrid, R, "Y");

    const keel::interp::Interp2Method m = parseInterp2Method(method, /*allowSeparable=*/true);
    if (m == keel::interp::Interp2Method::Cubic) {
        if (R < 3 || C < 3)
            throw Error("interp2: 'cubic' requires at least 3 points in each dimension",
                         0, 0, "interp2", "", "numkit:interp2:cubicSize");
        if (!keel::interp::is_uniform_grid({xGrid, C}) || !keel::interp::is_uniform_grid({yGrid, R}))
            throw Error("interp2: 'cubic' requires a uniformly-spaced grid",
                         0, 0, "interp2", "", "numkit:interp2:cubicNonUniform");
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> Vd(R * C, scratch.allocator());
    if (V.type() == ValueType::DOUBLE)
        std::memcpy(Vd.data(), V.doubleData(), R * C * sizeof(double));
    else
        for (keel::usize i = 0; i < R * C; ++i) Vd[i] = V.elemAsDouble(i);

    const auto &xd = Xq.dims();
    const auto &yd = Yq.dims();
    const bool sameShape = (xd.rows() == yd.rows() && xd.cols() == yd.cols());
    const bool xqIsVec = xd.isVector() || Xq.isScalar();
    const bool yqIsVec = yd.isVector() || Yq.isScalar();

    if (sameShape || Xq.isScalar() || Yq.isScalar()) {
        const Value &shapeRef = (Xq.numel() >= Yq.numel()) ? Xq : Yq;
        const keel::usize nq = shapeRef.numel();
        keel::Array<double> xqv(nq, scratch.allocator());
        keel::Array<double> yqv(nq, scratch.allocator());
        for (keel::usize i = 0; i < nq; ++i) {
            xqv[i] = Xq.isScalar() ? Xq.elemAsDouble(0) : Xq.elemAsDouble(i);
            yqv[i] = Yq.isScalar() ? Yq.elemAsDouble(0) : Yq.elemAsDouble(i);
        }
        auto out = Value::matrix(shapeRef.dims().rows(), shapeRef.dims().cols(), ValueType::DOUBLE, mr);
        (void)keel::interp::interp2_pointwise(Vd.as_ref(), R, C, {xGrid, C}, {yGrid, R},
                                      xqv.as_ref(), yqv.as_ref(), {out.doubleDataMut(), nq},
                                      m, scratch.allocator());
        return out;
    }

    const bool xqRow = (xd.rows() == 1), xqCol = (xd.cols() == 1);
    const bool yqRow = (yd.rows() == 1), yqCol = (yd.cols() == 1);
    if (xqIsVec && yqIsVec && ((xqRow && yqCol) || (xqCol && yqRow))) {
        const keel::usize nx = Xq.numel();
        const keel::usize ny = Yq.numel();
        keel::Array<double> xqv(nx, scratch.allocator());
        keel::Array<double> yqv(ny, scratch.allocator());
        for (keel::usize j = 0; j < nx; ++j) xqv[j] = Xq.elemAsDouble(j);
        for (keel::usize i = 0; i < ny; ++i) yqv[i] = Yq.elemAsDouble(i);
        auto out = Value::matrix(ny, nx, ValueType::DOUBLE, mr);
        (void)keel::interp::interp2_grid(Vd.as_ref(), R, C, {xGrid, C}, {yGrid, R},
                                 xqv.as_ref(), yqv.as_ref(), {out.doubleDataMut(), nx * ny},
                                 m, scratch.allocator());
        return out;
    }

    throw Error("interp2: Xq and Yq must have the same size",
                 0, 0, "interp2", "", "numkit:interp2:queryShape");
}

[[maybe_unused]] Value interp3Impl(const Value &V, const double *xGrid, keel::usize xN,
                                   const double *yGrid, keel::usize yN, const double *zGrid,
                                   keel::usize zN, const Value &Xq, const Value &Yq,
                                   const Value &Zq, keel::StringRef method, keel::Allocator *mr)
{
    if (V.type() == ValueType::COMPLEX)
        throw Error("interp3: complex inputs are not supported",
                     0, 0, "interp3", "", "numkit:interp3:complex");
    if (!V.dims().is3D())
        throw Error("interp3: V must be a 3D array",
                     0, 0, "interp3", "", "numkit:interp3:rank");
    if (Xq.numel() != Yq.numel() || Xq.numel() != Zq.numel())
        throw Error("interp3: Xq, Yq, Zq must have the same numel",
                     0, 0, "interp3", "", "numkit:interp3:queryShape");

    const keel::usize R = V.dims().rows();
    const keel::usize C = V.dims().cols();
    const keel::usize P = V.dims().pages();
    if (xN != C || yN != R || zN != P)
        throw Error("interp3: grid lengths must equal V's dim sizes",
                     0, 0, "interp3", "", "numkit:interp3:gridSize");
    validateMonotonicAscending(xGrid, C, "X");
    validateMonotonicAscending(yGrid, R, "Y");
    validateMonotonicAscending(zGrid, P, "Z");

    const keel::interp::Interp2Method m = parseInterp2Method(method);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> Vd(R * C * P, scratch.allocator());
    if (V.type() == ValueType::DOUBLE)
        std::memcpy(Vd.data(), V.doubleData(), R * C * P * sizeof(double));
    else
        for (keel::usize i = 0; i < R * C * P; ++i) Vd[i] = V.elemAsDouble(i);

    const auto &qd = Xq.dims();
    const keel::usize nq = Xq.numel();
    keel::Array<double> xqv(nq, scratch.allocator());
    keel::Array<double> yqv(nq, scratch.allocator());
    keel::Array<double> zqv(nq, scratch.allocator());
    for (keel::usize i = 0; i < nq; ++i) {
        xqv[i] = Xq.elemAsDouble(i);
        yqv[i] = Yq.elemAsDouble(i);
        zqv[i] = Zq.elemAsDouble(i);
    }
    auto out = Value::matrix(qd.rows(), qd.cols(), ValueType::DOUBLE, mr);
    (void)keel::interp::interp3_pointwise(Vd.as_ref(), R, C, P, {xGrid, C}, {yGrid, R}, {zGrid, P},
                                  xqv.as_ref(), yqv.as_ref(), zqv.as_ref(),
                                  {out.doubleDataMut(), nq}, m);
    return out;
}

} // namespace

} // namespace numkit::builtin
