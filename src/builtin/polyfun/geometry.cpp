// math/src/geom/geom.cpp
//
// Computational-geometry Value adapters delegating to keel::geom (<keel/geom.h>).

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/geom.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/builtin/polyfun.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cstring>

namespace numkit::builtin {

namespace {

keel::Array<keel::f64> toDoubleArray(const Value &v, keel::Allocator *alloc)
{
    const keel::usize n = v.numel();
    keel::Array<keel::f64> out(n, alloc);
    if (v.type() == ValueType::DOUBLE && n > 0) {
        std::memcpy(out.data(), v.doubleData(), n * sizeof(keel::f64));
    } else {
        for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
    }
    return out;
}

} // namespace

// ── polyarea ─────────────────────────────────────────────────────────

Value polyarea(const Value &x, const Value &y, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    if (y.numel() != n)
        throw Error("polyarea: x and y must have the same numel",
                     0, 0, "polyarea", "", "numkit:polyarea:shape");
    if (n < 3)
        return Value::scalar(0.0, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xd = toDoubleArray(x, scratch.allocator());
    auto yd = toDoubleArray(y, scratch.allocator());
    return Value::scalar(keel::geom::polyarea(xd.as_ref(), yd.as_ref()), mr);
}

// ── inpolygon ────────────────────────────────────────────────────────

Value inpolygon(const Value &xq, const Value &yq, const Value &xv,
                const Value &yv, keel::Allocator *mr)
{
    if (xq.numel() != yq.numel())
        throw Error("inpolygon: xq and yq must have the same numel",
                     0, 0, "inpolygon", "", "numkit:inpolygon:queryShape");
    if (xv.numel() != yv.numel())
        throw Error("inpolygon: xv and yv must have the same numel",
                     0, 0, "inpolygon", "", "numkit:inpolygon:polyShape");
    const keel::usize nQ = xq.numel();
    auto out = Value::matrix(xq.dims().rows(), xq.dims().cols(), ValueType::LOGICAL, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xqd = toDoubleArray(xq, scratch.allocator());
    auto yqd = toDoubleArray(yq, scratch.allocator());
    auto xvd = toDoubleArray(xv, scratch.allocator());
    auto yvd = toDoubleArray(yv, scratch.allocator());
    keel::geom::inpolygon(xqd.as_ref(), yqd.as_ref(), xvd.as_ref(), yvd.as_ref(),
                    keel::MutArrayRef<keel::u8>(out.logicalDataMut(), nQ));
    return out;
}

// ── convhull ─────────────────────────────────────────────────────────

Value convhull(const Value &x, const Value &y, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    if (y.numel() != n)
        throw Error("convhull: x and y must have the same numel",
                     0, 0, "convhull", "", "numkit:convhull:shape");
    if (n < 3) {
        auto out = Value::matrix(n + 1, 1, ValueType::DOUBLE, mr);
        double *dst = out.doubleDataMut();
        for (keel::usize i = 0; i < n; ++i) dst[i] = static_cast<double>(i + 1);
        dst[n] = 1.0;
        return out;
    }
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xd = toDoubleArray(x, scratch.allocator());
    auto yd = toDoubleArray(y, scratch.allocator());
    keel::Array<keel::usize> hull = keel::geom::convhull(xd.as_ref(), yd.as_ref(), scratch.allocator());
    auto out = Value::matrix(hull.size(), 1, ValueType::DOUBLE, mr);
    double *dst = out.doubleDataMut();
    for (keel::usize i = 0; i < hull.size(); ++i)
        dst[i] = static_cast<double>(hull[i] + 1);
    return out;
}

// ── boundary ─────────────────────────────────────────────────────────

Value boundary(const Value &x, const Value &y, double shrink, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    if (y.numel() != n)
        throw Error("boundary: x and y must have the same numel",
                     0, 0, "boundary", "", "numkit:boundary:shape");
    if (n < 4)
        return convhull(x, y, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xd = toDoubleArray(x, scratch.allocator());
    auto yd = toDoubleArray(y, scratch.allocator());
    keel::Array<keel::usize> poly =
        keel::geom::boundary(xd.as_ref(), yd.as_ref(), shrink, scratch.allocator());
    auto out = Value::matrix(poly.size(), 1, ValueType::DOUBLE, mr);
    double *dst = out.doubleDataMut();
    for (keel::usize i = 0; i < poly.size(); ++i)
        dst[i] = static_cast<double>(poly[i] + 1);
    return out;
}

// ── delaunay ─────────────────────────────────────────────────────────

Value delaunay(const Value &x, const Value &y, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    if (y.numel() != n)
        throw Error("delaunay: x and y must have the same numel",
                     0, 0, "delaunay", "", "numkit:delaunay:shape");
    if (n < 3)
        return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xd = toDoubleArray(x, scratch.allocator());
    auto yd = toDoubleArray(y, scratch.allocator());
    keel::Array<keel::geom::Triangle> tris =
        keel::geom::delaunay(xd.as_ref(), yd.as_ref(), scratch.allocator());
    const keel::usize M = tris.size();
    auto out = Value::matrix(M, 3, ValueType::DOUBLE, mr);
    double *dst = out.doubleDataMut();
    for (keel::usize i = 0; i < M; ++i) {
        dst[0 * M + i] = static_cast<double>(tris[i].v0 + 1);
        dst[1 * M + i] = static_cast<double>(tris[i].v1 + 1);
        dst[2 * M + i] = static_cast<double>(tris[i].v2 + 1);
    }
    return out;
}

// ── griddata (2-D) ───────────────────────────────────────────────────

Value griddata(const Value &x, const Value &y, const Value &v,
               const Value &xqIn, const Value &yqIn, keel::StringRef methodIn,
               keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    if (y.numel() != n || v.numel() != n)
        throw Error("griddata: x, y, v must have the same numel",
                     0, 0, "griddata", "", "numkit:griddata:shape");

    keel::geom::ScatteredMethod method = keel::geom::ScatteredMethod::Linear;
    if (methodIn.empty() || methodIn.equals_ignore_ascii_case("linear")) {
        method = keel::geom::ScatteredMethod::Linear;
    } else if (methodIn.equals_ignore_ascii_case("nearest")) {
        method = keel::geom::ScatteredMethod::Nearest;
    } else if (methodIn.equals_ignore_ascii_case("v4")) {
        method = keel::geom::ScatteredMethod::BiharmonicV4;
    } else if (methodIn.equals_ignore_ascii_case("natural")) {
        method = keel::geom::ScatteredMethod::Natural;
    } else if (methodIn.equals_ignore_ascii_case("cubic")) {
        method = keel::geom::ScatteredMethod::Cubic;
    } else {
        throw Error(keel::format("griddata: unknown method '{}'", methodIn),
                     0, 0, "griddata", "", "numkit:griddata:badMethod");
    }

    Value xqExpanded, yqExpanded;
    const bool bothVectors = xqIn.dims().isVector() && yqIn.dims().isVector();
    const bool diffOrient  = (xqIn.dims().rows() == 1) != (yqIn.dims().rows() == 1);
    if (bothVectors && (xqIn.numel() != yqIn.numel() || (diffOrient && xqIn.numel() > 1))) {
        const keel::usize rows = yqIn.numel();
        const keel::usize cols = xqIn.numel();
        xqExpanded = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
        yqExpanded = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
        double *xd = xqExpanded.doubleDataMut();
        double *yd = yqExpanded.doubleDataMut();
        for (keel::usize c = 0; c < cols; ++c) {
            const double xv = xqIn.elemAsDouble(c);
            for (keel::usize r = 0; r < rows; ++r) {
                xd[c * rows + r] = xv;
                yd[c * rows + r] = yqIn.elemAsDouble(r);
            }
        }
    }
    const Value &xq = xqExpanded.isUnset() ? xqIn : xqExpanded;
    const Value &yq = yqExpanded.isUnset() ? yqIn : yqExpanded;

    if (xq.numel() != yq.numel())
        throw Error("griddata: xq and yq must have the same numel",
                     0, 0, "griddata", "", "numkit:griddata:queryShape");
    const keel::usize nq = xq.numel();
    auto out = Value::matrix(xq.dims().rows(), xq.dims().cols(), ValueType::DOUBLE, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xd = toDoubleArray(x, scratch.allocator());
    auto yd = toDoubleArray(y, scratch.allocator());
    auto vd = toDoubleArray(v, scratch.allocator());
    auto xqd = toDoubleArray(xq, scratch.allocator());
    auto yqd = toDoubleArray(yq, scratch.allocator());
    (void)keel::geom::griddata(xd.as_ref(), yd.as_ref(), vd.as_ref(), xqd.as_ref(), yqd.as_ref(),
                         keel::MutArrayRef<keel::f64>(out.doubleDataMut(), nq), method,
                         scratch.allocator());
    return out;
}

Value griddata(const Value &x, const Value &y, const Value &v,
               const Value &xq, const Value &yq, keel::Allocator *mr)
{
    return griddata(x, y, v, xq, yq, "linear", mr);
}

// ── histcounts2 ──────────────────────────────────────────────────────

Value histcounts2(const Value &x, const Value &y, const Value &xedgesV,
                  const Value &yedgesV, keel::Allocator *mr)
{
    const keel::usize n = std::min(x.numel(), y.numel());
    if (xedgesV.numel() < 2 || yedgesV.numel() < 2 || n == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);

    const keel::usize nx = xedgesV.numel() - 1;
    const keel::usize ny = yedgesV.numel() - 1;
    auto out = Value::matrix(nx, ny, ValueType::DOUBLE, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto xd = toDoubleArray(x, scratch.allocator());
    auto yd = toDoubleArray(y, scratch.allocator());
    auto xe = toDoubleArray(xedgesV, scratch.allocator());
    auto ye = toDoubleArray(yedgesV, scratch.allocator());
    keel::geom::histcounts2(xd.as_ref(), yd.as_ref(), xe.as_ref(), ye.as_ref(),
                      keel::MutArrayRef<keel::f64>(out.doubleDataMut(), nx * ny));
    return out;
}

// ── griddatan ────────────────────────────────────────────────────────

Value griddatan(const Value &Xv, const Value &vv, const Value &xi,
                keel::StringRef method, keel::Allocator *mr)
{
    const keel::usize m = Xv.dims().rows();
    const keel::usize n = Xv.dims().cols();
    if (vv.numel() != m)
        throw Error("griddatan: length(v) must equal rows(X)",
                     0, 0, "griddatan", "", "numkit:griddatan:shape");
    const keel::usize k    = xi.dims().rows();
    const keel::usize nQry = xi.dims().cols();
    if (nQry != n)
        throw Error("griddatan: cols(xi) must equal cols(X)",
                     0, 0, "griddatan", "", "numkit:griddatan:queryDim");
    auto out = Value::matrix(k, 1, ValueType::DOUBLE, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (method == "nearest") {
        auto xd = toDoubleArray(Xv, scratch.allocator());
        auto vd = toDoubleArray(vv, scratch.allocator());
        auto xid = toDoubleArray(xi, scratch.allocator());
        (void)keel::geom::griddatan_nearest(xd.as_ref(), m, n, vd.as_ref(), xid.as_ref(), k,
                                      keel::MutArrayRef<keel::f64>(out.doubleDataMut(), k));
        return out;
    }

    if (method == "linear") {
        if (n != 2)
            throw Error("griddatan: 'linear' method requires n == 2 (use "
                        "'nearest' for higher dimensions; N-D Delaunay "
                        "is a v1 KNOWN GAP)",
                         0, 0, "griddatan", "",
                         "numkit:griddatan:linearNDUnsupported");
        auto xd = toDoubleArray(Xv, scratch.allocator());
        auto vd = toDoubleArray(vv, scratch.allocator());
        auto xid = toDoubleArray(xi, scratch.allocator());
        (void)keel::geom::griddata(keel::ArrayRef<keel::f64>(xd.data(), m),
                             keel::ArrayRef<keel::f64>(xd.data() + m, m),
                             vd.as_ref(),
                             keel::ArrayRef<keel::f64>(xid.data(), k),
                             keel::ArrayRef<keel::f64>(xid.data() + k, k),
                             keel::MutArrayRef<keel::f64>(out.doubleDataMut(), k),
                             keel::geom::ScatteredMethod::Linear, scratch.allocator());
        return out;
    }

    throw Error(keel::format("griddatan: unknown method '{}' (supported: 'linear', 'nearest')", method),
                 0, 0, "griddatan", "", "numkit:griddatan:badMethod");
}

Value griddatan(const Value &Xv, const Value &vv, const Value &xi, keel::Allocator *mr)
{
    return griddatan(Xv, vv, xi, "linear", mr);
}

// ── matchpairs ───────────────────────────────────────────────────────

MatchpairsResult matchpairs(const Value &C, double cU, keel::StringRef mode,
                            keel::Allocator *mr)
{
    if (C.type() == ValueType::COMPLEX)
        throw Error("matchpairs: complex Cost not supported",
                     0, 0, "matchpairs", "", "numkit:matchpairs:complex");
    bool maximise = false;
    if (mode == "max") maximise = true;
    else if (!mode.empty() && mode != "min")
        throw Error("matchpairs: mode must be 'min' or 'max'",
                     0, 0, "matchpairs", "", "numkit:matchpairs:badMode");

    const keel::usize rows = C.dims().rows();
    const keel::usize cols = C.dims().cols();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto cd = toDoubleArray(C, scratch.allocator());
    keel::geom::MatchpairsOutput mp =
        keel::geom::matchpairs(cd.as_ref(), rows, cols, cU, maximise, scratch.allocator());

    MatchpairsResult result;
    const keel::usize nm = mp.matches.size();
    result.M = Value::matrix(nm, 2, ValueType::DOUBLE, mr);
    double *md = result.M.doubleDataMut();
    for (keel::usize k = 0; k < nm; ++k) {
        md[k] = static_cast<double>(mp.matches[k].row + 1);
        md[nm + k] = static_cast<double>(mp.matches[k].col + 1);
    }
    result.uR = Value::matrix(mp.unmatched_rows.size(), 1, ValueType::DOUBLE, mr);
    for (keel::usize k = 0; k < mp.unmatched_rows.size(); ++k)
        result.uR.doubleDataMut()[k] = static_cast<double>(mp.unmatched_rows[k] + 1);
    result.uC = Value::matrix(mp.unmatched_cols.size(), 1, ValueType::DOUBLE, mr);
    for (keel::usize k = 0; k < mp.unmatched_cols.size(); ++k)
        result.uC.doubleDataMut()[k] = static_cast<double>(mp.unmatched_cols[k] + 1);
    return result;
}

} // namespace numkit::builtin
