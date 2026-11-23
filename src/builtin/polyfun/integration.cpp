// toolboxes/builtin/src/math/integration/integration.cpp
//
// Numerical-calculus builtins:
//   - gradient / gradient2 — central differences (vector + 2D matrix)
//   - cumtrapz             — cumulative trapezoidal integration (1-D)
//   - integral             — adaptive Gauss-Kronrod definite integral
// fzero lives in math/optim/fzero.cpp (uses the same callback helper).

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <keel/tuple.h>
#include <numkit/builtin/polyfun.hpp>
#include "integration_detail.hpp"

#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/ops/callback_eval.hpp>

#include <algorithm>

namespace numkit::builtin {

namespace cb = ::numkit::ops;

using namespace detail;   // integration impl-helpers (gradientND/cumtrapz*/trapzImpl/...)

namespace detail {

// One-pass forward/central/backward difference along a contiguous slice
// of length n with stride 1 (delegates to keel::quad::gradient_strided).
void gradient1D(const keel::f64 *src, keel::f64 *dst, keel::usize n, keel::f64 h)
{
    keel::quad::gradient_strided(src, dst, n, 1, h);
}

// 2D gradient along dim-2 (columns) for a matrix in column-major layout.
void gradientAlongCols(const keel::f64 *src, keel::f64 *dst, keel::usize R, keel::usize C, keel::f64 h)
{
    if (C == 0) return;
    for (keel::usize r = 0; r < R; ++r)
        keel::quad::gradient_strided(src + r, dst + r, C, R, h);
}

// 2D gradient along dim-1 (rows) for a matrix in column-major layout.
void gradientAlongRows(const keel::f64 *src, keel::f64 *dst, keel::usize R, keel::usize C, keel::f64 h)
{
    if (R == 0) return;
    for (keel::usize c = 0; c < C; ++c)
        keel::quad::gradient_strided(src + c * R, dst + c * R, R, 1, h);
}

// Gradient along an arbitrary dimension `dim` (0-based) of an N-D array in
// column-major layout. Central differences on the interior, one-sided at the
// ends, uniform scalar spacing h. A singleton dimension yields all zeros.
void gradientAlongDim(const keel::f64 *src, keel::f64 *dst, const Dims &shape,
                      keel::i32 dim, keel::f64 h)
{
    const keel::usize N = shape.numel();
    if (N == 0) return;
    const keel::usize L = shape.dim(dim);
    if (L <= 1) { for (keel::usize i = 0; i < N; ++i) dst[i] = 0.0; return; }
    keel::usize inner = 1;
    for (keel::i32 k = 0; k < dim; ++k) inner *= shape.dim(k);
    const keel::usize block = L * inner;
    const keel::usize outer = N / block;
    for (keel::usize o = 0; o < outer; ++o) {
        for (keel::usize in = 0; in < inner; ++in) {
            const keel::usize base = o * block + in;
            keel::quad::gradient_strided(src + base, dst + base, L, inner, h);
        }
    }
}

// One dimension's discrete-Laplacian term (g) along a strided line of
// length n (element stride s, uniform spacing h). Delegates to keel::quad.
void del2Line(const keel::f64 *src, keel::f64 *dst, keel::usize n, keel::usize s, keel::f64 h)
{
    keel::quad::del2_strided(src, dst, n, s, h);
}

Value toDoubleCopy(const Value &x, keel::Allocator *mr)
{
    auto r = createLike(x, ValueType::DOUBLE, mr);
    if (x.type() == ValueType::DOUBLE) {
        keel::copy_n(r.doubleDataMut(), x.doubleData(), x.numel());
    } else {
        keel::f64 *dst = r.doubleDataMut();
        for (keel::usize i = 0; i < x.numel(); ++i)
            dst[i] = x.elemAsDouble(i);
    }
    return r;
}

} // namespace

namespace detail {
// Real / imaginary parts of a COMPLEX value as same-shape DOUBLE arrays, and
// the inverse combine. (toDoubleCopy can't be used on complex — elemAsDouble
// rejects a nonzero imaginary part.)
Value realPartCopy(const Value &f, keel::Allocator *mr)
{
    auto r = createLike(f, ValueType::DOUBLE, mr);
    keel::f64 *d = r.doubleDataMut();
    const keel::c64 *c = f.complexData();
    const keel::usize n = f.numel();
    for (keel::usize i = 0; i < n; ++i) d[i] = c[i].real();
    return r;
}
Value imagPartCopy(const Value &f, keel::Allocator *mr)
{
    auto r = createLike(f, ValueType::DOUBLE, mr);
    keel::f64 *d = r.doubleDataMut();
    const keel::c64 *c = f.complexData();
    const keel::usize n = f.numel();
    for (keel::usize i = 0; i < n; ++i) d[i] = c[i].imag();
    return r;
}
Value combineComplexParts(const Value &re, const Value &im, keel::Allocator *mr)
{
    auto out = createLike(re, ValueType::COMPLEX, mr);
    keel::c64 *o = out.complexDataMut();
    const keel::f64 *r = re.doubleData(), *m = im.doubleData();
    const keel::usize n = out.numel();
    for (keel::usize i = 0; i < n; ++i) o[i] = keel::c64(r[i], m[i]);
    return out;
}
} // namespace

Value gradient(const Value &f, keel::f64 h, keel::Allocator *mr)
{
    if (h <= 0)
        throw Error("gradient: spacing h must be positive",
                     0, 0, "gradient", "", "numkit:gradient:badSpacing");
    // Complex: gradient real + imaginary parts separately, recombine (MATLAB).
    if (f.type() == ValueType::COMPLEX) {
        Value gr = gradient(realPartCopy(f, mr), h, mr);
        Value gi = gradient(imagPartCopy(f, mr), h, mr);
        return combineComplexParts(gr, gi, mr);
    }

    auto src = toDoubleCopy(f, mr);
    auto out = createLike(f, ValueType::DOUBLE, mr);
    const auto &d = f.dims();

    if (f.dims().isVector() || f.isScalar()) {
        gradient1D(src.doubleData(), out.doubleDataMut(), f.numel(), h);
        return out;
    }
    if (d.is3D() || d.ndim() > 2) {
        // N-D single output = gradient along the dim-2 (x / column) direction,
        // i.e. 0-based dim 1, matching MATLAB. (bugs/builtin/gradient-3d.md)
        gradientAlongDim(src.doubleData(), out.doubleDataMut(), d, 1, h);
        return out;
    }
    gradientAlongCols(src.doubleData(), out.doubleDataMut(),
                      d.rows(), d.cols(), h);
    return out;
}

Value del2(const Value &u, keel::f64 h, keel::Allocator *mr)
{
    if (h <= 0)
        throw Error("del2: spacing h must be positive",
                     0, 0, "del2", "", "numkit:del2:badSpacing");
    if (u.type() == ValueType::COMPLEX)
        throw Error("del2: complex inputs are not supported",
                     0, 0, "del2", "", "numkit:del2:complex");
    const auto &d = u.dims();
    if (d.is3D() || d.ndim() > 2)
        throw Error("del2: only 1D vector and 2D matrix inputs are supported",
                     0, 0, "del2", "", "numkit:del2:rank");

    auto src = toDoubleCopy(u, mr);
    auto out = createLike(u, ValueType::DOUBLE, mr);
    const keel::f64 *sd = src.doubleData();
    keel::f64 *od = out.doubleDataMut();
    const keel::usize N = u.numel();
    for (keel::usize i = 0; i < N; ++i) od[i] = 0.0;
    if (N == 0) return out;

    // MATLAB divides the summed per-dimension Laplacian by ndims(U); a
    // vector still has ndims == 2, so the divisor is 2 for both shapes.
    if (u.dims().isVector() || u.isScalar()) {
        keel::ArenaTemp arena = keel::get_scratch(mr);
        auto g = keel::Array<keel::f64>(arena.allocator()); g.assign(N, 0.0);
        del2Line(sd, g.data(), N, 1, h);
        for (keel::usize i = 0; i < N; ++i) od[i] = g[i] * 0.5;
        return out;
    }
    const keel::usize R = d.rows(), C = d.cols();
    keel::ArenaTemp arena = keel::get_scratch(mr);
    auto g1 = keel::Array<keel::f64>(arena.allocator()); g1.assign(N, 0.0);
    auto g2 = keel::Array<keel::f64>(arena.allocator()); g2.assign(N, 0.0);
    // dim-1: down each column (stride 1, length R).
    for (keel::usize c = 0; c < C; ++c) del2Line(sd + c * R, g1.data() + c * R, R, 1, h);
    // dim-2: across each row (stride R, length C).
    for (keel::usize r = 0; r < R; ++r) del2Line(sd + r, g2.data() + r, C, R, h);
    for (keel::usize i = 0; i < N; ++i) od[i] = (g1[i] + g2[i]) * 0.5;
    return out;
}

keel::Tuple<Value, Value>
gradient2(const Value &f, keel::f64 hx, keel::f64 hy, keel::Allocator *mr)
{
    if (hx <= 0 || hy <= 0)
        throw Error("gradient: spacing arguments must be positive",
                     0, 0, "gradient", "", "numkit:gradient:badSpacing");
    // Complex: gradient real + imaginary parts separately, recombine (MATLAB).
    if (f.type() == ValueType::COMPLEX) {
        auto [grx, gry] = gradient2(realPartCopy(f, mr), hx, hy, mr);
        auto [gix, giy] = gradient2(imagPartCopy(f, mr), hx, hy, mr);
        return keel::make_tuple(combineComplexParts(grx, gix, mr),
                               combineComplexParts(gry, giy, mr));
    }
    const auto &d = f.dims();
    if (d.is3D() || d.ndim() > 2)
        throw Error("gradient: 2-output form requires a 2D matrix input",
                     0, 0, "gradient", "", "numkit:gradient:rank");

    auto src = toDoubleCopy(f, mr);
    auto fx = createLike(f, ValueType::DOUBLE, mr);
    auto fy = createLike(f, ValueType::DOUBLE, mr);

    if (f.dims().isVector() || f.isScalar()) {
        gradient1D(src.doubleData(), fx.doubleDataMut(), f.numel(), hx);
        gradient1D(src.doubleData(), fy.doubleDataMut(), f.numel(), hy);
        return keel::make_tuple(keel::move(fx), keel::move(fy));
    }
    gradientAlongCols(src.doubleData(), fx.doubleDataMut(),
                      d.rows(), d.cols(), hx);
    gradientAlongRows(src.doubleData(), fy.doubleDataMut(),
                      d.rows(), d.cols(), hy);
    return keel::make_tuple(keel::move(fx), keel::move(fy));
}

namespace detail {
// N-D gradient: emit `nout` arrays. Output o is taken along dimension perm(o)
// where perm = {1, 0, 2, 3, ...} (0-based) — i.e. out0 = dim-2 (x), out1 =
// dim-1 (y), out_k = dim-(k+1) for k>=2, matching MATLAB. `hs[o]` is the
// scalar spacing for output o. Complex F is gradiented part-wise then
// recombined. Throws if more outputs than dimensions are requested.
keel::Array<Value> gradientND(const Value &f, const keel::f64 *hs, keel::usize nh,
                                   keel::usize nout, keel::Allocator *mr)
{
    if (f.type() == ValueType::COMPLEX) {
        auto re = gradientND(realPartCopy(f, mr), hs, nh, nout, mr);
        auto im = gradientND(imagPartCopy(f, mr), hs, nh, nout, mr);
        keel::Array<Value> out(mr);
        out.reserve(nout);
        for (keel::usize o = 0; o < nout; ++o)
            out.push_back(combineComplexParts(re[o], im[o], mr));
        return out;
    }

    const Dims &shape = f.dims();
    const keel::i32 R = shape.ndim();
    auto src = toDoubleCopy(f, mr);
    keel::Array<Value> outs(mr);
    outs.reserve(nout);
    for (keel::usize o = 0; o < nout; ++o) {
        const keel::i32 dim = (o == 0) ? 1 : (o == 1) ? 0 : keel::as<keel::i32>(o);
        if (dim >= R)
            throw Error("gradient: too many output arguments for the input "
                        "dimensionality",
                        0, 0, "gradient", "", "numkit:gradient:nargout");
        const keel::f64 h = (o < nh) ? hs[o] : 1.0;
        if (h <= 0)
            throw Error("gradient: spacing arguments must be positive",
                        0, 0, "gradient", "", "numkit:gradient:badSpacing");
        Value out = createLike(f, ValueType::DOUBLE, mr);
        gradientAlongDim(src.doubleData(), out.doubleDataMut(), shape, dim, h);
        outs.push_back(keel::move(out));
    }
    return outs;
}
} // namespace

// ── cumtrapz ─────────────────────────────────────────────────────────
namespace detail {

Value cumtrapzVector(const keel::f64 *y, const keel::f64 *x, keel::usize n, const Dims &shape, bool unitSpacing, keel::Allocator *mr)
{
    auto out = Value::matrix(shape.rows(), shape.cols(), ValueType::DOUBLE, mr);
    keel::quad::cumtrapz_strided(y, out.doubleDataMut(), n, 1, 1, unitSpacing ? nullptr : x, 1);
    return out;
}

} // namespace

namespace detail {

// Matrix form: integrate along columns. xData==nullptr -> unit spacing.
// xData layout: same shape as src, used per-column.
Value cumtrapzMatrixCols(const keel::f64 *src, const keel::f64 *xData, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();
    if (rows == 0 || cols == 0) return out;
    for (keel::usize c = 0; c < cols; ++c) {
        const keel::f64 *xCol = xData ? (xData + c * rows) : nullptr;
        keel::quad::cumtrapz_strided(src + c * rows, dst + c * rows, rows, 1, 1, xCol, 1);
    }
    return out;
}

// Row-wise (dim 2) cumulative trapezoid: each row integrated across cols.
Value cumtrapzMatrixRows(const keel::f64 *src, const keel::f64 *xData, keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    keel::f64 *dst = out.doubleDataMut();
    if (rows == 0 || cols == 0) return out;
    for (keel::usize r = 0; r < rows; ++r) {
        const keel::f64 *xRow = xData ? (xData + r) : nullptr;
        keel::quad::cumtrapz_strided(src + r, dst + r, cols, rows, rows, xRow, rows);
    }
    return out;
}

// ── COMPLEX counterparts (the integration variable x stays real) ────────
Value cumtrapzVectorC(const keel::c64 *y, const keel::f64 *x, keel::usize n, const Dims &shape,
                      bool unitSpacing, keel::Allocator *mr)
{
    auto out = Value::matrix(shape.rows(), shape.cols(), ValueType::COMPLEX, mr);
    keel::quad::cumtrapz_strided_c64(y, out.complexDataMut(), n, 1, 1, unitSpacing ? nullptr : x, 1);
    return out;
}

Value cumtrapzMatrixColsC(const keel::c64 *src, const keel::f64 *xData, keel::usize rows, keel::usize cols,
                          keel::Allocator *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::COMPLEX, mr);
    keel::c64 *dst = out.complexDataMut();
    if (rows == 0 || cols == 0) return out;
    for (keel::usize c = 0; c < cols; ++c) {
        const keel::f64 *xCol = xData ? (xData + c * rows) : nullptr;
        keel::quad::cumtrapz_strided_c64(src + c * rows, dst + c * rows, rows, 1, 1, xCol, 1);
    }
    return out;
}

Value cumtrapzMatrixRowsC(const keel::c64 *src, const keel::f64 *xData, keel::usize rows, keel::usize cols,
                          keel::Allocator *mr)
{
    auto out = Value::matrix(rows, cols, ValueType::COMPLEX, mr);
    keel::c64 *dst = out.complexDataMut();
    if (rows == 0 || cols == 0) return out;
    for (keel::usize r = 0; r < rows; ++r) {
        const keel::f64 *xRow = xData ? (xData + r) : nullptr;
        keel::quad::cumtrapz_strided_c64(src + r, dst + r, cols, rows, rows, xRow, rows);
    }
    return out;
}

} // namespace

// cumtrapz(Y, dim): unit-spacing cumulative trapezoid along dim (1 or 2).
// A vector is treated as the matrix it is (1×N or N×1): integrating along
// the singleton dimension is a no-op (all zeros), matching MATLAB — e.g.
// cumtrapz([1 2 3 4], 1) → [0 0 0 0], cumtrapz([1 2 3 4], 2) → cumulative.
Value cumtrapzDim(const Value &y, keel::i32 dim, keel::Allocator *mr)
{
    const keel::usize rows = y.dims().rows(), cols = y.dims().cols();
    if (dim <= 0) dim = 1;
    if (y.type() == ValueType::COMPLEX) {
        const keel::c64 *yc = y.complexData();
        if (dim == 2) return cumtrapzMatrixRowsC(yc, nullptr, rows, cols, mr);
        return cumtrapzMatrixColsC(yc, nullptr, rows, cols, mr);
    }
    auto ys = toDoubleCopy(y, mr);
    if (dim == 2)
        return cumtrapzMatrixRows(ys.doubleData(), nullptr, rows, cols, mr);
    return cumtrapzMatrixCols(ys.doubleData(), nullptr, rows, cols, mr);
}

Value cumtrapz(const Value &y, keel::Allocator *mr)
{
    if (y.type() == ValueType::COMPLEX) {
        const keel::c64 *yc = y.complexData();
        if (y.dims().isVector() || y.isScalar())
            return cumtrapzVectorC(yc, nullptr, y.numel(), y.dims(), /*unitSpacing=*/true, mr);
        return cumtrapzMatrixColsC(yc, nullptr, y.dims().rows(), y.dims().cols(), mr);
    }
    auto ys = toDoubleCopy(y, mr);
    if (y.dims().isVector() || y.isScalar()) {
        return cumtrapzVector(ys.doubleData(), nullptr, y.numel(), y.dims(), /*unitSpacing=*/true, mr);
    }
    return cumtrapzMatrixCols(ys.doubleData(), nullptr, y.dims().rows(), y.dims().cols(), mr);
}

Value cumtrapz(const Value &x, const Value &y, keel::Allocator *mr)
{
    // The X coordinate must be real; Y may be complex.
    if (x.type() == ValueType::COMPLEX)
        throw Error("cumtrapz: the X coordinate must be real",
                     0, 0, "cumtrapz", "", "numkit:cumtrapz:complexX");
    const bool yIsC = (y.type() == ValueType::COMPLEX);
    Value ys;
    if (!yIsC) ys = toDoubleCopy(y, mr);
    const keel::f64 *yd = yIsC ? nullptr : ys.doubleData();
    const keel::c64 *yc = yIsC ? y.complexData() : nullptr;

    if (y.dims().isVector() || y.isScalar()) {
        if (!x.dims().isVector() && !x.isScalar())
            throw Error("cumtrapz: when y is a vector, x must also be a vector",
                         0, 0, "cumtrapz", "", "numkit:cumtrapz:shapeMismatch");
        if (x.numel() != y.numel())
            throw Error("cumtrapz: x and y must have the same length",
                         0, 0, "cumtrapz", "", "numkit:cumtrapz:lengthMismatch");
        auto xs = toDoubleCopy(x, mr);
        return yIsC ? cumtrapzVectorC(yc, xs.doubleData(), y.numel(), y.dims(), /*unitSpacing=*/false, mr)
                    : cumtrapzVector(yd, xs.doubleData(), y.numel(), y.dims(), /*unitSpacing=*/false, mr);
    }

    // Matrix y. x may be a vector (broadcast across every column) or
    // matrix of the same shape as y (per-column spacing).
    const keel::usize rows = y.dims().rows();
    const keel::usize cols = y.dims().cols();
    auto xs = toDoubleCopy(x, mr);
    if (x.dims().isVector() && x.numel() == rows) {
        // Broadcast x to every column.
        auto xMat = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
        keel::f64 *dx = xMat.doubleDataMut();
        const keel::f64 *src = xs.doubleData();
        for (keel::usize c = 0; c < cols; ++c)
            keel::copy_n(dx + c * rows, src, rows);
        return yIsC ? cumtrapzMatrixColsC(yc, dx, rows, cols, mr)
                    : cumtrapzMatrixCols(yd, dx, rows, cols, mr);
    }
    if (x.dims().rows() != rows || x.dims().cols() != cols)
        throw Error("cumtrapz: x size must match y or be a column-length vector",
                     0, 0, "cumtrapz", "", "numkit:cumtrapz:shapeMismatch");
    return yIsC ? cumtrapzMatrixColsC(yc, xs.doubleData(), rows, cols, mr)
                : cumtrapzMatrixCols(yd, xs.doubleData(), rows, cols, mr);
}

// ── integral (adaptive Gauss-Kronrod via keel::quad) ───────────────────
namespace detail {

static keel::c64 evalScalarOrComplex(FnHandle fn, keel::f64 x, bool &sawComplex,
                                   keel::Allocator *mr)
{
    Value in = Value::scalar(x, mr);
    Value out[1];
    fn(keel::ArrayRef<Value>(&in, 1), keel::MutArrayRef<Value>(out, 1), mr);
    if (out[0].type() == ValueType::COMPLEX) {
        sawComplex = true;
        return out[0].toComplex();
    }
    return keel::c64(out[0].toScalar(), 0.0);
}

} // namespace

Value integral(FnHandle fn, keel::f64 a, keel::f64 b, keel::f64 absTol,
               keel::Allocator *mr)
{
    if (!keel::is_finite(a) || !keel::is_finite(b))
        throw Error("integral: bounds must be finite",
                     0, 0, "integral", "", "numkit:integral:badBounds");
    if (absTol <= 0)
        throw Error("integral: absTol must be positive",
                     0, 0, "integral", "", "numkit:integral:badTol");
    if (a == b) return Value::scalar(0.0, mr);
    constexpr keel::i32 kMaxDepth = 20;
    bool sawComplex = false;
    auto eval_fn = [&](keel::f64 x) -> keel::c64 {
        return detail::evalScalarOrComplex(fn, x, sawComplex, mr);
    };
    const auto res = keel::quad::integral_c64(eval_fn, a, b, absTol, kMaxDepth);
    if (sawComplex && res.value.imag() != 0.0)
        return Value::complexScalar(res.value.real(), res.value.imag(), mr);
    return Value::scalar(res.value.real(), mr);
}

Value integral2(FnHandle fn, keel::f64 a, keel::f64 b, keel::f64 c, keel::f64 d,
                keel::f64 absTol, keel::Allocator *mr)
{
    if (!keel::is_finite(a) || !keel::is_finite(b) || !keel::is_finite(c) || !keel::is_finite(d))
        throw Error("integral2: bounds must be finite",
                     0, 0, "integral2", "", "numkit:integral2:badBounds");
    if (absTol <= 0)
        throw Error("integral2: absTol must be positive",
                     0, 0, "integral2", "", "numkit:integral2:badTol");
    if (a == b || c == d) return Value::scalar(0.0, mr);
    constexpr keel::i32 kMaxDepth = 18;
    bool sawComplex = false;

    auto batch_fn = [&](keel::ArrayRef<keel::f64> xs, keel::ArrayRef<keel::f64> ys,
                        keel::MutArrayRef<keel::c64> out_buf) -> bool {
        try {
            Value X = Value::matrix(15, 15, ValueType::DOUBLE, mr);
            Value Y = Value::matrix(15, 15, ValueType::DOUBLE, mr);
            keel::copy_n(X.doubleDataMut(), xs.data(), 225);
            keel::copy_n(Y.doubleDataMut(), ys.data(), 225);
            Value xy[2] = {keel::move(X), keel::move(Y)};
            Value out[1];
            fn(keel::ArrayRef<Value>(xy, 2), keel::MutArrayRef<Value>(out, 1), mr);
            if (out[0].numel() == 225) {
                if (out[0].type() == ValueType::COMPLEX) {
                    sawComplex = true;
                    keel::copy_n(out_buf.data(), out[0].complexData(), 225);
                    return true;
                }
                if (out[0].type() == ValueType::DOUBLE) {
                    const keel::f64 *dd = out[0].doubleData();
                    for (keel::usize k = 0; k < 225; ++k) out_buf[k] = keel::c64(dd[k], 0.0);
                    return true;
                }
            } else if (out[0].isScalar()) {
                // Constant function f(X,Y) = const
                keel::c64 cv = (out[0].type() == ValueType::COMPLEX)
                    ? (sawComplex = true, out[0].toComplex())
                    : keel::c64(out[0].toScalar(), 0.0);
                for (keel::usize k = 0; k < 225; ++k) out_buf[k] = cv;
                return true;
            }
            return false;
        } catch (...) {
            return false;
        }
    };

    auto scalar_fn = [&](keel::f64 xv, keel::f64 yv) -> keel::c64 {
        Value xy[2] = {Value::scalar(xv, mr), Value::scalar(yv, mr)};
        Value out[1];
        fn(keel::ArrayRef<Value>(xy, 2), keel::MutArrayRef<Value>(out, 1), mr);
        if (out[0].type() == ValueType::COMPLEX) {
            sawComplex = true;
            return out[0].toComplex();
        }
        return keel::c64(out[0].toScalar(), 0.0);
    };

    const auto res = keel::quad::integral2_batch_c64(batch_fn, scalar_fn, a, b, c, d, absTol, kMaxDepth);
    if (sawComplex && res.value.imag() != 0.0)
        return Value::complexScalar(res.value.real(), res.value.imag(), mr);
    return Value::scalar(res.value.real(), mr);
}

Value integral3(FnHandle fn, keel::f64 a, keel::f64 b, keel::f64 c, keel::f64 d,
                keel::f64 e, keel::f64 f, keel::f64 absTol,
                keel::Allocator *mr)
{
    keel::f64 xval = 0.0;
    auto inner2D = [&fn, &xval](keel::ArrayRef<Value> ayz, keel::MutArrayRef<Value> oyz,
                                keel::Allocator *m) {
        if (ayz[0].numel() > 1) {
            Value X = Value::matrix(ayz[0].dims().rows(), ayz[0].dims().cols(), ValueType::DOUBLE, m);
            keel::f64 *xd = X.doubleDataMut();
            for (keel::usize k = 0; k < X.numel(); ++k) xd[k] = xval;
            Value xyz[3] = { keel::move(X), ayz[0], ayz[1] };
            fn(keel::ArrayRef<Value>(xyz, 3), oyz, m);
        } else {
            Value xyz[3] = { Value::scalar(xval, m), ayz[0], ayz[1] };
            fn(keel::ArrayRef<Value>(xyz, 3), oyz, m);
        }
    };
    auto outerFn = [&](keel::ArrayRef<Value> ax, keel::MutArrayRef<Value> ox,
                       keel::Allocator *m) {
        xval = ax[0].toScalar();
        ox[0] = integral2(inner2D, c, d, e, f, absTol, m);
    };
    return integral(outerFn, a, b, absTol, mr);
}

namespace detail {

// Trapezoidal integral of `y` along 1-based `dim`. `xData` (length = size of
// y along dim) gives non-uniform spacing; nullptr = unit spacing. A vector
// reduces to a scalar; a matrix reduces along `dim` (MATLAB trapz semantics:
// trapz(M) = 1xC over columns, trapz(M,2) = Rx1 over rows).
Value trapzImpl(const Value &y, keel::i32 dim, const keel::f64 *xData,
                keel::Allocator *mr)
{
    const keel::usize R = y.dims().rows(), C = y.dims().cols();
    if (y.type() == ValueType::COMPLEX) {
        const keel::c64 *yc = y.complexData();
        if (dim != 2) {
            Value out = Value::matrix(1, C, ValueType::COMPLEX, mr);
            keel::c64 *o = out.complexDataMut();
            for (keel::usize c = 0; c < C; ++c)
                o[c] = keel::quad::trapz_strided_c64(yc + c * R, R, 1, xData, 1);
            return out;
        }
        Value out = Value::matrix(R, 1, ValueType::COMPLEX, mr);
        keel::c64 *o = out.complexDataMut();
        for (keel::usize r = 0; r < R; ++r)
            o[r] = keel::quad::trapz_strided_c64(yc + r, C, R, xData, 1);
        return out;
    }
    const keel::f64 *yd = y.doubleData();
    if (dim != 2) {  // dim 1 (default for column reduction)
        Value out = Value::matrix(1, C, ValueType::DOUBLE, mr);
        keel::f64 *o = out.doubleDataMut();
        for (keel::usize c = 0; c < C; ++c)
            o[c] = keel::quad::trapz_strided(yd + c * R, R, 1, xData, 1);
        return out;
    }
    Value out = Value::matrix(R, 1, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize r = 0; r < R; ++r)
        o[r] = keel::quad::trapz_strided(yd + r, C, R, xData, 1);
    return out;
}

keel::i32 trapzFirstNonSingletonDim(const Value &y)
{
    const auto &d = y.dims();
    for (keel::i32 k = 0; k < d.ndim(); ++k)
        if (d.dim(k) > 1) return k + 1;
    return 1;
}

} // namespace

// ── trapz (uniform / explicit spacing) — public compute ────────
Value trapz(const Value &y, keel::Allocator *mr)
{
    if (y.numel() == 0) return Value::scalar(0.0, mr);
    // Integrate along the first non-singleton dimension (MATLAB): a vector
    // reduces to a scalar, a matrix reduces over its columns.
    return trapzImpl(y, trapzFirstNonSingletonDim(y), nullptr, mr);
}

Value trapz(const Value &x, const Value &y, keel::Allocator *mr)
{
    const keel::i32 dim = trapzFirstNonSingletonDim(y);
    const keel::usize along = (dim != 2) ? y.dims().rows() : y.dims().cols();
    if (x.numel() != along)
        throw Error("trapz: x and y must have same length",
                     0, 0, "trapz", "", "numkit:trapz:lengthMismatch");
    return trapzImpl(y, dim, x.doubleData(), mr);
}

} // namespace numkit::builtin
