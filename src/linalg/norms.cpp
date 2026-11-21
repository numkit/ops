// toolboxes/linalg/src/norms.cpp
//
// Vector / matrix norms â€” implementations and engine adapters.
// Migrated from toolboxes/builtin/src/language/arrays/{matrix,predicates}.cpp.
//
// `norm_value` for the matrix 2-norm routes through the SVD kernel
// in nk/linalg/decompositions.hpp.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <numkit/linalg/norms.hpp>

#include <numkit/linalg/decompositions.hpp> // svd_values
#include "linalg_detail.hpp"           // shared argcheck cores
// Compute-only TU: Value substrate + Error, no engine. The norm/vecnorm
// builtins (CallContext wrappers) live in norms_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace numkit::linalg {

namespace {

bool isVectorShape(const Value &x)
{
    if (x.dims().ndim() != 2) return false;
    return x.dims().dim(0) == 1 || x.dims().dim(1) == 1;
}

inline keel::usize lin(keel::usize i, keel::usize j, keel::usize R) { return i + j * R; }

// Magnitude of linear element i, complex-aware (|z| for COMPLEX, |x| for
// real). Mirrors the getAbs branch in vecnorm so the scalar norm paths norm
// complex inputs by element magnitude, matching MATLAB.
inline keel::f64 absLin(const Value &x, keel::usize i)
{
    return x.isComplex() ? keel::abs(x.complexData()[i])
                         : std::fabs(x.elemAsDouble(i));
}

// vecnorm's own contract: like the float family but logical is rejected
// too (reference-probed 2026-09-18: "First argument must be single or
// double").
void vecnorm_argcheck(const Value &A)
{
    const ValueType t = A.type();
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX
        || t == ValueType::SINGLE)
        return;
    throw Error(keel::format("vecnorm: input 1 (A) must be a single or double array "
                             "(got a {} {} array).",
                             detail::shapeOf(A), mtypeName(t)),
                0, 0, "vecnorm", "", "numkit:vecnorm:inputType");
}

} // anonymous namespace

Value norm_value(const Value &x, keel::f64 p, keel::Allocator *mr)
{
    detail::requireDoubleInput(x, "norm", "x");
    const bool wantSingle = x.type() == ValueType::SINGLE;
    if (x.numel() == 0)
        return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);

    if (isVectorShape(x)) {
        const keel::usize n = x.numel();
        if (p == 2.0) {
            keel::f64 s = 0.0;
            for (keel::usize i = 0; i < n; ++i) { const keel::f64 a = absLin(x, i); s += a * a; }
            return detail::narrowDoubleToSingleIf(Value::scalar(std::sqrt(s), mr), wantSingle, mr);
        } else if (p == 1.0) {
            keel::f64 s = 0.0;
            for (keel::usize i = 0; i < n; ++i) s += absLin(x, i);
            return detail::narrowDoubleToSingleIf(Value::scalar(s, mr), wantSingle, mr);
        } else {
            keel::f64 s = 0.0;
            for (keel::usize i = 0; i < n; ++i) s += std::pow(absLin(x, i), p);
            return detail::narrowDoubleToSingleIf(Value::scalar(std::pow(s, 1.0 / p), mr), wantSingle, mr);
        }
    }

    // Matrix forms.
    if (x.dims().ndim() != 2)
        throw Error("norm: input must be vector or 2D matrix",
                    0, 0, "norm", "", "numkit:norm:badShape");
    const keel::usize m = keel::as<keel::usize>(x.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(x.dims().dim(1));

    if (p == 2.0) {
        auto sv = svd_values(x, mr);
        if (sv.numel() == 0)
            return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);
        return detail::narrowDoubleToSingleIf(Value::scalar(sv.elemAsDouble(0), mr), wantSingle, mr);
    }
    if (p == 1.0) {
        keel::f64 mx = 0.0;
        for (keel::usize j = 0; j < n; ++j) {
            keel::f64 s = 0.0;
            for (keel::usize i = 0; i < m; ++i) s += absLin(x, i + j * m);
            mx = std::max(mx, s);
        }
        return detail::narrowDoubleToSingleIf(Value::scalar(mx, mr), wantSingle, mr);
    }
    throw Error("norm: matrix p-norms only support 1, 2, inf, 'fro'",
                0, 0, "norm", "", "numkit:norm:badP");
}

Value norm_inf(const Value &x, keel::Allocator *mr)
{
    detail::requireDoubleInput(x, "norm", "x");
    const bool wantSingle = x.type() == ValueType::SINGLE;
    if (x.numel() == 0)
        return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);
    if (isVectorShape(x)) {
        const keel::usize n = x.numel();
        keel::f64 mx = 0.0;
        for (keel::usize i = 0; i < n; ++i)
            mx = std::max(mx, absLin(x, i));
        return detail::narrowDoubleToSingleIf(Value::scalar(mx, mr), wantSingle, mr);
    }
    // Matrix inf-norm: max row sum.
    if (x.dims().ndim() != 2)
        throw Error("norm: input must be vector or 2D matrix",
                    0, 0, "norm", "", "numkit:norm:badShape");
    const keel::usize m = keel::as<keel::usize>(x.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(x.dims().dim(1));
    keel::f64 mx = 0.0;
    for (keel::usize i = 0; i < m; ++i) {
        keel::f64 s = 0.0;
        for (keel::usize j = 0; j < n; ++j) s += absLin(x, i + j * m);
        mx = std::max(mx, s);
    }
    return detail::narrowDoubleToSingleIf(Value::scalar(mx, mr), wantSingle, mr);
}

Value norm_fro(const Value &x, keel::Allocator *mr)
{
    detail::requireDoubleInput(x, "norm", "x");
    const bool wantSingle = x.type() == ValueType::SINGLE;
    const keel::usize n = x.numel();
    if (n == 0)
        return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);
    keel::f64 s = 0.0;
    for (keel::usize i = 0; i < n; ++i) { const keel::f64 a = absLin(x, i); s += a * a; }
    return detail::narrowDoubleToSingleIf(Value::scalar(std::sqrt(s), mr), wantSingle, mr);
}

Value vecnorm(const Value &A, keel::f64 p, keel::i32 dim, keel::Allocator *mr)
{
    vecnorm_argcheck(A);
    if (A.dims().is3D())
        throw Error("vecnorm: 3-D arrays not supported",
                    0, 0, "vecnorm", "", "numkit:vecnorm:3D");
    const bool wantSingle = A.type() == ValueType::SINGLE;
    const keel::usize R = A.dims().rows();
    const keel::usize C = A.dims().cols();

    // Default dim: first non-singleton (1-based). Empties default to 1.
    if (dim == 0) {
        if (R == 0 && C == 0) dim = 1;
        else if (R != 1)      dim = 1;
        else                  dim = 2;
    }
    if (dim != 1 && dim != 2)
        throw Error("vecnorm: dim must be 1 or 2",
                    0, 0, "vecnorm", "", "numkit:vecnorm:BadDim");

    auto p_norm = [&](auto getAbs, keel::usize n) -> keel::f64 {
        if (n == 0) return 0.0;
        if (std::isinf(p) && p > 0) {
            keel::f64 m = 0.0;
            for (keel::usize k = 0; k < n; ++k) {
                keel::f64 v = getAbs(k);
                if (std::isnan(v)) return std::numeric_limits<keel::f64>::quiet_NaN();
                if (v > m) m = v;
            }
            return m;
        }
        if (std::isinf(p) && p < 0) {
            keel::f64 m = std::numeric_limits<keel::f64>::infinity();
            for (keel::usize k = 0; k < n; ++k) {
                keel::f64 v = getAbs(k);
                if (std::isnan(v)) return std::numeric_limits<keel::f64>::quiet_NaN();
                if (v < m) m = v;
            }
            return m;
        }
        if (p == 2.0) {
            keel::f64 s = 0.0;
            for (keel::usize k = 0; k < n; ++k) {
                keel::f64 v = getAbs(k);
                s += v * v;
            }
            return std::sqrt(s);
        }
        keel::f64 s = 0.0;
        for (keel::usize k = 0; k < n; ++k) s += std::pow(getAbs(k), p);
        return std::pow(s, 1.0 / p);
    };

    auto getAbs = [&](keel::usize i, keel::usize j) -> keel::f64 {
        if (A.isComplex()) return keel::abs(A.complexElem(i, j));
        return keel::abs(A.elemAsDouble(lin(i, j, R)));
    };

    if (R == 0 && C == 0)
        return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);

    if (dim == 1) {
        Value out = Value::matrix(1, C, ValueType::DOUBLE, mr);
        if (C == 0) return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize j = 0; j < C; ++j) {
            od[j] = p_norm([&](keel::usize k){ return getAbs(k, j); }, R);
        }
        return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
    }
    Value out = Value::matrix(R, 1, ValueType::DOUBLE, mr);
    if (R == 0) return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < R; ++i) {
        od[i] = p_norm([&](keel::usize k){ return getAbs(i, k); }, C);
    }
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Engine adapters â€” registered in LinalgLibrary::install
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

} // namespace numkit::linalg
