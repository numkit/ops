// toolboxes/linalg/src/norms.cpp
//
// Vector / matrix norms â€” implementations and engine adapters.
// Migrated from toolboxes/builtin/src/language/arrays/{matrix,predicates}.cpp.
//
// `norm_value` for the matrix 2-norm routes through the SVD kernel
// in nk/linalg/decompositions.hpp.

#include <keel/core.h>
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
inline double absLin(const Value &x, keel::usize i)
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
    throw Error("vecnorm: input 1 (A) must be a single or double array "
                "(got a " + detail::shapeOf(A) + " " + mtypeName(t)
                    + " array).",
                0, 0, "vecnorm", "", "numkit:vecnorm:inputType");
}

} // anonymous namespace

Value norm_value(const Value &x, double p, keel::Allocator *mr)
{
    detail::requireDoubleInput(x, "norm", "x");
    const bool wantSingle = x.type() == ValueType::SINGLE;
    if (x.numel() == 0)
        return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);

    if (isVectorShape(x)) {
        const keel::usize n = x.numel();
        if (p == 2.0) {
            double s = 0.0;
            for (keel::usize i = 0; i < n; ++i) { const double a = absLin(x, i); s += a * a; }
            return detail::narrowDoubleToSingleIf(Value::scalar(std::sqrt(s), mr), wantSingle, mr);
        } else if (p == 1.0) {
            double s = 0.0;
            for (keel::usize i = 0; i < n; ++i) s += absLin(x, i);
            return detail::narrowDoubleToSingleIf(Value::scalar(s, mr), wantSingle, mr);
        } else {
            double s = 0.0;
            for (keel::usize i = 0; i < n; ++i) s += std::pow(absLin(x, i), p);
            return detail::narrowDoubleToSingleIf(Value::scalar(std::pow(s, 1.0 / p), mr), wantSingle, mr);
        }
    }

    // Matrix forms.
    if (x.dims().ndim() != 2)
        throw Error("norm: input must be vector or 2D matrix",
                    0, 0, "norm", "", "numkit:norm:badShape");
    const keel::usize m = static_cast<keel::usize>(x.dims().dim(0));
    const keel::usize n = static_cast<keel::usize>(x.dims().dim(1));

    if (p == 2.0) {
        auto sv = svd_values(x, mr);
        if (sv.numel() == 0)
            return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);
        return detail::narrowDoubleToSingleIf(Value::scalar(sv.elemAsDouble(0), mr), wantSingle, mr);
    }
    if (p == 1.0) {
        double mx = 0.0;
        for (keel::usize j = 0; j < n; ++j) {
            double s = 0.0;
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
        double mx = 0.0;
        for (keel::usize i = 0; i < n; ++i)
            mx = std::max(mx, absLin(x, i));
        return detail::narrowDoubleToSingleIf(Value::scalar(mx, mr), wantSingle, mr);
    }
    // Matrix inf-norm: max row sum.
    if (x.dims().ndim() != 2)
        throw Error("norm: input must be vector or 2D matrix",
                    0, 0, "norm", "", "numkit:norm:badShape");
    const keel::usize m = static_cast<keel::usize>(x.dims().dim(0));
    const keel::usize n = static_cast<keel::usize>(x.dims().dim(1));
    double mx = 0.0;
    for (keel::usize i = 0; i < m; ++i) {
        double s = 0.0;
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
    double s = 0.0;
    for (keel::usize i = 0; i < n; ++i) { const double a = absLin(x, i); s += a * a; }
    return detail::narrowDoubleToSingleIf(Value::scalar(std::sqrt(s), mr), wantSingle, mr);
}

Value vecnorm(const Value &A, double p, int dim, keel::Allocator *mr)
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

    auto p_norm = [&](auto getAbs, keel::usize n) -> double {
        if (n == 0) return 0.0;
        if (std::isinf(p) && p > 0) {
            double m = 0.0;
            for (keel::usize k = 0; k < n; ++k) {
                double v = getAbs(k);
                if (std::isnan(v)) return std::numeric_limits<double>::quiet_NaN();
                if (v > m) m = v;
            }
            return m;
        }
        if (std::isinf(p) && p < 0) {
            double m = std::numeric_limits<double>::infinity();
            for (keel::usize k = 0; k < n; ++k) {
                double v = getAbs(k);
                if (std::isnan(v)) return std::numeric_limits<double>::quiet_NaN();
                if (v < m) m = v;
            }
            return m;
        }
        if (p == 2.0) {
            double s = 0.0;
            for (keel::usize k = 0; k < n; ++k) {
                double v = getAbs(k);
                s += v * v;
            }
            return std::sqrt(s);
        }
        double s = 0.0;
        for (keel::usize k = 0; k < n; ++k) s += std::pow(getAbs(k), p);
        return std::pow(s, 1.0 / p);
    };

    auto getAbs = [&](keel::usize i, keel::usize j) -> double {
        if (A.isComplex()) return keel::abs(A.complexElem(i, j));
        return keel::abs(A.elemAsDouble(lin(i, j, R)));
    };

    if (R == 0 && C == 0)
        return detail::narrowDoubleToSingleIf(Value::scalar(0.0, mr), wantSingle, mr);

    if (dim == 1) {
        Value out = Value::matrix(1, C, ValueType::DOUBLE, mr);
        if (C == 0) return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
        double *od = out.doubleDataMut();
        for (keel::usize j = 0; j < C; ++j) {
            od[j] = p_norm([&](keel::usize k){ return getAbs(k, j); }, R);
        }
        return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
    }
    Value out = Value::matrix(R, 1, ValueType::DOUBLE, mr);
    if (R == 0) return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
    double *od = out.doubleDataMut();
    for (keel::usize i = 0; i < R; ++i) {
        od[i] = p_norm([&](keel::usize k){ return getAbs(i, k); }, C);
    }
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Engine adapters â€” registered in LinalgLibrary::install
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

} // namespace numkit::linalg
