// ops/src/fused/fused_kernels_portable.cpp
//
// Scalar fallbacks for the fused element-wise kernels (non-SIMD builds). One
// file for all kernels — scalar loops are tiny and carry no Highway
// multi-target / inliner-budget cost, so there is no reason to split them.
// Semantics match the SIMD versions bit-for-bit (std::fmin/fmax NaN rules).

#include <keel/core.h>
#include <keel/math.h>
#include <numkit/ops/fused/fused_kernels.hpp>

#include <cstddef>

namespace numkit::ops {

void fusedAffineClamp(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                      keel::f64 lo, keel::f64 hi, keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 v = scale * x[i] + offset;
        out[i] = keel::fmax(lo, keel::fmin(hi, v));
    }
}

void fusedAffineClampMinOuter(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                              keel::f64 lo, keel::f64 hi, keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 v = scale * x[i] + offset;
        out[i] = keel::fmin(hi, keel::fmax(lo, v));
    }
}

void fusedAffineClampShiftDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                              keel::f64 lo, keel::f64 hi, keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 v = (x[i] - sub) / div;
        out[i] = keel::fmax(lo, keel::fmin(hi, v));
    }
}

void fusedAffineClampMinOuterShiftDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                                      keel::f64 lo, keel::f64 hi, keel::f64 *out,
                                      keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 v = (x[i] - sub) / div;
        out[i] = keel::fmin(hi, keel::fmax(lo, v));
    }
}

// The kernels that carry a small-N inline gate in their Highway TU (fusedAffine,
// fusedAbs*, fusedSq*) forward to the shared scalar bodies in fused_scalar.cpp
// so the gate and this fallback are one definition (can't drift) — see that
// file + bugs/ops/cheap-elementwise-simd-small-n. The rest keep their body
// inline (no gate shares them).
void fusedAffine(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                 keel::f64 *out, keel::usize n) {
    fusedAffineScalar(x, scale, offset, out, n);
}

void fusedAxpby(const keel::f64 *x, keel::f64 a, const keel::f64 *y, keel::f64 b,
                keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = a * x[i] + b * y[i];
}

void fusedShiftScaleMul(const keel::f64 *x, keel::f64 sub, keel::f64 mul,
                        keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = (x[i] - sub) * mul;
}

void fusedShiftScaleDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                        keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = (x[i] - sub) / div;
}

void fusedAbsAffine(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                    keel::f64 *out, keel::usize n) {
    fusedAbsAffineScalar(x, scale, offset, out, n);
}

void fusedAbsDiff(const keel::f64 *x, const keel::f64 *y, keel::f64 *out, keel::usize n) {
    fusedAbsDiffScalar(x, y, out, n);
}

void fusedAbsShiftDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                      keel::f64 *out, keel::usize n) {
    fusedAbsShiftDivScalar(x, sub, div, out, n);
}

void fusedUnaryAffine(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                      UnaryAffineFn fn, keel::f64 *out, keel::usize n) {
    switch (fn) {
        case UnaryAffineFn::Sqrt:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::sqrt(scale * x[i] + offset);
            break;
        case UnaryAffineFn::Floor:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::floor(scale * x[i] + offset);
            break;
        case UnaryAffineFn::Ceil:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::ceil(scale * x[i] + offset);
            break;
        case UnaryAffineFn::Fix:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::trunc(scale * x[i] + offset);
            break;
        case UnaryAffineFn::Round:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::round(scale * x[i] + offset);
            break;
    }
}

void fusedUnaryShiftDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                        UnaryAffineFn fn, keel::f64 *out, keel::usize n) {
    switch (fn) {
        case UnaryAffineFn::Sqrt:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::sqrt((x[i] - sub) / div);
            break;
        case UnaryAffineFn::Floor:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::floor((x[i] - sub) / div);
            break;
        case UnaryAffineFn::Ceil:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::ceil((x[i] - sub) / div);
            break;
        case UnaryAffineFn::Fix:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::trunc((x[i] - sub) / div);
            break;
        case UnaryAffineFn::Round:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::round((x[i] - sub) / div);
            break;
    }
}

void fusedSqAffine(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                   keel::f64 *out, keel::usize n) {
    fusedSqAffineScalar(x, scale, offset, out, n);
}

void fusedSqDiff(const keel::f64 *x, const keel::f64 *y, keel::f64 *out, keel::usize n) {
    fusedSqDiffScalar(x, y, out, n);
}

void fusedSqShiftDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                     keel::f64 *out, keel::usize n) {
    fusedSqShiftDivScalar(x, sub, div, out, n);
}

void fusedSqrtSumSq(const keel::f64 *x, const keel::f64 *y, keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = keel::sqrt(x[i] * x[i] + y[i] * y[i]);
}

void fusedSoftThreshold(const keel::f64 *x, keel::f64 t, keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 v = x[i];
        const keel::f64 s = keel::is_nan(v) ? v : (v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0));
        out[i] = s * keel::fmax(0.0, keel::abs(v) - t);
    }
}

void fusedTransAffine(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                      TransAffineFn fn, keel::f64 *out, keel::usize n) {
    // Non-SIMD build: nk's exp/expm1 are scalar std:: here too, so the
    // affine + scalar transcendental matches the per-op path bit-for-bit.
    switch (fn) {
        case TransAffineFn::Exp:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::exp(scale * x[i] + offset);
            break;
        case TransAffineFn::Expm1:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::expm1(scale * x[i] + offset);
            break;
        case TransAffineFn::Log:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::log(scale * x[i] + offset);
            break;
        case TransAffineFn::Log2:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::log2(scale * x[i] + offset);
            break;
        case TransAffineFn::Log10:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::log10(scale * x[i] + offset);
            break;
        case TransAffineFn::Sin:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::sin(scale * x[i] + offset);
            break;
        case TransAffineFn::Cos:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::cos(scale * x[i] + offset);
            break;
        case TransAffineFn::Tanh:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::tanh(scale * x[i] + offset);
            break;
        case TransAffineFn::Sinh:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::sinh(scale * x[i] + offset);
            break;
        case TransAffineFn::Atan:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::atan(scale * x[i] + offset);
            break;
        case TransAffineFn::Asinh:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::asinh(scale * x[i] + offset);
            break;
        case TransAffineFn::Asin:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::asin(scale * x[i] + offset);
            break;
        case TransAffineFn::Acos:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::acos(scale * x[i] + offset);
            break;
        case TransAffineFn::Acosh:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::acosh(scale * x[i] + offset);
            break;
        case TransAffineFn::Atanh:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::atanh(scale * x[i] + offset);
            break;
        case TransAffineFn::Log1p:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::log1p(scale * x[i] + offset);
            break;
        case TransAffineFn::Cosh:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::cosh(scale * x[i] + offset);
            break;
        case TransAffineFn::Tan:
            for (keel::usize i = 0; i < n; ++i) out[i] = keel::tan(scale * x[i] + offset);
            break;
    }
}

void fusedTransShiftDiv(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                        TransAffineFn fn, keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 v = (x[i] - sub) / div;
        switch (fn) {
            case TransAffineFn::Exp:   out[i] = keel::exp(v);   break;
            case TransAffineFn::Expm1: out[i] = keel::expm1(v); break;
            case TransAffineFn::Log:   out[i] = keel::log(v);   break;
            case TransAffineFn::Log2:  out[i] = keel::log2(v);  break;
            case TransAffineFn::Log10: out[i] = keel::log10(v); break;
            case TransAffineFn::Sin:   out[i] = keel::sin(v);   break;
            case TransAffineFn::Cos:   out[i] = keel::cos(v);   break;
            case TransAffineFn::Tanh:  out[i] = keel::tanh(v);  break;
            case TransAffineFn::Sinh:  out[i] = keel::sinh(v);  break;
            case TransAffineFn::Atan:  out[i] = keel::atan(v);  break;
            case TransAffineFn::Asinh: out[i] = keel::asinh(v); break;
            case TransAffineFn::Asin:  out[i] = keel::asin(v);  break;
            case TransAffineFn::Acos:  out[i] = keel::acos(v);  break;
            case TransAffineFn::Acosh: out[i] = keel::acosh(v); break;
            case TransAffineFn::Atanh: out[i] = keel::atanh(v); break;
            case TransAffineFn::Log1p: out[i] = keel::log1p(v); break;
            case TransAffineFn::Cosh:  out[i] = keel::cosh(v);  break;
            case TransAffineFn::Tan:   out[i] = keel::tan(v);   break;
        }
    }
}

} // namespace numkit::ops
