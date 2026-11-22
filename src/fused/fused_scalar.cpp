// ops/src/fused/fused_scalar.cpp
//
// Single source of truth for the SCALAR bodies of the fused element-wise
// kernels that carry a small-N inline gate in their Highway TU (fusedAffine,
// fusedAbs*, fusedSq*). Compiled in BOTH the SIMD and portable builds:
//
//   * the portable fallbacks (fused_kernels_portable.cpp) forward here, and
//   * the Highway dispatchers' small-N gate calls these directly instead of
//     re-inlining the loop.
//
// One body, two consumers → the gate can no longer drift from the fallback
// (footgun #2), and because the loop lives in a lambda-free TU it
// auto-vectorizes regardless of how the Highway dispatcher captures its
// parallel_for worker — sidestepping the [&]-capture-escape that silently
// descalarized an inline gate loop on MSVC (footgun #1; see
// bugs/ops/cheap-elementwise-simd-small-n). NUMKIT_NOINLINE keeps the body from
// being merged back into an escaping-lambda context if IPO/LTCG is ever turned
// on (off today, so the separate TU already suffices).
//
// Semantics are bit-identical to the per-op path: each body is one chain of
// plain IEEE ops — the same the Highway lanes compute — so the gate ↔ SIMD
// result matches element-for-element. Bodies copied verbatim from the former
// inline gates / fused_kernels_portable.cpp.

#include <keel/core.h>
#include <keel/math.h>
#include <numkit/ops/compiler.hpp>
#include <numkit/ops/fused/fused_kernels.hpp>

#include <cstddef>

namespace numkit::ops {

NUMKIT_NOINLINE void fusedAffineScalar(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                                       keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = scale * x[i] + offset;
}

NUMKIT_NOINLINE void fusedAbsAffineScalar(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                                          keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = keel::abs(scale * x[i] + offset);
}

NUMKIT_NOINLINE void fusedAbsShiftDivScalar(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                                            keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = keel::abs((x[i] - sub) / div);
}

NUMKIT_NOINLINE void fusedAbsDiffScalar(const keel::f64 *x, const keel::f64 *y,
                                        keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) out[i] = keel::abs(x[i] - y[i]);
}

NUMKIT_NOINLINE void fusedSqAffineScalar(const keel::f64 *x, keel::f64 scale, keel::f64 offset,
                                         keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 v = scale * x[i] + offset;
        out[i] = v * v;
    }
}

NUMKIT_NOINLINE void fusedSqShiftDivScalar(const keel::f64 *x, keel::f64 sub, keel::f64 div,
                                           keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 v = (x[i] - sub) / div;
        out[i] = v * v;
    }
}

NUMKIT_NOINLINE void fusedSqDiffScalar(const keel::f64 *x, const keel::f64 *y,
                                       keel::f64 *out, keel::usize n) {
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 v = x[i] - y[i];
        out[i] = v * v;
    }
}

} // namespace numkit::ops
