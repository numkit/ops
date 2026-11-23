// toolboxes/builtin/src/lang/operators/backends/binary_ops_loops.hpp
//
// Private declarations for the backend-split binary-op inner loops.
// Implemented by exactly one of backends/MStdBinaryOps_{portable,simd}.cpp
// depending on NUMKIT_WITH_SIMD. Callers (binary_ops.cpp) use these
// only for the 2D same-shape DOUBLE fast path — every other shape
// (broadcasting, 3D, type mismatch) goes through the general
// elementwiseDouble() helper in helpers.hpp unchanged.

#pragma once

#include <keel/core.h>

namespace numkit::ops::detail {

void plusLoop   (const keel::f64 *a, const keel::f64 *b, keel::f64 *out, keel::usize n);
void minusLoop  (const keel::f64 *a, const keel::f64 *b, keel::f64 *out, keel::usize n);
void timesLoop  (const keel::f64 *a, const keel::f64 *b, keel::f64 *out, keel::usize n);
void rdivideLoop(const keel::f64 *a, const keel::f64 *b, keel::f64 *out, keel::usize n);

// Shared scalar bodies for the small-N gate ↔ portable fallback. The *Loop
// entry points above forward here (portable) or call it from their small-N gate
// (Highway), so the gate and the fallback are one definition and the loop lives
// in a lambda-free TU (binary_scalar.cpp) that auto-vectorizes regardless of the
// dispatcher's parallel_for capture. See bugs/ops/cheap-elementwise-simd-small-n.
void plusScalar   (const keel::f64 *a, const keel::f64 *b, keel::f64 *out, keel::usize n);
void minusScalar  (const keel::f64 *a, const keel::f64 *b, keel::f64 *out, keel::usize n);
void timesScalar  (const keel::f64 *a, const keel::f64 *b, keel::f64 *out, keel::usize n);
void rdivideScalar(const keel::f64 *a, const keel::f64 *b, keel::f64 *out, keel::usize n);

// Matrix multiply over column-major real doubles.
//  A: M×K at memory layout a[k*M + i]
//  B: K×N at memory layout b[j*K + k]
//  C: M×N written as c[j*M + i]; caller allocates, this function zeros
// and then accumulates. Loop order is (j, k, i) — SAXPY down columns of
// A with the (k,j) element of B as the scalar — cache-friendly and
// directly vectorisable with MulAdd.
void matmulDoubleLoop(const keel::f64 *a, const keel::f64 *b, keel::f64 *c,
                      keel::usize M, keel::usize N, keel::usize K);

} // namespace numkit::ops::detail
