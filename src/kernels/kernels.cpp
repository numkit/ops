// nk/ops/kernels.cpp
//
// The public kernel facade (nk/ops/kernels.hpp) — thin forwarders to the
// backend-split detail loops. The facade itself is allowed to reach into
// detail; its callers are not. Keeping the forwarders in one base TU (always
// compiled, no SIMD specialisation of its own) means the public symbol resolves
// to whichever backend (Highway / portable) was linked, with no drift.
#include <keel/core.h>
#include <keel/complex.h>
#include <numkit/ops/kernels.hpp>

#include <numkit/ops/blas3.hpp>              // ops::gemm (Highway complex GEMM)
#include <numkit/ops/binary_ops.hpp>        // detail::matmulDoubleLoop / *Loop
#include <numkit/ops/fused/fused_kernels.hpp>  // fusedTransAffine / TransAffineFn

namespace numkit::ops {

void matmulDouble(const keel::f64 *a, const keel::f64 *b, keel::f64 *c,
                  keel::usize M, keel::usize N, keel::usize K)
{
    detail::matmulDoubleLoop(a, b, c, M, N, K);
}

void matmulComplex(const keel::c64 *a, const keel::c64 *b,
                   keel::c64 *c, keel::usize M, keel::usize N, keel::usize K)
{
    if (M == 0 || N == 0)
        return;
    gemm(MatrixTranspose::NoTrans, MatrixTranspose::NoTrans,
         M, N, K,
         keel::c64(1.0, 0.0), a, M,
         b, K,
         keel::c64(0.0, 0.0), c, M);
}

void plusDouble(const keel::f64 *a, const keel::f64 *b, keel::f64 *out, keel::usize n)
{
    detail::plusLoop(a, b, out, n);
}
void minusDouble(const keel::f64 *a, const keel::f64 *b, keel::f64 *out, keel::usize n)
{
    detail::minusLoop(a, b, out, n);
}
void timesDouble(const keel::f64 *a, const keel::f64 *b, keel::f64 *out, keel::usize n)
{
    detail::timesLoop(a, b, out, n);
}
void rdivideDouble(const keel::f64 *a, const keel::f64 *b, keel::f64 *out, keel::usize n)
{
    detail::rdivideLoop(a, b, out, n);
}

// Transcendentals: forward to the SIMD fusedTransAffine (scale=1, offset=0).
// The set is real-total (no complex-domain decline), so no pre-scan is needed.
void sinDouble(const keel::f64 *x, keel::f64 *out, keel::usize n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Sin, out, n); }
void cosDouble(const keel::f64 *x, keel::f64 *out, keel::usize n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Cos, out, n); }
void tanDouble(const keel::f64 *x, keel::f64 *out, keel::usize n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Tan, out, n); }
void atanDouble(const keel::f64 *x, keel::f64 *out, keel::usize n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Atan, out, n); }
void sinhDouble(const keel::f64 *x, keel::f64 *out, keel::usize n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Sinh, out, n); }
void coshDouble(const keel::f64 *x, keel::f64 *out, keel::usize n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Cosh, out, n); }
void tanhDouble(const keel::f64 *x, keel::f64 *out, keel::usize n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Tanh, out, n); }
void expDouble(const keel::f64 *x, keel::f64 *out, keel::usize n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Exp, out, n); }
void asinhDouble(const keel::f64 *x, keel::f64 *out, keel::usize n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Asinh, out, n); }
void expm1Double(const keel::f64 *x, keel::f64 *out, keel::usize n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Expm1, out, n); }

} // namespace numkit::ops
