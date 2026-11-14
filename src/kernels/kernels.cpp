// nk/ops/kernels.cpp
//
// The public kernel facade (nk/ops/kernels.hpp) — thin forwarders to the
// backend-split detail loops. The facade itself is allowed to reach into
// detail; its callers are not. Keeping the forwarders in one base TU (always
// compiled, no SIMD specialisation of its own) means the public symbol resolves
// to whichever backend (Highway / portable) was linked, with no drift.
#include <numkit/ops/kernels.hpp>

#include <numkit/ops/blas3.hpp>              // ops::gemm (Highway complex GEMM)
#include <numkit/ops/binary_ops.hpp>        // detail::matmulDoubleLoop / *Loop
#include <numkit/ops/fused/fused_kernels.hpp>  // fusedTransAffine / TransAffineFn

namespace numkit::ops {

void matmulDouble(const double *a, const double *b, double *c,
                  std::size_t M, std::size_t N, std::size_t K)
{
    detail::matmulDoubleLoop(a, b, c, M, N, K);
}

void matmulComplex(const std::complex<double> *a, const std::complex<double> *b,
                   std::complex<double> *c, std::size_t M, std::size_t N, std::size_t K)
{
    if (M == 0 || N == 0)
        return;
    gemm(MatrixTranspose::NoTrans, MatrixTranspose::NoTrans,
         M, N, K,
         std::complex<double>(1.0, 0.0), a, M,
         b, K,
         std::complex<double>(0.0, 0.0), c, M);
}

void plusDouble(const double *a, const double *b, double *out, std::size_t n)
{
    detail::plusLoop(a, b, out, n);
}
void minusDouble(const double *a, const double *b, double *out, std::size_t n)
{
    detail::minusLoop(a, b, out, n);
}
void timesDouble(const double *a, const double *b, double *out, std::size_t n)
{
    detail::timesLoop(a, b, out, n);
}
void rdivideDouble(const double *a, const double *b, double *out, std::size_t n)
{
    detail::rdivideLoop(a, b, out, n);
}

// Transcendentals: forward to the SIMD fusedTransAffine (scale=1, offset=0).
// The set is real-total (no complex-domain decline), so no pre-scan is needed.
void sinDouble(const double *x, double *out, std::size_t n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Sin, out, n); }
void cosDouble(const double *x, double *out, std::size_t n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Cos, out, n); }
void tanDouble(const double *x, double *out, std::size_t n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Tan, out, n); }
void atanDouble(const double *x, double *out, std::size_t n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Atan, out, n); }
void sinhDouble(const double *x, double *out, std::size_t n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Sinh, out, n); }
void coshDouble(const double *x, double *out, std::size_t n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Cosh, out, n); }
void tanhDouble(const double *x, double *out, std::size_t n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Tanh, out, n); }
void expDouble(const double *x, double *out, std::size_t n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Exp, out, n); }
void asinhDouble(const double *x, double *out, std::size_t n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Asinh, out, n); }
void expm1Double(const double *x, double *out, std::size_t n)
{ fusedTransAffine(x, 1.0, 0.0, TransAffineFn::Expm1, out, n); }

} // namespace numkit::ops
