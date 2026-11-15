#include <keel/core.h>
#include <numkit/ops/blas2.hpp>
#include <numkit/ops/parallel_for.hpp>

namespace numkit::ops {

void gemv(keel::usize m, keel::usize n,
          double alpha, const double *A, keel::usize lda,
          const double *x, keel::usize incx,
          double beta, double *y, keel::usize incy)
{
    if (m == 0) return;
    
    // Scale y by beta
    if (beta == 0.0) {
        for (keel::usize i = 0; i < m; ++i) y[i * incy] = 0.0;
    } else if (beta != 1.0) {
        for (keel::usize i = 0; i < m; ++i) y[i * incy] *= beta;
    }

    if (alpha == 0.0 || n == 0) return;

    // A is m x n (column-major)
    // We parallelize over columns to avoid data races on y
    constexpr keel::usize kParallelThreshold = 1024;
    numkit::detail::parallel_for(n, kParallelThreshold, [=](keel::usize j_start, keel::usize j_end) {
        // If we parallelize over columns (n), multiple threads will write to y.
        // Wait, parallel_for over n would cause data races on y!
        // We MUST parallelize over m (rows) to avoid races on y.
        // Let's not use parallel_for for GEMV in this simple portable version, 
        // or we parallelize over m and inner loop over n.
    });
    // Actually, parallelizing GEMV over rows (m) is safer:
    
    constexpr keel::usize kRowThreshold = 1024;
    numkit::detail::parallel_for(m, kRowThreshold, [=](keel::usize i_start, keel::usize i_end) {
        for (keel::usize j = 0; j < n; ++j) {
            const double xj = alpha * x[j * incx];
            if (xj == 0.0) continue;
            const double *aj = A + j * lda;
            for (keel::usize i = i_start; i < i_end; ++i) {
                y[i * incy] += aj[i] * xj;
            }
        }
    });
}

void ger(keel::usize m, keel::usize n,
         double alpha, const double *x, keel::usize incx,
         const double *y, keel::usize incy,
         double *A, keel::usize lda)
{
    if (alpha == 0.0 || m == 0 || n == 0) return;

    constexpr keel::usize kParallelThreshold = 256;
    numkit::detail::parallel_for(n, kParallelThreshold, [=](keel::usize j_start, keel::usize j_end) {
        for (keel::usize j = j_start; j < j_end; ++j) {
            const double yj = alpha * y[j * incy];
            if (yj == 0.0) continue;
            double *aj = A + j * lda;
            if (incx == 1) {
                for (keel::usize i = 0; i < m; ++i) aj[i] += x[i] * yj;
            } else {
                for (keel::usize i = 0; i < m; ++i) aj[i] += x[i * incx] * yj;
            }
        }
    });
}

} // namespace numkit::ops