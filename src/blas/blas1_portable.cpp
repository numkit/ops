#include <keel/core.h>
#include <numkit/ops/blas1.hpp>
#include <numkit/ops/parallel_for.hpp>

namespace numkit::ops {

void axpy(keel::usize n, double alpha, const double *x, double *y)
{
    if (alpha == 0.0 || n == 0) return;
    
    // We can use a simple threshold for parallel_for
    constexpr keel::usize kParallelThreshold = 256 * 1024;
    
    numkit::detail::parallel_for(n, kParallelThreshold, [=](keel::usize start, keel::usize end) {
        for (keel::usize i = start; i < end; ++i) {
            y[i] += alpha * x[i];
        }
    });
}

} // namespace numkit::ops