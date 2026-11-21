// ops/src/conv.cpp
//
// Direct real-double linear convolution, delegated to keel::signal.
// See conv.hpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/signal.h>
#include <numkit/ops/conv.hpp>

namespace numkit::ops {

keel::Array<double> convDirect(const double *a, keel::usize na,
                              const double *b, keel::usize nb,
                              keel::Allocator *mr)
{
    keel::usize nc = na + nb - 1;
    keel::Array<double> c(nc, mr);
    keel::signal::conv_1d(a, na, b, nb, c.data());
    return c;
}

void conv2Direct(const double *A, keel::usize M, keel::usize N,
                 const double *B, keel::usize P, keel::usize Q,
                 double *out)
{
    keel::signal::conv_2d(A, M, N, B, P, Q, out);
}

} // namespace numkit::ops
