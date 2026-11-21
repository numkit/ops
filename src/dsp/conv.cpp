// ops/src/conv.cpp
//
// Direct real-double linear convolution, delegated to keel::signal.
// See conv.hpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <numkit/ops/conv.hpp>

namespace numkit::ops {

keel::Array<double> convDirect(const double *a, keel::usize na,
                              const double *b, keel::usize nb,
                              keel::Allocator *mr)
{
    keel::usize nc = na + nb - 1;
    keel::Array<double> c(nc, mr);
    (void)keel::signal::conv_1d({a, na}, {b, nb}, c);
    return c;
}

void conv2Direct(const double *A, keel::usize M, keel::usize N,
                 const double *B, keel::usize P, keel::usize Q,
                 double *out)
{
    const keel::usize outLen = (M > 0 && N > 0 && P > 0 && Q > 0)
                                   ? (M + P - 1) * (N + Q - 1)
                                   : 0;
    (void)keel::signal::conv_2d({A, M * N}, M, N, {B, P * Q}, P, Q, {out, outLen});
}

} // namespace numkit::ops
