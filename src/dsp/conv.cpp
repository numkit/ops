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

keel::Array<keel::f64> convDirect(const keel::f64 *a, keel::usize na,
                              const keel::f64 *b, keel::usize nb,
                              keel::Allocator *mr)
{
    keel::usize nc = na + nb - 1;
    keel::Array<keel::f64> c(nc, mr);
    (void)keel::signal::conv_1d({a, na}, {b, nb}, c);
    return c;
}

void conv2Direct(const keel::f64 *A, keel::usize M, keel::usize N,
                 const keel::f64 *B, keel::usize P, keel::usize Q,
                 keel::f64 *out)
{
    const keel::usize outLen = (M > 0 && N > 0 && P > 0 && Q > 0)
                                   ? (M + P - 1) * (N + Q - 1)
                                   : 0;
    (void)keel::signal::conv_2d({A, M * N}, M, N, {B, P * Q}, P, Q, {out, outLen});
}

} // namespace numkit::ops
