// ops/src/conv.cpp
//
// Direct real-double linear convolution, moved verbatim from signal's
// dsp_helpers.hpp (convDirect) so the FIR kernel lives in the kernel layer.
// Portable C++; the inner accumulate is the natural SIMD target (a follow-on
// Highway backend can replace it without touching this signature or the
// toolbox call sites). See conv.hpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <numkit/ops/conv.hpp>

#include <algorithm>

namespace numkit::ops {

keel::Array<double> convDirect(const double *a, keel::usize na,
                              const double *b, keel::usize nb,
                              keel::Allocator *mr)
{
    keel::usize        nc = na + nb - 1;
    keel::Array<double> c(nc, mr);
    for (keel::usize i = 0; i < na; ++i)
        for (keel::usize j = 0; j < nb; ++j)
            c[i + j] += a[i] * b[j];
    return c;
}

void conv2Direct(const double *A, keel::usize M, keel::usize N,
                 const double *B, keel::usize P, keel::usize Q,
                 double *out)
{
    const keel::usize outR = M + P - 1;
    const keel::usize outC = N + Q - 1;
    std::fill_n(out, outR * outC, 0.0);
    for (keel::usize j = 0; j < Q; ++j) {
        for (keel::usize i = 0; i < P; ++i) {
            const double bij = B[j * P + i];
            if (bij == 0.0) continue;
            for (keel::usize cc = 0; cc < N; ++cc) {
                const keel::usize outCol = j + cc;
                for (keel::usize rr = 0; rr < M; ++rr) {
                    out[outCol * outR + (i + rr)] += A[cc * M + rr] * bij;
                }
            }
        }
    }
}

} // namespace numkit::ops
