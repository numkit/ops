// ops/src/iir_filter.cpp
//
// Direct-Form-II-transposed IIR filter recurrence (real + complex) and
// biquad kernels, delegated to keel::signal. See iir_filter.hpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <keel/ref.h>
#include <keel/signal.h>
#include <numkit/ops/iir_filter.hpp>

namespace numkit::ops {

keel::Array<double> applyFilterDf2t(const double *bn, keel::usize nb, const double *an,
                                   keel::usize na, const double *input, keel::usize len,
                                   const double *zi, keel::usize ziLen, double *zfOut,
                                   keel::Allocator *mr)
{
    keel::Array<double> out(len, mr);
    const keel::usize nfilt = keel::max(nb, na);
    const keel::usize zfLen = (zfOut && nfilt > 1) ? (nfilt - 1) : 0;
    const keel::ArrayRef<double> ziRef = zi ? keel::ArrayRef<double>(zi, ziLen) : keel::ArrayRef<double>{};
    const keel::MutArrayRef<double> zfRef = zfOut ? keel::MutArrayRef<double>(zfOut, zfLen) : keel::MutArrayRef<double>{};
    (void)keel::signal::filter_df2t({bn, nb}, {an, na}, {input, len}, out, ziRef, zfRef, mr);
    return out;
}

keel::Array<keel::c64> applyFilterDf2tComplex(const keel::c64 *bn, keel::usize nb, const keel::c64 *an,
                                           keel::usize na, const keel::c64 *input, keel::usize len,
                                           const keel::c64 *zi, keel::usize ziLen, keel::c64 *zfOut,
                                           keel::Allocator *mr)
{
    keel::Array<keel::c64> out(len, mr);
    const keel::usize nfilt = keel::max(nb, na);
    const keel::usize zfLen = (zfOut && nfilt > 1) ? (nfilt - 1) : 0;
    const keel::ArrayRef<keel::c64> ziRef = zi ? keel::ArrayRef<keel::c64>(zi, ziLen) : keel::ArrayRef<keel::c64>{};
    const keel::MutArrayRef<keel::c64> zfRef = zfOut ? keel::MutArrayRef<keel::c64>(zfOut, zfLen) : keel::MutArrayRef<keel::c64>{};
    (void)keel::signal::filter_df2t_c64({bn, nb}, {an, na}, {input, len}, out, ziRef, zfRef, mr);
    return out;
}

void biquadDf2t(double b0, double b1, double b2, double a1, double a2,
                const double *x, double *y, keel::usize n)
{
    (void)keel::signal::biquad_df2t(b0, b1, b2, a1, a2, {x, n}, {y, n});
}

void biquadDf2tWithState(double b0, double b1, double b2, double a1, double a2,
                         const double *x, double *y, keel::usize n,
                         double s1_init, double s2_init)
{
    (void)keel::signal::biquad_df2t_state(b0, b1, b2, a1, a2, {x, n}, {y, n}, s1_init, s2_init);
}

} // namespace numkit::ops
