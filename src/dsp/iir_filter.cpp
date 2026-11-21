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

keel::Array<keel::f64> applyFilterDf2t(const keel::f64 *bn, keel::usize nb, const keel::f64 *an,
                                   keel::usize na, const keel::f64 *input, keel::usize len,
                                   const keel::f64 *zi, keel::usize ziLen, keel::f64 *zfOut,
                                   keel::Allocator *mr)
{
    keel::Array<keel::f64> out(len, mr);
    const keel::usize nfilt = keel::max(nb, na);
    const keel::usize zfLen = (zfOut && nfilt > 1) ? (nfilt - 1) : 0;
    const keel::ArrayRef<keel::f64> ziRef = zi ? keel::ArrayRef<keel::f64>(zi, ziLen) : keel::ArrayRef<keel::f64>{};
    const keel::MutArrayRef<keel::f64> zfRef = zfOut ? keel::MutArrayRef<keel::f64>(zfOut, zfLen) : keel::MutArrayRef<keel::f64>{};
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

void biquadDf2t(keel::f64 b0, keel::f64 b1, keel::f64 b2, keel::f64 a1, keel::f64 a2,
                const keel::f64 *x, keel::f64 *y, keel::usize n)
{
    (void)keel::signal::biquad_df2t(b0, b1, b2, a1, a2, {x, n}, {y, n});
}

void biquadDf2tWithState(keel::f64 b0, keel::f64 b1, keel::f64 b2, keel::f64 a1, keel::f64 a2,
                         const keel::f64 *x, keel::f64 *y, keel::usize n,
                         keel::f64 s1_init, keel::f64 s2_init)
{
    (void)keel::signal::biquad_df2t_state(b0, b1, b2, a1, a2, {x, n}, {y, n}, s1_init, s2_init);
}

} // namespace numkit::ops
