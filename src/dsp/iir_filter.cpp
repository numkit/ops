// ops/src/iir_filter.cpp
//
// Direct-Form-II-transposed IIR filter recurrence (real + complex), moved
// verbatim from signal's filter.cpp so the kernel lives in the kernel layer
// (ops, core-free). The toolbox keeps the Value-API (filter/filtfilt) and the
// Value-level marshalling helpers (toComplexBuf, firstNonSingletonExtent). The
// recurrence is sequential — no SIMD backend. See iir_filter.hpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <numkit/ops/iir_filter.hpp>

#include <algorithm>

namespace numkit::ops {

keel::Array<double> applyFilterDf2t(const double *bn, keel::usize nb, const double *an,
                                   keel::usize na, const double *input, keel::usize len,
                                   const double *zi, keel::usize ziLen, double *zfOut,
                                   keel::Allocator *mr)
{
    const keel::usize  nfilt = std::max(nb, na);
    keel::Array<double> out(len, mr);
    keel::Array<double> z(nfilt, mr);
    for (keel::usize i = 0; i < nfilt; ++i)
        z[i] = (zi && i < ziLen) ? zi[i] : 0.0;
    for (keel::usize n = 0; n < len; ++n) {
        out[n] = (nb > 0 ? bn[0] : 0.0) * input[n] + z[0];
        for (keel::usize i = 1; i < nfilt; ++i) {
            z[i - 1] = (i < nb ? bn[i] : 0.0) * input[n]
                       - (i < na ? an[i] : 0.0) * out[n]
                       + (i < nfilt - 1 ? z[i] : 0.0);
        }
    }
    if (zfOut)
        for (keel::usize i = 0; i + 1 < nfilt; ++i) zfOut[i] = z[i];
    return out;
}

keel::Array<keel::c64> applyFilterDf2tComplex(const keel::c64 *bn, keel::usize nb, const keel::c64 *an,
                                           keel::usize na, const keel::c64 *input, keel::usize len,
                                           const keel::c64 *zi, keel::usize ziLen, keel::c64 *zfOut,
                                           keel::Allocator *mr)
{
    const keel::usize   nfilt = std::max(nb, na);
    keel::Array<keel::c64> out(len, mr);
    keel::Array<keel::c64> z(nfilt, mr);
    const keel::c64       zero(0.0, 0.0);
    for (keel::usize i = 0; i < nfilt; ++i)
        z[i] = (zi && i < ziLen) ? zi[i] : zero;
    for (keel::usize n = 0; n < len; ++n) {
        out[n] = (nb > 0 ? bn[0] : zero) * input[n] + z[0];
        for (keel::usize i = 1; i < nfilt; ++i) {
            z[i - 1] = (i < nb ? bn[i] : zero) * input[n]
                       - (i < na ? an[i] : zero) * out[n]
                       + (i < nfilt - 1 ? z[i] : zero);
        }
    }
    if (zfOut)
        for (keel::usize i = 0; i + 1 < nfilt; ++i) zfOut[i] = z[i];
    return out;
}

void biquadDf2t(double b0, double b1, double b2, double a1, double a2,
                const double *x, double *y, keel::usize n)
{
    double s1 = 0.0, s2 = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        const double xi = x[i];
        const double yi = b0 * xi + s1;
        s1 = b1 * xi - a1 * yi + s2;
        s2 = b2 * xi - a2 * yi;
        y[i] = yi;
    }
}

void biquadDf2tWithState(double b0, double b1, double b2, double a1, double a2,
                         const double *x, double *y, keel::usize n,
                         double s1_init, double s2_init)
{
    double s1 = s1_init, s2 = s2_init;
    for (keel::usize i = 0; i < n; ++i) {
        const double xi = x[i];
        const double yi = b0 * xi + s1;
        s1 = b1 * xi - a1 * yi + s2;
        s2 = b2 * xi - a2 * yi;
        y[i] = yi;
    }
}

} // namespace numkit::ops
