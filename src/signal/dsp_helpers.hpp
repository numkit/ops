#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <numkit/value/value.hpp>
#include <numkit/ops/fft/fft_radix2.hpp>   // numkit::fftRadix2(buf,N,W) base butterfly (ops-layer)

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <cstddef>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit {

// ============================================================
// Twiddle-factor precomputation
//
// W[k] = exp(dir * 2πi * k / N) for k in [0, N/2). Stage `len` of an
// N-point radix-2 FFT uses W[0], W[step], W[2·step], …,
// W[(len/2 - 1)·step] where step = N/len. A single table of length
// N/2 thus covers every stage — removing the rotating `w *= wlen`
// from the inner loop and breaking the data dependency that prevents
// SIMD vectorisation (relevant for Phase 8e.2+).
//
// dir=+1 matches the current sign convention used by fftRadix2 and
// its conjugate-trick inverse wrapper in fft.cpp.
// ============================================================
inline void fillFftTwiddles(keel::c64 *W, keel::usize N, int dir)
{
    const keel::usize half = N / 2;
    if (half == 0) return;
    const double base = dir * 2.0 * M_PI / keel::as<double>(N);
    for (keel::usize k = 0; k < half; ++k) {
        const double a = base * keel::as<double>(k);
        W[k] = keel::c64(std::cos(a), std::sin(a));
    }
}

// The base radix-2 butterfly — fftRadix2(Complex*, size_t, const Complex*) — now
// lives in <nk/ops/fft/fft_radix2.hpp> (included above): it is an ops-layer
// numerical kernel shared with ops/src/fft_portable.cpp (previously this signal
// header was its only home, which made ops reach up into the signal toolbox — a
// layering inversion that broke SIMD-off builds). The convenience overloads below
// build the twiddle table and delegate to that base.

// Convenience overload: builds a one-shot twiddle table from the
// caller-provided arena (`mr`) on each call. Still used by callers that
// haven't hoisted the table outside their own loops (spectral_analysis/,
// time_frequency/, transforms/hilbert.cpp). For hot paths
// (fftAlongDim in fft.cpp), prefer the primary overload + fillFftTwiddles
// so the table cost amortises.
inline void fftRadix2(keel::Allocator *mr,
                      keel::c64 *buf, keel::usize N, int dir)
{
    if (N <= 1) return;
    keel::Array<keel::c64> W(N / 2, mr);
    fillFftTwiddles(W.data(), N, dir);
    fftRadix2(buf, N, W.data());
}

// Container overload — buf.data() + buf.size() + dir. Handles both
// std::vector<Complex> and std::pmr::vector<Complex>.
template <typename Container>
inline auto fftRadix2(keel::Allocator *mr, Container &buf, int dir)
    -> decltype(buf.data(), buf.size(), void())
{
    fftRadix2(mr, buf.data(), buf.size(), dir);
}

// ============================================================
// Next power of 2 >= n
// ============================================================
inline keel::usize nextPow2(keel::usize n)
{
    if (n == 0)
        return 1;
    keel::usize p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

// ============================================================
// Prepare complex buffer from Value (real or complex)
// with zero-padding to fftLen.
// ============================================================
inline keel::Array<keel::c64> prepareFFTBuffer(keel::Allocator *mr,
                                            const Value &x,
                                            keel::usize inputLen, keel::usize fftLen)
{
    keel::Array<keel::c64> buf(fftLen, mr);
    if (x.isComplex()) {
        const keel::c64 *src = x.complexData();
        for (keel::usize i = 0; i < inputLen; ++i)
            buf[i] = src[i];
    } else {
        const double *src = x.doubleData();
        for (keel::usize i = 0; i < inputLen; ++i)
            buf[i] = keel::c64(src[i], 0.0);
    }
    return buf;
}

// ============================================================
// Pack complex buffer into Value (1 x outLen).
// Pointer + size, not a container reference, so the same helper
// composes with std::vector, std::pmr::vector, raw arrays, etc.
// ============================================================
inline Value packComplexResult(const keel::c64 *buf, keel::usize outLen, keel::Allocator *mr)
{
    auto r = Value::complexMatrix(1, outLen, mr);
    keel::c64 *dst = r.complexDataMut();
    for (keel::usize i = 0; i < outLen; ++i)
        dst[i] = buf[i];
    return r;
}

// Direct convolution O(n*m) moved to the kernel layer: numkit::ops::convDirect
// (nk/ops/conv.hpp). convFFT stays here — it is built on the signal-local
// radix-2 FFT helpers below, not ops::fft.

// ============================================================
// FFT-based convolution O(N log N)
// ============================================================
inline keel::Array<double> convFFT(keel::Allocator *mr,
                                  const double *a, keel::usize na,
                                  const double *b, keel::usize nb)
{
    keel::usize nc = na + nb - 1;
    keel::usize fftLen = nextPow2(nc);

    keel::Array<keel::c64> fa(fftLen, mr);
    keel::Array<keel::c64> fb(fftLen, mr);
    for (keel::usize i = 0; i < na; ++i) fa[i] = keel::c64(a[i], 0.0);
    for (keel::usize i = 0; i < nb; ++i) fb[i] = keel::c64(b[i], 0.0);

    // Forward twiddles used for both fa and fb; inverse twiddles used
    // once for fa. Hoisting them here saves two tables' worth of work
    // vs the convenience overload.
    keel::Array<keel::c64> W_fwd(fftLen / 2, mr);
    keel::Array<keel::c64> W_inv(fftLen / 2, mr);
    fillFftTwiddles(W_fwd.data(), fftLen, +1);
    fillFftTwiddles(W_inv.data(), fftLen, -1);

    fftRadix2(fa.data(), fftLen, W_fwd.data());
    fftRadix2(fb.data(), fftLen, W_fwd.data());

    for (keel::usize i = 0; i < fftLen; ++i)
        fa[i] *= fb[i];

    fftRadix2(fa.data(), fftLen, W_inv.data());

    double invN = 1.0 / keel::as<double>(fftLen);
    keel::Array<double> c(nc, mr);
    for (keel::usize i = 0; i < nc; ++i)
        c[i] = fa[i].real() * invN;
    return c;
}

// Threshold for switching conv from direct to FFT
constexpr keel::usize CONV_FFT_THRESHOLD = 500;

} // namespace numkit
