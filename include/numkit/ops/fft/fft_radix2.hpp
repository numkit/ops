#pragma once
//
// Scalar iterative radix-2 Cooley-Tukey FFT butterfly (in-place).
//
// This is an ops-layer numerical kernel: it operates on a raw contiguous
// Complex buffer + a precomputed twiddle table, with no dependency on the
// engine, the filesystem, or any toolbox. It was previously defined in
// toolboxes/signal/src/dsp_helpers.hpp, which made the ops-layer portable
// FFT (ops/src/fft_portable.cpp) reach UP into the signal toolbox (a layering
// inversion that only surfaced on SIMD-off builds, where fft_portable.cpp is
// compiled). Hosting the butterfly here keeps ops self-contained; signal's
// dsp_helpers.hpp now includes this header for its convenience overloads.
//
#include <keel/core.h>
#include <keel/complex.h>
#include <numkit/value/value.hpp>   // numkit::Complex

#include <algorithm>                // std::swap
#include <cstddef>

namespace numkit {

// Iterative radix-2 Cooley-Tukey FFT (in-place). N must be a power of 2.
// Takes a precomputed twiddle table W of length N/2 (see fillFftTwiddles in
// signal's dsp_helpers.hpp for how to build one). Decoupled from any container
// type so callers can back the buffer with std::vector, std::pmr::vector, a
// stack array, or any other contiguous Complex storage.
inline void fftRadix2(keel::c64 *buf, keel::usize N, const keel::c64 *W)
{
    if (N <= 1)
        return;

    // Bit-reversal permutation
    for (keel::usize i = 1, j = 0; i < N; ++i) {
        keel::usize bit = N >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(buf[i], buf[j]);
    }

    // Butterfly stages — look up twiddles from the precomputed table.
    for (keel::usize len = 2; len <= N; len <<= 1) {
        const keel::usize step = N / len;
        for (keel::usize i = 0; i < N; i += len) {
            for (keel::usize j = 0; j < len / 2; ++j) {
                const keel::c64 w = W[j * step];
                const keel::c64 u = buf[i + j];
                const keel::c64 v = buf[i + j + len / 2] * w;
                buf[i + j]           = u + v;
                buf[i + j + len / 2] = u - v;
            }
        }
    }
}

} // namespace numkit
