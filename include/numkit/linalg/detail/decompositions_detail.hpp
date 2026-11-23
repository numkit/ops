// toolboxes/linalg/src/decompositions_detail.hpp
//
// PRIVATE (src-level) header â€” NOT part of the public linalg API.
//
// Shared raw-buffer / low-level factorisation kernels between compute
// (decompositions.cpp) and its register half (decompositions_reg.cpp).

#pragma once

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <keel/tuple.h>

#include <numkit/value/value.hpp>
#include <numkit/ops/blas.hpp>
#include "linalg_detail.hpp"

namespace numkit::linalg {

// Build upper-triangular R (column-major, nÃ—n) such that R'Â·R = A, reading the
// upper triangle of A. Returns the 1-based column index where the
// factorization broke down (non-positive pivot), or 0 on success.
template <typename T>
keel::usize cholUpperFactor(const T *a, T *r, keel::usize n) {
    keel::fill(r, r + n * n, T(0));
    for (keel::usize j = 0; j < n; ++j) {
        T s = a[j + j * n];
        for (keel::usize k = 0; k < j; ++k) {
            if constexpr (detail::is_complex_v<T>) {
                s -= detail::abs_sq(r[k + j * n]);
            } else {
                s -= r[k + j * n] * r[k + j * n];
            }
        }
        if constexpr (detail::is_complex_v<T>) {
            if (s.real() <= 0.0 || keel::abs(s.imag()) > 1e-12 * (1.0 + keel::abs(s.real()))) return j + 1;
            r[j + j * n] = T(keel::sqrt(s.real()), 0.0);
        } else {
            if (s <= 0.0) return j + 1;
            r[j + j * n] = keel::sqrt(s);
        }
        const T inv_diag = T(1) / r[j + j * n];
        for (keel::usize i = j + 1; i < n; ++i) {
            T t = a[j + i * n];
            if constexpr (detail::is_complex_v<T>) {
                const T a_ji = a[i + j * n];
                if (keel::abs(t - keel::conj(a_ji)) > 1e-12 * (1.0 + keel::abs(t))) return j + 1;
            }
            for (keel::usize k = 0; k < j; ++k) {
                if constexpr (detail::is_complex_v<T>) {
                    t -= keel::conj(r[k + j * n]) * r[k + i * n];
                } else {
                    t -= r[k + j * n] * r[k + i * n];
                }
            }
            r[j + i * n] = t * inv_diag;
        }
    }
    return 0;
}

// Transpose (conjugate transpose for complex) a square kÃ—k column-major matrix into a fresh Value.
template <typename T>
Value transposeSquare(const T *src, keel::usize k, keel::Allocator *mr) {
    Value out = detail::make_matrix<T>(k, k, mr);
    T *d = detail::get_data_mut<T>(out);
    for (keel::usize col = 0; col < k; ++col) {
        for (keel::usize row = 0; row < k; ++row) {
            d[row + col * k] = detail::conj_if_complex(src[col + row * k]);
        }
    }
    return out;
}

// Column-pivoted QR (the [Q,R,P] form).
keel::Tuple<Value, Value>
qr_pivoted(const Value &A, keel::Array<keel::usize> &perm,
           keel::Allocator *mr = nullptr);

} // namespace numkit::linalg
