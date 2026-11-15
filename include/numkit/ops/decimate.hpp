#pragma once
//
// decimate.hpp — line-series downsampling for fast Plot rendering.
//
// Engine-side twin of the frontend ide/src/components/v3/decimate.js: the
// same M4 / LTTB algorithms so a series decimated here (Phase 2, binary
// tiles) renders identically to one decimated in JS (Phase 1). A plot is
// only ~W pixels wide, so a series of N≫W points is downsampled to O(W)
// points over the visible x-range before it ever crosses to JS.
//
//   M4   — per pixel-column keep {first, min, max, last}. Pixel-faithful:
//          smooth data stays a thin line; spikes and the true oscillation
//          extent survive. Default.
//   LTTB — Largest-Triangle-Three-Buckets: smoother for trends, ~1 point
//          per column, but can hide narrow spikes. Opt-in.
//
// x is assumed ascending (plot(y) → 1..N; plot(x,y) with sorted x). A
// non-finite y is a gap; it is kept when it lands on a bucket endpoint.

#include <keel/core.h>
#include <vector>
#include <cstddef>
#include <cmath>
#include <algorithm>

namespace numkit {

struct DecimatedSeries {
    std::vector<keel::f64> x;
    std::vector<keel::f64> y;
};

// First index i with x[i] >= target (x ascending); n if none.
inline keel::usize decimLowerBound(const keel::f64 *x, keel::usize n, keel::f64 target) {
    keel::usize lo = 0, hi = n;
    while (lo < hi) {
        keel::usize mid = (lo + hi) >> 1;
        if (x[mid] < target) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// Index range [i0, i1) covering the visible x-window, padded one sample each
// side so the line enters/leaves the viewport. Clamped to [0, n].
inline void decimVisibleRange(const keel::f64 *x, keel::usize n, keel::f64 x0, keel::f64 x1,
                              keel::usize &i0, keel::usize &i1) {
    if (n == 0) { i0 = i1 = 0; return; }
    i0 = decimLowerBound(x, n, x0);
    i1 = decimLowerBound(x, n, x1);
    if (i0 > 0) i0 -= 1;
    if (i1 < n) i1 += 1;
    if (i1 > n) i1 = n;
    if (i1 < i0) i1 = i0;
}

// M4: {first, min, max, last} per pixel column over the visible range.
// At most 4*width points, in x order.
inline DecimatedSeries decimateM4(const keel::f64 *x, const keel::f64 *y, keel::usize n,
                                  keel::f64 x0, keel::f64 x1, int width) {
    DecimatedSeries out;
    keel::usize i0, i1;
    decimVisibleRange(x, n, x0, x1, i0, i1);
    if (i1 <= i0) return out;
    const int cols = width < 1 ? 1 : width;
    const keel::f64 span = (x1 - x0) != 0.0 ? (x1 - x0) : 1.0;
    auto bucketOf = [&](keel::f64 xv) -> int {
        int b = static_cast<int>(std::floor(((xv - x0) / span) * cols));
        if (b < 0) b = 0; else if (b >= cols) b = cols - 1;
        return b;
    };

    out.x.reserve(static_cast<keel::usize>(cols) * 4);
    out.y.reserve(static_cast<keel::usize>(cols) * 4);
    int curBucket = -1;
    keel::usize bFirst = i0, bLast = i0, bMinI = i0, bMaxI = i0;
    keel::f64 bMinY = 0.0, bMaxY = 0.0;
    auto flush = [&]() {
        if (curBucket < 0) return;
        keel::usize idxs[4] = { bFirst, bMinI, bMaxI, bLast };
        std::sort(idxs, idxs + 4);
        long long prev = -1;
        for (keel::usize ii : idxs) {
            if (static_cast<long long>(ii) != prev) {
                out.x.push_back(x[ii]);
                out.y.push_back(y[ii]);
                prev = static_cast<long long>(ii);
            }
        }
    };

    for (keel::usize i = i0; i < i1; ++i) {
        keel::f64 xv = x[i], yv = y[i];
        int b = bucketOf(xv);
        if (b != curBucket) {
            flush();
            curBucket = b;
            bFirst = i; bLast = i; bMinI = i; bMaxI = i;
            bMinY = std::isfinite(yv) ? yv : INFINITY;
            bMaxY = std::isfinite(yv) ? yv : -INFINITY;
        } else {
            bLast = i;
            if (std::isfinite(yv)) {
                if (yv < bMinY) { bMinY = yv; bMinI = i; }
                if (yv > bMaxY) { bMaxY = yv; bMaxI = i; }
            }
        }
    }
    flush();
    return out;
}

// M2: {min, max} per pixel column (2 points vs M4's 4) — a lighter mode for
// dense data; keeps spikes / true extent. Mirrors decimate.js decimateM2.
inline DecimatedSeries decimateM2(const keel::f64 *x, const keel::f64 *y, keel::usize n,
                                  keel::f64 x0, keel::f64 x1, int width) {
    DecimatedSeries out;
    keel::usize i0, i1;
    decimVisibleRange(x, n, x0, x1, i0, i1);
    if (i1 <= i0) return out;
    const int cols = width < 1 ? 1 : width;
    const keel::f64 span = (x1 - x0) != 0.0 ? (x1 - x0) : 1.0;
    auto bucketOf = [&](keel::f64 xv) -> int {
        int b = static_cast<int>(std::floor(((xv - x0) / span) * cols));
        if (b < 0) b = 0; else if (b >= cols) b = cols - 1;
        return b;
    };
    out.x.reserve(static_cast<keel::usize>(cols) * 2);
    out.y.reserve(static_cast<keel::usize>(cols) * 2);
    int curBucket = -1;
    keel::usize bMinI = i0, bMaxI = i0;
    keel::f64 bMinY = 0.0, bMaxY = 0.0;
    auto flush = [&]() {
        if (curBucket < 0) return;
        keel::usize a = std::min(bMinI, bMaxI), b = std::max(bMinI, bMaxI);
        out.x.push_back(x[a]); out.y.push_back(y[a]);
        if (b != a) { out.x.push_back(x[b]); out.y.push_back(y[b]); }
    };
    for (keel::usize i = i0; i < i1; ++i) {
        keel::f64 xv = x[i], yv = y[i];
        int bk = bucketOf(xv);
        if (bk != curBucket) {
            flush();
            curBucket = bk; bMinI = i; bMaxI = i;
            bMinY = std::isfinite(yv) ? yv : INFINITY;
            bMaxY = std::isfinite(yv) ? yv : -INFINITY;
        } else if (std::isfinite(yv)) {
            if (yv < bMinY) { bMinY = yv; bMinI = i; }
            if (yv > bMaxY) { bMaxY = yv; bMaxI = i; }
        }
    }
    flush();
    return out;
}

// LTTB: ~threshold points preserving visual shape (smooth trends).
inline DecimatedSeries decimateLTTB(const keel::f64 *x, const keel::f64 *y, keel::usize n,
                                    keel::f64 x0, keel::f64 x1, int threshold) {
    DecimatedSeries out;
    keel::usize i0, i1;
    decimVisibleRange(x, n, x0, x1, i0, i1);
    const keel::usize cnt = i1 - i0;
    if (cnt == 0) return out;
    if (threshold < 3 || static_cast<keel::usize>(threshold) >= cnt) {
        for (keel::usize i = i0; i < i1; ++i) { out.x.push_back(x[i]); out.y.push_back(y[i]); }
        return out;
    }
    out.x.reserve(static_cast<keel::usize>(threshold));
    out.y.reserve(static_cast<keel::usize>(threshold));
    const keel::f64 bucketSize = static_cast<keel::f64>(cnt - 2) / (threshold - 2);
    keel::usize a = i0;                       // first point is always kept
    out.x.push_back(x[i0]); out.y.push_back(y[i0]);
    for (int i = 0; i < threshold - 2; ++i) {
        keel::usize avgStart = i0 + static_cast<keel::usize>(std::floor((i + 1) * bucketSize)) + 1;
        keel::usize avgEnd   = i0 + static_cast<keel::usize>(std::floor((i + 2) * bucketSize)) + 1;
        if (avgEnd > i1) avgEnd = i1;
        keel::f64 avgX = 0.0, avgY = 0.0;
        const keel::usize avgN = avgEnd > avgStart ? (avgEnd - avgStart) : 1;
        for (keel::usize j = avgStart; j < avgEnd; ++j) { avgX += x[j]; avgY += y[j]; }
        avgX /= avgN; avgY /= avgN;
        const keel::usize rangeStart = i0 + static_cast<keel::usize>(std::floor(i * bucketSize)) + 1;
        const keel::usize rangeEnd   = i0 + static_cast<keel::usize>(std::floor((i + 1) * bucketSize)) + 1;
        const keel::f64 ax = x[a], ay = y[a];
        keel::f64 maxArea = -1.0;
        keel::usize maxIdx = rangeStart;
        for (keel::usize j = rangeStart; j < rangeEnd && j < i1; ++j) {
            keel::f64 area = std::fabs((ax - avgX) * (y[j] - ay) - (ax - x[j]) * (avgY - ay));
            if (area > maxArea) { maxArea = area; maxIdx = j; }
        }
        out.x.push_back(x[maxIdx]); out.y.push_back(y[maxIdx]);
        a = maxIdx;
    }
    out.x.push_back(x[i1 - 1]); out.y.push_back(y[i1 - 1]);
    return out;
}

// Dispatcher. algo: 0 = M4 (default), 1 = LTTB, 2 = none (raw visible
// slice). Also returns raw when the visible count is already <= 2*width.
enum class DecimAlgo { M4 = 0, LTTB = 1, None = 2, M2 = 3 };

inline DecimatedSeries decimateSeries(const keel::f64 *x, const keel::f64 *y, keel::usize n,
                                      keel::f64 x0, keel::f64 x1, int width, DecimAlgo algo) {
    const int w = width < 1 ? 1 : width;
    keel::usize i0, i1;
    decimVisibleRange(x, n, x0, x1, i0, i1);
    const keel::usize cnt = i1 - i0;
    if (algo == DecimAlgo::None || cnt <= static_cast<keel::usize>(w) * 2) {
        DecimatedSeries out;
        for (keel::usize i = i0; i < i1; ++i) { out.x.push_back(x[i]); out.y.push_back(y[i]); }
        return out;
    }
    if (algo == DecimAlgo::LTTB) return decimateLTTB(x, y, n, x0, x1, w);
    if (algo == DecimAlgo::M2)   return decimateM2(x, y, n, x0, x1, w);
    return decimateM4(x, y, n, x0, x1, w);
}

// ── LOD pyramid (engine twin of decimate.js buildPyramid/decimateLOD) ───
//
// Returns the COARSER levels only (raw stays the implicit finest level, so
// we never copy the full series). level[0] ≈ raw/8, each next ≤ half, all
// extrema-preserving (M4). Stops at ~baseTarget points. O(N) total.
inline std::vector<DecimatedSeries> buildPyramid(const keel::f64 *x, const keel::f64 *y,
                                                 keel::usize n, keel::usize baseTarget = 8000) {
    std::vector<DecimatedSeries> levels;
    const keel::f64 *cx = x;
    const keel::f64 *cy = y;
    keel::usize s = n;
    while (s > baseTarget) {
        const int cols = static_cast<int>(std::max<keel::usize>(1, (s + 7) / 8));
        DecimatedSeries d = decimateM4(cx, cy, s, cx[0], cx[s - 1], cols);
        if (d.x.size() >= s) break;            // no progress
        levels.push_back(std::move(d));
        cx = levels.back().x.data();
        cy = levels.back().y.data();
        s = levels.back().x.size();
    }
    return levels;
}

// Decimate from the coarsest of {raw, pyramid...} that still has ≥ 2*width
// points in [x0,x1] (finer as you zoom in) → O(width) per call at any zoom.
inline DecimatedSeries decimateLOD(const keel::f64 *rawX, const keel::f64 *rawY, keel::usize rawN,
                                   const std::vector<DecimatedSeries> &pyramid,
                                   keel::f64 x0, keel::f64 x1, int width, DecimAlgo algo) {
    const int w = width < 1 ? 1 : width;
    const keel::f64 *bx = rawX;
    const keel::f64 *by = rawY;
    keel::usize bn = rawN;
    for (int i = static_cast<int>(pyramid.size()) - 1; i >= -1; --i) {
        const keel::f64 *lx; const keel::f64 *ly; keel::usize ln;
        if (i >= 0) { lx = pyramid[i].x.data(); ly = pyramid[i].y.data(); ln = pyramid[i].x.size(); }
        else        { lx = rawX; ly = rawY; ln = rawN; }
        keel::usize a, b;
        decimVisibleRange(lx, ln, x0, x1, a, b);
        if (b - a >= static_cast<keel::usize>(w) * 2 || i == -1) { bx = lx; by = ly; bn = ln; break; }
    }
    return decimateSeries(bx, by, bn, x0, x1, w, algo);
}

}  // namespace numkit
