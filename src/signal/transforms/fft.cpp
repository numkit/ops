// ops/src/signal/transforms/fft.cpp
//
// Public C++ API for 1-D / 2-D / N-D FFT, IFFT, CZT, and interpft, plus
// PRIVATE kernel-dispatch bridges onto standalone keel::fft.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/transforms/fft.hpp>
#include <numkit/ops/fft/fft_kernels.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "../dsp_helpers.hpp"
#include "../signal_detail.hpp"

#include "keel/fft.h"

#include <algorithm>
#include <cmath>

// ── Low-level kernel forwarders to keel::fft_kernels ───────────────────
namespace numkit::ops::detail {

void fftRadix2Impl(keel::c64 *buf, keel::usize N, const keel::c64 *W) {
    keel::fft_kernels::fft_radix2_inplace(buf, N, W);
}

void fftStockhamDispatch(keel::c64 *buf, keel::usize N, const keel::c64 *W) {
    keel::fft_kernels::fft_stockham_inplace(buf, N, W);
}

void fftRadix2SoaDispatch(keel::c64 *buf, keel::usize N, const keel::c64 *W) {
    keel::fft_kernels::fft_radix2_soa_inplace(buf, N, W);
}

void fftRadix4Pow4SoaDispatch(keel::c64 *buf, keel::usize N, const keel::c64 *W) {
    keel::fft_kernels::fft_radix4_soa_inplace(buf, N, W);
}

void fftSoaStagesDispatch(keel::f64 *re, keel::f64 *im, keel::usize N, const keel::c64 *W) {
    keel::fft_kernels::fft_soa_stages_inplace(re, im, N, W);
}

} // namespace numkit::ops::detail

namespace numkit::signal {

// ── Shared algorithm core (delegated to keel::fft) ─────────────────────
static Value fftAlongDim(const Value &x_in, keel::usize N_req, int dim, int dir,
                         keel::Allocator *mr) {
    Value xPromoted;
    const bool needsPromote = (x_in.type() != ValueType::DOUBLE && !x_in.isComplex());
    if (needsPromote) {
        const auto &di = x_in.dims();
        xPromoted = di.is3D()
            ? Value::matrix3d(di.rows(), di.cols(), di.pages(), ValueType::DOUBLE, mr)
            : Value::matrix(di.rows(), di.cols(), ValueType::DOUBLE, mr);
        double *pd = xPromoted.doubleDataMut();
        for (keel::usize i = 0; i < x_in.numel(); ++i)
            pd[i] = x_in.elemAsDouble(i);
    }
    const Value &x = needsPromote ? xPromoted : x_in;
    const auto &d = x.dims();
    const keel::usize R = d.rows();
    const keel::usize C = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;

    keel::usize axisLen = 0;
    switch (dim) {
        case 1: axisLen = R; break;
        case 2: axisLen = C; break;
        case 3: axisLen = P; break;
        default: break;
    }

    const keel::usize outAxisLen = (N_req > 0) ? N_req : axisLen;

    if (axisLen <= 1 && outAxisLen > 1) {
        throw Error("fft: extending dimension beyond ndims is not supported "
                    "when the axis length is 1",
                    0, 0, "fft", "", "numkit:fft:extendDim");
    }

    keel::usize outR = R, outC = C, outP = P;
    switch (dim) {
        case 1: outR = outAxisLen; break;
        case 2: outC = outAxisLen; break;
        case 3: outP = outAxisLen; break;
        default: break;
    }
    const bool outIs3D = d.is3D();

    if (outR == 0 || outC == 0 || outP == 0 || x.numel() == 0) {
        if (dir == -1) {
            return outIs3D
                ? Value::matrix3d(outR, outC, outP, ValueType::DOUBLE, mr)
                : Value::matrix(outR, outC, ValueType::DOUBLE, mr);
        }
        return outIs3D
            ? Value::matrix3d(outR, outC, outP, ValueType::COMPLEX, mr)
            : Value::complexMatrix(outR, outC, mr);
    }

    auto result = outIs3D
        ? Value::matrix3d(outR, outC, outP, ValueType::COMPLEX, mr)
        : Value::complexMatrix(outR, outC, mr);

    const keel::usize dims[3] = {R, C, P};
    const auto layout = keel::FftBatchLayout::from_col_major(
        keel::ArrayRef<keel::usize>(dims, 3),
        static_cast<keel::usize>(dim - 1),
        outAxisLen,
        outAxisLen);

    keel::MutArrayRef<keel::c64> dstRef(result.complexDataMut(), result.numel());

    if (x.isComplex()) {
        const auto kdir = (dir == +1) ? keel::FftDirection::Forward : keel::FftDirection::Inverse;
        keel::fft_batch(
            keel::ArrayRef<keel::c64>(x.complexData(), x.numel()),
            dstRef,
            layout,
            kdir,
            keel::FftNormalization::Backward);
    } else if (dir == +1) {
        keel::fft_batch_real(
            keel::ArrayRef<keel::f64>(x.doubleData(), x.numel()),
            dstRef,
            layout,
            keel::FftNormalization::Backward);
    } else {
        // Real-input IFFT: IFFT(x) = conj(FFT(x)) / N since x is real.
        keel::fft_batch_real(
            keel::ArrayRef<keel::f64>(x.doubleData(), x.numel()),
            dstRef,
            layout,
            keel::FftNormalization::None);
        const double invN = 1.0 / static_cast<double>(outAxisLen);
        keel::c64 *out = result.complexDataMut();
        const keel::usize total = result.numel();
        for (keel::usize i = 0; i < total; ++i) {
            out[i] = keel::c64(out[i].real() * invN, -out[i].imag() * invN);
        }
    }

    // ifft: downgrade to real when every imaginary part is within tolerance.
    if (dir == -1) {
        bool allReal = true;
        const keel::c64 *out = result.complexData();
        for (keel::usize i = 0; i < result.numel() && allReal; ++i) {
            if (keel::abs(out[i].imag()) > 1e-10)
                allReal = false;
        }
        if (allReal) {
            auto realOut = outIs3D
                ? Value::matrix3d(outR, outC, outP, ValueType::DOUBLE, mr)
                : Value::matrix(outR, outC, ValueType::DOUBLE, mr);
            for (keel::usize i = 0; i < realOut.numel(); ++i)
                realOut.doubleDataMut()[i] = result.complexData()[i].real();
            return realOut;
        }
    }

    return result;
}

// ── Public API ─────────────────────────────────────────────────────────

static int resolveDefaultDim(const Value &x)
{
    const auto &d = x.dims();
    if (d.rows() > 1) return 1;
    if (d.cols() > 1) return 2;
    if (d.is3D() && d.pages() > 1) return 3;
    return 1;
}

Value fft(const Value &x, int n, int dim, keel::Allocator *mr)
{
    if (dim < 0 || dim > 3)
        throw Error("fft: dim must be 0 (auto), 1, 2, or 3",
                     0, 0, "fft", "", "numkit:fft:invalidDim");
    if (dim == 0) dim = resolveDefaultDim(x);

    const keel::usize N = (n < 0) ? 0u : static_cast<keel::usize>(n);
    return fftAlongDim(x, N, dim, /*dir=*/1, mr);
}

Value ifft(const Value &X, int n, int dim, keel::Allocator *mr)
{
    if (dim < 0 || dim > 3)
        throw Error("ifft: dim must be 0 (auto), 1, 2, or 3",
                     0, 0, "ifft", "", "numkit:ifft:invalidDim");
    if (dim == 0) dim = resolveDefaultDim(X);

    const keel::usize N = (n < 0) ? 0u : static_cast<keel::usize>(n);
    return fftAlongDim(X, N, dim, /*dir=*/-1, mr);
}

Value ifftSymmetric(const Value &X, int n, int dim, keel::Allocator *mr)
{
    detail::requireNumericOrLogicalInput(X, "ifft", "X", 1);
    const auto &d = X.dims();
    if (d.ndim() > 2)
        throw Error("ifft: the 'symmetric' option supports vectors and matrices only",
                     0, 0, "ifft", "", "numkit:ifft:symmetricNdims");
    int useDim = (dim == 0) ? resolveDefaultDim(X) : dim;
    if (useDim != 1 && useDim != 2)
        throw Error("ifft: the 'symmetric' option supports dim 1 or 2",
                     0, 0, "ifft", "", "numkit:ifft:symmetricDim");

    const keel::usize R = d.rows(), C = d.cols();
    const keel::usize origLen = (useDim == 1) ? R : C;
    const keel::usize L       = (n > 0) ? static_cast<keel::usize>(n) : origLen;

    auto getC = [&](keel::usize idx) -> keel::c64 {
        return X.isComplex() ? X.complexData()[idx]
                             : keel::c64(X.elemAsDouble(idx), 0.0);
    };

    const keel::usize outR = (useDim == 1) ? L : R;
    const keel::usize outC = (useDim == 1) ? C : L;
    Value Xh = Value::matrix(outR, outC, ValueType::COMPLEX, mr);
    if (Xh.numel() == 0) return ifft(Xh, static_cast<int>(L), useDim, mr);
    keel::c64 *h = Xh.complexDataMut();
    for (keel::usize i = 0; i < Xh.numel(); ++i) h[i] = keel::c64(0.0, 0.0);

    const keel::usize nslices = (useDim == 1) ? C : R;
    for (keel::usize s = 0; s < nslices; ++s) {
        auto inIdx  = [&](keel::usize j) { return (useDim == 1) ? (s * R + j) : (s + j * R); };
        auto outIdx = [&](keel::usize j) { return (useDim == 1) ? (s * outR + j) : (s + j * outR); };
        for (keel::usize j = 0; j < L; ++j)
            h[outIdx(j)] = (j < origLen) ? getC(inIdx(j)) : keel::c64(0.0, 0.0);
        h[outIdx(0)] = keel::c64(h[outIdx(0)].real(), 0.0);
        for (keel::usize k = 1; k <= (L - 1) / 2; ++k)
            h[outIdx(L - k)] = keel::conj(h[outIdx(k)]);
        if (L >= 2 && (L % 2) == 0)
            h[outIdx(L / 2)] = keel::c64(h[outIdx(L / 2)].real(), 0.0);
    }

    Value Y = ifft(Xh, static_cast<int>(L), useDim, mr);
    if (!Y.isComplex()) return Y;
    Value re = Value::matrix(Y.dims().rows(), Y.dims().cols(), ValueType::DOUBLE, mr);
    const keel::c64 *yc = Y.complexData();
    double *rd = re.doubleDataMut();
    for (keel::usize i = 0; i < Y.numel(); ++i) rd[i] = yc[i].real();
    return re;
}

Value ifft2Symmetric(const Value &X, keel::Allocator *mr)
{
    const auto &d = X.dims();
    if (d.ndim() > 2)
        throw Error("ifft2: the 'symmetric' option supports 2-D inputs only",
                     0, 0, "ifft2", "", "numkit:ifft2:symmetricNdims");
    const keel::usize R = d.rows(), C = d.cols();
    Value y = X;
    if (C > 1) y = ifftSymmetric(y, -1, 2, mr);
    if (R > 1) y = ifftSymmetric(y, -1, 1, mr);
    if (R <= 1 && C <= 1) {
        const keel::c64 z = X.isComplex() ? X.complexData()[0]
                                        : keel::c64(X.elemAsDouble(0), 0.0);
        return Value::scalar(z.real(), mr);
    }
    return y;
}

Value fft2(const Value &X, int m, int n, keel::Allocator *mr)
{
    Value step1 = fft(X, m, 1, mr);
    return fft(step1, n, 2, mr);
}

Value ifft2(const Value &X, int m, int n, keel::Allocator *mr)
{
    Value step1 = ifft(X, m, 1, mr);
    return ifft(step1, n, 2, mr);
}

namespace {

inline int effectiveNdim(const Value &X)
{
    const auto &d = X.dims();
    if (d.is3D()) return 3;
    if (d.cols() > 1 || d.rows() == 0) return 2;
    return 2;
}

Value fftnImpl(const Value &X, keel::ArrayRef<keel::usize> sz,
               Value (*op)(const Value &, int, int, keel::Allocator *),
               keel::Allocator *mr)
{
    if (X.isEmpty()) return X;
    const int ndim = effectiveNdim(X);
    if (sz.size() > static_cast<keel::usize>(ndim))
        throw Error("fftn: size vector length exceeds ndims(X)",
                     0, 0, "fftn", "", "numkit:fftn:badSize");
    Value Y = X;
    for (int d = 1; d <= ndim; ++d) {
        int n = -1;
        if (static_cast<keel::usize>(d) <= sz.size())
            n = static_cast<int>(sz[d - 1]);
        Y = op(Y, n, d, mr);
    }
    return Y;
}

} // anonymous namespace

Value fftn(const Value &X, keel::ArrayRef<keel::usize> sz,
           keel::Allocator *mr)
{
    return fftnImpl(X, sz, &fft, mr);
}

Value ifftn(const Value &X, keel::ArrayRef<keel::usize> sz,
            keel::Allocator *mr)
{
    return fftnImpl(X, sz, &ifft, mr);
}

static Value cztVector(const keel::c64 *xData, keel::usize N, int m, keel::c64 w, keel::c64 a, keel::Allocator *mr)
{
if (m <= 0) return Value::complexMatrix(0, 0, mr);
    const keel::usize M = static_cast<keel::usize>(m);
    if (N == 0) {
        return Value::complexMatrix(0, 0, mr);
    }

    keel::usize L = 1;
    while (L < N + M - 1) L <<= 1;

    Value g = Value::complexMatrix(L, 1, mr);
    keel::c64 *gd = g.complexDataMut();
    for (keel::usize n = 0; n < N; ++n) {
        const double n2_half = 0.5 * static_cast<double>(n) * static_cast<double>(n);
        const keel::c64 a_negn = keel::pow(a, -static_cast<double>(n));
        const keel::c64 w_n2   = keel::pow(w,  n2_half);
        gd[n] = xData[n] * a_negn * w_n2;
    }
    for (keel::usize n = N; n < L; ++n) gd[n] = keel::c64(0, 0);

    Value h = Value::complexMatrix(L, 1, mr);
    keel::c64 *hd = h.complexDataMut();
    for (keel::usize k = 0; k < M; ++k) {
        const double k2_half = 0.5 * static_cast<double>(k) * static_cast<double>(k);
        hd[k] = keel::pow(w, -k2_half);
    }
    for (keel::usize n = M; n < L; ++n) hd[n] = keel::c64(0, 0);
    for (keel::usize n = 1; n < N; ++n) {
        const double n2_half = 0.5 * static_cast<double>(n) * static_cast<double>(n);
        hd[L - n] = keel::pow(w, -n2_half);
    }

    Value G = fft(g, static_cast<int>(L), 1, mr);
    Value H = fft(h, static_cast<int>(L), 1, mr);
    const keel::c64 *Gd = G.complexData();
    const keel::c64 *Hd = H.complexData();
    Value P = Value::complexMatrix(L, 1, mr);
    keel::c64 *Pd = P.complexDataMut();
    for (keel::usize i = 0; i < L; ++i) Pd[i] = Gd[i] * Hd[i];
    Value yL = ifft(P, static_cast<int>(L), 1, mr);

    Value Y = Value::complexMatrix(1, M, mr);
    keel::c64 *Yd = Y.complexDataMut();
    if (yL.isComplex()) {
        const keel::c64 *yd = yL.complexData();
        for (keel::usize k = 0; k < M; ++k) {
            const double k2_half = 0.5 * static_cast<double>(k) * static_cast<double>(k);
            Yd[k] = yd[k] * keel::pow(w, k2_half);
        }
    } else {
        const double *yd = yL.doubleData();
        for (keel::usize k = 0; k < M; ++k) {
            const double k2_half = 0.5 * static_cast<double>(k) * static_cast<double>(k);
            Yd[k] = keel::c64(yd[k], 0.0) * keel::pow(w, k2_half);
        }
    }
    return Y;
}

Value czt(const Value &x, int m, keel::c64 w, keel::c64 a, keel::Allocator *mr)
{
detail::requireFloatInput(x, "czt");
    if (x.isEmpty()) return Value::complexMatrix(0, 0, mr);

    const auto &d = x.dims();
    const keel::usize R = d.rows();
    const keel::usize C = d.cols();

    const bool isVector = (R == 1 || C == 1) && !d.is3D();
    if (isVector) {
        const keel::usize N = x.numel();
        keel::ArenaTemp arena = keel::get_scratch(mr);
        keel::Array<keel::c64> buf(N, arena.allocator());
        if (x.isComplex()) {
            const keel::c64 *xc = x.complexData();
            for (keel::usize i = 0; i < N; ++i) buf[i] = xc[i];
        } else {
            for (keel::usize i = 0; i < N; ++i) buf[i] = keel::c64(x.elemAsDouble(i), 0.0);
        }
        Value y = cztVector(buf.data(), N, m, w, a, mr);
        if (R == 1) return y;
        Value yCol = Value::complexMatrix(static_cast<keel::usize>(m), 1, mr);
        keel::c64 *col = yCol.complexDataMut();
        const keel::c64 *row = y.complexData();
        for (int k = 0; k < m; ++k) col[k] = row[k];
        return yCol;
    }

    const int nd = d.ndim();
    keel::Array<keel::usize> outDims(static_cast<keel::usize>(nd), mr);
    outDims[0] = static_cast<keel::usize>(m);
    for (int i = 1; i < nd; ++i)
        outDims[static_cast<keel::usize>(i)] = d.dim(i);
    Value out = (nd <= 2)
        ? Value::complexMatrix(static_cast<keel::usize>(m), C, mr)
        : Value::matrixND(outDims.data(), nd, ValueType::COMPLEX, mr);
    keel::c64 *od = out.complexDataMut();
    const keel::usize nCols = (R > 0) ? (x.numel() / R) : 0;
    keel::ArenaTemp arena = keel::get_scratch(mr);
    keel::Array<keel::c64> col(R, arena.allocator());
    for (keel::usize c = 0; c < nCols; ++c) {
        if (x.isComplex()) {
            const keel::c64 *xc = x.complexData();
            for (keel::usize r = 0; r < R; ++r) col[r] = xc[c * R + r];
        } else {
            for (keel::usize r = 0; r < R; ++r) col[r] = keel::c64(x.elemAsDouble(c * R + r), 0.0);
        }
        Value y = cztVector(col.data(), R, m, w, a, mr);
        const keel::c64 *yd = y.complexData();
        for (int k = 0; k < m; ++k) od[c * m + k] = yd[k];
    }
    return out;
}

Value interpft(const Value &x, int n, int dim, keel::Allocator *mr)
{
    if (n <= 0)
        throw Error("interpft: output length n must be positive",
                     0, 0, "interpft", "", "numkit:interpft:badN");
    if (dim < 0 || dim > 3)
        throw Error("interpft: dim must be 0 (auto), 1, 2, or 3",
                     0, 0, "interpft", "", "numkit:interpft:badDim");

    int useDim = dim;
    if (useDim == 0) {
        const auto &d = x.dims();
        if (d.rows() > 1)                   useDim = 1;
        else if (d.cols() > 1)              useDim = 2;
        else if (d.is3D() && d.pages() > 1) useDim = 3;
        else                                useDim = 1;
    }

    const auto &d = x.dims();
    keel::usize mLen = 1;
    if      (useDim == 1) mLen = d.rows();
    else if (useDim == 2) mLen = d.cols();
    else if (useDim == 3) mLen = d.pages();
    if (mLen == 0)
        return x;
    if (static_cast<keel::usize>(n) == mLen)
        return x;

    Value X = fft(x, /*n=*/-1, useDim, mr);

    const keel::usize nOut = static_cast<keel::usize>(n);
    const keel::usize half = mLen / 2;

    auto pickN = [&](keel::usize r, keel::usize c) -> Value {
        Value Y = Value::complexMatrix(r, c, mr);
        auto *dst = Y.complexDataMut();
        for (keel::usize i = 0; i < r * c; ++i) dst[i] = {0.0, 0.0};
        return Y;
    };

    auto scaleVal = [&](Value v) -> Value {
        const double s = static_cast<double>(nOut) / static_cast<double>(mLen);
        const keel::usize total = v.numel();
        if (v.type() == ValueType::COMPLEX) {
            auto *p = v.complexDataMut();
            for (keel::usize i = 0; i < total; ++i) p[i] *= s;
        } else if (v.type() == ValueType::DOUBLE) {
            auto *p = v.doubleDataMut();
            for (keel::usize i = 0; i < total; ++i) p[i] *= s;
        }
        return v;
    };

    if (useDim == 1 && (d.ndim() <= 2)) {
        const keel::usize cols = d.cols();
        Value Y = pickN(nOut, cols);
        const auto *src = X.complexData();
        auto *dst       = Y.complexDataMut();
        for (keel::usize c = 0; c < cols; ++c) {
            const keel::usize srcCol = c * mLen;
            const keel::usize dstCol = c * nOut;
            for (keel::usize i = 0; i <= half && i < nOut; ++i)
                dst[dstCol + i] = src[srcCol + i];
            const keel::usize backLen = mLen - half - 1;
            for (keel::usize i = 0; i < backLen; ++i) {
                if (nOut - backLen + i < nOut)
                    dst[dstCol + nOut - backLen + i] =
                        src[srcCol + half + 1 + i];
            }
            if ((mLen & 1u) == 0u && nOut > mLen) {
                const keel::usize mid = half;
                dst[dstCol + mid] = src[srcCol + mid] * 0.5;
                if (nOut - mid < nOut)
                    dst[dstCol + nOut - mid] = src[srcCol + mid] * 0.5;
            }
        }
        return scaleVal(ifft(Y, /*n=*/-1, 1, mr));
    }

    if (useDim == 2 && (d.ndim() <= 2)) {
        const keel::usize rows = d.rows();
        Value Y = pickN(rows, nOut);
        const auto *src = X.complexData();
        auto *dst       = Y.complexDataMut();
        const keel::usize backLen = mLen - half - 1;
        for (keel::usize r = 0; r < rows; ++r) {
            for (keel::usize i = 0; i <= half && i < nOut; ++i)
                dst[i * rows + r] = src[i * rows + r];
            for (keel::usize i = 0; i < backLen; ++i) {
                if (nOut - backLen + i < nOut)
                    dst[(nOut - backLen + i) * rows + r] =
                        src[(half + 1 + i) * rows + r];
            }
            if ((mLen & 1u) == 0u && nOut > mLen) {
                const keel::usize mid = half;
                dst[mid * rows + r] = src[mid * rows + r] * 0.5;
                if (nOut - mid < nOut)
                    dst[(nOut - mid) * rows + r] = src[mid * rows + r] * 0.5;
            }
        }
        return scaleVal(ifft(Y, /*n=*/-1, 2, mr));
    }

    throw Error("interpft: dim 3 / N-D inputs not yet supported",
                 0, 0, "interpft", "", "numkit:interpft:dimUnsupported");
}

} // namespace numkit::signal
