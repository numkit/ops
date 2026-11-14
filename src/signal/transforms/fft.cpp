// ops/src/signal/transforms/fft.cpp
//
// Public C++ API for 1-D / 2-D / N-D FFT, IFFT, CZT, and interpft, plus
// PRIVATE kernel-dispatch bridges onto standalone keel::fft.

#include <numkit/signal/transforms/fft.hpp>
#include <numkit/ops/fft/fft_kernels.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "../dsp_helpers.hpp"
#include "../signal_detail.hpp"

#include "keel/fft.h"
#include "keel/fft_interop_std.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory_resource>
#include <vector>

// ── Low-level kernel forwarders to keel::fft_kernels ───────────────────
namespace numkit::ops::detail {

void fftRadix2Impl(Complex *buf, std::size_t N, const Complex *W) {
    keel::fft_kernels::fft_radix2_inplace(
        reinterpret_cast<keel::c64 *>(buf),
        N,
        reinterpret_cast<const keel::c64 *>(W));
}

void fftStockhamDispatch(Complex *buf, std::size_t N, const Complex *W) {
    keel::fft_kernels::fft_stockham_inplace(
        reinterpret_cast<keel::c64 *>(buf),
        N,
        reinterpret_cast<const keel::c64 *>(W));
}

void fftRadix2SoaDispatch(Complex *buf, std::size_t N, const Complex *W) {
    keel::fft_kernels::fft_radix2_soa_inplace(
        reinterpret_cast<keel::c64 *>(buf),
        N,
        reinterpret_cast<const keel::c64 *>(W));
}

void fftRadix4Pow4SoaDispatch(Complex *buf, std::size_t N, const Complex *W) {
    keel::fft_kernels::fft_radix4_soa_inplace(
        reinterpret_cast<keel::c64 *>(buf),
        N,
        reinterpret_cast<const keel::c64 *>(W));
}

void fftSoaStagesDispatch(double *re, double *im, std::size_t N, const Complex *W) {
    keel::fft_kernels::fft_soa_stages_inplace(
        re, im, N, reinterpret_cast<const keel::c64 *>(W));
}

} // namespace numkit::ops::detail

namespace numkit::signal {

// ── Shared algorithm core (delegated to keel::fft) ─────────────────────
static Value fftAlongDim(const Value &x_in, size_t N_req, int dim, int dir,
                         std::pmr::memory_resource *mr) {
    Value xPromoted;
    const bool needsPromote = (x_in.type() != ValueType::DOUBLE && !x_in.isComplex());
    if (needsPromote) {
        const auto &di = x_in.dims();
        xPromoted = di.is3D()
            ? Value::matrix3d(di.rows(), di.cols(), di.pages(), ValueType::DOUBLE, mr)
            : Value::matrix(di.rows(), di.cols(), ValueType::DOUBLE, mr);
        double *pd = xPromoted.doubleDataMut();
        for (size_t i = 0; i < x_in.numel(); ++i)
            pd[i] = x_in.elemAsDouble(i);
    }
    const Value &x = needsPromote ? xPromoted : x_in;
    const auto &d = x.dims();
    const size_t R = d.rows();
    const size_t C = d.cols();
    const size_t P = d.is3D() ? d.pages() : 1;

    size_t axisLen = 0;
    switch (dim) {
        case 1: axisLen = R; break;
        case 2: axisLen = C; break;
        case 3: axisLen = P; break;
        default: break;
    }

    const size_t outAxisLen = (N_req > 0) ? N_req : axisLen;

    if (axisLen <= 1 && outAxisLen > 1) {
        throw Error("fft: extending dimension beyond ndims is not supported "
                    "when the axis length is 1",
                    0, 0, "fft", "", "numkit:fft:extendDim");
    }

    size_t outR = R, outC = C, outP = P;
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

    auto dstRef = keel::as_mut_complex_ref(result.complexDataMut(), result.numel());

    if (x.isComplex()) {
        const auto kdir = (dir == +1) ? keel::FftDirection::Forward : keel::FftDirection::Inverse;
        keel::fft_batch(
            keel::as_complex_ref(x.complexData(), x.numel()),
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
        Complex *out = result.complexDataMut();
        const size_t total = result.numel();
        for (size_t i = 0; i < total; ++i) {
            out[i] = Complex(out[i].real() * invN, -out[i].imag() * invN);
        }
    }

    // ifft: downgrade to real when every imaginary part is within tolerance.
    if (dir == -1) {
        bool allReal = true;
        const Complex *out = result.complexData();
        for (size_t i = 0; i < result.numel() && allReal; ++i) {
            if (std::abs(out[i].imag()) > 1e-10)
                allReal = false;
        }
        if (allReal) {
            auto realOut = outIs3D
                ? Value::matrix3d(outR, outC, outP, ValueType::DOUBLE, mr)
                : Value::matrix(outR, outC, ValueType::DOUBLE, mr);
            for (size_t i = 0; i < realOut.numel(); ++i)
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

Value fft(const Value &x, int n, int dim, std::pmr::memory_resource *mr)
{
    if (dim < 0 || dim > 3)
        throw Error("fft: dim must be 0 (auto), 1, 2, or 3",
                     0, 0, "fft", "", "numkit:fft:invalidDim");
    if (dim == 0) dim = resolveDefaultDim(x);

    const size_t N = (n < 0) ? 0u : static_cast<size_t>(n);
    return fftAlongDim(x, N, dim, /*dir=*/1, mr);
}

Value ifft(const Value &X, int n, int dim, std::pmr::memory_resource *mr)
{
    if (dim < 0 || dim > 3)
        throw Error("ifft: dim must be 0 (auto), 1, 2, or 3",
                     0, 0, "ifft", "", "numkit:ifft:invalidDim");
    if (dim == 0) dim = resolveDefaultDim(X);

    const size_t N = (n < 0) ? 0u : static_cast<size_t>(n);
    return fftAlongDim(X, N, dim, /*dir=*/-1, mr);
}

Value ifftSymmetric(const Value &X, int n, int dim, std::pmr::memory_resource *mr)
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

    const std::size_t R = d.rows(), C = d.cols();
    const std::size_t origLen = (useDim == 1) ? R : C;
    const std::size_t L       = (n > 0) ? static_cast<std::size_t>(n) : origLen;

    auto getC = [&](std::size_t idx) -> Complex {
        return X.isComplex() ? X.complexData()[idx]
                             : Complex(X.elemAsDouble(idx), 0.0);
    };

    const std::size_t outR = (useDim == 1) ? L : R;
    const std::size_t outC = (useDim == 1) ? C : L;
    Value Xh = Value::matrix(outR, outC, ValueType::COMPLEX, mr);
    if (Xh.numel() == 0) return ifft(Xh, static_cast<int>(L), useDim, mr);
    Complex *h = Xh.complexDataMut();
    for (std::size_t i = 0; i < Xh.numel(); ++i) h[i] = Complex(0.0, 0.0);

    const std::size_t nslices = (useDim == 1) ? C : R;
    for (std::size_t s = 0; s < nslices; ++s) {
        auto inIdx  = [&](std::size_t j) { return (useDim == 1) ? (s * R + j) : (s + j * R); };
        auto outIdx = [&](std::size_t j) { return (useDim == 1) ? (s * outR + j) : (s + j * outR); };
        for (std::size_t j = 0; j < L; ++j)
            h[outIdx(j)] = (j < origLen) ? getC(inIdx(j)) : Complex(0.0, 0.0);
        h[outIdx(0)] = Complex(h[outIdx(0)].real(), 0.0);
        for (std::size_t k = 1; k <= (L - 1) / 2; ++k)
            h[outIdx(L - k)] = std::conj(h[outIdx(k)]);
        if (L >= 2 && (L % 2) == 0)
            h[outIdx(L / 2)] = Complex(h[outIdx(L / 2)].real(), 0.0);
    }

    Value Y = ifft(Xh, static_cast<int>(L), useDim, mr);
    if (!Y.isComplex()) return Y;
    Value re = Value::matrix(Y.dims().rows(), Y.dims().cols(), ValueType::DOUBLE, mr);
    const Complex *yc = Y.complexData();
    double *rd = re.doubleDataMut();
    for (std::size_t i = 0; i < Y.numel(); ++i) rd[i] = yc[i].real();
    return re;
}

Value ifft2Symmetric(const Value &X, std::pmr::memory_resource *mr)
{
    const auto &d = X.dims();
    if (d.ndim() > 2)
        throw Error("ifft2: the 'symmetric' option supports 2-D inputs only",
                     0, 0, "ifft2", "", "numkit:ifft2:symmetricNdims");
    const std::size_t R = d.rows(), C = d.cols();
    Value y = X;
    if (C > 1) y = ifftSymmetric(y, -1, 2, mr);
    if (R > 1) y = ifftSymmetric(y, -1, 1, mr);
    if (R <= 1 && C <= 1) {
        const Complex z = X.isComplex() ? X.complexData()[0]
                                        : Complex(X.elemAsDouble(0), 0.0);
        return Value::scalar(z.real(), mr);
    }
    return y;
}

Value fft2(const Value &X, int m, int n, std::pmr::memory_resource *mr)
{
    Value step1 = fft(X, m, 1, mr);
    return fft(step1, n, 2, mr);
}

Value ifft2(const Value &X, int m, int n, std::pmr::memory_resource *mr)
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

Value fftnImpl(const Value &X, Span<const std::size_t> sz,
               Value (*op)(const Value &, int, int, std::pmr::memory_resource *),
               std::pmr::memory_resource *mr)
{
    if (X.isEmpty()) return X;
    const int ndim = effectiveNdim(X);
    if (sz.size() > static_cast<std::size_t>(ndim))
        throw Error("fftn: size vector length exceeds ndims(X)",
                     0, 0, "fftn", "", "numkit:fftn:badSize");
    Value Y = X;
    for (int d = 1; d <= ndim; ++d) {
        int n = -1;
        if (static_cast<std::size_t>(d) <= sz.size())
            n = static_cast<int>(sz[d - 1]);
        Y = op(Y, n, d, mr);
    }
    return Y;
}

} // anonymous namespace

Value fftn(const Value &X, Span<const std::size_t> sz,
           std::pmr::memory_resource *mr)
{
    return fftnImpl(X, sz, &fft, mr);
}

Value ifftn(const Value &X, Span<const std::size_t> sz,
            std::pmr::memory_resource *mr)
{
    return fftnImpl(X, sz, &ifft, mr);
}

static Value cztVector(const Complex *xData, std::size_t N, int m, Complex w, Complex a, std::pmr::memory_resource *mr)
{
    using Cd = std::complex<double>;
    if (m <= 0) return Value::complexMatrix(0, 0, mr);
    const std::size_t M = static_cast<std::size_t>(m);
    if (N == 0) {
        return Value::complexMatrix(0, 0, mr);
    }

    std::size_t L = 1;
    while (L < N + M - 1) L <<= 1;

    Value g = Value::complexMatrix(L, 1, mr);
    Cd *gd = g.complexDataMut();
    for (std::size_t n = 0; n < N; ++n) {
        const double n2_half = 0.5 * static_cast<double>(n) * static_cast<double>(n);
        const Cd a_negn = std::pow(a, -static_cast<double>(n));
        const Cd w_n2   = std::pow(w,  n2_half);
        gd[n] = xData[n] * a_negn * w_n2;
    }
    for (std::size_t n = N; n < L; ++n) gd[n] = Cd(0, 0);

    Value h = Value::complexMatrix(L, 1, mr);
    Cd *hd = h.complexDataMut();
    for (std::size_t k = 0; k < M; ++k) {
        const double k2_half = 0.5 * static_cast<double>(k) * static_cast<double>(k);
        hd[k] = std::pow(w, -k2_half);
    }
    for (std::size_t n = M; n < L; ++n) hd[n] = Cd(0, 0);
    for (std::size_t n = 1; n < N; ++n) {
        const double n2_half = 0.5 * static_cast<double>(n) * static_cast<double>(n);
        hd[L - n] = std::pow(w, -n2_half);
    }

    Value G = fft(g, static_cast<int>(L), 1, mr);
    Value H = fft(h, static_cast<int>(L), 1, mr);
    const Cd *Gd = G.complexData();
    const Cd *Hd = H.complexData();
    Value P = Value::complexMatrix(L, 1, mr);
    Cd *Pd = P.complexDataMut();
    for (std::size_t i = 0; i < L; ++i) Pd[i] = Gd[i] * Hd[i];
    Value yL = ifft(P, static_cast<int>(L), 1, mr);

    Value Y = Value::complexMatrix(1, M, mr);
    Cd *Yd = Y.complexDataMut();
    if (yL.isComplex()) {
        const Cd *yd = yL.complexData();
        for (std::size_t k = 0; k < M; ++k) {
            const double k2_half = 0.5 * static_cast<double>(k) * static_cast<double>(k);
            Yd[k] = yd[k] * std::pow(w, k2_half);
        }
    } else {
        const double *yd = yL.doubleData();
        for (std::size_t k = 0; k < M; ++k) {
            const double k2_half = 0.5 * static_cast<double>(k) * static_cast<double>(k);
            Yd[k] = Cd(yd[k], 0.0) * std::pow(w, k2_half);
        }
    }
    return Y;
}

Value czt(const Value &x, int m, Complex w, Complex a, std::pmr::memory_resource *mr)
{
    using Cd = std::complex<double>;
    detail::requireFloatInput(x, "czt");
    if (x.isEmpty()) return Value::complexMatrix(0, 0, mr);

    const auto &d = x.dims();
    const std::size_t R = d.rows();
    const std::size_t C = d.cols();

    const bool isVector = (R == 1 || C == 1) && !d.is3D();
    if (isVector) {
        const std::size_t N = x.numel();
        ScratchArena arena(mr);
        ScratchVec<Cd> buf(N, &arena);
        if (x.isComplex()) {
            const Cd *xc = x.complexData();
            for (std::size_t i = 0; i < N; ++i) buf[i] = xc[i];
        } else {
            for (std::size_t i = 0; i < N; ++i) buf[i] = Cd(x.elemAsDouble(i), 0.0);
        }
        Value y = cztVector(buf.data(), N, m, w, a, mr);
        if (R == 1) return y;
        Value yCol = Value::complexMatrix(static_cast<std::size_t>(m), 1, mr);
        Cd *col = yCol.complexDataMut();
        const Cd *row = y.complexData();
        for (int k = 0; k < m; ++k) col[k] = row[k];
        return yCol;
    }

    const int nd = d.ndim();
    std::vector<std::size_t> outDims(static_cast<std::size_t>(nd));
    outDims[0] = static_cast<std::size_t>(m);
    for (int i = 1; i < nd; ++i)
        outDims[static_cast<std::size_t>(i)] = d.dim(i);
    Value out = (nd <= 2)
        ? Value::complexMatrix(static_cast<std::size_t>(m), C, mr)
        : Value::matrixND(outDims.data(), nd, ValueType::COMPLEX, mr);
    Cd *od = out.complexDataMut();
    const std::size_t nCols = (R > 0) ? (x.numel() / R) : 0;
    ScratchArena arena(mr);
    ScratchVec<Cd> col(R, &arena);
    for (std::size_t c = 0; c < nCols; ++c) {
        if (x.isComplex()) {
            const Cd *xc = x.complexData();
            for (std::size_t r = 0; r < R; ++r) col[r] = xc[c * R + r];
        } else {
            for (std::size_t r = 0; r < R; ++r) col[r] = Cd(x.elemAsDouble(c * R + r), 0.0);
        }
        Value y = cztVector(col.data(), R, m, w, a, mr);
        const Cd *yd = y.complexData();
        for (int k = 0; k < m; ++k) od[c * m + k] = yd[k];
    }
    return out;
}

Value interpft(const Value &x, int n, int dim, std::pmr::memory_resource *mr)
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
    size_t mLen = 1;
    if      (useDim == 1) mLen = d.rows();
    else if (useDim == 2) mLen = d.cols();
    else if (useDim == 3) mLen = d.pages();
    if (mLen == 0)
        return x;
    if (static_cast<size_t>(n) == mLen)
        return x;

    Value X = fft(x, /*n=*/-1, useDim, mr);

    const size_t nOut = static_cast<size_t>(n);
    const size_t half = mLen / 2;

    auto pickN = [&](size_t r, size_t c) -> Value {
        Value Y = Value::complexMatrix(r, c, mr);
        auto *dst = Y.complexDataMut();
        for (size_t i = 0; i < r * c; ++i) dst[i] = {0.0, 0.0};
        return Y;
    };

    auto scaleVal = [&](Value v) -> Value {
        const double s = static_cast<double>(nOut) / static_cast<double>(mLen);
        const size_t total = v.numel();
        if (v.type() == ValueType::COMPLEX) {
            auto *p = v.complexDataMut();
            for (size_t i = 0; i < total; ++i) p[i] *= s;
        } else if (v.type() == ValueType::DOUBLE) {
            auto *p = v.doubleDataMut();
            for (size_t i = 0; i < total; ++i) p[i] *= s;
        }
        return v;
    };

    if (useDim == 1 && (d.ndim() <= 2)) {
        const size_t cols = d.cols();
        Value Y = pickN(nOut, cols);
        const auto *src = X.complexData();
        auto *dst       = Y.complexDataMut();
        for (size_t c = 0; c < cols; ++c) {
            const size_t srcCol = c * mLen;
            const size_t dstCol = c * nOut;
            for (size_t i = 0; i <= half && i < nOut; ++i)
                dst[dstCol + i] = src[srcCol + i];
            const size_t backLen = mLen - half - 1;
            for (size_t i = 0; i < backLen; ++i) {
                if (nOut - backLen + i < nOut)
                    dst[dstCol + nOut - backLen + i] =
                        src[srcCol + half + 1 + i];
            }
            if ((mLen & 1u) == 0u && nOut > mLen) {
                const size_t mid = half;
                dst[dstCol + mid] = src[srcCol + mid] * 0.5;
                if (nOut - mid < nOut)
                    dst[dstCol + nOut - mid] = src[srcCol + mid] * 0.5;
            }
        }
        return scaleVal(ifft(Y, /*n=*/-1, 1, mr));
    }

    if (useDim == 2 && (d.ndim() <= 2)) {
        const size_t rows = d.rows();
        Value Y = pickN(rows, nOut);
        const auto *src = X.complexData();
        auto *dst       = Y.complexDataMut();
        const size_t backLen = mLen - half - 1;
        for (size_t r = 0; r < rows; ++r) {
            for (size_t i = 0; i <= half && i < nOut; ++i)
                dst[i * rows + r] = src[i * rows + r];
            for (size_t i = 0; i < backLen; ++i) {
                if (nOut - backLen + i < nOut)
                    dst[(nOut - backLen + i) * rows + r] =
                        src[(half + 1 + i) * rows + r];
            }
            if ((mLen & 1u) == 0u && nOut > mLen) {
                const size_t mid = half;
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
