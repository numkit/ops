// toolboxes/signal/src/convolution/convolution.cpp
//
// Public C++ API for convolution and friends. See convolution.hpp for
// contracts. Algorithms unchanged from the previous lambda form — only
// moved into named free functions that take std::pmr::memory_resource* explicitly.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/convolution/convolution.hpp>

#include <numkit/ops/conv.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <tuple>
#include <numkit/value/value_type.hpp>

#include "../dsp_helpers.hpp"
#include "../signal_detail.hpp"  // shared argcheck cores

#include <algorithm>
#include <cmath>

namespace numkit::signal {

// ── conv argcheck ─────────────────────────────────────────────────────
// Reference (R2025b) rejects char/cell with a numeric-or-logical class,
// matrices and empties with AorBNotVector, and a numeric shape argument
// with unknownShapeParameter — all before any computation.
static void conv_argcheck(const Value &a, const Value &b, keel::StringRef shape)
{
    detail::requireNumericSignalInput(a, "conv", "a", 1);
    detail::requireNumericSignalInput(b, "conv", "b", 2);
    detail::requireVectorInput(a, "conv", "a", 1);
    detail::requireVectorInput(b, "conv", "b", 2);
    detail::validateShapeOption(shape, "conv");
}

// ── conv ──────────────────────────────────────────────────────────────
// Choose direct vs FFT convolution by ACTUAL cost, not na·nb alone. Direct
// costs na·nb MACs; FFT costs ~K transforms of the zero-padded length. The
// old `na·nb > T²` rule wrongly chose FFT for a long vector × tiny kernel
// (e.g. 1e6 × 3 → a 1e6-point FFT for a 3-tap conv), ~100× slower than direct.
inline bool conv_use_fft(keel::usize na, keel::usize nb)
{
    if (na == 0 || nb == 0) return false;
    const double direct = static_cast<double>(na) * static_cast<double>(nb);
    keel::usize need = na + nb - 1, nfft = 1;
    while (nfft < need) nfft <<= 1;
    const double fft = 6.0 * static_cast<double>(nfft) *
                       std::log2(static_cast<double>(nfft < 2 ? 2 : nfft));
    return direct > fft;
}

// MATLAB conv promotes integer/logical inputs to double — the result is
// always double, never the integer class (unlike kron/cross). Promote up
// front so the real fast-path's doubleData() accessors are valid.
static Value convPromoteToDouble(const Value &v, keel::Allocator *mr)
{
    const auto &d = v.dims();
    Value r = d.is3D() ? Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr)
                       : Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    const keel::usize n = v.numel();
    double *dst = r.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) dst[i] = v.elemAsDouble(i);
    return r;
}

// MATLAB conv output orientation (probed vs R2025b): the result follows
// the LONGER vector; equal lengths give a COLUMN (row only when the first
// input is a row AND strictly longer than a column second input, or both
// are rows).
static bool convRowOut(const Value &a, const Value &b)
{
    auto isCol = [](const Value &v) {
        return v.dims().cols() == 1 && v.numel() > 1;
    };
    if (a.numel() > b.numel()) return !isCol(a);
    if (b.numel() > a.numel()) return !isCol(b);
    return !isCol(a) && !isCol(b);
}

Value conv(const Value &aIn, const Value &bIn, keel::StringRef shape, keel::Allocator *mr)
{
    conv_argcheck(aIn, bIn, shape);
    const bool wantSingle = (aIn.type() == ValueType::SINGLE ||
                             bIn.type() == ValueType::SINGLE);
    auto needsPromote = [](const Value &v) {
        return !v.isComplex() && (v.isLogical() || isIntegerType(v.type()) || v.type() == ValueType::SINGLE);
    };
    Value aHold, bHold;
    const bool aProm = needsPromote(aIn);
    const bool bProm = needsPromote(bIn);
    if (aProm) aHold = convPromoteToDouble(aIn, mr);
    if (bProm) bHold = convPromoteToDouble(bIn, mr);
    const Value &a = aProm ? aHold : aIn;
    const Value &b = bProm ? bHold : bIn;

    const keel::usize na = a.numel(), nb = b.numel();

    // Complex inputs: convolution is BILINEAR, so the real/imag split used for
    // the linear ops does NOT apply — do a genuine complex multiply-accumulate
    // full[n] = sum_k a[k]·b[n-k] (direct; correctness over an FFT path), then
    // the same shape trim. Handles complex×complex and complex×real.
    if (a.isComplex() || b.isComplex()) {
        keel::ArenaTemp cscratch = keel::get_scratch(mr);
        const keel::usize nfull = (na == 0 || nb == 0) ? 0 : na + nb - 1;
        auto toC = [&](const Value &v, keel::usize n) {
            keel::Array<keel::c64> out(n, cscratch.allocator());
            if (v.isComplex()) {
                const keel::c64 *cd = v.complexData();
                for (keel::usize i = 0; i < n; ++i) out[i] = cd[i];
            } else {
                for (keel::usize i = 0; i < n; ++i) out[i] = keel::c64(v.elemAsDouble(i), 0.0);
            }
            return out;
        };
        auto av = toC(a, na);
        auto bv = toC(b, nb);
        keel::Array<keel::c64> full(nfull, cscratch.allocator());
        for (keel::usize i = 0; i < nfull; ++i) full[i] = keel::c64(0.0, 0.0);
        for (keel::usize i = 0; i < na; ++i)
            for (keel::usize j = 0; j < nb; ++j)
                full[i + j] += av[i] * bv[j];

        keel::usize outStartC = 0, outLenC = nfull;
        if (shape == "same") { outLenC = na; outStartC = nb / 2; }
        else if (shape == "valid") {
            outLenC = (na >= nb) ? na - nb + 1 : nb - na + 1;
            outStartC = std::min(na, nb) - 1;
        }
        auto rc = convRowOut(a, b)
                      ? Value::matrix(1, outLenC, ValueType::COMPLEX, mr)
                      : Value::matrix(outLenC, 1, ValueType::COMPLEX, mr);
        keel::c64 *rcd = rc.complexDataMut();
        for (keel::usize i = 0; i < outLenC; ++i) rcd[i] = full[outStartC + i];
        return rc;
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto c = conv_use_fft(na, nb)
        ? convFFT  (scratch.allocator(), a.doubleData(), na, b.doubleData(), nb)
        : ops::convDirect(a.doubleData(), na, b.doubleData(), nb, scratch.allocator());

    const keel::usize nc = c.size();
    keel::usize outStart = 0, outLen = nc;
    if (shape == "same") {
        // MATLAB: 'same' is the central part the SAME SIZE AS THE FIRST input
        // (length na), taken starting at floor(nb/2) (0-based) of the full
        // convolution. conv([1 2 3 4],[1 1],'same')=[3 5 7 4] (not [1 3 5 7]);
        // conv([1 2],[1 1 1 1 1],'same')=[3 3] (length 2, not 5).
        outLen = na;
        outStart = nb / 2;
    } else if (shape == "valid") {
        outLen = (na >= nb) ? na - nb + 1 : nb - na + 1;
        outStart = std::min(na, nb) - 1;
    }

    auto r = convRowOut(a, b)
                 ? Value::matrix(1, outLen, ValueType::DOUBLE, mr)
                 : Value::matrix(outLen, 1, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < outLen; ++i)
        r.doubleDataMut()[i] = c[outStart + i];
    return detail::narrowDoubleToSingleIf(r, wantSingle, mr);
}

// ── deconv ────────────────────────────────────────────────────────────
std::tuple<Value, Value>
deconv(const Value &bIn, const Value &aIn, keel::Allocator *mr)
{
    // MATLAB deconv promotes integer/logical inputs to double — quotient and
    // remainder are always double (never the integer class). Promote up front
    // so the doubleData() accessors below are valid (same rule as conv).
    auto needsPromote = [](const Value &v) {
        return !v.isComplex() && (v.isLogical() || isIntegerType(v.type()));
    };
    Value bHold, aHold;
    const bool bProm = needsPromote(bIn);
    const bool aProm = needsPromote(aIn);
    if (bProm) bHold = convPromoteToDouble(bIn, mr);
    if (aProm) aHold = convPromoteToDouble(aIn, mr);
    const Value &b = bProm ? bHold : bIn;
    const Value &a = aProm ? aHold : aIn;

    const keel::usize nb = b.numel(), na = a.numel();
    if (na > nb) {
        // MATLAB: a divisor longer than the dividend can't be divided once —
        // quotient is the scalar 0, remainder is the numerator unchanged.
        auto qv = Value::matrix(1, 1, ValueType::DOUBLE, mr);
        qv.doubleDataMut()[0] = 0.0;
        auto rv = Value::matrix(1, nb, ValueType::DOUBLE, mr);
        const double *bd = b.doubleData();
        for (keel::usize i = 0; i < nb; ++i) rv.doubleDataMut()[i] = bd[i];
        return std::make_tuple(std::move(qv), std::move(rv));
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> rem(keel::ArrayRef<double>(b.doubleData(), nb), scratch.allocator());
    const double *ad = a.doubleData();

    const keel::usize nq = nb - na + 1;
    auto q = keel::Array<double>(nq, scratch.allocator());

    const double a0 = ad[0];
    if (a0 == 0.0)
        throw Error("deconv: leading coefficient is zero",
                     0, 0, "deconv", "", "numkit:deconv:zeroLead");

    for (keel::usize i = 0; i < nq; ++i) {
        q[i] = rem[i] / a0;
        for (keel::usize j = 0; j < na; ++j)
            rem[i + j] -= q[i] * ad[j];
    }

    auto qv = Value::matrix(1, nq, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < nq; ++i)
        qv.doubleDataMut()[i] = q[i];

    auto rv = Value::matrix(1, nb, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < nb; ++i)
        rv.doubleDataMut()[i] = rem[i];

    return std::make_tuple(std::move(qv), std::move(rv));
}

// ── xcorr ─────────────────────────────────────────────────────────────
std::tuple<Value, Value>
xcorr(const Value &xIn, const Value &yIn, keel::Allocator *mr)
{
    if (xIn.isComplex() || yIn.isComplex()) {
        const keel::usize nx = xIn.numel(), ny = yIn.numel();
        const keel::usize maxLen = std::max(nx, ny);
        const keel::usize nc = (maxLen > 0) ? (2 * maxLen - 1) : 0;
        const int maxLag = static_cast<int>(maxLen) - 1;
        const bool colOut = (xIn.dims().rows() > 1 && xIn.dims().cols() == 1);
        auto r = colOut ? Value::matrix(nc, 1, ValueType::COMPLEX, mr)
                        : Value::matrix(1, nc, ValueType::COMPLEX, mr);
        keel::c64 *rd = r.complexDataMut();
        auto getC = [](const Value &v, keel::usize i) -> keel::c64 {
            if (v.isComplex()) return v.complexData()[i];
            return keel::c64(v.elemAsDouble(i), 0.0);
        };
        for (int m = -maxLag; m <= maxLag; ++m) {
            keel::c64 acc(0.0, 0.0);
            for (keel::usize n = 0; n < maxLen; ++n) {
                long long idxX = static_cast<long long>(n) + m;
                if (idxX >= 0 && static_cast<keel::usize>(idxX) < nx && n < ny) {
                    acc += getC(xIn, static_cast<keel::usize>(idxX)) * keel::conj(getC(yIn, n));
                }
            }
            rd[m + maxLag] = acc;
        }
        auto lags = Value::matrix(1, nc, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < nc; ++i)
            lags.doubleDataMut()[i] = static_cast<double>(static_cast<int>(i) - maxLag);
        return std::make_tuple(std::move(r), std::move(lags));
    }
    Value xHold, yHold;
    const Value &x = (xIn.type() == ValueType::DOUBLE) ? xIn : (xHold = convPromoteToDouble(xIn, mr));
    const Value &y = (yIn.type() == ValueType::DOUBLE) ? yIn : (yHold = convPromoteToDouble(yIn, mr));
    const double *xd = x.doubleData();
    const keel::usize nx = x.numel();
    const double *yd = y.doubleData();
    const keel::usize ny = y.numel();

    const keel::usize maxLen = std::max(nx, ny);
    const keel::usize nc = (maxLen > 0) ? (2 * maxLen - 1) : 0;
    const int maxLag = static_cast<int>(maxLen) - 1;
    const bool colOut = (x.dims().rows() > 1 && x.dims().cols() == 1);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto r = colOut ? Value::matrix(nc, 1, ValueType::DOUBLE, mr)
                    : Value::matrix(1, nc, ValueType::DOUBLE, mr);
    double *rd = r.doubleDataMut();

    if (nc > 0 && conv_use_fft(maxLen, maxLen)) {
        const keel::usize fftLen = nextPow2(nc);
        keel::Array<keel::c64> fx(fftLen, scratch.allocator());
        keel::Array<keel::c64> fy(fftLen, scratch.allocator());
        for (keel::usize i = 0; i < nx; ++i) fx[i] = keel::c64(xd[i], 0.0);
        for (keel::usize i = 0; i < ny; ++i) fy[i] = keel::c64(yd[i], 0.0);

        keel::Array<keel::c64> W_fwd(fftLen / 2, scratch.allocator());
        keel::Array<keel::c64> W_inv(fftLen / 2, scratch.allocator());
        fillFftTwiddles(W_fwd.data(), fftLen, +1);
        fillFftTwiddles(W_inv.data(), fftLen, -1);

        fftRadix2(fx.data(), fftLen, W_fwd.data());
        fftRadix2(fy.data(), fftLen, W_fwd.data());

        for (keel::usize i = 0; i < fftLen; ++i)
            fx[i] *= keel::conj(fy[i]);

        fftRadix2(fx.data(), fftLen, W_inv.data());

        const double invN = 1.0 / static_cast<double>(fftLen);
        const keel::usize negLags = static_cast<keel::usize>(maxLag);
        for (keel::usize i = 0; i < negLags; ++i)
            rd[i] = fx[fftLen - negLags + i].real() * invN;
        for (keel::usize i = 0; i <= negLags; ++i)
            rd[negLags + i] = fx[i].real() * invN;
    } else if (nc > 0) {
        auto xPad = keel::Array<double>(maxLen, scratch.allocator());
        auto yRev = keel::Array<double>(maxLen, scratch.allocator());
        for (keel::usize i = 0; i < maxLen; ++i) {
            xPad[i] = (i < nx) ? xd[i] : 0.0;
            const keel::usize yIdx = maxLen - 1 - i;
            yRev[i] = (yIdx < ny) ? yd[yIdx] : 0.0;
        }
        auto c = ops::convDirect(xPad.data(), maxLen, yRev.data(), maxLen, scratch.allocator());
        for (keel::usize i = 0; i < nc; ++i)
            rd[i] = c[i];
    }

    auto lags = Value::matrix(1, nc, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < nc; ++i)
        lags.doubleDataMut()[i] = static_cast<double>(static_cast<int>(i) - maxLag);

    return std::make_tuple(std::move(r), std::move(lags));
}

// ── Pack 36: xcov ────────────────────────────────────────────────────
// Cross-covariance = xcorr of the mean-removed signals, with MATLAB's
// scaleopt and maxlag handling:
//   'none'     (default) raw sum
//   'biased'   divide every lag by N
//   'unbiased' divide lag m by (N - |m|)
//   'coeff'    divide by sqrt(Cxx(0)*Cyy(0)) so the auto-cov peak is 1
// where N = max(numel(x), numel(y)). maxlag < 0 means "full" (N-1);
// otherwise the result is cropped (or zero-padded) to lags -maxlag..maxlag.
std::tuple<Value, Value>
xcov(const Value &x, const Value &y, int maxlag,
     keel::StringRef scaleopt, keel::Allocator *mr)
{
    auto center = [mr](const Value &v) -> Value {
        const keel::usize n = v.numel();
        Value c = Value::matrix(1, n, ValueType::DOUBLE, mr);
        if (n == 0) return c;
        const double *vd = v.doubleData();
        double sum = 0.0;
        for (keel::usize i = 0; i < n; ++i) sum += vd[i];
        const double m = sum / static_cast<double>(n);
        double *cd = c.doubleDataMut();
        for (keel::usize i = 0; i < n; ++i) cd[i] = vd[i] - m;
        return c;
    };
    Value xc = center(x);
    Value yc = center(y);
    const keel::usize nx = xc.numel(), ny = yc.numel();
    const keel::usize N  = std::max(nx, ny);

    auto [cfull, lagsfull] = xcorr(xc, yc, mr);
    const keel::usize nc = cfull.numel();
    double *cd = cfull.doubleDataMut();
    const double *ld = lagsfull.doubleData();

    // Case-insensitive scaleopt.
    std::string opt(scaleopt.data(), scaleopt.size());
    for (char &ch : opt) if (ch >= 'A' && ch <= 'Z') ch = char(ch + 32);

    if (opt == "biased") {
        const double inv = (N > 0) ? 1.0 / static_cast<double>(N) : 0.0;
        for (keel::usize i = 0; i < nc; ++i) cd[i] *= inv;
    } else if (opt == "unbiased") {
        for (keel::usize i = 0; i < nc; ++i) {
            const double div = static_cast<double>(N) - keel::abs(ld[i]);
            cd[i] = (div > 0.0) ? cd[i] / div : 0.0;
        }
    } else if (opt == "coeff" || opt == "normalized") {
        double c0x = 0.0, c0y = 0.0;
        const double *xd = xc.doubleData();
        const double *yd = yc.doubleData();
        for (keel::usize i = 0; i < nx; ++i) c0x += xd[i] * xd[i];
        for (keel::usize i = 0; i < ny; ++i) c0y += yd[i] * yd[i];
        const double denom = std::sqrt(c0x * c0y);
        if (denom > 0.0)
            for (keel::usize i = 0; i < nc; ++i) cd[i] /= denom;
    } else if (!(opt.empty() || opt == "none")) {
        throw Error("xcov: scaleopt must be 'none', 'biased', 'unbiased', or 'coeff'",
                     0, 0, "xcov", "", "numkit:xcov:badScaleopt");
    }

    const int fullMaxLag = (N > 0) ? static_cast<int>(N) - 1 : 0;
    if (maxlag < 0) maxlag = fullMaxLag;
    if (maxlag == fullMaxLag)
        return std::make_tuple(std::move(cfull), std::move(lagsfull));

    // Crop (maxlag < full) or zero-pad (maxlag > full) about lag 0, which
    // sits at index fullMaxLag in the full vector.
    const int center0 = fullMaxLag;
    const int outLen = 2 * maxlag + 1;
    Value cOut = Value::matrix(1, outLen, ValueType::DOUBLE, mr);
    Value lOut = Value::matrix(1, outLen, ValueType::DOUBLE, mr);
    double *co = cOut.doubleDataMut();
    double *lo = lOut.doubleDataMut();
    for (int m = -maxlag; m <= maxlag; ++m) {
        const int src = center0 + m;
        const int dst = m + maxlag;
        lo[dst] = static_cast<double>(m);
        co[dst] = (src >= 0 && src < static_cast<int>(nc)) ? cd[src] : 0.0;
    }
    return std::make_tuple(std::move(cOut), std::move(lOut));
}

// ── Pack 36: conv2 / filter2 / convn ─────────────────────────────────
// The direct 2-D convolution kernel moved to the kernel layer —
// numkit::ops::conv2Direct (nk/ops/conv.hpp), the 2-D companion to
// ops::convDirect. The Value-level shape-crop + validation stay here.
namespace {

// Crop a "full" 2-D conv result to MATLAB's "same" or "valid" shape.
// Returns a fresh Value sized appropriately. fullR/fullC are dims of
// the full result; A is the first input (size M×N), B is the second
// (size P×Q).
Value cropConv2(const double *full, keel::usize fullR, keel::usize fullC, keel::usize M, keel::usize N, keel::usize P, keel::usize Q, keel::StringRef shape, keel::Allocator *mr)
{
    if (shape == "full") {
        auto out = Value::matrix(fullR, fullC, ValueType::DOUBLE, mr);
        keel::mem_copy(out.doubleDataMut(), full, fullR * fullC * sizeof(double));
        return out;
    }
    if (shape == "same") {
        // Center-crop to size of A. MATLAB picks r0 = floor(P/2),
        // c0 = floor(Q/2) (verified empirically against R2025b for
        // both even and odd P, Q).
        const keel::usize outR = M, outC = N;
        const keel::usize r0 = P / 2;
        const keel::usize c0 = Q / 2;
        auto out = Value::matrix(outR, outC, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (keel::usize cc = 0; cc < outC; ++cc)
            for (keel::usize rr = 0; rr < outR; ++rr)
                od[cc * outR + rr] = full[(cc + c0) * fullR + (rr + r0)];
        return out;
    }
    if (shape == "valid") {
        // Output is max(M-P+1, 0) × max(N-Q+1, 0).
        const keel::usize outR = (M >= P) ? M - P + 1 : 0;
        const keel::usize outC = (N >= Q) ? N - Q + 1 : 0;
        auto out = Value::matrix(outR, outC, ValueType::DOUBLE, mr);
        if (outR == 0 || outC == 0) return out;
        double *od = out.doubleDataMut();
        const keel::usize r0 = P - 1;
        const keel::usize c0 = Q - 1;
        for (keel::usize cc = 0; cc < outC; ++cc)
            for (keel::usize rr = 0; rr < outR; ++rr)
                od[cc * outR + rr] = full[(cc + c0) * fullR + (rr + r0)];
        return out;
    }
    throw Error("conv2: shape must be 'full', 'same', or 'valid'",
                 0, 0, "conv2", "", "numkit:conv2:badShape");
}

static void convMatrixArgcheck(const Value &v, std::string_view fn,
                               std::string_view argName, int argNum)
{
    const ValueType t = v.type();
    if (t == ValueType::DOUBLE || t == ValueType::SINGLE || t == ValueType::COMPLEX)
        return;
    if (t == ValueType::LOGICAL || isIntegerType(t))
        return;                       // promoted by the caller
    detail::throwInputType(v, fn, argName, argNum, "numeric or logical");
}

static Value convPromoteIfNeeded(const Value &v, keel::Allocator *mr)
{
    if (!v.isComplex() && (v.isLogical() || isIntegerType(v.type()) || v.type() == ValueType::SINGLE))
        return convPromoteToDouble(v, mr);
    return Value();                   // empty ⇒ use v unchanged
}

static Value allocLike(const Value &v, ValueType t, keel::Allocator *mr)
{
    const auto &d = v.dims();
    if (d.ndim() <= 2) return Value::matrix(d.rows(), d.cols(), t, mr);
    if (d.ndim() == 3) return Value::matrix3d(d.rows(), d.cols(), d.pages(), t, mr);
    std::vector<keel::usize> od(static_cast<keel::usize>(d.ndim()));
    for (int k = 0; k < d.ndim(); ++k) od[static_cast<keel::usize>(k)] = d.dim(k);
    return Value::matrixND(od.data(), d.ndim(), t, mr);
}

static Value extractPartValue(const Value &v, bool imagPart, keel::Allocator *mr)
{
    Value out = allocLike(v, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    const keel::usize n = v.numel();
    if (v.isComplex()) {
        const keel::c64 *cd = v.complexData();
        for (keel::usize i = 0; i < n; ++i)
            od[i] = imagPart ? cd[i].imag() : cd[i].real();
    } else if (imagPart) {
        std::fill(od, od + n, 0.0);
    } else {
        for (keel::usize i = 0; i < n; ++i)
            od[i] = v.elemAsDouble(i);
    }
    return out;
}

} // namespace

Value conv2(const Value &A, const Value &B, keel::StringRef shape, keel::Allocator *mr)
{
    convMatrixArgcheck(A, "conv2", "A", 1);
    convMatrixArgcheck(B, "conv2", "B", 2);
    detail::validateShapeOption(shape, "conv2");
    if (A.dims().ndim() > 2 || B.dims().ndim() > 2)
        throw Error("conv2: N-D arrays are not supported",
                     0, 0, "conv2", "", "numkit:conv2:nd");
    const bool wantSingle = (A.type() == ValueType::SINGLE ||
                             B.type() == ValueType::SINGLE);
    Value aHold = convPromoteIfNeeded(A, mr);
    Value bHold = convPromoteIfNeeded(B, mr);
    const Value &a = aHold.isEmpty() ? A : aHold;
    const Value &b = bHold.isEmpty() ? B : bHold;

    const keel::usize M = a.dims().rows(), N = a.dims().cols();
    const keel::usize P = b.dims().rows(), Q = b.dims().cols();
    const bool anyCplx = a.isComplex() || b.isComplex();
    if (M == 0 || N == 0 || P == 0 || Q == 0) {
        return Value::matrix(0, 0, anyCplx ? ValueType::COMPLEX : (wantSingle ? ValueType::SINGLE : ValueType::DOUBLE), mr);
    }
    if (anyCplx) {
        Value ar = extractPartValue(a, false, mr);
        Value ai = extractPartValue(a, true, mr);
        Value br = extractPartValue(b, false, mr);
        Value bi = extractPartValue(b, true, mr);
        Value rr = conv2(ar, br, shape, mr);
        const keel::usize nOut = rr.numel();
        Value ri = b.isComplex() ? conv2(ar, bi, shape, mr) : allocLike(rr, ValueType::DOUBLE, mr);
        Value ir = a.isComplex() ? conv2(ai, br, shape, mr) : allocLike(rr, ValueType::DOUBLE, mr);
        Value ii = (a.isComplex() && b.isComplex()) ? conv2(ai, bi, shape, mr) : allocLike(rr, ValueType::DOUBLE, mr);
        Value out = allocLike(rr, ValueType::COMPLEX, mr);
        keel::c64 *od = out.complexDataMut();
        const double *rrd = rr.doubleData();
        const double *rid = ri.doubleData();
        const double *ird = ir.doubleData();
        const double *iid = ii.doubleData();
        for (keel::usize i = 0; i < nOut; ++i)
            od[i] = keel::c64(rrd[i] - iid[i], rid[i] + ird[i]);
        return out;
    }

    const keel::usize fullR = M + P - 1, fullC = N + Q - 1;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> full(fullR * fullC, scratch.allocator());
    ops::conv2Direct(a.doubleData(), M, N, b.doubleData(), P, Q, full.data());

    return detail::narrowDoubleToSingleIf(
        cropConv2(full.data(), fullR, fullC, M, N, P, Q, shape, mr), wantSingle, mr);
}

Value filter2(const Value &h, const Value &X, keel::StringRef shape, keel::Allocator *mr)
{
    convMatrixArgcheck(h, "filter2", "h", 1);
    convMatrixArgcheck(X, "filter2", "X", 2);
    detail::validateShapeOption(shape, "filter2");
    if (h.dims().ndim() > 2 || X.dims().ndim() > 2)
        throw Error("filter2: N-D arrays are not supported",
                     0, 0, "filter2", "", "numkit:filter2:nd");
    const bool wantSingle = (h.type() == ValueType::SINGLE ||
                             X.type() == ValueType::SINGLE);
    Value hHold = convPromoteIfNeeded(h, mr);
    Value xHold = convPromoteIfNeeded(X, mr);
    const Value &hh = hHold.isEmpty() ? h : hHold;
    const Value &xx = xHold.isEmpty() ? X : xHold;

    const keel::usize P = hh.dims().rows(), Q = hh.dims().cols();
    if (hh.isComplex()) {
        auto hf = Value::matrix(P, Q, ValueType::COMPLEX, mr);
        const keel::c64 *src = hh.complexData();
        keel::c64 *dst       = hf.complexDataMut();
        for (keel::usize j = 0; j < Q; ++j)
            for (keel::usize i = 0; i < P; ++i)
                dst[j * P + i] = src[(Q - 1 - j) * P + (P - 1 - i)];
        return conv2(xx, hf, shape, mr);
    }
    auto hf = Value::matrix(P, Q, ValueType::DOUBLE, mr);
    const double *src = hh.doubleData();
    double *dst       = hf.doubleDataMut();
    for (keel::usize j = 0; j < Q; ++j)
        for (keel::usize i = 0; i < P; ++i)
            dst[j * P + i] = src[(Q - 1 - j) * P + (P - 1 - i)];
    return detail::narrowDoubleToSingleIf(conv2(xx, hf, shape, mr), wantSingle, mr);
}

Value convn(const Value &A, const Value &B, keel::StringRef shape, keel::Allocator *mr)
{
    convMatrixArgcheck(A, "convn", "A", 1);
    convMatrixArgcheck(B, "convn", "B", 2);
    detail::validateShapeOption(shape, "convn");
    const bool wantSingle = (A.type() == ValueType::SINGLE ||
                             B.type() == ValueType::SINGLE);
    Value aHold = convPromoteIfNeeded(A, mr);
    Value bHold = convPromoteIfNeeded(B, mr);
    const Value &a = aHold.isEmpty() ? A : aHold;
    const Value &b = bHold.isEmpty() ? B : bHold;
    if (a.isComplex() || b.isComplex()) {
        Value ar = extractPartValue(a, false, mr);
        Value ai = extractPartValue(a, true, mr);
        Value br = extractPartValue(b, false, mr);
        Value bi = extractPartValue(b, true, mr);
        Value rr = convn(ar, br, shape, mr);
        const keel::usize nOut = rr.numel();
        Value ri = b.isComplex() ? convn(ar, bi, shape, mr) : allocLike(rr, ValueType::DOUBLE, mr);
        Value ir = a.isComplex() ? convn(ai, br, shape, mr) : allocLike(rr, ValueType::DOUBLE, mr);
        Value ii = (a.isComplex() && b.isComplex()) ? convn(ai, bi, shape, mr) : allocLike(rr, ValueType::DOUBLE, mr);
        Value out = allocLike(rr, ValueType::COMPLEX, mr);
        keel::c64 *od = out.complexDataMut();
        const double *rrd = rr.doubleData();
        const double *rid = ri.doubleData();
        const double *ird = ir.doubleData();
        const double *iid = ii.doubleData();
        for (keel::usize i = 0; i < nOut; ++i)
            od[i] = keel::c64(rrd[i] - iid[i], rid[i] + ird[i]);
        return out;
    }
    const int da = a.dims().ndim(), db = b.dims().ndim();
    const int nd = std::max(da, db);
    if (nd <= 1) {
        return detail::narrowDoubleToSingleIf(conv(a, b, shape, mr), wantSingle, mr);
    }
    if (nd == 2) {
        return detail::narrowDoubleToSingleIf(conv2(a, b, shape, mr), wantSingle, mr);
    }
    if (nd > 3) {
        std::vector<keel::usize> sa(static_cast<keel::usize>(nd), 1);
        std::vector<keel::usize> sb(static_cast<keel::usize>(nd), 1);
        std::vector<keel::usize> sfull(static_cast<keel::usize>(nd), 1);
        for (int d = 0; d < nd; ++d) {
            if (d < da) sa[static_cast<keel::usize>(d)] = a.dims().dim(d);
            if (d < db) sb[static_cast<keel::usize>(d)] = b.dims().dim(d);
            if (sa[static_cast<keel::usize>(d)] == 0 || sb[static_cast<keel::usize>(d)] == 0)
                return Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
            sfull[static_cast<keel::usize>(d)] = sa[static_cast<keel::usize>(d)] + sb[static_cast<keel::usize>(d)] - 1;
        }
        std::vector<keel::usize> strideFull(static_cast<keel::usize>(nd), 1);
        for (int d = 1; d < nd; ++d)
            strideFull[static_cast<keel::usize>(d)] =
                strideFull[static_cast<keel::usize>(d - 1)] * sfull[static_cast<keel::usize>(d - 1)];

        Value fullV = Value::matrixND(sfull.data(), nd, ValueType::DOUBLE, mr);
        double *out = fullV.doubleDataMut();
        const double *Ad = a.doubleData();
        const double *Bd = b.doubleData();
        const keel::usize na = a.numel(), nb = b.numel();

        std::vector<keel::usize> ca(static_cast<keel::usize>(nd), 0);
        std::vector<keel::usize> cb(static_cast<keel::usize>(nd), 0);
        for (keel::usize ib = 0; ib < nb; ++ib) {
            keel::usize tmpB = ib;
            keel::usize baseOffB = 0;
            for (int d = 0; d < nd; ++d) {
                const keel::usize c = tmpB % sb[static_cast<keel::usize>(d)];
                tmpB /= sb[static_cast<keel::usize>(d)];
                cb[static_cast<keel::usize>(d)] = c;
                baseOffB += c * strideFull[static_cast<keel::usize>(d)];
            }
            const double bv = Bd[ib];
            if (bv == 0.0) continue;
            for (keel::usize ia = 0; ia < na; ++ia) {
                keel::usize tmpA = ia;
                keel::usize outIdx = baseOffB;
                for (int d = 0; d < nd; ++d) {
                    const keel::usize c = tmpA % sa[static_cast<keel::usize>(d)];
                    tmpA /= sa[static_cast<keel::usize>(d)];
                    outIdx += c * strideFull[static_cast<keel::usize>(d)];
                }
                out[outIdx] += Ad[ia] * bv;
            }
        }

        if (shape == "full") return detail::narrowDoubleToSingleIf(fullV, wantSingle, mr);

        std::vector<keel::usize> sout(static_cast<keel::usize>(nd), 1);
        std::vector<keel::usize> off(static_cast<keel::usize>(nd), 0);
        if (shape == "same") {
            for (int d = 0; d < nd; ++d) {
                sout[static_cast<keel::usize>(d)] = sa[static_cast<keel::usize>(d)];
                off[static_cast<keel::usize>(d)] = sb[static_cast<keel::usize>(d)] / 2;
            }
        } else if (shape == "valid") {
            for (int d = 0; d < nd; ++d) {
                if (sa[static_cast<keel::usize>(d)] < sb[static_cast<keel::usize>(d)])
                    return Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
                sout[static_cast<keel::usize>(d)] = sa[static_cast<keel::usize>(d)] - sb[static_cast<keel::usize>(d)] + 1;
                off[static_cast<keel::usize>(d)] = sb[static_cast<keel::usize>(d)] - 1;
            }
        } else {
            throw Error("convn: shape must be 'full', 'same', or 'valid'",
                         0, 0, "convn", "", "numkit:convn:badShape");
        }

        Value cropV = Value::matrixND(sout.data(), nd, ValueType::DOUBLE, mr);
        double *cd = cropV.doubleDataMut();
        const keel::usize ncrop = cropV.numel();
        for (keel::usize ic = 0; ic < ncrop; ++ic) {
            keel::usize tmp = ic;
            keel::usize fullIdx = 0;
            for (int d = 0; d < nd; ++d) {
                const keel::usize c = tmp % sout[static_cast<keel::usize>(d)];
                tmp /= sout[static_cast<keel::usize>(d)];
                fullIdx += (c + off[static_cast<keel::usize>(d)]) * strideFull[static_cast<keel::usize>(d)];
            }
            cd[ic] = out[fullIdx];
        }
        return detail::narrowDoubleToSingleIf(cropV, wantSingle, mr);
    }
    // 3-D: direct nested-loop convolution.
    const keel::usize M = a.dims().rows(), N = a.dims().cols();
    const keel::usize Mp = (a.dims().ndim() == 3) ? a.dims().pages() : 1;
    const keel::usize P = b.dims().rows(), Q = b.dims().cols();
    const keel::usize Pp = (b.dims().ndim() == 3) ? b.dims().pages() : 1;
    const keel::usize outR = M + P - 1, outC = N + Q - 1, outP = Mp + Pp - 1;
    if (M == 0 || N == 0 || P == 0 || Q == 0 || Mp == 0 || Pp == 0) {
        return Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
    }

    auto outV = Value::matrix3d(outR, outC, outP, ValueType::DOUBLE, mr);
    double *out = outV.doubleDataMut();
    const keel::usize pageStrideOut = outR * outC;
    const keel::usize pageStrideA   = M * N;
    const keel::usize pageStrideB   = P * Q;
    const double *Ad = a.doubleData();
    const double *Bd = b.doubleData();
    std::fill_n(out, outR * outC * outP, 0.0);

    for (keel::usize bp = 0; bp < Pp; ++bp) {
        for (keel::usize ap = 0; ap < Mp; ++ap) {
            const keel::usize op = ap + bp;
            for (keel::usize bj = 0; bj < Q; ++bj) {
                for (keel::usize bi = 0; bi < P; ++bi) {
                    const double bv = Bd[bp * pageStrideB + bj * P + bi];
                    if (bv == 0.0) continue;
                    for (keel::usize ac = 0; ac < N; ++ac) {
                        for (keel::usize ar = 0; ar < M; ++ar) {
                            const double av = Ad[ap * pageStrideA + ac * M + ar];
                            out[op * pageStrideOut
                                + (ac + bj) * outR
                                + (ar + bi)] += av * bv;
                        }
                    }
                }
            }
        }
    }

    if (shape == "full") return detail::narrowDoubleToSingleIf(outV, wantSingle, mr);
    if (shape == "same") {
        // Center-crop to A's shape. floor(P/2) offset matches MATLAB's
        // conv2 / convn 'same' rule for both even and odd kernel sizes.
        const keel::usize r0 = P  / 2;
        const keel::usize c0 = Q  / 2;
        const keel::usize p0 = Pp / 2;
        auto crop = Value::matrix3d(M, N, Mp, ValueType::DOUBLE, mr);
        double *cd = crop.doubleDataMut();
        for (keel::usize pp = 0; pp < Mp; ++pp)
            for (keel::usize cc = 0; cc < N; ++cc)
                for (keel::usize rr = 0; rr < M; ++rr)
                    cd[pp * M * N + cc * M + rr] =
                        out[(pp + p0) * pageStrideOut
                            + (cc + c0) * outR
                            + (rr + r0)];
        return detail::narrowDoubleToSingleIf(crop, wantSingle, mr);
    }
    if (shape == "valid") {
        const keel::usize outR2 = (M >= P)  ? M  - P  + 1 : 0;
        const keel::usize outC2 = (N >= Q)  ? N  - Q  + 1 : 0;
        const keel::usize outP2 = (Mp >= Pp)? Mp - Pp + 1 : 0;
        if (outR2 == 0 || outC2 == 0 || outP2 == 0)
            return Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
        auto crop = Value::matrix3d(outR2, outC2, outP2, ValueType::DOUBLE, mr);
        double *cd = crop.doubleDataMut();
        for (keel::usize pp = 0; pp < outP2; ++pp)
            for (keel::usize cc = 0; cc < outC2; ++cc)
                for (keel::usize rr = 0; rr < outR2; ++rr)
                    cd[pp * outR2 * outC2 + cc * outR2 + rr] =
                        out[(pp + Pp - 1) * pageStrideOut
                            + (cc + Q - 1) * outR
                            + (rr + P - 1)];
        return detail::narrowDoubleToSingleIf(crop, wantSingle, mr);
    }
    throw Error("convn: shape must be 'full', 'same', or 'valid'",
                 0, 0, "convn", "", "numkit:convn:badShape");
}

} // namespace numkit::signal
