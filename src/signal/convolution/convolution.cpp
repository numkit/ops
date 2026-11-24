// toolboxes/signal/src/convolution/convolution.cpp
//
// Public C++ API for convolution and friends. See convolution.hpp for
// contracts. Algorithms unchanged from the previous lambda form — only
// moved into named free functions that take std::pmr::memory_resource* explicitly.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <keel/signal.h>
#include <numkit/signal/convolution/convolution.hpp>

#include <numkit/ops/conv.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <keel/tuple.h>
#include <numkit/value/value_type.hpp>

#include "../dsp_helpers.hpp"
#include "../signal_detail.hpp"  // shared argcheck cores


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
    const keel::f64 direct = keel::as<keel::f64>(na) * keel::as<keel::f64>(nb);
    keel::usize need = na + nb - 1, nfft = 1;
    while (nfft < need) nfft <<= 1;
    const keel::f64 fft = 6.0 * keel::as<keel::f64>(nfft) *
                       keel::log2(keel::as<keel::f64>(nfft < 2 ? 2 : nfft));
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
    keel::f64 *dst = r.doubleDataMut();
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
            outStartC = keel::min(na, nb) - 1;
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
        outStart = keel::min(na, nb) - 1;
    }

    auto r = convRowOut(a, b)
                 ? Value::matrix(1, outLen, ValueType::DOUBLE, mr)
                 : Value::matrix(outLen, 1, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < outLen; ++i)
        r.doubleDataMut()[i] = c[outStart + i];
    return detail::narrowDoubleToSingleIf(r, wantSingle, mr);
}

// ── deconv ────────────────────────────────────────────────────────────
keel::Tuple<Value, Value>
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
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> q(scratch.allocator()), rem(scratch.allocator());
    const keel::signal::SignalStatus st = keel::signal::deconv(
        keel::ArrayRef<keel::f64>(b.doubleData(), nb),
        keel::ArrayRef<keel::f64>(a.doubleData(), na), q, rem, scratch.allocator());
    if (st == keel::signal::SignalStatus::kZeroLeadingCoeff)
        throw Error("deconv: leading coefficient is zero",
                     0, 0, "deconv", "", "numkit:deconv:zeroLead");

    auto qv = Value::matrix(1, q.size(), ValueType::DOUBLE, mr);
    if (!q.empty()) keel::copy_n(qv.doubleDataMut(), q.data(), q.size());
    auto rv = Value::matrix(1, rem.size(), ValueType::DOUBLE, mr);
    if (!rem.empty()) keel::copy_n(rv.doubleDataMut(), rem.data(), rem.size());
    return keel::make_tuple(keel::move(qv), keel::move(rv));
}

// ── xcorr ─────────────────────────────────────────────────────────────
keel::Tuple<Value, Value>
xcorr(const Value &xIn, const Value &yIn, keel::Allocator *mr)
{
    if (xIn.isComplex() || yIn.isComplex()) {
        const keel::usize nx = xIn.numel(), ny = yIn.numel();
        const keel::usize maxLen = keel::max(nx, ny);
        const keel::usize nc = (maxLen > 0) ? (2 * maxLen - 1) : 0;
        const bool colOut = (xIn.dims().rows() > 1 && xIn.dims().cols() == 1);
        auto r = colOut ? Value::matrix(nc, 1, ValueType::COMPLEX, mr)
                        : Value::matrix(1, nc, ValueType::COMPLEX, mr);
        auto lags = Value::matrix(1, nc, ValueType::DOUBLE, mr);
        if (nc == 0) return keel::make_tuple(keel::move(r), keel::move(lags));

        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<keel::c64> xc(nx, scratch.allocator()), yc(ny, scratch.allocator());
        if (xIn.isComplex()) keel::copy_n(xc.data(), xIn.complexData(), nx);
        else for (keel::usize i = 0; i < nx; ++i) xc[i] = keel::c64(xIn.elemAsDouble(i), 0.0);
        if (yIn.isComplex()) keel::copy_n(yc.data(), yIn.complexData(), ny);
        else for (keel::usize i = 0; i < ny; ++i) yc[i] = keel::c64(yIn.elemAsDouble(i), 0.0);

        (void)keel::signal::xcorr_c64(
            xc.as_ref(), yc.as_ref(),
            keel::MutArrayRef<keel::c64>(r.complexDataMut(), nc),
            keel::MutArrayRef<keel::f64>(lags.doubleDataMut(), nc),
            scratch.allocator());
        return keel::make_tuple(keel::move(r), keel::move(lags));
    }
    Value xHold, yHold;
    const Value &x = (xIn.type() == ValueType::DOUBLE) ? xIn : (xHold = convPromoteToDouble(xIn, mr));
    const Value &y = (yIn.type() == ValueType::DOUBLE) ? yIn : (yHold = convPromoteToDouble(yIn, mr));
    const keel::usize nx = x.numel(), ny = y.numel();
    const keel::usize maxLen = keel::max(nx, ny);
    const keel::usize nc = (maxLen > 0) ? (2 * maxLen - 1) : 0;
    const bool colOut = (x.dims().rows() > 1 && x.dims().cols() == 1);

    auto r = colOut ? Value::matrix(nc, 1, ValueType::DOUBLE, mr)
                    : Value::matrix(1, nc, ValueType::DOUBLE, mr);
    auto lags = Value::matrix(1, nc, ValueType::DOUBLE, mr);
    if (nc > 0) {
        (void)keel::signal::xcorr(
            keel::ArrayRef<keel::f64>(x.doubleData(), nx),
            keel::ArrayRef<keel::f64>(y.doubleData(), ny),
            keel::MutArrayRef<keel::f64>(r.doubleDataMut(), nc),
            keel::MutArrayRef<keel::f64>(lags.doubleDataMut(), nc), mr);
    }
    return keel::make_tuple(keel::move(r), keel::move(lags));
}

// ── Pack 36: xcov ────────────────────────────────────────────────────
keel::Tuple<Value, Value>
xcov(const Value &x, const Value &y, keel::i32 maxlag,
     keel::StringRef scaleopt, keel::Allocator *mr)
{
    keel::signal::XcorrScale sc = keel::signal::XcorrScale::kNone;
    if (scaleopt.equals_ignore_ascii_case("biased")) {
        sc = keel::signal::XcorrScale::kBiased;
    } else if (scaleopt.equals_ignore_ascii_case("unbiased")) {
        sc = keel::signal::XcorrScale::kUnbiased;
    } else if (scaleopt.equals_ignore_ascii_case("coeff") ||
               scaleopt.equals_ignore_ascii_case("normalized")) {
        sc = keel::signal::XcorrScale::kCoeff;
    } else if (!(scaleopt.empty() || scaleopt.equals_ignore_ascii_case("none"))) {
        throw Error("xcov: scaleopt must be 'none', 'biased', 'unbiased', or 'coeff'",
                     0, 0, "xcov", "", "numkit:xcov:badScaleopt");
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> cOut(scratch.allocator()), lOut(scratch.allocator());
    (void)keel::signal::xcov(
        keel::ArrayRef<keel::f64>(x.doubleData(), x.numel()),
        keel::ArrayRef<keel::f64>(y.doubleData(), y.numel()),
        maxlag, sc, cOut, lOut, scratch.allocator());

    const bool colOut = (x.dims().rows() > 1 && x.dims().cols() == 1) && (maxlag < 0);
    Value cv = colOut ? Value::matrix(cOut.size(), 1, ValueType::DOUBLE, mr)
                      : Value::matrix(1, cOut.size(), ValueType::DOUBLE, mr);
    Value lv = Value::matrix(1, lOut.size(), ValueType::DOUBLE, mr);
    if (!cOut.empty()) keel::copy_n(cv.doubleDataMut(), cOut.data(), cOut.size());
    if (!lOut.empty()) keel::copy_n(lv.doubleDataMut(), lOut.data(), lOut.size());
    return keel::make_tuple(keel::move(cv), keel::move(lv));
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
Value cropConv2(const keel::f64 *full, keel::usize fullR, keel::usize fullC, keel::usize M, keel::usize N, keel::usize P, keel::usize Q, keel::StringRef shape, keel::Allocator *mr)
{
    if (shape == "full") {
        auto out = Value::matrix(fullR, fullC, ValueType::DOUBLE, mr);
        keel::copy_n(out.doubleDataMut(), full, fullR * fullC);
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
        keel::f64 *od = out.doubleDataMut();
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
        keel::f64 *od = out.doubleDataMut();
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

static void convMatrixArgcheck(const Value &v, keel::StringRef fn,
                               keel::StringRef argName, keel::i32 argNum)
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
    keel::Array<keel::usize> od(d.ndim(), mr);
    for (keel::i32 k = 0; k < d.ndim(); ++k) od[keel::as<keel::usize>(k)] = d.dim(k);
    return Value::matrixND(od.data(), d.ndim(), t, mr);
}

static Value extractPartValue(const Value &v, bool imagPart, keel::Allocator *mr)
{
    Value out = allocLike(v, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::usize n = v.numel();
    if (v.isComplex()) {
        const keel::c64 *cd = v.complexData();
        for (keel::usize i = 0; i < n; ++i)
            od[i] = imagPart ? cd[i].imag() : cd[i].real();
    } else if (imagPart) {
        keel::fill(od, od + n, 0.0);
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
        const keel::f64 *rrd = rr.doubleData();
        const keel::f64 *rid = ri.doubleData();
        const keel::f64 *ird = ir.doubleData();
        const keel::f64 *iid = ii.doubleData();
        for (keel::usize i = 0; i < nOut; ++i)
            od[i] = keel::c64(rrd[i] - iid[i], rid[i] + ird[i]);
        return out;
    }

    const keel::usize fullR = M + P - 1, fullC = N + Q - 1;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> full(fullR * fullC, scratch.allocator());
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
    const keel::f64 *src = hh.doubleData();
    keel::f64 *dst       = hf.doubleDataMut();
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
        const keel::f64 *rrd = rr.doubleData();
        const keel::f64 *rid = ri.doubleData();
        const keel::f64 *ird = ir.doubleData();
        const keel::f64 *iid = ii.doubleData();
        for (keel::usize i = 0; i < nOut; ++i)
            od[i] = keel::c64(rrd[i] - iid[i], rid[i] + ird[i]);
        return out;
    }
    const keel::i32 da = a.dims().ndim(), db = b.dims().ndim();
    const keel::i32 nd = keel::max(da, db);
    if (nd <= 1) {
        return detail::narrowDoubleToSingleIf(conv(a, b, shape, mr), wantSingle, mr);
    }
    if (nd == 2) {
        return detail::narrowDoubleToSingleIf(conv2(a, b, shape, mr), wantSingle, mr);
    }
    if (nd > 3) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        const keel::usize ndu = keel::as<keel::usize>(nd);
        keel::Array<keel::usize> sa(ndu, keel::as<keel::usize>(1), scratch.allocator());
        keel::Array<keel::usize> sb(ndu, keel::as<keel::usize>(1), scratch.allocator());
        keel::Array<keel::usize> sfull(ndu, keel::as<keel::usize>(1), scratch.allocator());
        for (keel::i32 d = 0; d < nd; ++d) {
            if (d < da) sa[keel::as<keel::usize>(d)] = a.dims().dim(d);
            if (d < db) sb[keel::as<keel::usize>(d)] = b.dims().dim(d);
            if (sa[keel::as<keel::usize>(d)] == 0 || sb[keel::as<keel::usize>(d)] == 0)
                return Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
            sfull[keel::as<keel::usize>(d)] = sa[keel::as<keel::usize>(d)] + sb[keel::as<keel::usize>(d)] - 1;
        }
        keel::Array<keel::usize> strideFull(ndu, keel::as<keel::usize>(1), scratch.allocator());
        for (keel::i32 d = 1; d < nd; ++d)
            strideFull[keel::as<keel::usize>(d)] =
                strideFull[keel::as<keel::usize>(d - 1)] * sfull[keel::as<keel::usize>(d - 1)];

        Value fullV = Value::matrixND(sfull.data(), nd, ValueType::DOUBLE, mr);
        keel::f64 *out = fullV.doubleDataMut();
        const keel::f64 *Ad = a.doubleData();
        const keel::f64 *Bd = b.doubleData();
        const keel::usize na = a.numel(), nb = b.numel();

        keel::Array<keel::usize> cb(ndu, keel::as<keel::usize>(0), scratch.allocator());
        for (keel::usize ib = 0; ib < nb; ++ib) {
            keel::usize tmpB = ib;
            keel::usize baseOffB = 0;
            for (keel::i32 d = 0; d < nd; ++d) {
                const keel::usize c = tmpB % sb[keel::as<keel::usize>(d)];
                tmpB /= sb[keel::as<keel::usize>(d)];
                cb[keel::as<keel::usize>(d)] = c;
                baseOffB += c * strideFull[keel::as<keel::usize>(d)];
            }
            const keel::f64 bv = Bd[ib];
            if (bv == 0.0) continue;
            for (keel::usize ia = 0; ia < na; ++ia) {
                keel::usize tmpA = ia;
                keel::usize outIdx = baseOffB;
                for (keel::i32 d = 0; d < nd; ++d) {
                    const keel::usize c = tmpA % sa[keel::as<keel::usize>(d)];
                    tmpA /= sa[keel::as<keel::usize>(d)];
                    outIdx += c * strideFull[keel::as<keel::usize>(d)];
                }
                out[outIdx] += Ad[ia] * bv;
            }
        }

        if (shape == "full") return detail::narrowDoubleToSingleIf(fullV, wantSingle, mr);

        keel::Array<keel::usize> sout(ndu, keel::as<keel::usize>(1), scratch.allocator());
        keel::Array<keel::usize> off(ndu, keel::as<keel::usize>(0), scratch.allocator());
        if (shape == "same") {
            for (keel::i32 d = 0; d < nd; ++d) {
                sout[keel::as<keel::usize>(d)] = sa[keel::as<keel::usize>(d)];
                off[keel::as<keel::usize>(d)] = sb[keel::as<keel::usize>(d)] / 2;
            }
        } else if (shape == "valid") {
            for (keel::i32 d = 0; d < nd; ++d) {
                if (sa[keel::as<keel::usize>(d)] < sb[keel::as<keel::usize>(d)])
                    return Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
                sout[keel::as<keel::usize>(d)] = sa[keel::as<keel::usize>(d)] - sb[keel::as<keel::usize>(d)] + 1;
                off[keel::as<keel::usize>(d)] = sb[keel::as<keel::usize>(d)] - 1;
            }
        } else {
            throw Error("convn: shape must be 'full', 'same', or 'valid'",
                         0, 0, "convn", "", "numkit:convn:badShape");
        }

        Value cropV = Value::matrixND(sout.data(), nd, ValueType::DOUBLE, mr);
        keel::f64 *cd = cropV.doubleDataMut();
        const keel::usize ncrop = cropV.numel();
        for (keel::usize ic = 0; ic < ncrop; ++ic) {
            keel::usize tmp = ic;
            keel::usize fullIdx = 0;
            for (keel::i32 d = 0; d < nd; ++d) {
                const keel::usize c = tmp % sout[keel::as<keel::usize>(d)];
                tmp /= sout[keel::as<keel::usize>(d)];
                fullIdx += (c + off[keel::as<keel::usize>(d)]) * strideFull[keel::as<keel::usize>(d)];
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
    keel::f64 *out = outV.doubleDataMut();
    const keel::usize pageStrideOut = outR * outC;
    const keel::usize pageStrideA   = M * N;
    const keel::usize pageStrideB   = P * Q;
    const keel::f64 *Ad = a.doubleData();
    const keel::f64 *Bd = b.doubleData();
    keel::fill_n(out, outR * outC * outP, 0.0);

    for (keel::usize bp = 0; bp < Pp; ++bp) {
        for (keel::usize ap = 0; ap < Mp; ++ap) {
            const keel::usize op = ap + bp;
            for (keel::usize bj = 0; bj < Q; ++bj) {
                for (keel::usize bi = 0; bi < P; ++bi) {
                    const keel::f64 bv = Bd[bp * pageStrideB + bj * P + bi];
                    if (bv == 0.0) continue;
                    for (keel::usize ac = 0; ac < N; ++ac) {
                        for (keel::usize ar = 0; ar < M; ++ar) {
                            const keel::f64 av = Ad[ap * pageStrideA + ac * M + ar];
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
        keel::f64 *cd = crop.doubleDataMut();
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
        keel::f64 *cd = crop.doubleDataMut();
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
