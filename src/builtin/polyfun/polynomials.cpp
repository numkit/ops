// toolboxes/builtin/src/math/elementary/polynomials.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/builtin/polyfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/ops/poly_helpers.hpp>

#include <numkit/builtin/elmat.hpp>  // poly_of_matrix

#include <cmath>
#include <cstring>
#include <tuple>

#include "polynomials_detail.hpp"

namespace numkit::builtin {

Value roots(const Value &p, keel::Allocator *mr)
{
    if (!p.dims().isVector() && !p.isScalar() && !p.isEmpty())
        throw Error("roots: argument must be a vector",
                     0, 0, "roots", "", "numkit:roots:notVector");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const keel::usize n = p.numel();
    keel::Array<keel::c64> rs(scratch.allocator());
    if (p.type() == ValueType::COMPLEX) {
        rs = numkit::ops::polyRootsDurandKerner(p.complexData(), n, scratch.allocator());
    } else {
        auto coeffs = keel::Array<double>(n, scratch.allocator());
        for (keel::usize i = 0; i < n; ++i)
            coeffs[i] = p.elemAsDouble(i);
        rs = numkit::ops::polyRootsDurandKerner(coeffs.data(), coeffs.size(), scratch.allocator());
    }
    const keel::usize k = rs.size();

    // If every root is real and input was real, return a real column. Otherwise return COMPLEX.
    bool anyComplex = (p.type() == ValueType::COMPLEX);
    for (const auto &r : rs)
        if (keel::abs(r.imag()) > 1e-12 * (keel::abs(r.real()) + 1.0)) {
            anyComplex = true;
            break;
        }

    if (!anyComplex) {
        auto out = Value::matrix(k, 1, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < k; ++i)
            out.doubleDataMut()[i] = rs[i].real();
        return out;
    }
    auto out = Value::complexMatrix(k, 1, mr);
    for (keel::usize i = 0; i < k; ++i)
        out.complexDataMut()[i] = rs[i];
    return out;
}

// ── polyder / polyint ───────────────────────────────────────────────

Value polyder(const Value &p, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto pv = readPolyAsDouble(p, "polyder", scratch.allocator());
    auto deriv = polyderRaw(pv.data(), pv.size(), scratch.allocator());
    // MATLAB convention: polynomial coefficient vectors are canonical
    // (no leading zeros), so a polyder result like [0, 1, 2] must be
    // trimmed to [1, 2]. See BUGS.md #12.
    trimLeadingZeros(deriv);
    return rowFromVec(deriv.data(), deriv.size(), mr);
}

std::tuple<Value, Value>
polyder(const Value &b, const Value &a, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto bv = readPolyAsDouble(b, "polyder", scratch.allocator());
    auto av = readPolyAsDouble(a, "polyder", scratch.allocator());
    auto bp = polyderRaw(bv.data(), bv.size(), scratch.allocator());
    auto ap = polyderRaw(av.data(), av.size(), scratch.allocator());
    // num = a * b' - b * a'
    auto t1 = polyConv(av.data(), av.size(), bp.data(), bp.size(), scratch.allocator());
    auto t2 = polyConv(bv.data(), bv.size(), ap.data(), ap.size(), scratch.allocator());
    // Align lengths (pad with leading zeros so subtraction is safe).
    auto prependZeros = [](auto &vec, keel::usize pad) {
        const keel::usize oldN = vec.size();
        vec.resize(oldN + pad, 0.0);
        for (keel::usize i = oldN; i > 0; --i) vec[i - 1 + pad] = vec[i - 1];
        for (keel::usize i = 0; i < pad; ++i) vec[i] = 0.0;
    };
    if (t1.size() < t2.size()) prependZeros(t1, t2.size() - t1.size());
    if (t2.size() < t1.size()) prependZeros(t2, t1.size() - t2.size());
    keel::Array<double> num(t1.size(), scratch.allocator());
    for (keel::usize i = 0; i < t1.size(); ++i) num[i] = t1[i] - t2[i];
    trimLeadingZeros(num);
    auto den = polyConv(av.data(), av.size(), av.data(), av.size(), scratch.allocator());
    trimLeadingZeros(den);
    return std::make_tuple(rowFromVec(num.data(), num.size(), mr),
                           rowFromVec(den.data(), den.size(), mr));
}

// Read a vector input of (real or COMPLEX) numbers as a Complex vector.

Value polyint(const Value &p, double k, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto pv = readPolyAsDouble(p, "polyint", scratch.allocator());
    if (pv.empty()) {
        // ∫ 0 dx = k.
        auto out = Value::matrix(1, 1, ValueType::DOUBLE, mr);
        out.doubleDataMut()[0] = k;
        return out;
    }
    const keel::usize n = pv.size();
    auto r = keel::Array<double>(n + 1, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) {
        const double exponent = static_cast<double>(n - i);
        r[i] = pv[i] / exponent;
    }
    r[n] = k;
    return rowFromVec(r.data(), r.size(), mr);
}

// ── tf2zp / zp2tf ───────────────────────────────────────────────────
std::tuple<Value, Value, Value>
tf2zp(const Value &b, const Value &a, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto bv = readPolyAsDouble(b, "tf2zp", scratch.allocator());
    auto av = readPolyAsDouble(a, "tf2zp", scratch.allocator());
    if (av.empty() || av[0] == 0.0)
        throw Error("tf2zp: leading denominator coefficient must be non-zero",
                     0, 0, "tf2zp", "", "numkit:tf2zp:badDen");
    // MATLAB strips leading zeros from the numerator before forming the
    // ZPK gain: b = [0 0 0 0 1] is the polynomial "1", so the gain is 1,
    // NOT b(1) = 0. The root finder (polyRootsDurandKerner) already ignores
    // leading zeros for the zeros, so a naive k = bv[0]/av[0] gave a bogus
    // gain of 0 whenever b was zero-padded (e.g. zp2tf output fed into
    // lp2lp/lp2bp via tf2zpk).
    keel::usize bnz = 0;
    while (bnz < bv.size() && bv[bnz] == 0.0) ++bnz;
    if (bv.empty() || bnz == bv.size()) {
        // Numerator is empty or all zeros → no zeros, gain 0.
        auto z = Value::matrix(0, 1, ValueType::DOUBLE, mr);
        auto pRoots = numkit::ops::polyRootsDurandKerner(av.data(), av.size(), scratch.allocator());
        auto p = realColIfFlat(pRoots.data(), pRoots.size(), mr);
        auto k = Value::scalar(0.0, mr);
        return std::make_tuple(std::move(z), std::move(p), std::move(k));
    }
    auto zRoots = numkit::ops::polyRootsDurandKerner(bv.data(), bv.size(), scratch.allocator());
    auto pRoots = numkit::ops::polyRootsDurandKerner(av.data(), av.size(), scratch.allocator());
    const double k = bv[bnz] / av[0];

    return std::make_tuple(realColIfFlat(zRoots.data(), zRoots.size(), mr),
                           realColIfFlat(pRoots.data(), pRoots.size(), mr),
                           Value::scalar(k, mr));
}

std::tuple<Value, Value>
zp2tf(const Value &z, const Value &p, double k, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto zv = readVecAsComplex(z, "zp2tf", scratch.allocator());
    auto pv = readVecAsComplex(p, "zp2tf", scratch.allocator());
    auto bRaw = numkit::ops::polyExpandFromRoots(zv.data(), zv.size(), scratch.allocator());
    auto aRaw = numkit::ops::polyExpandFromRoots(pv.data(), pv.size(), scratch.allocator());
    for (auto &v : bRaw) v *= k;
    // MATLAB returns b with the SAME length as a: when there are fewer
    // zeros than poles, the numerator is left-padded with zeros (the
    // surplus high-order coefficients are 0). E.g. one zero, two poles ->
    // b = [0 b0 b1], not [b0 b1].
    if (bRaw.size() < aRaw.size()) {
        const keel::usize off = aRaw.size() - bRaw.size();
        keel::Array<double> bPad(aRaw.size(), scratch.allocator());
        for (keel::usize i = 0; i < aRaw.size(); ++i) bPad[i] = 0.0;
        for (keel::usize i = 0; i < bRaw.size(); ++i) bPad[off + i] = bRaw[i];
        return std::make_tuple(rowFromVec(bPad.data(), bPad.size(), mr),
                               rowFromVec(aRaw.data(), aRaw.size(), mr));
    }
    return std::make_tuple(rowFromVec(bRaw.data(), bRaw.size(), mr),
                           rowFromVec(aRaw.data(), aRaw.size(), mr));
}

// ── Pack 29: poly / polyvalm / polydiv ───────────────────────────────

Value poly(const Value &r, keel::Allocator *mr)
{
    // Vector-of-roots → coefficient row. Real-only path uses the
    // existing helper (which drops the imaginary residue, fine when
    // the roots have come from `roots(p)` of a real polynomial).
    if (r.isEmpty()) return rowFromVec(nullptr, 0, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const keel::usize n = r.numel();
    auto cv = keel::Array<keel::c64>(n, scratch.allocator());
    if (r.isComplex()) {
        const keel::c64 *p = r.complexData();
        for (keel::usize i = 0; i < n; ++i) cv[i] = p[i];
    } else {
        const double *p = r.doubleData();
        for (keel::usize i = 0; i < n; ++i) cv[i] = keel::c64(p[i], 0.0);
    }
    auto coeffs = numkit::ops::polyExpandFromRoots(cv.data(), n, scratch.allocator());
    return rowFromVec(coeffs.data(), coeffs.size(), mr);
}

Value polyvalm(const Value &p, const Value &A, keel::Allocator *mr)
{
    // Contract probed R2025b (probe66/67): p accepts double/single/
    // logical/char (char codes, logical 0/1 — both widen to double) and
    // complex; an INTEGER on either side rejects (superiorfloat class:
    // "Inputs must be floats"); A must be square double/single (complex
    // computes); the EMPTY polynomial is the ZERO matrix (polyvalm([], X)
    // = zeros(size(X)), NOT the identity); empty A → 0x0; scalar A works.
    for (const Value *v : {&p, &A}) {
        if (isIntegerType(v->type()))
            throw Error("polyvalm: inputs must be floats, namely single "
                        "or double (got a "
                        + std::string(mtypeName(v->type())) + " array).",
                        0, 0, "polyvalm", "",
                        "numkit:polyvalm:superiorfloat");
    }
    if (A.dims().ndim() > 2 || A.dims().rows() != A.dims().cols())
        throw Error("polyvalm: matrix must be square.",
                     0, 0, "polyvalm", "", "numkit:polyvalm:NonSquareMatrix");
    const keel::usize n = A.dims().rows();
    const keel::usize k = p.numel();

    const bool wantSingle = p.type() == ValueType::SINGLE
        || A.type() == ValueType::SINGLE;

    // Empty polynomial → the zero matrix (0x0 for an empty A).
    if (k == 0) {
        Value z = Value::matrix(n, n, ValueType::DOUBLE, mr);
        std::fill(z.doubleDataMut(), z.doubleDataMut() + n * n, 0.0);
        return z;
    }

    const bool pCx = p.isComplex(), aCx = A.isComplex();
    auto pv = [&](keel::usize i) -> keel::c64 {
        return pCx ? p.complexElem(i)
                   : keel::c64(p.elemAsDouble(i), 0.0);
    };
    auto av = [&](keel::usize i) -> keel::c64 {
        return aCx ? A.complexElem(i)
                   : keel::c64(A.elemAsDouble(i), 0.0);
    };

    // Horner at the matrix level: result = result * A + p_i * I
    // (complex accumulation; a real result narrows at the end).
    std::vector<keel::c64> result(n * n, keel::c64(0.0, 0.0));
    const keel::c64 p0 = pv(0);
    for (keel::usize i = 0; i < n; ++i) result[i * n + i] = p0;
    std::vector<keel::c64> next(n * n);
    for (keel::usize step = 1; step < k; ++step) {
        // next = result * A (column-major)
        for (keel::usize i = 0; i < n; ++i)
            for (keel::usize j = 0; j < n; ++j) {
                keel::c64 s(0.0, 0.0);
                for (keel::usize kk = 0; kk < n; ++kk)
                    s += result[kk * n + i] * av(j * n + kk);
                next[j * n + i] = s;
            }
        const keel::c64 pi = pv(step);
        for (keel::usize i = 0; i < n; ++i) next[i * n + i] += pi;
        result.swap(next);
    }

    bool anyImag = false;
    for (const keel::c64 &c : result)
        if (c.imag() != 0.0) { anyImag = true; break; }
    if (anyImag) {
        auto out = Value::complexMatrix(n, n, mr);
        std::copy(result.begin(), result.end(), out.complexDataMut());
        return out;
    }
    if (wantSingle) {
        Value s = Value::matrix(n, n, ValueType::SINGLE, mr);
        float *sd = s.singleDataMut();
        for (keel::usize i = 0; i < n * n; ++i)
            sd[i] = static_cast<float>(result[i].real());
        return s;
    }
    auto out = Value::matrix(n, n, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (keel::usize i = 0; i < n * n; ++i) od[i] = result[i].real();
    return out;
}

Value packComplexOrReal(const std::vector<keel::c64> &v, keel::Allocator *mr)
{
    bool anyComplex = false;
    for (const auto &z : v)
        if (keel::abs(z.imag()) > 1e-10 * (keel::abs(z.real()) + 1.0)) {
            anyComplex = true; break;
        }
    if (!anyComplex) {
        auto out = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < v.size(); ++i)
            out.doubleDataMut()[i] = v[i].real();
        return out;
    }
    auto out = Value::complexMatrix(v.size(), 1, mr);
    for (keel::usize i = 0; i < v.size(); ++i)
        out.complexDataMut()[i] = v[i];
    return out;
}

PolyDiv polydiv(const Value &b, const Value &a, keel::Allocator *mr)
{
    if (a.isEmpty() || a.numel() == 0)
        throw Error("polydiv: divisor must be non-empty",
                     0, 0, "polydiv", "", "numkit:polydiv:emptyA");
    const keel::usize na = a.numel();
    const keel::usize nb = b.numel();
    // Strip leading zeros from a (matches MATLAB behaviour).
    keel::usize aOff = 0;
    while (aOff + 1 < na && a.doubleData()[aOff] == 0.0) ++aOff;
    const keel::usize aEff = na - aOff;
    if (aEff == 0 || a.doubleData()[aOff] == 0.0)
        throw Error("polydiv: divisor is zero", 0, 0, "polydiv", "",
                     "numkit:polydiv:zeroDivisor");

    if (nb < aEff) {
        // Quotient is 0; remainder == b.
        auto q = Value::matrix(1, 1, ValueType::DOUBLE, mr);
        q.doubleDataMut()[0] = 0.0;
        return { std::move(q), b };
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto bb = keel::Array<double>(nb, scratch.allocator());
    for (keel::usize i = 0; i < nb; ++i) bb[i] = b.doubleData()[i];
    const double aLead = a.doubleData()[aOff];
    const keel::usize qLen = nb - aEff + 1;
    auto qv = keel::Array<double>(qLen, scratch.allocator());
    for (keel::usize i = 0; i < qLen; ++i) qv[i] = 0.0;

    for (keel::usize i = 0; i < qLen; ++i) {
        const double coef = bb[i] / aLead;
        qv[i] = coef;
        for (keel::usize j = 0; j < aEff; ++j)
            bb[i + j] -= coef * a.doubleData()[aOff + j];
    }
    // Remainder: bb[qLen .. nb-1]; trim leading zeros.
    keel::usize rOff = qLen;
    while (rOff < nb && bb[rOff] == 0.0) ++rOff;
    Value rOut;
    if (rOff >= nb) {
        rOut = Value::matrix(1, 1, ValueType::DOUBLE, mr);
        rOut.doubleDataMut()[0] = 0.0;
    } else {
        rOut = rowFromVec(bb.data() + rOff, nb - rOff, mr);
    }
    return { rowFromVec(qv.data(), qLen, mr), std::move(rOut) };
}

// ── Pack 36: padecoef ────────────────────────────────────────────────
//
// Coefficients of the (N,N) Padé approximant of e^{-T s}:
//
//     e^{-Ts}  ≈  P(-Ts) / P(Ts)
//     P(x)    = Σ_{k=0..N} a_k x^k
//     a_k     = (2N-k)! N! / ((2N)! k! (N-k)!)
//
// Stable forward recurrence:
//     a_0 = 1
//     a_{k+1} / a_k = (N - k) / ((2N - k) (k + 1))
//
// In MATLAB convention (descending power), the coefficient of s^{N-k}
// is `a_k * (-T)^k` for the numerator and `a_k * T^k` for the
// denominator. After computing both vectors we divide through by
// den[0] so the leading denominator coefficient is 1 (matches
// `padecoef(T,N)` in MATLAB).
PadeCoef padecoef(double T, int N, keel::Allocator *mr)
{
    if (N < 0)
        throw Error("padecoef: order N must be >= 0",
                     0, 0, "padecoef", "", "numkit:padecoef:badN");

    const keel::usize n = static_cast<keel::usize>(N) + 1;

    // a[k] for k = 0..N.
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> a(n, scratch.allocator());
    a[0] = 1.0;
    for (int k = 0; k < N; ++k)
        a[k + 1] = a[k] * static_cast<double>(N - k)
                       / (static_cast<double>(2 * N - k) * (k + 1));

    auto numV = Value::matrix(1, n, ValueType::DOUBLE, mr);
    auto denV = Value::matrix(1, n, ValueType::DOUBLE, mr);
    double *num = numV.doubleDataMut();
    double *den = denV.doubleDataMut();

    // Descending: index 0 is s^N, index N is s^0. So position N-k stores
    // the k-th-power-of-s coefficient.
    double powNegT = 1.0, powT = 1.0;
    for (int k = 0; k <= N; ++k) {
        num[N - k] = a[k] * powNegT;
        den[N - k] = a[k] * powT;
        powNegT *= -T;
        powT    *=  T;
    }

    // Normalize so leading denominator coefficient is 1.
    const double scale = den[0];
    if (scale != 0.0) {
        for (keel::usize i = 0; i < n; ++i) {
            num[i] /= scale;
            den[i] /= scale;
        }
    }
    return { std::move(numV), std::move(denV) };
}


// ── residue / residuez — partial fraction expansion ─────────────────
//
// Forward form `[r, p, k] = residue(b, a)` only. The inverse form
// `[b, a] = residue(r, p, k)` is a documented v1 gap.
//
// Algorithm (distinct poles):
//   1. polydiv b/a  → quotient k (direct term), remainder b_rem (deg < deg a)
//   2. roots(a)     → poles p (complex)
//   3. Detect repeated poles within `1e-6 · max(1,|p|)` — throw if any.
//   4. a'(s) = polyder(a)
//   5. For each pole p_i, r_i = b_rem(p_i) / a'(p_i)  (Horner, complex)
//
// residuez is the same machinery in z-domain. We use the substitution
// from MATLAB's `residuez.m`: reverse coefficient order to convert
// between B(z^-1) / A(z^-1) (z-domain convention) and the standard
// polynomial form, then apply the same residue formula. Direct term k
// in z-domain is the polynomial in z^-1 — we return it in the same
// MATLAB convention.

namespace {

// Horner's method in complex domain. Coefficients in descending order.
keel::c64 hornerCx(const double *coeffs, keel::usize n, keel::c64 x)
{
    if (n == 0) return keel::c64(0, 0);
    keel::c64 acc(coeffs[0], 0.0);
    for (keel::usize i = 1; i < n; ++i)
        acc = acc * x + keel::c64(coeffs[i], 0.0);
    return acc;
}

// Strip leading zeros and verify result is non-empty.
std::vector<double> readPolyStripped(const Value &p, const char *fn)
{
    if (p.type() == ValueType::COMPLEX)
        throw Error(std::string(fn) + ": complex coefficients not supported in v1",
                     0, 0, fn, "", std::string("numkit:") + fn + ":complex");
    if (!p.isEmpty() && !p.dims().isVector() && !p.isScalar())
        throw Error(std::string(fn) + ": arguments must be vectors",
                     0, 0, fn, "", std::string("numkit:") + fn + ":notVector");
    std::vector<double> v(p.numel());
    for (keel::usize i = 0; i < v.size(); ++i) v[i] = p.elemAsDouble(i);
    keel::usize lo = 0;
    while (lo + 1 < v.size() && v[lo] == 0.0) ++lo;
    return std::vector<double>(v.begin() + lo, v.end());
}

// Returns true if any pair of poles is within tol·max(1,|p|).
bool hasRepeatedPoles(const std::vector<keel::c64> &p)
{
    for (keel::usize i = 0; i < p.size(); ++i)
        for (keel::usize j = i + 1; j < p.size(); ++j) {
            const double scale = std::max(1.0, std::max(keel::abs(p[i]), keel::abs(p[j])));
            if (keel::abs(p[i] - p[j]) < 1e-6 * scale)
                return true;
        }
    return false;
}

// Pack a complex column. If all entries are real (within tol), return
// a real column instead.

Value packDirectTerm(const std::vector<double> &k, keel::Allocator *mr)
{
    if (k.empty() || (k.size() == 1 && k[0] == 0.0))
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto out = Value::matrix(1, k.size(), ValueType::DOUBLE, mr);
    std::memcpy(out.doubleDataMut(), k.data(), k.size() * sizeof(double));
    return out;
}

// Evaluate B(z) at z = p, where B is in z^-1 ascending coefficient order
// (MATLAB z-domain convention: b[0] + b[1]·z^-1 + ... + b[n]·z^-n).
keel::c64 evalZPolyCx(const double *b, keel::usize n, keel::c64 p)
{
    if (n == 0) return keel::c64(0, 0);
    keel::c64 acc(0, 0);
    const keel::c64 inv = keel::c64(1, 0) / p;
    keel::c64 term(1, 0);   // (1/p)^k
    for (keel::usize k = 0; k < n; ++k) {
        acc += keel::c64(b[k], 0) * term;
        term *= inv;
    }
    return acc;
}

ResidueResult residueS(const Value &b, const Value &a,
                       keel::Allocator *mr)
{
    const char *fn = "residue";
    auto A = readPolyStripped(a, fn);
    auto B = readPolyStripped(b, fn);
    if (A.empty() || A[0] == 0.0)
        throw Error("residue: denominator must be non-zero",
                     0, 0, fn, "", "numkit:residue:zeroDenom");

    // Polynomial division: B = K·A + R   (deg R < deg A).
    std::vector<double> K;
    std::vector<double> R = B;
    if (B.size() >= A.size()) {
        const keel::usize qLen = B.size() - A.size() + 1;
        K.assign(qLen, 0.0);
        std::vector<double> bb = B;
        const double aLead = A[0];
        for (keel::usize i = 0; i < qLen; ++i) {
            const double coef = bb[i] / aLead;
            K[i] = coef;
            for (keel::usize j = 0; j < A.size(); ++j)
                bb[i + j] -= coef * A[j];
        }
        keel::usize rOff = qLen;
        while (rOff < bb.size() && bb[rOff] == 0.0) ++rOff;
        R.assign(bb.begin() + rOff, bb.end());
    }

    // Roots of A → s-domain poles.
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto rs = numkit::ops::polyRootsDurandKerner(A.data(), A.size(), scratch.allocator());
    std::vector<keel::c64> poles(rs.begin(), rs.end());

    // Group the roots into distinct poles. Durand-Kerner converges only
    // LINEARLY on multiple roots, so a true m-fold pole arrives as a
    // cluster spread up to ~1e-4*scale — cluster with that tolerance
    // (the old 1e-6 check missed such clusters entirely and the
    // distinct-pole cover-up then produced garbage), take the centroid,
    // and POLISH it: after deflating (s-p)^(m-1) out of A the root is
    // simple, so Newton converges quadratically to machine precision.
    // Groups keep first-occurrence order. Within a group MATLAB returns
    // the partial-fraction coefficients in ASCENDING power order —
    // r = [c1 c2 ... cm] for c1/(s-p) + ... + cm/(s-p)^m (probed R2025b:
    // residue([1 0 0],[1 3 3 1]) gives r = [1 -2 1] for (s+1)^(1..3)).
    struct PoleGroup { keel::c64 p; keel::usize m; };
    std::vector<PoleGroup> groups;
    {
        auto polyDivRoot = [](const std::vector<keel::c64> &c, keel::c64 p) {
            std::vector<keel::c64> q(c.size() - 1);
            if (q.empty()) return q;
            q[0] = c[0];
            for (keel::usize i = 1; i + 1 < c.size(); ++i)
                q[i] = c[i] + p * q[i - 1];
            return q;
        };
        auto hornerC = [](const std::vector<keel::c64> &c, keel::c64 x) {
            if (c.empty()) return keel::c64(0, 0);
            keel::c64 acc = c[0];
            for (keel::usize i = 1; i < c.size(); ++i) acc = acc * x + c[i];
            return acc;
        };
        std::vector<keel::c64> Acx(A.size());
        for (keel::usize i = 0; i < A.size(); ++i) Acx[i] = keel::c64(A[i], 0.0);
        std::vector<bool> grouped(poles.size(), false);
        for (keel::usize i = 0; i < poles.size(); ++i) {
            if (grouped[i]) continue;
            grouped[i] = true;
            PoleGroup g{poles[i], 1};
            for (keel::usize j = i + 1; j < poles.size(); ++j) {
                if (grouped[j]) continue;
                const double scale = std::max(1.0,
                    std::max(keel::abs(poles[i]), keel::abs(poles[j])));
                if (keel::abs(poles[j] - poles[i]) < 5e-4 * scale) {
                    grouped[j] = true;
                    ++g.m;
                    g.p += poles[j];
                }
            }
            if (g.m > 1) {
                g.p /= static_cast<double>(g.m);
                // A has REAL coefficients here (complex rejected at the
                // door), so a multiple root clustered near the real axis
                // IS real — the DK cluster need not be conjugate-
                // symmetric and its centroid carries O(spread) imaginary
                // dust. Snap it; Schroeder then stays in real arithmetic.
                if (keel::abs(g.p.imag()) < 5e-4 * (1.0 + keel::abs(g.p)))
                    g.p = keel::c64(g.p.real(), 0.0);
                // Schroeder's iteration for a known-multiplicity root:
                // p <- p - m*A(p)/A'(p) is QUADRATIC for an m-fold
                // root (plain Newton is only linear there), and it
                // works on A directly — no deflation noise floor.
                const double mD = static_cast<double>(g.m);
                for (int it = 0; it < 30; ++it) {
                    const keel::c64 fv = hornerC(Acx, g.p);
                    keel::c64 dacc(0, 0);
                    for (keel::usize k = 0; k + 1 < Acx.size(); ++k)
                        dacc = dacc * g.p + Acx[k] * static_cast<double>(Acx.size() - 1 - k);
                    if (keel::abs(dacc) < 1e-300) break;
                    const keel::c64 step = mD * fv / dacc;
                    g.p -= step;
                    if (keel::abs(step) < 1e-15 * (1.0 + keel::abs(g.p))) break;
                }
            }
            groups.push_back(g);
        }
    }
    const bool anyRepeated = groups.size() != poles.size();

    // Derivative coefficients (descending).
    std::vector<double> Aprime;
    if (A.size() > 1) {
        const keel::usize n = A.size() - 1;
        Aprime.assign(n, 0.0);
        for (keel::usize i = 0; i < n; ++i)
            Aprime[i] = A[i] * static_cast<double>(n - i);
    }

    std::vector<keel::c64> residues;
    std::vector<keel::c64> polesOut;
    residues.reserve(poles.size());
    polesOut.reserve(poles.size());

    if (!anyRepeated) {
        // Residues via standard cover-up: r_i = R(p_i) / A'(p_i).
        if (!R.empty()) {
            for (keel::usize i = 0; i < poles.size(); ++i) {
                const keel::c64 num = hornerCx(R.data(), R.size(), poles[i]);
                const keel::c64 den = hornerCx(Aprime.data(), Aprime.size(), poles[i]);
                if (keel::abs(den) < 1e-300)
                    throw Error("residue: derivative vanishes at a pole — "
                                "likely repeated pole undetected by tolerance",
                                 0, 0, fn, "", "numkit:residue:denomZero");
                residues.push_back(num / den);
            }
        } else {
            residues.assign(poles.size(), keel::c64(0, 0));
        }
        polesOut = poles;
    } else {
        // Repeated poles: Taylor-series method. For a pole p of
        // multiplicity m, write N/D = H(s)/(s-p)^m with
        // H(s) = R(s) / (A(s)/(s-p)^m) analytic at p, expand
        // H(s) = sum h_j (s-p)^j; then the coefficient of (s-p)^(-i)
        // is h_{m-i}. H's Taylor coefficients come from the Taylor
        // series of R and of the reciprocal of A/(s-p)^m at p.
        // Synthetic division of complex descending coeffs by (s - p).
        auto divideOut = [](std::vector<keel::c64> c, keel::c64 p) {
            std::vector<keel::c64> q(c.size() - 1);
            if (q.empty()) return q;
            q[0] = c[0];
            for (keel::usize i = 1; i + 1 < c.size(); ++i)
                q[i] = c[i] + p * q[i - 1];
            return q;
        };
        // Taylor coefficients 0..order-1 of a polynomial (descending
        // coeffs) at s = p, via the derivative chain / factorial.
        auto taylorCoeffs = [](const auto &c, keel::c64 p, keel::usize order) {
            std::vector<keel::c64> t;
            std::vector<keel::c64> q(c.begin(), c.end());
            std::vector<keel::c64> fact(1, keel::c64(1, 0));
            for (keel::usize d = 0; d < order; ++d) {
                // Horner at p.
                keel::c64 acc(0, 0);
                if (!q.empty()) {
                    acc = q[0];
                    for (keel::usize i = 1; i < q.size(); ++i)
                        acc = acc * p + q[i];
                }
                t.push_back(acc / fact.back());
                // fact = (d+1)!
                fact.push_back(fact.back() * keel::c64(static_cast<double>(d) + 1, 0));
                // q := q' (formal derivative, descending coeffs).
                if (q.size() > 1) {
                    const keel::usize n = q.size() - 1;
                    std::vector<keel::c64> dq(n);
                    for (keel::usize i = 0; i < n; ++i)
                        dq[i] = q[i] * static_cast<double>(n - i);
                    q = std::move(dq);
                } else {
                    q.clear();
                }
            }
            return t;
        };

        for (const auto &g : groups) {
            if (g.m == 1) {
                const keel::c64 num = hornerCx(R.data(), R.empty() ? 0 : R.size(), g.p);
                const keel::c64 den = hornerCx(Aprime.data(), Aprime.size(), g.p);
                if (keel::abs(den) < 1e-300)
                    throw Error("residue: derivative vanishes at a pole — "
                                "likely repeated pole undetected by tolerance",
                                 0, 0, fn, "", "numkit:residue:denomZero");
                residues.push_back(R.empty() ? keel::c64(0, 0) : num / den);
                polesOut.push_back(g.p);
                continue;
            }
            // D_k = A / (s-p)^m (synthetic division m times).
            std::vector<keel::c64> Dk(A.size());
            for (keel::usize i = 0; i < A.size(); ++i) Dk[i] = keel::c64(A[i], 0.0);
            for (keel::usize d = 0; d < g.m && Dk.size() > 1; ++d)
                Dk = divideOut(Dk, g.p);
            const auto dser = taylorCoeffs(Dk, g.p, g.m);   // D_k series
            if (dser.empty() || keel::abs(dser[0]) < 1e-300)
                throw Error("residue: deflated denominator vanishes at a "
                            "repeated pole",
                             0, 0, fn, "", "numkit:residue:denomZero");
            // Reciprocal series of D_k at p, order m.
            std::vector<keel::c64> recip(g.m, keel::c64(0, 0));
            recip[0] = keel::c64(1, 0) / dser[0];
            for (keel::usize j = 1; j < g.m; ++j) {
                keel::c64 s(0, 0);
                for (keel::usize i = 1; i <= j; ++i) {
                    keel::c64 di = i < dser.size() ? dser[i] : keel::c64(0, 0);
                    s += di * recip[j - i];
                }
                recip[j] = -s / dser[0];
            }
            // R series at p, order m.
            std::vector<keel::c64> rser(g.m, keel::c64(0, 0));
            if (!R.empty()) {
                auto rt = taylorCoeffs(R, g.p, g.m);
                for (keel::usize j = 0; j < g.m && j < rt.size(); ++j)
                    rser[j] = rt[j];
            }
            // h_j = sum r_i recip_{j-i}; residues for powers 1..m =
            // h_{m-1}, h_{m-2}, ..., h_0.
            std::vector<keel::c64> h(g.m, keel::c64(0, 0));
            for (keel::usize j = 0; j < g.m; ++j)
                for (keel::usize i = 0; i <= j; ++i)
                    h[j] += rser[i] * recip[j - i];
            for (keel::usize i = 0; i < g.m; ++i) {
                residues.push_back(h[g.m - 1 - i]);
                polesOut.push_back(g.p);
            }
        }
    }

    return {
        packComplexOrReal(residues, mr),
        packComplexOrReal(polesOut, mr),
        packDirectTerm(K, mr),
    };
}

ResidueResult residueZ(const Value &b, const Value &a,
                       keel::Allocator *mr)
{
    const char *fn = "residuez";
    // Port of MATLAB residuez.m (R2025b, read via `type`): the impulse/
    // S-matrix method — simple poles by cover-up at 1/p over the
    // FLIPPED polynomials, repeated poles by least-squares over the
    // cumulative-filter basis (bugs/closed/signal/residuez-repeated-poles).
    // z-domain convention: a[0] + a[1]·z^-1 + ... ascending.
    auto av = readPolyStripped(a, fn);
    auto bv = readPolyStripped(b, fn);
    if (av.empty() || av[0] == 0.0)
        throw Error("residuez: denominator a[0] must be non-zero",
                     0, 0, fn, "", "numkit:residuez:zeroDenom");
    const double a0 = av[0];
    std::vector<double> A = av, B = bv;
    for (auto &x : A) x /= a0;
    for (auto &x : B) x /= a0;

    const keel::usize LA = A.size(), LB0 = B.size();
    if (LA == 1) {
        // No poles: everything is the direct term.
        return {packComplexOrReal({}, mr), packComplexOrReal({}, mr),
                packDirectTerm(B, mr)};
    }
    const keel::usize Nres = LA - 1;

    keel::ArenaTemp scratch = keel::get_scratch(mr);

    // Direct terms: [k, brem] = deconv(fliplr(B), fliplr(A)), k flipped;
    // the remainder keeps the strictly-proper part.
    std::vector<double> K;
    std::vector<double> Bp = B;                 // strictly-proper numerator
    if (LB0 > Nres) {
        std::vector<double> bf(Bp.rbegin(), Bp.rend());
        std::vector<double> af(A.rbegin(), A.rend());
        const keel::usize qLen = bf.size() - af.size() + 1;
        K.assign(qLen, 0.0);
        for (keel::usize i = 0; i < qLen; ++i) {
            const double coef = bf[i] / af[0];
            K[i] = coef;
            for (keel::usize j = 0; j < af.size(); ++j)
                bf[i + j] -= coef * af[j];
        }
        std::reverse(K.begin(), K.end());
        keel::usize rOff = qLen;
        while (rOff < bf.size() && bf[rOff] == 0.0) ++rOff;
        Bp.assign(bf.rbegin(), bf.rend() - rOff);   // flip back, strip zeros
        if (Bp.empty()) Bp = {0.0};
    }

    // Poles and mpoles grouping (tol 1e-3 relative, groups consecutive,
    // order = descending |p| — MATLAB mpoles' sort).
    auto rs = numkit::ops::polyRootsDurandKerner(A.data(), A.size(), scratch.allocator());
    std::vector<keel::c64> poles(rs.begin(), rs.end());
    {
        // Real coefficients => a pole carrying only DK conjugate-dust on
        // its imaginary part IS real (LAPACK returns these real via
        // balancing; DK leaves up to ~1e-4 dust on multiple roots that
        // otherwise complexifies the whole S/least-squares pipeline).
        // A conjugate pair this close to the axis merges to a double
        // real pole — consistent with the mpoles grouping tolerance.
        for (auto &pz : poles)
            if (keel::abs(pz.imag()) < 1e-4 * (1.0 + keel::abs(pz)))
                pz = keel::c64(pz.real(), 0.0);
        std::vector<std::pair<double, keel::usize>> keyed;
        keyed.reserve(poles.size());
        for (keel::usize i = 0; i < poles.size(); ++i)
            keyed.emplace_back(-keel::abs(poles[i]), i);   // stable desc |p|
        std::stable_sort(keyed.begin(), keyed.end());
        std::vector<keel::c64> sorted;
        sorted.reserve(poles.size());
        for (auto &kv : keyed) sorted.push_back(poles[kv.second]);
        poles = std::move(sorted);
    }
    constexpr double kMpolesTol = 1e-3;
    keel::usize m = poles.size();
    std::vector<keel::usize> mults;
    {
        // mpoles semantics: take the first unprocessed pole, gather ALL
        // poles within tol*|p| ANYWHERE in the remaining list, emit them
        // CONSECUTIVELY (groups compact together — a bare in-place
        // marking split interleaved groups like [2 1 -1 1]).
        std::vector<bool> used(m, false);
        std::vector<keel::c64> grouped;
        grouped.reserve(m);
        for (keel::usize i = 0; i < m; ++i) {
            if (used[i]) continue;
            bool allNonzero = true;
            for (keel::usize k = i; k < m; ++k)
                if (!used[k] && keel::abs(poles[k]) == 0.0) { allNonzero = false; break; }
            const double thresh = allNonzero ? kMpolesTol * keel::abs(poles[i])
                                             : kMpolesTol;
            std::vector<keel::usize> members;
            for (keel::usize k = i; k < m; ++k)
                if (!used[k] && keel::abs(poles[k] - poles[i]) < thresh) {
                    used[k] = true;
                    members.push_back(k);
                }
            for (keel::usize d = 0; d < members.size(); ++d) {
                grouped.push_back(poles[members[d]]);
                mults.push_back(d + 1);
            }
        }
        poles = std::move(grouped);
        m = poles.size();
    }

    // Direct-form IIR filter (local: the signal layer's filter is above
    // this layer). y(n) = sum b_k x(n-k) - sum a_k y(n-k), a[0]==1.
    auto zfilter = [](const std::vector<keel::c64> &bb,
                      const std::vector<keel::c64> &aa,
                      const std::vector<keel::c64> &xx) {
        std::vector<keel::c64> y(xx.size(), keel::c64(0, 0));
        for (keel::usize n = 0; n < xx.size(); ++n) {
            keel::c64 acc(0, 0);
            for (keel::usize k = 0; k < bb.size() && k <= n; ++k)
                acc += bb[k] * xx[n - k];
            for (keel::usize k = 1; k < aa.size() && k <= n; ++k)
                acc -= aa[k] * y[n - k];
            y[n] = acc;
        }
        return y;
    };

    // h = filter(Bp, A, imp), imp = [1 0 ... 0] of length N+1.
    const keel::usize NH = m + 1;
    std::vector<keel::c64> imp(NH, keel::c64(0, 0));
    imp[0] = keel::c64(1, 0);
    std::vector<keel::c64> Bc(Bp.size()), Ac(A.size());
    for (keel::usize i = 0; i < Bp.size(); ++i) Bc[i] = keel::c64(Bp[i], 0.0);
    for (keel::usize i = 0; i < A.size(); ++i) Ac[i] = keel::c64(A[i], 0.0);
    std::vector<keel::c64> h = zfilter(Bc, Ac, imp);

    // S columns: first-of-group from imp, continuations cumulative.
    std::vector<std::vector<keel::c64>> S(m);
    for (keel::usize j = 0; j < m; ++j) {
        if (mults[j] > 1)
            S[j] = zfilter({keel::c64(1, 0)}, {keel::c64(1, 0), -poles[j]}, S[j - 1]);
        else
            S[j] = zfilter({keel::c64(1, 0)}, {keel::c64(1, 0), -poles[j]}, imp);
    }

    // Simple-pole cover-up at 1/p over flipped polys (poly in z^-1
    // evaluated at 1/p == polynomial with reversed coefficient order).
    std::vector<keel::c64> residues(m, keel::c64(0, 0));
    std::vector<bool> jdone(m, false);
    std::vector<keel::c64> bflip(Bp.rbegin(), Bp.rend());
    for (keel::usize i = 0; i < m; ++i) {
        keel::usize ii = 0;
        if (i == m - 1) ii = m;
        else if (mults[i + 1] == 1) ii = i + 1;
        if (ii == 0) continue;
        // Other poles = all except this pole's group [i-mults[i]+1, i].
        std::vector<keel::c64> others;
        for (keel::usize j = 0; j < m; ++j)
            if (j < i - mults[i] + 1 || j > i) others.push_back(poles[j]);
        // temp = poly(others) with DESCENDING coefficients, then flipped.
        std::vector<keel::c64> temp{keel::c64(1, 0)};
        for (const auto &o : others) {
            std::vector<keel::c64> next(temp.size() + 1, keel::c64(0, 0));
            for (keel::usize t = 0; t < temp.size(); ++t) {
                next[t] += temp[t];
                next[t + 1] -= o * temp[t];
            }
            temp = std::move(next);
        }
        // polyval at 1/p (Horner on flipped coefficient order).
        auto polyvalAt = [](const std::vector<keel::c64> &c, keel::c64 x) {
            keel::c64 acc(0, 0);
            for (const auto &co : c) acc = acc * x + co;
            return acc;
        };
        const keel::c64 xInv = keel::c64(1, 0) / poles[i];
        // MATLAB evaluates fliplr(poly(others)) — the ASCENDING
        // coefficient order (a polynomial in z^-1 at z = p). Evaluating
        // the descending build directly differs by (1/p)^deg — a sign
        // for odd degrees at negative poles.
        const keel::c64 den = polyvalAt(
            std::vector<keel::c64>(temp.rbegin(), temp.rend()), xInv);
        if (keel::abs(den) < 1e-300)
            throw Error("residuez: cover-up denominator vanishes",
                         0, 0, fn, "", "numkit:residuez:denomZero");
        residues[i] = polyvalAt(bflip, xInv) / den;
        jdone[i] = true;
    }

    // Repeated-pole residues: least squares S(:,jkl) \ h via normal
    // equations (the basis is small and well-conditioned; MATLAB's \
    // is QR on the same system).
    std::vector<keel::usize> jkl;
    for (keel::usize j = 0; j < m; ++j)
        if (!jdone[j]) jkl.push_back(j);
    if (!jkl.empty()) {
        for (keel::usize j = 0; j < m; ++j)
            if (jdone[j]) {
                for (keel::usize n = 0; n < NH; ++n)
                    h[n] -= S[j][n] * residues[j];
            }
        const keel::usize M = jkl.size();
        std::vector<std::vector<keel::c64>> ATA(M, std::vector<keel::c64>(M));
        std::vector<keel::c64> ATb(M, keel::c64(0, 0));
        for (keel::usize r = 0; r < M; ++r) {
            for (keel::usize n = 0; n < NH; ++n)
                ATb[r] += keel::conj(S[jkl[r]][n]) * h[n];
            for (keel::usize c = 0; c < M; ++c) {
                keel::c64 acc(0, 0);
                for (keel::usize n = 0; n < NH; ++n)
                    acc += keel::conj(S[jkl[r]][n]) * S[jkl[c]][n];
                ATA[r][c] = acc;
            }
        }
        // Gaussian elimination with partial pivot on ATA t = ATb.
        for (keel::usize col = 0; col < M; ++col) {
            keel::usize piv = col;
            for (keel::usize r = col + 1; r < M; ++r)
                if (keel::abs(ATA[r][col]) > keel::abs(ATA[piv][col])) piv = r;
            std::swap(ATA[col], ATA[piv]);
            std::swap(ATb[col], ATb[piv]);
            const keel::c64 d = ATA[col][col];
            for (keel::usize c = col; c < M; ++c) ATA[col][c] /= d;
            ATb[col] /= d;
            for (keel::usize r = 0; r < M; ++r) {
                if (r == col) continue;
                const keel::c64 f = ATA[r][col];
                for (keel::usize c = col; c < M; ++c) ATA[r][c] -= f * ATA[col][c];
                ATb[r] -= f * ATb[col];
            }
        }
        for (keel::usize r = 0; r < M && r < jkl.size(); ++r)
            residues[jkl[r]] = ATb[r];
    }

    return {
        packComplexOrReal(residues, mr),
        packComplexOrReal(poles, mr),
        packDirectTerm(K, mr),
    };
}

} // namespace

ResidueResult residue(const Value &b, const Value &a,
                      keel::Allocator *mr)
{
    return residueS(b, a, mr);
}

ResidueResult residuez(const Value &b, const Value &a,
                       keel::Allocator *mr)
{
    return residueZ(b, a, mr);
}


// ════════════════════════════════════════════════════════════════════════
// Curve fitting / evaluation (moved from toolboxes/fit)
// ════════════════════════════════════════════════════════════════════════

Value polyfit(const Value &x, const Value &y, int deg, keel::Allocator *mr)
{
    const keel::usize m = x.numel();
    const int np = deg + 1;

    if (m == 0)
        throw Error("polyfit: not enough data points",
                     0, 0, "polyfit", "", "numkit:polyfit:tooFewPoints");
    if (static_cast<keel::usize>(np) > m) {
        Value pSub = polyfit(x, y, static_cast<int>(m) - 1, mr);
        auto p = Value::matrix(1, np, ValueType::DOUBLE, mr);
        double *pd = p.doubleDataMut();
        const int offset = np - static_cast<int>(m);
        for (int j = 0; j < offset; ++j)
            pd[j] = 0.0;
        for (keel::usize j = 0; j < m; ++j)
            pd[offset + static_cast<int>(j)] = pSub.doubleData()[j];
        return p;
    }

    const double *xd = x.doubleData();
    const double *yd = y.doubleData();

    keel::ArenaTemp scratch = keel::get_scratch(mr);

    // Vandermonde matrix A[j, i] = x[i]^(deg - j), stored column-major.
    auto A = keel::Array<double>(m * np, scratch.allocator());
    for (keel::usize i = 0; i < m; ++i)
        for (int j = 0; j < np; ++j)
            A[j * m + i] = std::pow(xd[i], deg - j);

    // Solve A * p = y via Householder QR on Vandermonde matrix A (avoiding
    // normal equations A^T A, which squares the condition number).
    auto rhs = keel::Array<double>(m, scratch.allocator());
    for (keel::usize i = 0; i < m; ++i)
        rhs[i] = yd[i];

    for (int k = 0; k < np; ++k) {
        double norm2 = 0.0;
        for (keel::usize i = static_cast<keel::usize>(k); i < m; ++i) {
            const double val = A[k * m + i];
            norm2 += val * val;
        }
        const double normX = std::sqrt(norm2);
        if (normX == 0.0) continue;
        const double akk = A[k * m + k];
        const double alpha = (akk >= 0.0) ? -normX : normX;
        const double v0 = akk - alpha;
        A[k * m + k] = v0;
        double vNorm2 = v0 * v0;
        for (keel::usize i = static_cast<keel::usize>(k) + 1; i < m; ++i) {
            const double vi = A[k * m + i];
            vNorm2 += vi * vi;
        }
        if (vNorm2 > 0.0) {
            const double beta = 2.0 / vNorm2;
            for (int c = k + 1; c < np; ++c) {
                double dot = 0.0;
                for (keel::usize i = static_cast<keel::usize>(k); i < m; ++i)
                    dot += A[k * m + i] * A[c * m + i];
                const double scale = beta * dot;
                for (keel::usize i = static_cast<keel::usize>(k); i < m; ++i)
                    A[c * m + i] -= scale * A[k * m + i];
            }
            double dotY = 0.0;
            for (keel::usize i = static_cast<keel::usize>(k); i < m; ++i)
                dotY += A[k * m + i] * rhs[i];
            const double scaleY = beta * dotY;
            for (keel::usize i = static_cast<keel::usize>(k); i < m; ++i)
                rhs[i] -= scaleY * A[k * m + i];
        }
        A[k * m + k] = alpha;
    }

    auto p = Value::matrix(1, np, ValueType::DOUBLE, mr);
    double *pd = p.doubleDataMut();
    for (int k = np - 1; k >= 0; --k) {
        double s = rhs[k];
        for (int c = k + 1; c < np; ++c)
            s -= A[c * m + k] * pd[c];
        const double diag = A[k * m + k];
        pd[k] = (keel::abs(diag) > 0.0) ? (s / diag) : 0.0;
    }
    return p;
}

Value polyval(const Value &p, const Value &x, keel::Allocator *mr)
{
    const double *pd = p.doubleData();
    const keel::usize np = p.numel();
    const keel::usize nx = x.numel();
    const double *xd = x.doubleData();

    auto r = createLike(x, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < nx; ++i) {
        double val = pd[0];
        for (keel::usize j = 1; j < np; ++j)
            val = val * xd[i] + pd[j];
        r.doubleDataMut()[i] = val;
    }
    return r;
}

} // namespace numkit::builtin
