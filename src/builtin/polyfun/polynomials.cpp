// toolboxes/builtin/src/math/elementary/polynomials.cpp
//
// Value-level polynomial builtins delegating numerical kernels to keel::poly.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/poly.h>
#include <keel/ref.h>

#include <numkit/builtin/elmat.hpp>  // poly_of_matrix
#include <numkit/builtin/polyfun.hpp>
#include <numkit/ops/helpers.hpp>
#include <numkit/ops/poly_helpers.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cstring>
#include <tuple>
#include <utility>

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
        rs = keel::poly::roots(keel::ArrayRef<keel::c64>(p.complexData(), n), scratch.allocator());
    } else {
        auto coeffs = keel::Array<double>(n, scratch.allocator());
        for (keel::usize i = 0; i < n; ++i)
            coeffs[i] = p.elemAsDouble(i);
        rs = keel::poly::roots(keel::ArrayRef<keel::f64>(coeffs.data(), coeffs.size()), scratch.allocator());
    }
    const keel::usize k = rs.size();

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
    auto deriv = keel::poly::deriv(keel::ArrayRef<keel::f64>(pv.data(), pv.size()), scratch.allocator());
    return rowFromVec(deriv.data(), deriv.size(), mr);
}

std::tuple<Value, Value>
polyder(const Value &b, const Value &a, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto bv = readPolyAsDouble(b, "polyder", scratch.allocator());
    auto av = readPolyAsDouble(a, "polyder", scratch.allocator());
    auto res = keel::poly::deriv_ratio(keel::ArrayRef<keel::f64>(bv.data(), bv.size()),
                                       keel::ArrayRef<keel::f64>(av.data(), av.size()),
                                       scratch.allocator());
    return std::make_tuple(rowFromVec(res.num.data(), res.num.size(), mr),
                           rowFromVec(res.den.data(), res.den.size(), mr));
}

Value polyint(const Value &p, double k, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto pv = readPolyAsDouble(p, "polyint", scratch.allocator());
    auto r = keel::poly::integ(keel::ArrayRef<keel::f64>(pv.data(), pv.size()), k, scratch.allocator());
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
    keel::usize bnz = 0;
    while (bnz < bv.size() && bv[bnz] == 0.0) ++bnz;
    if (bv.empty() || bnz == bv.size()) {
        auto z = Value::matrix(0, 1, ValueType::DOUBLE, mr);
        auto pRoots = keel::poly::roots(keel::ArrayRef<keel::f64>(av.data(), av.size()), scratch.allocator());
        auto p = realColIfFlat(pRoots.data(), pRoots.size(), mr);
        auto k = Value::scalar(0.0, mr);
        return std::make_tuple(std::move(z), std::move(p), std::move(k));
    }
    auto zRoots = keel::poly::roots(keel::ArrayRef<keel::f64>(bv.data(), bv.size()), scratch.allocator());
    auto pRoots = keel::poly::roots(keel::ArrayRef<keel::f64>(av.data(), av.size()), scratch.allocator());
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
    auto bRaw = keel::poly::from_roots(keel::ArrayRef<keel::c64>(zv.data(), zv.size()), scratch.allocator());
    auto aRaw = keel::poly::from_roots(keel::ArrayRef<keel::c64>(pv.data(), pv.size()), scratch.allocator());
    for (auto &v : bRaw) v *= k;
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

// ── poly / polyvalm / polydiv ────────────────────────────────────────

Value poly(const Value &r, keel::Allocator *mr)
{
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
    auto coeffs = keel::poly::from_roots(keel::ArrayRef<keel::c64>(cv.data(), n), scratch.allocator());
    return rowFromVec(coeffs.data(), coeffs.size(), mr);
}

Value polyvalm(const Value &p, const Value &A, keel::Allocator *mr)
{
    for (const Value *v : {&p, &A}) {
        if (isIntegerType(v->type()))
            throw Error(keel::format("polyvalm: inputs must be floats, namely single "
                                     "or double (got a {} array).",
                                     mtypeName(v->type())),
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

    if (k == 0) {
        Value z = Value::matrix(n, n, ValueType::DOUBLE, mr);
        std::fill(z.doubleDataMut(), z.doubleDataMut() + n * n, 0.0);
        return z;
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const bool pCx = p.isComplex(), aCx = A.isComplex();
    keel::Array<keel::c64> pBuf(k, scratch.allocator());
    for (keel::usize i = 0; i < k; ++i)
        pBuf[i] = pCx ? p.complexElem(i) : keel::c64(p.elemAsDouble(i), 0.0);
    keel::Array<keel::c64> aBuf(n * n, scratch.allocator());
    for (keel::usize i = 0; i < n * n; ++i)
        aBuf[i] = aCx ? A.complexElem(i) : keel::c64(A.elemAsDouble(i), 0.0);

    auto result = keel::poly::polyvalm(keel::ArrayRef<keel::c64>(pBuf.data(), k),
                                       keel::ArrayRef<keel::c64>(aBuf.data(), n * n),
                                       n, scratch.allocator());

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

Value packComplexOrReal(keel::ArrayRef<keel::c64> v, keel::Allocator *mr)
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
    keel::usize aOff = 0;
    while (aOff + 1 < na && a.doubleData()[aOff] == 0.0) ++aOff;
    const keel::usize aEff = na - aOff;
    if (aEff == 0 || a.doubleData()[aOff] == 0.0)
        throw Error("polydiv: divisor is zero", 0, 0, "polydiv", "",
                     "numkit:polydiv:zeroDivisor");

    if (nb < aEff) {
        auto q = Value::matrix(1, 1, ValueType::DOUBLE, mr);
        q.doubleDataMut()[0] = 0.0;
        return { std::move(q), b };
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto res = keel::poly::div(keel::ArrayRef<keel::f64>(b.doubleData(), nb),
                               keel::ArrayRef<keel::f64>(a.doubleData(), na),
                               scratch.allocator());
    return { rowFromVec(res.q.data(), res.q.size(), mr),
             rowFromVec(res.r.data(), res.r.size(), mr) };
}

// ── padecoef ─────────────────────────────────────────────────────────

PadeCoef padecoef(double T, int N, keel::Allocator *mr)
{
    if (N < 0)
        throw Error("padecoef: order N must be >= 0",
                     0, 0, "padecoef", "", "numkit:padecoef:badN");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto res = keel::poly::pade_delay(T, static_cast<keel::i32>(N), scratch.allocator());
    return { rowFromVec(res.num.data(), res.num.size(), mr),
             rowFromVec(res.den.data(), res.den.size(), mr) };
}

// ── residue / residuez — partial fraction expansion ──────────────────

namespace {

keel::Array<double> readPolyRealForResidue(const Value &p, const char *fn, keel::Allocator *alloc)
{
    if (p.type() == ValueType::COMPLEX)
        throw Error(keel::format("{}: complex coefficients not supported in v1", fn),
                     0, 0, fn, "", keel::format("numkit:{}:complex", fn));
    if (!p.isEmpty() && !p.dims().isVector() && !p.isScalar())
        throw Error(keel::format("{}: arguments must be vectors", fn),
                     0, 0, fn, "", keel::format("numkit:{}:notVector", fn));
    const keel::usize n = p.numel();
    keel::Array<double> v(n, alloc);
    for (keel::usize i = 0; i < n; ++i) v[i] = p.elemAsDouble(i);
    return v;
}

Value packDirectTerm(keel::ArrayRef<double> k, keel::Allocator *mr)
{
    if (k.empty() || (k.size() == 1 && k[0] == 0.0))
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto out = Value::matrix(1, k.size(), ValueType::DOUBLE, mr);
    std::memcpy(out.doubleDataMut(), k.data(), k.size() * sizeof(double));
    return out;
}

} // namespace

ResidueResult residue(const Value &b, const Value &a, keel::Allocator *mr)
{
    const char *fn = "residue";
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Allocator *salloc = scratch.allocator();
    auto A = readPolyRealForResidue(a, fn, salloc);
    auto B = readPolyRealForResidue(b, fn, salloc);
    auto res = keel::poly::residue(keel::ArrayRef<keel::f64>(B.data(), B.size()),
                                   keel::ArrayRef<keel::f64>(A.data(), A.size()),
                                   salloc);
    if (res.status == keel::poly::Status::ZeroDivisor)
        throw Error("residue: denominator must be non-zero",
                     0, 0, fn, "", "numkit:residue:zeroDenom");
    if (res.status == keel::poly::Status::DegeneratePole)
        throw Error("residue: derivative vanishes at a pole — "
                    "likely repeated pole undetected by tolerance",
                     0, 0, fn, "", "numkit:residue:denomZero");
    return {
        packComplexOrReal(keel::ArrayRef<keel::c64>(res.r.data(), res.r.size()), mr),
        packComplexOrReal(keel::ArrayRef<keel::c64>(res.p.data(), res.p.size()), mr),
        packDirectTerm(keel::ArrayRef<keel::f64>(res.k.data(), res.k.size()), mr),
    };
}

ResidueResult residuez(const Value &b, const Value &a, keel::Allocator *mr)
{
    const char *fn = "residuez";
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Allocator *salloc = scratch.allocator();
    auto A = readPolyRealForResidue(a, fn, salloc);
    auto B = readPolyRealForResidue(b, fn, salloc);
    auto res = keel::poly::residuez(keel::ArrayRef<keel::f64>(B.data(), B.size()),
                                    keel::ArrayRef<keel::f64>(A.data(), A.size()),
                                    salloc);
    if (res.status == keel::poly::Status::ZeroDivisor)
        throw Error("residuez: denominator a[0] must be non-zero",
                     0, 0, fn, "", "numkit:residuez:zeroDenom");
    if (res.status == keel::poly::Status::DegeneratePole)
        throw Error("residuez: cover-up denominator vanishes",
                     0, 0, fn, "", "numkit:residuez:denomZero");
    return {
        packComplexOrReal(keel::ArrayRef<keel::c64>(res.r.data(), res.r.size()), mr),
        packComplexOrReal(keel::ArrayRef<keel::c64>(res.p.data(), res.p.size()), mr),
        packDirectTerm(keel::ArrayRef<keel::f64>(res.k.data(), res.k.size()), mr),
    };
}

// ── Curve fitting / evaluation ───────────────────────────────────────

Value polyfit(const Value &x, const Value &y, int deg, keel::Allocator *mr)
{
    const keel::usize m = x.numel();
    if (m == 0)
        throw Error("polyfit: not enough data points",
                     0, 0, "polyfit", "", "numkit:polyfit:tooFewPoints");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto res = keel::poly::polyfit(keel::ArrayRef<keel::f64>(x.doubleData(), m),
                                   keel::ArrayRef<keel::f64>(y.doubleData(), y.numel()),
                                   static_cast<keel::i32>(deg), scratch.allocator());
    return rowFromVec(res.coeffs.data(), res.coeffs.size(), mr);
}

Value polyval(const Value &p, const Value &x, keel::Allocator *mr)
{
    auto r = createLike(x, ValueType::DOUBLE, mr);
    keel::poly::polyval_batch(keel::ArrayRef<keel::f64>(p.doubleData(), p.numel()),
                              keel::ArrayRef<keel::f64>(x.doubleData(), x.numel()),
                              keel::MutArrayRef<keel::f64>(r.doubleDataMut(), x.numel()));
    return r;
}

} // namespace numkit::builtin
