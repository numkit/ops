// toolboxes/builtin/src/math/elementary/special.cpp
//
// Special functions — thin Value adapters delegating to keel::specfun (<keel/specfun.h>).

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/specfun.h>
#include <numkit/builtin/specfun.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


namespace numkit::builtin {

Value gammaFn(const Value &x, keel::Allocator *mr) { return gamma(x, mr); }

Value gamma(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](keel::f64 v) { return keel::specfun::gamma(v); }, mr);
}

Value gammaln(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](keel::f64 v) { return keel::specfun::gammaln(v); }, mr);
}

// erf is backend-split (SLEEF-ported Highway dd kernel) and lives in
// special_highway.cpp + special_portable.cpp, like exp / log.

Value erfc(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](keel::f64 v) { return keel::specfun::erfc(v); }, mr);
}

Value erfinv(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](keel::f64 v) { return keel::specfun::erfinv(v); }, mr);
}

Value erfcinv(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](keel::f64 v) { return keel::specfun::erfcinv(v); }, mr);
}

Value erfcx(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](keel::f64 v) { return keel::specfun::erfcx(v); }, mr);
}

Value beta(const Value &z, const Value &w, keel::Allocator *mr)
{
    return elementwiseDouble(z, w, [](keel::f64 zz, keel::f64 ww) { return keel::specfun::beta(zz, ww); }, mr);
}

Value betaln(const Value &z, const Value &w, keel::Allocator *mr)
{
    return elementwiseDouble(z, w, [](keel::f64 zz, keel::f64 ww) { return keel::specfun::betaln(zz, ww); }, mr);
}

Value expint(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](keel::f64 v) { return keel::specfun::expint(v); }, mr);
}

Value psi(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](keel::f64 v) { return keel::specfun::psi(v); }, mr);
}

Value gammainc(const Value &x, const Value &a, keel::Allocator *mr)
{
    return elementwiseDouble(x, a, [](keel::f64 xx, keel::f64 aa) { return keel::specfun::gammainc(xx, aa); }, mr);
}

Value betainc(const Value &x, const Value &a, const Value &b, keel::Allocator *mr)
{
    if (x.isScalar() && a.isScalar() && b.isScalar()) {
        return Value::scalar(keel::specfun::betainc(x.toScalar(), a.toScalar(), b.toScalar()), mr);
    }
    const keel::usize nx = x.numel(), na = a.numel(), nb = b.numel();
    const keel::usize n = keel::max({nx, na, nb});
    auto pickShape = [&]() -> const Value & {
        if (nx == n) return x;
        if (na == n) return a;
        return b;
    };
    auto r = createLike(pickShape(), ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 xi = (nx == 1) ? x.toScalar() : x.doubleData()[i];
        const keel::f64 ai = (na == 1) ? a.toScalar() : a.doubleData()[i];
        const keel::f64 bi = (nb == 1) ? b.toScalar() : b.doubleData()[i];
        r.doubleDataMut()[i] = keel::specfun::betainc(xi, ai, bi);
    }
    return r;
}

Value legendre(keel::i32 n, const Value &x, keel::Allocator *mr)
{
    if (n < 0) {
        throw Error("legendre: n must be >= 0", 0, 0, "legendre", "", "numkit:legendre:badN");
    }
    const keel::usize L = x.numel();
    const keel::usize rows = keel::as<keel::usize>(n + 1);
    auto r = Value::matrix(rows, L, ValueType::DOUBLE, mr);
    keel::f64 *dst = r.doubleDataMut();
    for (keel::usize k = 0; k < L; ++k) {
        keel::specfun::legendre(keel::as<keel::i32>(n), x.elemAsDouble(k),
                       keel::MutArrayRef<keel::f64>(dst + k * rows, rows));
    }
    return r;
}

Value besselj(const Value &nu, const Value &x, keel::Allocator *mr)
{
    return elementwiseDouble(nu, x, [](keel::f64 n, keel::f64 xx) { return keel::specfun::besselj(n, xx); }, mr);
}

Value bessely(const Value &nu, const Value &x, keel::Allocator *mr)
{
    return elementwiseDouble(nu, x, [](keel::f64 n, keel::f64 xx) { return keel::specfun::bessely(n, xx); }, mr);
}

Value besseli(const Value &nu, const Value &x, keel::Allocator *mr)
{
    return elementwiseDouble(nu, x, [](keel::f64 n, keel::f64 xx) { return keel::specfun::besseli(n, xx); }, mr);
}

Value besselk(const Value &nu, const Value &x, keel::Allocator *mr)
{
    return elementwiseDouble(nu, x, [](keel::f64 n, keel::f64 xx) { return keel::specfun::besselk(n, xx); }, mr);
}

Value besselh(const Value &nu, keel::i32 k, const Value &x, keel::Allocator *mr)
{
    if (k != 1 && k != 2) {
        throw Error("besselh: k must be 1 or 2", 0, 0, "besselh", "", "numkit:besselh:badK");
    }
    if (nu.isScalar() && x.isScalar()) {
        return Value::complexScalar(
            keel::specfun::besselh(nu.toScalar(), keel::as<keel::i32>(k), x.toScalar()), mr);
    }
    const Value &shape = !nu.isScalar() ? nu : x;
    auto r = createLike(shape, ValueType::COMPLEX, mr);
    keel::c64 *dst = r.complexDataMut();
    const keel::usize N = shape.numel();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 n = nu.isScalar() ? nu.toScalar() : nu.doubleData()[i];
        const keel::f64 xx = x.isScalar() ? x.toScalar() : x.doubleData()[i];
        dst[i] = keel::specfun::besselh(n, keel::as<keel::i32>(k), xx);
    }
    return r;
}

EllipKE ellipke(const Value &m, keel::Allocator *mr)
{
    if (m.isScalar()) {
        const keel::specfun::EllipKE ke = keel::specfun::ellipke(m.toScalar());
        return {Value::scalar(ke.k, mr), Value::scalar(ke.e, mr)};
    }
    auto K = createLike(m, ValueType::DOUBLE, mr);
    auto E = createLike(m, ValueType::DOUBLE, mr);
    const keel::usize n = m.numel();
    for (keel::usize i = 0; i < n; ++i) {
        const keel::specfun::EllipKE ke = keel::specfun::ellipke(m.doubleData()[i]);
        K.doubleDataMut()[i] = ke.k;
        E.doubleDataMut()[i] = ke.e;
    }
    return {keel::move(K), keel::move(E)};
}

Value airy(keel::i32 k, const Value &x, keel::Allocator *mr)
{
    if (k < 0 || k > 3) {
        throw Error(keel::format("airy: kind k must be 0..3 (got {})", k),
                    0, 0, "airy", "", "numkit:airy:badK");
    }
    return unaryDouble(x, [k](keel::f64 v) { return keel::specfun::airy(keel::as<keel::i32>(k), v); }, mr);
}

Value gammaincinv(const Value &P, const Value &a, keel::Allocator *mr)
{
    return elementwiseDouble(
        P, a, [](keel::f64 pp, keel::f64 aa) { return keel::specfun::gammaincinv(pp, aa); }, mr);
}

Value betaincinv(const Value &P, const Value &a, const Value &b, keel::Allocator *mr)
{
    if (P.isScalar() && a.isScalar() && b.isScalar()) {
        return Value::scalar(keel::specfun::betaincinv(P.toScalar(), a.toScalar(), b.toScalar()), mr);
    }
    const keel::usize np = P.numel(), na = a.numel(), nb = b.numel();
    const keel::usize n = keel::max({np, na, nb});
    auto pickShape = [&]() -> const Value & {
        if (np == n) return P;
        if (na == n) return a;
        return b;
    };
    auto r = createLike(pickShape(), ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 pi = (np == 1) ? P.toScalar() : P.doubleData()[i];
        const keel::f64 ai = (na == 1) ? a.toScalar() : a.doubleData()[i];
        const keel::f64 bi = (nb == 1) ? b.toScalar() : b.doubleData()[i];
        r.doubleDataMut()[i] = keel::specfun::betaincinv(pi, ai, bi);
    }
    return r;
}

EllipJ ellipj(const Value &u, const Value &m, keel::Allocator *mr)
{
    if (u.isScalar() && m.isScalar()) {
        const keel::specfun::EllipJ ej = keel::specfun::ellipj(u.toScalar(), m.toScalar());
        return {Value::scalar(ej.sn, mr), Value::scalar(ej.cn, mr), Value::scalar(ej.dn, mr)};
    }
    const keel::usize nu = u.numel(), nm = m.numel();
    const keel::usize n = keel::max(nu, nm);
    const Value &shape = (nu == n) ? u : m;
    auto sn = createLike(shape, ValueType::DOUBLE, mr);
    auto cn = createLike(shape, ValueType::DOUBLE, mr);
    auto dn = createLike(shape, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 ui = (nu == 1) ? u.toScalar() : u.doubleData()[i];
        const keel::f64 mi = (nm == 1) ? m.toScalar() : m.doubleData()[i];
        const keel::specfun::EllipJ ej = keel::specfun::ellipj(ui, mi);
        sn.doubleDataMut()[i] = ej.sn;
        cn.doubleDataMut()[i] = ej.cn;
        dn.doubleDataMut()[i] = ej.dn;
    }
    return {keel::move(sn), keel::move(cn), keel::move(dn)};
}

Value marcumq(const Value &a_v, const Value &b_v, keel::i32 m, keel::Allocator *mr)
{
    if (m < 1) m = 1;
    const keel::usize Na = a_v.numel(), Nb = b_v.numel();
    const keel::usize N = keel::max(Na, Nb);
    Value out = Value::matrix((Na > Nb) ? Na : Nb, 1, ValueType::DOUBLE, mr);
    if (N == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 a = a_v.elemAsDouble(Na > 1 ? i : 0);
        const keel::f64 b = b_v.elemAsDouble(Nb > 1 ? i : 0);
        od[i] = keel::specfun::marcumq(a, b, keel::as<keel::i32>(m));
    }
    return out;
}

} // namespace numkit::builtin
