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

#include <algorithm>
#include <utility>

namespace numkit::builtin {

Value gammaFn(const Value &x, keel::Allocator *mr) { return gamma(x, mr); }

Value gamma(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](double v) { return keel::gamma(v); }, mr);
}

Value gammaln(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](double v) { return keel::gammaln(v); }, mr);
}

// erf is backend-split (SLEEF-ported Highway dd kernel) and lives in
// special_highway.cpp + special_portable.cpp, like exp / log.

Value erfc(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](double v) { return keel::erfc(v); }, mr);
}

Value erfinv(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](double v) { return keel::erfinv(v); }, mr);
}

Value erfcinv(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](double v) { return keel::erfcinv(v); }, mr);
}

Value erfcx(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](double v) { return keel::erfcx(v); }, mr);
}

Value beta(const Value &z, const Value &w, keel::Allocator *mr)
{
    return elementwiseDouble(z, w, [](double zz, double ww) { return keel::beta(zz, ww); }, mr);
}

Value betaln(const Value &z, const Value &w, keel::Allocator *mr)
{
    return elementwiseDouble(z, w, [](double zz, double ww) { return keel::betaln(zz, ww); }, mr);
}

Value expint(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](double v) { return keel::expint(v); }, mr);
}

Value psi(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](double v) { return keel::psi(v); }, mr);
}

Value gammainc(const Value &x, const Value &a, keel::Allocator *mr)
{
    return elementwiseDouble(x, a, [](double xx, double aa) { return keel::gammainc(xx, aa); }, mr);
}

Value betainc(const Value &x, const Value &a, const Value &b, keel::Allocator *mr)
{
    if (x.isScalar() && a.isScalar() && b.isScalar()) {
        return Value::scalar(keel::betainc(x.toScalar(), a.toScalar(), b.toScalar()), mr);
    }
    const keel::usize nx = x.numel(), na = a.numel(), nb = b.numel();
    const keel::usize n = std::max({nx, na, nb});
    auto pickShape = [&]() -> const Value & {
        if (nx == n) return x;
        if (na == n) return a;
        return b;
    };
    auto r = createLike(pickShape(), ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < n; ++i) {
        const double xi = (nx == 1) ? x.toScalar() : x.doubleData()[i];
        const double ai = (na == 1) ? a.toScalar() : a.doubleData()[i];
        const double bi = (nb == 1) ? b.toScalar() : b.doubleData()[i];
        r.doubleDataMut()[i] = keel::betainc(xi, ai, bi);
    }
    return r;
}

Value legendre(int n, const Value &x, keel::Allocator *mr)
{
    if (n < 0) {
        throw Error("legendre: n must be >= 0", 0, 0, "legendre", "", "numkit:legendre:badN");
    }
    const keel::usize L = x.numel();
    const keel::usize rows = static_cast<keel::usize>(n + 1);
    auto r = Value::matrix(rows, L, ValueType::DOUBLE, mr);
    double *dst = r.doubleDataMut();
    for (keel::usize k = 0; k < L; ++k) {
        keel::legendre(static_cast<keel::i32>(n), x.elemAsDouble(k),
                       keel::MutArrayRef<keel::f64>(dst + k * rows, rows));
    }
    return r;
}

Value besselj(const Value &nu, const Value &x, keel::Allocator *mr)
{
    return elementwiseDouble(nu, x, [](double n, double xx) { return keel::besselj(n, xx); }, mr);
}

Value bessely(const Value &nu, const Value &x, keel::Allocator *mr)
{
    return elementwiseDouble(nu, x, [](double n, double xx) { return keel::bessely(n, xx); }, mr);
}

Value besseli(const Value &nu, const Value &x, keel::Allocator *mr)
{
    return elementwiseDouble(nu, x, [](double n, double xx) { return keel::besseli(n, xx); }, mr);
}

Value besselk(const Value &nu, const Value &x, keel::Allocator *mr)
{
    return elementwiseDouble(nu, x, [](double n, double xx) { return keel::besselk(n, xx); }, mr);
}

Value besselh(const Value &nu, int k, const Value &x, keel::Allocator *mr)
{
    if (k != 1 && k != 2) {
        throw Error("besselh: k must be 1 or 2", 0, 0, "besselh", "", "numkit:besselh:badK");
    }
    if (nu.isScalar() && x.isScalar()) {
        return Value::complexScalar(
            keel::besselh(nu.toScalar(), static_cast<keel::i32>(k), x.toScalar()), mr);
    }
    const Value &shape = !nu.isScalar() ? nu : x;
    auto r = createLike(shape, ValueType::COMPLEX, mr);
    keel::c64 *dst = r.complexDataMut();
    const keel::usize N = shape.numel();
    for (keel::usize i = 0; i < N; ++i) {
        const double n = nu.isScalar() ? nu.toScalar() : nu.doubleData()[i];
        const double xx = x.isScalar() ? x.toScalar() : x.doubleData()[i];
        dst[i] = keel::besselh(n, static_cast<keel::i32>(k), xx);
    }
    return r;
}

EllipKE ellipke(const Value &m, keel::Allocator *mr)
{
    if (m.isScalar()) {
        const keel::EllipKE ke = keel::ellipke(m.toScalar());
        return {Value::scalar(ke.k, mr), Value::scalar(ke.e, mr)};
    }
    auto K = createLike(m, ValueType::DOUBLE, mr);
    auto E = createLike(m, ValueType::DOUBLE, mr);
    const keel::usize n = m.numel();
    for (keel::usize i = 0; i < n; ++i) {
        const keel::EllipKE ke = keel::ellipke(m.doubleData()[i]);
        K.doubleDataMut()[i] = ke.k;
        E.doubleDataMut()[i] = ke.e;
    }
    return {std::move(K), std::move(E)};
}

Value airy(int k, const Value &x, keel::Allocator *mr)
{
    if (k < 0 || k > 3) {
        throw Error(keel::format("airy: kind k must be 0..3 (got {})", k),
                    0, 0, "airy", "", "numkit:airy:badK");
    }
    return unaryDouble(x, [k](double v) { return keel::airy(static_cast<keel::i32>(k), v); }, mr);
}

Value gammaincinv(const Value &P, const Value &a, keel::Allocator *mr)
{
    return elementwiseDouble(
        P, a, [](double pp, double aa) { return keel::gammaincinv(pp, aa); }, mr);
}

Value betaincinv(const Value &P, const Value &a, const Value &b, keel::Allocator *mr)
{
    if (P.isScalar() && a.isScalar() && b.isScalar()) {
        return Value::scalar(keel::betaincinv(P.toScalar(), a.toScalar(), b.toScalar()), mr);
    }
    const keel::usize np = P.numel(), na = a.numel(), nb = b.numel();
    const keel::usize n = std::max({np, na, nb});
    auto pickShape = [&]() -> const Value & {
        if (np == n) return P;
        if (na == n) return a;
        return b;
    };
    auto r = createLike(pickShape(), ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < n; ++i) {
        const double pi = (np == 1) ? P.toScalar() : P.doubleData()[i];
        const double ai = (na == 1) ? a.toScalar() : a.doubleData()[i];
        const double bi = (nb == 1) ? b.toScalar() : b.doubleData()[i];
        r.doubleDataMut()[i] = keel::betaincinv(pi, ai, bi);
    }
    return r;
}

EllipJ ellipj(const Value &u, const Value &m, keel::Allocator *mr)
{
    if (u.isScalar() && m.isScalar()) {
        const keel::EllipJ ej = keel::ellipj(u.toScalar(), m.toScalar());
        return {Value::scalar(ej.sn, mr), Value::scalar(ej.cn, mr), Value::scalar(ej.dn, mr)};
    }
    const keel::usize nu = u.numel(), nm = m.numel();
    const keel::usize n = std::max(nu, nm);
    const Value &shape = (nu == n) ? u : m;
    auto sn = createLike(shape, ValueType::DOUBLE, mr);
    auto cn = createLike(shape, ValueType::DOUBLE, mr);
    auto dn = createLike(shape, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < n; ++i) {
        const double ui = (nu == 1) ? u.toScalar() : u.doubleData()[i];
        const double mi = (nm == 1) ? m.toScalar() : m.doubleData()[i];
        const keel::EllipJ ej = keel::ellipj(ui, mi);
        sn.doubleDataMut()[i] = ej.sn;
        cn.doubleDataMut()[i] = ej.cn;
        dn.doubleDataMut()[i] = ej.dn;
    }
    return {std::move(sn), std::move(cn), std::move(dn)};
}

Value marcumq(const Value &a_v, const Value &b_v, int m, keel::Allocator *mr)
{
    if (m < 1) m = 1;
    const keel::usize Na = a_v.numel(), Nb = b_v.numel();
    const keel::usize N = std::max(Na, Nb);
    Value out = Value::matrix((Na > Nb) ? Na : Nb, 1, ValueType::DOUBLE, mr);
    if (N == 0) return out;
    double *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const double a = a_v.elemAsDouble(Na > 1 ? i : 0);
        const double b = b_v.elemAsDouble(Nb > 1 ? i : 0);
        od[i] = keel::marcumq(a, b, static_cast<keel::i32>(m));
    }
    return out;
}

} // namespace numkit::builtin
