// toolboxes/signal/src/digital_filtering/filter.cpp

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/signal/digital_filtering/filter.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>
#include "filter_detail.hpp"


namespace numkit::signal {

// The raw-buffer IIR recurrence kernels (applyFilterDf2t / ...Complex) moved to
// the kernel layer — numkit::ops::applyFilterDf2t{,Complex} (nk/ops/
// iir_filter.hpp), re-exported into this namespace by filter_detail.hpp. The
// Value-level marshalling helpers below stay in signal.

// Gather a Value into a Complex buffer (real element -> Complex(v, 0)).
keel::Array<keel::c64> toComplexBuf(const Value &v, keel::usize n, keel::Allocator *mr)
{
    keel::Array<keel::c64> out(n, mr);
    if (v.isComplex()) {
        const keel::c64 *c = v.complexData();
        for (keel::usize i = 0; i < n; ++i) out[i] = c[i];
    } else {
        for (keel::usize i = 0; i < n; ++i) out[i] = keel::c64(v.elemAsDouble(i), 0.0);
    }
    return out;
}

// MATLAB filter() operates along the first array dimension of x whose size
// is not 1. Because the data is stored column-major, every "signal" along
// that dimension is a contiguous run of `L` samples (all lower dimensions
// have size 1, so their combined stride is 1) and the whole buffer splits
// cleanly into numel/L such runs. Returns L; for a scalar/empty input it
// returns numel (a single — possibly empty — run).
keel::usize firstNonSingletonExtent(const Value &x)
{
    const Dims &d = x.dims();
    const keel::i32 nd = d.ndim();
    for (keel::i32 k = 0; k < nd; ++k)
        if (d.dim(k) != 1) return d.dim(k);
    return x.numel();
}

// [Phase 2b] end promoted filter kernels

// ── filter ────────────────────────────────────────────────────────────
// filter_argcheck: the checker + dispatch-plan resolver. The rejection
// behavior (which inputs fail, which argument is blamed) is validated
// against the reference engine; the strings are numkit-voiced and carry the
// received value/shape (user rule 2026-09-17).

namespace {

// A "vector": at most 2D with one singleton extent — and non-empty
// (empty b or a is rejected as not-a-vector, like the reference).
bool isVectorValue(const Value &v)
{
    if (v.numel() == 0)
        return false;
    const Dims &d = v.dims();
    return d.ndim() <= 2 && (d.dim(0) == 1 || d.dim(1) == 1);
}

void requireNumericOrLogical(const Value &v, const char *name)
{
    if (!v.isNumeric() && !v.isLogical())
        throw Error(keel::format("filter: {} must be numeric or logical (type is {}).",
                                 name, mtypeName(v.type())),
                    0, 0, "filter", "", "numkit:filter:inputType");
}

} // namespace

FilterPlan filter_argcheck(const Value &b, const Value &a, const Value &x,
                           const Value &zi)
{
    requireNumericOrLogical(b, "b");
    requireNumericOrLogical(a, "a");
    requireNumericOrLogical(x, "x");
    const bool hasZi = zi.numel() > 0;
    if (hasZi)
        requireNumericOrLogical(zi, "zi");

    if (!isVectorValue(b) || !isVectorValue(a)) {
        const Value &bad = isVectorValue(b) ? a : b;
        const char *name = isVectorValue(b) ? "a" : "b";
        const Dims &d = bad.dims();
        throw Error(keel::format("filter: b and a must be non-empty vectors ({} is {}x{}).",
                                 name, d.dim(0), d.dim(1)),
                    0, 0, "filter", "", "numkit:filter:notVector");
    }

    // a(1) != 0 — complex inputs report through the complex view.
    if (a.isComplex()) {
        if (a.complexData()[0] == keel::c64(0.0, 0.0))
            throw Error("filter: first denominator coefficient a(1) must "
                        "be non-zero (got 0).",
                        0, 0, "filter", "", "numkit:filter:zeroLead");
    } else if (a.elemAsDouble(0) == 0.0) {
        throw Error("filter: first denominator coefficient a(1) must be "
                    "non-zero (got 0).",
                    0, 0, "filter", "", "numkit:filter:zeroLead");
    }

    FilterPlan p;
    p.nb = b.numel();
    p.na = a.numel();
    p.nx = x.numel();
    p.complex = b.isComplex() || a.isComplex() || x.isComplex()
                || (hasZi && zi.isComplex());
    p.sigLen = (p.nx == 0) ? 0 : firstNonSingletonExtent(x);
    p.nSig = (p.sigLen > 0) ? p.nx / p.sigLen : 0;
    p.stateLen = keel::max(p.nb, p.na) - 1;

    if (hasZi) {
        const keel::usize ziLen = zi.numel();
        const bool perSignal = p.nSig > 1 && ziLen == p.stateLen * p.nSig;
        if (!(ziLen == p.stateLen || perSignal))
            throw Error(
                keel::format("filter: zi must have max(length(a),length(b))-1 = {} elements per signal (got {}).",
                             p.stateLen, ziLen),
                0, 0, "filter", "", "numkit:filter:invalidInitialConditions");
    }
    return p;
}

// Shared DF2T driver for both public forms. All dispatch decisions come
// from the plan — nothing is re-derived here (this is the code the
// [y,zf] tail in filter_reg used to duplicate). zi == empty Value means
// zero initial state; zf is always produced (the plain form just drops
// it — its cost is one write per state element).
static FilterState runFilter(const FilterPlan &p, const Value &b,
                             const Value &a, const Value &x, const Value &zi,
                             keel::Allocator *mr)
{
    const bool hasZi = zi.numel() > 0;
    const keel::usize zfCols = (p.nSig > 0) ? p.nSig : 1;

    if (p.complex) {
        keel::ArenaTemp cs = keel::get_scratch(mr);
        auto bc = toComplexBuf(b, p.nb, cs.allocator());
        auto ac = toComplexBuf(a, p.na, cs.allocator());
        auto xc = toComplexBuf(x, p.nx, cs.allocator());
        const keel::c64 a0 = ac[0];
        for (keel::usize i = 0; i < p.nb; ++i) bc[i] /= a0;
        for (keel::usize i = 0; i < p.na; ++i) ac[i] /= a0;
        keel::Array<keel::c64> zic = hasZi ? toComplexBuf(zi, zi.numel(), cs.allocator())
                                        : keel::Array<keel::c64>(0, cs.allocator());
        const keel::c64 *ziPtr = hasZi ? zic.data() : nullptr;
        const keel::usize ziLen = hasZi ? zi.numel() : 0;
        const bool ziPerSig = hasZi && p.stateLen > 0 && p.nSig > 1
                              && ziLen == p.stateLen * p.nSig;

        Value rc = createLike(x, ValueType::COMPLEX, mr);
        keel::c64 *yc = rc.complexDataMut();
        Value zfv = Value::matrix(p.stateLen, zfCols, ValueType::COMPLEX, mr);
        keel::c64 *zfData = zfv.complexDataMut();
        for (keel::usize i = 0; i < p.stateLen * zfCols; ++i)
            zfData[i] = keel::c64(0.0, 0.0);
        // Empty input: the "final state" is just the seed zi (or zeros).
        if (p.nSig == 0 && hasZi && p.stateLen)
            for (keel::usize i = 0; i < p.stateLen && i < ziLen; ++i)
                zfData[i] = ziPtr[i];
        for (keel::usize s = 0; s < p.nSig; ++s) {
            const keel::c64 *ziRun = nullptr;
            keel::usize ziRunLen = 0;
            if (hasZi) {
                ziRun    = ziPerSig ? ziPtr + s * p.stateLen : ziPtr;
                ziRunLen = ziPerSig ? p.stateLen : ziLen;
            }
            keel::Array<keel::c64> zf(p.stateLen, cs.allocator());
            auto out = applyFilterDf2tComplex(bc.data(), p.nb, ac.data(),
                                              p.na, xc.data() + s * p.sigLen,
                                              p.sigLen, ziRun, ziRunLen,
                                              p.stateLen ? zf.data() : nullptr, cs.allocator());
            for (keel::usize i = 0; i < p.sigLen; ++i)
                yc[s * p.sigLen + i] = out[i];
            if (p.stateLen)
                for (keel::usize i = 0; i < p.stateLen; ++i)
                    zfData[s * p.stateLen + i] = zf[i];
        }
        return {keel::move(rc), keel::move(zfv)};
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto bn = keel::Array<keel::f64>(p.nb, scratch.allocator());
    auto an = keel::Array<keel::f64>(p.na, scratch.allocator());
    const keel::f64 a0 = a.elemAsDouble(0);
    for (keel::usize i = 0; i < p.nb; ++i) bn[i] = b.elemAsDouble(i) / a0;
    for (keel::usize i = 0; i < p.na; ++i) an[i] = a.elemAsDouble(i) / a0;

    // Logical/integer inputs are legal filter inputs — materialize the
    // non-double ones so the kernel sees contiguous double runs.
    keel::Array<keel::f64> xmat(0, scratch.allocator());
    const keel::f64 *xd = nullptr;
    if (x.type() == ValueType::DOUBLE) {
        xd = x.doubleData();
    } else {
        xmat = keel::Array<keel::f64>(p.nx, scratch.allocator());
        for (keel::usize i = 0; i < p.nx; ++i) xmat[i] = x.elemAsDouble(i);
        xd = xmat.data();
    }
    keel::Array<keel::f64> zimat(0, scratch.allocator());
    const keel::f64 *ziPtr = nullptr;
    keel::usize ziLen = 0;
    if (hasZi) {
        ziLen = zi.numel();
        if (zi.type() == ValueType::DOUBLE) {
            ziPtr = zi.doubleData();
        } else {
            zimat = keel::Array<keel::f64>(ziLen, scratch.allocator());
            for (keel::usize i = 0; i < ziLen; ++i) zimat[i] = zi.elemAsDouble(i);
            ziPtr = zimat.data();
        }
    }
    const bool ziPerSig = hasZi && p.stateLen > 0 && p.nSig > 1
                          && ziLen == p.stateLen * p.nSig;

    Value r = createLike(x, ValueType::DOUBLE, mr);
    keel::f64 *y = r.doubleDataMut();
    Value zfv = Value::matrix(p.stateLen, zfCols, ValueType::DOUBLE, mr);
    keel::f64 *zfData = zfv.doubleDataMut();
    for (keel::usize i = 0; i < p.stateLen * zfCols; ++i) zfData[i] = 0.0;
    if (p.nSig == 0 && hasZi && p.stateLen)
        for (keel::usize i = 0; i < p.stateLen && i < ziLen; ++i)
            zfData[i] = ziPtr[i];
    for (keel::usize s = 0; s < p.nSig; ++s) {
        const keel::f64 *ziRun = nullptr;
        keel::usize ziRunLen = 0;
        if (hasZi) {
            ziRun    = ziPerSig ? ziPtr + s * p.stateLen : ziPtr;
            ziRunLen = ziPerSig ? p.stateLen : ziLen;
        }
        keel::Array<keel::f64> zf(p.stateLen, scratch.allocator());
        auto out = applyFilterDf2t(bn.data(), p.nb, an.data(), p.na,
                                   xd + s * p.sigLen, p.sigLen,
                                   ziRun, ziRunLen,
                                   p.stateLen ? zf.data() : nullptr, scratch.allocator());
        keel::copy_n(y + s * p.sigLen, out.data(), p.sigLen);
        if (p.stateLen)
            keel::copy_n(zfData + s * p.stateLen, zf.data(), p.stateLen);
    }
    return {keel::move(r), keel::move(zfv)};
}

Value filter(const Value &b, const Value &a, const Value &x,
             keel::Allocator *mr)
{
    const FilterPlan p = filter_argcheck(b, a, x, Value());
    return runFilter(p, b, a, x, Value(), mr).y;
}

FilterState filter(const Value &b, const Value &a, const Value &x,
                   const Value &zi, keel::Allocator *mr)
{
    const FilterPlan p = filter_argcheck(b, a, x, zi);
    return runFilter(p, b, a, x, zi, mr);
}

Value filtfilt(const Value &b, const Value &a, const Value &x, keel::Allocator *mr)
{
    if (x.isComplex()) {
        const keel::usize nx = x.numel();
        const auto *xd = x.complexData();
        Value xRe = Value::matrix(x.dims().rows(), x.dims().cols(), ValueType::DOUBLE, mr);
        Value xIm = Value::matrix(x.dims().rows(), x.dims().cols(), ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < nx; ++i) {
            xRe.doubleDataMut()[i] = xd[i].real();
            xIm.doubleDataMut()[i] = xd[i].imag();
        }
        Value yRe = filtfilt(b, a, xRe, mr);
        Value yIm = filtfilt(b, a, xIm, mr);
        Value out = Value::matrix(x.dims().rows(), x.dims().cols(), ValueType::COMPLEX, mr);
        auto *od = out.complexDataMut();
        for (keel::usize i = 0; i < nx; ++i)
            od[i] = keel::c64(yRe.doubleData()[i], yIm.doubleData()[i]);
        return out;
    }
    const keel::usize nb = b.numel(), na = a.numel(), nx = x.numel();
    const keel::f64 *bd = b.doubleData();
    const keel::f64 *ad = a.doubleData();
    const keel::f64 *xd = x.doubleData();

    const keel::f64 a0 = ad[0];
    if (a0 == 0.0)
        throw Error("filtfilt: a(1) must be nonzero",
                     0, 0, "filtfilt", "", "numkit:filtfilt:zeroLead");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto bn = keel::Array<keel::f64>(nb, scratch.allocator());
    auto an = keel::Array<keel::f64>(na, scratch.allocator());
    for (keel::usize i = 0; i < nb; ++i)
        bn[i] = bd[i] / a0;
    for (keel::usize i = 0; i < na; ++i)
        an[i] = ad[i] / a0;

    const keel::usize nfilt = keel::max(nb, na);
    const keel::usize stateLen = (nfilt > 0) ? (nfilt - 1) : 0;

    // Steady-state step-response initial conditions zi: solve (I - A)*zi = rhs
    // where rhs[k] = b[k+1] - b[0]*a[k+1]. The companion/DF2T system telescopes:
    //   (1 + sum_{k=1..m} a[k]) * zi[0] = sum_{k=0..m-1} rhs[k]
    //   zi[k+1] = zi[k] + a[k+1]*zi[0] - rhs[k].
    auto zi = keel::Array<keel::f64>(stateLen, scratch.allocator());
    if (stateLen > 0) {
        auto rhs = keel::Array<keel::f64>(stateLen, scratch.allocator());
        keel::f64 sumRhs = 0.0;
        keel::f64 sumA = 1.0;
        for (keel::usize k = 0; k < stateLen; ++k) {
            const keel::f64 bk1 = (k + 1 < nb) ? bn[k + 1] : 0.0;
            const keel::f64 ak1 = (k + 1 < na) ? an[k + 1] : 0.0;
            rhs[k] = bk1 - bn[0] * ak1;
            sumRhs += rhs[k];
            sumA += ak1;
        }
        if (keel::abs(sumA) > 1e-14) {
            zi[0] = sumRhs / sumA;
            for (keel::usize k = 0; k + 1 < stateLen; ++k) {
                const keel::f64 ak1 = (k + 1 < na) ? an[k + 1] : 0.0;
                zi[k + 1] = zi[k] + ak1 * zi[0] - rhs[k];
            }
        }
    }

    // Edge-reflect padding, length 3 * (nfilt - 1) on each side (capped at nx - 1)
    keel::usize nEdge = 3 * stateLen;
    if (nx > 0 && nEdge >= nx)
        nEdge = nx - 1;
    else if (nx == 0)
        nEdge = 0;

    const keel::usize extLen = nx + 2 * nEdge;
    auto ext = keel::Array<keel::f64>(extLen, scratch.allocator());
    for (keel::usize i = 0; i < nEdge; ++i)
        ext[i] = 2.0 * xd[0] - xd[nEdge - i];
    for (keel::usize i = 0; i < nx; ++i)
        ext[nEdge + i] = xd[i];
    for (keel::usize i = 0; i < nEdge; ++i)
        ext[nEdge + nx + i] = 2.0 * xd[nx - 1] - xd[nx - 2 - i];

    auto ziScaled = keel::Array<keel::f64>(stateLen, scratch.allocator());
    if (stateLen > 0 && extLen > 0) {
        for (keel::usize k = 0; k < stateLen; ++k)
            ziScaled[k] = zi[k] * ext[0];
    }
    auto fwd = applyFilterDf2t(bn.data(), nb, an.data(), na, ext.data(), extLen,
                               stateLen > 0 ? ziScaled.data() : nullptr, stateLen, nullptr, scratch.allocator());
    keel::reverse(fwd.begin(), fwd.end());
    if (stateLen > 0 && !fwd.empty()) {
        for (keel::usize k = 0; k < stateLen; ++k)
            ziScaled[k] = zi[k] * fwd[0];
    }
    auto bwd = applyFilterDf2t(bn.data(), nb, an.data(), na, fwd.data(), fwd.size(),
                               stateLen > 0 ? ziScaled.data() : nullptr, stateLen, nullptr, scratch.allocator());
    keel::reverse(bwd.begin(), bwd.end());

    auto r = createLike(x, ValueType::DOUBLE, mr);
    keel::f64 *y = r.doubleDataMut();
    if (nx > 0)
        keel::copy_n(y, bwd.data() + nEdge, nx);
    return r;
}

} // namespace numkit::signal
