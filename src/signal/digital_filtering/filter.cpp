// toolboxes/signal/src/digital_filtering/filter.cpp

#include <numkit/signal/digital_filtering/filter.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>
#include "filter_detail.hpp"

#include <algorithm>
#include <cstring>
#include <memory_resource>

namespace numkit::signal {

// The raw-buffer IIR recurrence kernels (applyFilterDf2t / ...Complex) moved to
// the kernel layer — numkit::ops::applyFilterDf2t{,Complex} (nk/ops/
// iir_filter.hpp), re-exported into this namespace by filter_detail.hpp. The
// Value-level marshalling helpers below stay in signal.

// Gather a Value into a Complex buffer (real element -> Complex(v, 0)).
ScratchVec<Complex> toComplexBuf(const Value &v, size_t n, std::pmr::memory_resource *mr)
{
    ScratchVec<Complex> out(n, mr);
    if (v.isComplex()) {
        const Complex *c = v.complexData();
        for (size_t i = 0; i < n; ++i) out[i] = c[i];
    } else {
        for (size_t i = 0; i < n; ++i) out[i] = Complex(v.elemAsDouble(i), 0.0);
    }
    return out;
}

// MATLAB filter() operates along the first array dimension of x whose size
// is not 1. Because the data is stored column-major, every "signal" along
// that dimension is a contiguous run of `L` samples (all lower dimensions
// have size 1, so their combined stride is 1) and the whole buffer splits
// cleanly into numel/L such runs. Returns L; for a scalar/empty input it
// returns numel (a single — possibly empty — run).
size_t firstNonSingletonExtent(const Value &x)
{
    const Dims &d = x.dims();
    const int nd = d.ndim();
    for (int k = 0; k < nd; ++k)
        if (d.dim(k) != 1) return d.dim(k);
    return x.numel();
}

// [Phase 2b] end promoted filter kernels

// ── filter ────────────────────────────────────────────────────────────
// filter_argcheck: the checker + dispatch-plan resolver. The rejection
// behavior (which inputs fail, which argument is blamed) is validated
// against the reference engine; the strings are nk-voiced and carry the
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
        throw Error(std::string("filter: ") + name + " must be numeric or "
                    "logical (type is " + mtypeName(v.type()) + ").",
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
        throw Error(std::string("filter: b and a must be non-empty "
                    "vectors (") + name + " is "
                    + std::to_string(d.dim(0)) + "x"
                    + std::to_string(d.dim(1)) + ").",
                    0, 0, "filter", "", "numkit:filter:notVector");
    }

    // a(1) != 0 — complex inputs report through the complex view.
    if (a.isComplex()) {
        if (a.complexData()[0] == Complex(0.0, 0.0))
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
    p.stateLen = std::max(p.nb, p.na) - 1;

    if (hasZi) {
        const size_t ziLen = zi.numel();
        const bool perSignal = p.nSig > 1 && ziLen == p.stateLen * p.nSig;
        if (!(ziLen == p.stateLen || perSignal))
            throw Error(
                "filter: zi must have max(length(a),length(b))-1 = "
                + std::to_string(p.stateLen) + " elements per signal (got "
                + std::to_string(ziLen) + ").",
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
                             std::pmr::memory_resource *mr)
{
    const bool hasZi = zi.numel() > 0;
    const size_t zfCols = (p.nSig > 0) ? p.nSig : 1;

    if (p.complex) {
        ScratchArena cs(mr);
        auto bc = toComplexBuf(b, p.nb, &cs);
        auto ac = toComplexBuf(a, p.na, &cs);
        auto xc = toComplexBuf(x, p.nx, &cs);
        const Complex a0 = ac[0];
        for (size_t i = 0; i < p.nb; ++i) bc[i] /= a0;
        for (size_t i = 0; i < p.na; ++i) ac[i] /= a0;
        ScratchVec<Complex> zic = hasZi ? toComplexBuf(zi, zi.numel(), &cs)
                                        : ScratchVec<Complex>(0, &cs);
        const Complex *ziPtr = hasZi ? zic.data() : nullptr;
        const size_t ziLen = hasZi ? zi.numel() : 0;
        const bool ziPerSig = hasZi && p.stateLen > 0 && p.nSig > 1
                              && ziLen == p.stateLen * p.nSig;

        Value rc = createLike(x, ValueType::COMPLEX, mr);
        Complex *yc = rc.complexDataMut();
        Value zfv = Value::matrix(p.stateLen, zfCols, ValueType::COMPLEX, mr);
        Complex *zfData = zfv.complexDataMut();
        for (size_t i = 0; i < p.stateLen * zfCols; ++i)
            zfData[i] = Complex(0.0, 0.0);
        // Empty input: the "final state" is just the seed zi (or zeros).
        if (p.nSig == 0 && hasZi && p.stateLen)
            for (size_t i = 0; i < p.stateLen && i < ziLen; ++i)
                zfData[i] = ziPtr[i];
        for (size_t s = 0; s < p.nSig; ++s) {
            const Complex *ziRun = nullptr;
            size_t ziRunLen = 0;
            if (hasZi) {
                ziRun    = ziPerSig ? ziPtr + s * p.stateLen : ziPtr;
                ziRunLen = ziPerSig ? p.stateLen : ziLen;
            }
            ScratchVec<Complex> zf(p.stateLen, &cs);
            auto out = applyFilterDf2tComplex(bc.data(), p.nb, ac.data(),
                                              p.na, xc.data() + s * p.sigLen,
                                              p.sigLen, &cs, ziRun, ziRunLen,
                                              p.stateLen ? zf.data() : nullptr);
            for (size_t i = 0; i < p.sigLen; ++i)
                yc[s * p.sigLen + i] = out[i];
            if (p.stateLen)
                for (size_t i = 0; i < p.stateLen; ++i)
                    zfData[s * p.stateLen + i] = zf[i];
        }
        return {std::move(rc), std::move(zfv)};
    }

    ScratchArena scratch(mr);
    auto bn = ScratchVec<double>(p.nb, &scratch);
    auto an = ScratchVec<double>(p.na, &scratch);
    const double a0 = a.elemAsDouble(0);
    for (size_t i = 0; i < p.nb; ++i) bn[i] = b.elemAsDouble(i) / a0;
    for (size_t i = 0; i < p.na; ++i) an[i] = a.elemAsDouble(i) / a0;

    // Logical/integer inputs are legal filter inputs — materialize the
    // non-double ones so the kernel sees contiguous double runs.
    ScratchVec<double> xmat(0, &scratch);
    const double *xd = nullptr;
    if (x.type() == ValueType::DOUBLE) {
        xd = x.doubleData();
    } else {
        xmat = ScratchVec<double>(p.nx, &scratch);
        for (size_t i = 0; i < p.nx; ++i) xmat[i] = x.elemAsDouble(i);
        xd = xmat.data();
    }
    ScratchVec<double> zimat(0, &scratch);
    const double *ziPtr = nullptr;
    size_t ziLen = 0;
    if (hasZi) {
        ziLen = zi.numel();
        if (zi.type() == ValueType::DOUBLE) {
            ziPtr = zi.doubleData();
        } else {
            zimat = ScratchVec<double>(ziLen, &scratch);
            for (size_t i = 0; i < ziLen; ++i) zimat[i] = zi.elemAsDouble(i);
            ziPtr = zimat.data();
        }
    }
    const bool ziPerSig = hasZi && p.stateLen > 0 && p.nSig > 1
                          && ziLen == p.stateLen * p.nSig;

    Value r = createLike(x, ValueType::DOUBLE, mr);
    double *y = r.doubleDataMut();
    Value zfv = Value::matrix(p.stateLen, zfCols, ValueType::DOUBLE, mr);
    double *zfData = zfv.doubleDataMut();
    for (size_t i = 0; i < p.stateLen * zfCols; ++i) zfData[i] = 0.0;
    if (p.nSig == 0 && hasZi && p.stateLen)
        for (size_t i = 0; i < p.stateLen && i < ziLen; ++i)
            zfData[i] = ziPtr[i];
    for (size_t s = 0; s < p.nSig; ++s) {
        const double *ziRun = nullptr;
        size_t ziRunLen = 0;
        if (hasZi) {
            ziRun    = ziPerSig ? ziPtr + s * p.stateLen : ziPtr;
            ziRunLen = ziPerSig ? p.stateLen : ziLen;
        }
        ScratchVec<double> zf(p.stateLen, &scratch);
        auto out = applyFilterDf2t(bn.data(), p.nb, an.data(), p.na,
                                   xd + s * p.sigLen, p.sigLen, &scratch,
                                   ziRun, ziRunLen,
                                   p.stateLen ? zf.data() : nullptr);
        std::memcpy(y + s * p.sigLen, out.data(), p.sigLen * sizeof(double));
        if (p.stateLen)
            std::memcpy(zfData + s * p.stateLen, zf.data(),
                        p.stateLen * sizeof(double));
    }
    return {std::move(r), std::move(zfv)};
}

Value filter(const Value &b, const Value &a, const Value &x,
             std::pmr::memory_resource *mr)
{
    const FilterPlan p = filter_argcheck(b, a, x, Value());
    return runFilter(p, b, a, x, Value(), mr).y;
}

FilterState filter(const Value &b, const Value &a, const Value &x,
                   const Value &zi, std::pmr::memory_resource *mr)
{
    const FilterPlan p = filter_argcheck(b, a, x, zi);
    return runFilter(p, b, a, x, zi, mr);
}

Value filtfilt(const Value &b, const Value &a, const Value &x, std::pmr::memory_resource *mr)
{
    const size_t nb = b.numel(), na = a.numel(), nx = x.numel();
    const double *bd = b.doubleData();
    const double *ad = a.doubleData();
    const double *xd = x.doubleData();

    const double a0 = ad[0];
    if (a0 == 0.0)
        throw Error("filtfilt: a(1) must be nonzero",
                     0, 0, "filtfilt", "", "numkit:filtfilt:zeroLead");

    ScratchArena scratch(mr);
    auto bn = ScratchVec<double>(nb, &scratch);
    auto an = ScratchVec<double>(na, &scratch);
    for (size_t i = 0; i < nb; ++i)
        bn[i] = bd[i] / a0;
    for (size_t i = 0; i < na; ++i)
        an[i] = ad[i] / a0;

    const size_t nfilt = std::max(nb, na);

    // Edge-reflect padding, length 3 * nfilt on each side (capped at nx - 1)
    size_t nEdge = 3 * nfilt;
    if (nEdge >= nx)
        nEdge = nx - 1;

    const size_t extLen = nx + 2 * nEdge;
    auto ext = ScratchVec<double>(extLen, &scratch);
    for (size_t i = 0; i < nEdge; ++i)
        ext[i] = 2.0 * xd[0] - xd[nEdge - i];
    for (size_t i = 0; i < nx; ++i)
        ext[nEdge + i] = xd[i];
    for (size_t i = 0; i < nEdge; ++i)
        ext[nEdge + nx + i] = 2.0 * xd[nx - 1] - xd[nx - 2 - i];

    auto fwd = applyFilterDf2t(bn.data(), nb, an.data(), na, ext.data(), extLen, &scratch);
    std::reverse(fwd.begin(), fwd.end());
    auto bwd = applyFilterDf2t(bn.data(), nb, an.data(), na, fwd.data(), fwd.size(), &scratch);
    std::reverse(bwd.begin(), bwd.end());

    auto r = createLike(x, ValueType::DOUBLE, mr);
    double *y = r.doubleDataMut();
    std::memcpy(y, bwd.data() + nEdge, nx * sizeof(double));
    return r;
}

} // namespace numkit::signal
