// toolboxes/signal/src/digital_filtering/filter_detail.hpp
//
// Private (src-only) helpers shared between the engine-free compute in
// filter.cpp and its CallContext register half in filter_reg.cpp. NOT part of
// the public signal API (library_api.md forbids raw double*/Complex* on the
// public surface). The register wrapper reuses these to honour the zi/zf
// initial-condition forms of filter().
//
// The raw-buffer IIR recurrence kernels (applyFilterDf2t / ...Complex) moved to
// the kernel layer — numkit::ops::applyFilterDf2t{,Complex} (nk/ops/
// iir_filter.hpp). They are re-exported here as signal-scope aliases so the
// existing unqualified call sites (filter.cpp, filtfilt, filter_reg.cpp) need
// no churn. The Value-level marshalling helpers stay in signal.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <numkit/ops/iir_filter.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/scratch.hpp>

#include <cstddef>
#include <memory_resource>

namespace numkit::signal {

// IIR recurrence kernels now live in ops; re-export under the signal namespace.
using numkit::ops::applyFilterDf2t;
using numkit::ops::applyFilterDf2tComplex;

// Marshal a Value's first n elements into a complex scratch buffer.
ScratchVec<Complex> toComplexBuf(const Value &v, size_t n,
                                 std::pmr::memory_resource *mr);

// Length of x along its first non-singleton dimension.
size_t firstNonSingletonExtent(const Value &x);

// filter's checker + dispatch-plan resolver (the argcheck-separation
// pattern): validates the b/a/x/zi contracts with nk-voiced
// diagnostics and resolves the once-per-call dispatch decisions so the
// plain and [y, zf] paths share ONE derivation (the [y,zf] tail in
// filter_reg used to re-derive all of this). Read-only, allocation-free.
struct FilterPlan
{
    bool   complex;    // any of b/a/x/zi complex
    size_t nb, na, nx;
    size_t sigLen;     // first non-singleton extent of x
    size_t nSig;       // nx / sigLen (0 for empty x)
    size_t stateLen;   // max(nb, na) - 1
};

// checker+resolver, deliberately one pass (the <fn>_argcheck shape for
// Value-typed APIs, library_api.md §20): every check here inevitably
// computes a dispatch fact, and returning those facts as the plan makes
// "accepted" and "what the kernel executes" atomic — the kernel never
// re-inspects the Values, so validation and interpretation cannot
// drift apart. Validation is NOT optional for this surface: rejection
// behavior is part of the observable contract.
FilterPlan filter_argcheck(const Value &b, const Value &a, const Value &x,
                           const Value &zi);

} // namespace numkit::signal
