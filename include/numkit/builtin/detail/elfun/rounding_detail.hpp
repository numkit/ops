// toolboxes/.../rounding_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by rounding.cpp + rounding_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/helpers.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>


namespace numkit::builtin {

namespace {

// Shared shape: scalar shortcut → result; double-array → SIMD/portable
// loop; other numeric types → fall through to unaryDouble (which calls
// Value::elemAsDouble per element).
template <typename ScalarOp, typename SimdOp>
Value roundLikeDispatch(const Value &x, ScalarOp scalar, SimdOp simdLoop, keel::Allocator *mr)
{
    // floor/ceil/round/fix are the IDENTITY on integer-typed values; MATLAB
    // keeps the integer class. (The double path below would throw on integer
    // storage / drop the class for a scalar.)
    if (isIntegerType(x.type()))
        return copyIntegerSameClass(x, mr);
    if (x.isComplex()) {
        // MATLAB applies floor/ceil/round/fix component-wise to re and im
        // (e.g. floor(3+4.7i) = 3+4i). Checked before isScalar — toScalar()
        // rejects a complex scalar.
        auto cxOp = [&scalar](keel::c64 z) {
            return keel::c64(scalar(z.real()), scalar(z.imag()));
        };
        if (x.isScalar())
            return Value::complexScalar(cxOp(x.toComplex()), mr);
        return unaryComplex(x, cxOp, mr);
    }
    if (x.isScalar())
        return Value::scalar(scalar(x.toScalar()), mr);
    if (x.type() == ValueType::DOUBLE) {
        Value r = createLike(x, ValueType::DOUBLE, mr);
        if (x.numel() == 0) return r;
        simdLoop(x.doubleData(), r.doubleDataMut(), x.numel());
        return r;
    }
    return unaryDouble(x, scalar, mr);
}

// round(x, N): N decimal places (N may be negative). round(x, N,
// 'significant'): N significant digits. Round-half-away-from-zero (MATLAB).
inline keel::f64 roundNScalar(keel::f64 v, keel::i32 n, bool significant)
{
    if (!std::isfinite(v)) return v;
    keel::i32 digits = n;
    if (significant) {
        if (v == 0.0) return 0.0;
        digits = n - keel::as<keel::i32>(std::floor(std::log10(std::fabs(v)))) - 1;
    }
    const keel::f64 f = std::pow(10.0, digits);
    return std::round(v * f) / f;
}

} // namespace

// round(x,n)/round(x,n,'significant') worker (def in rounding.cpp, external).
Value roundN(const Value &x, keel::i32 n, bool significant, keel::Allocator *mr);

} // namespace numkit::builtin
