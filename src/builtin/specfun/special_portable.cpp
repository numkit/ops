// toolboxes/builtin/src/math/special/special_portable.cpp
//
// Reference scalar erf. Compiled when NUMKIT_WITH_SIMD=OFF; the Highway-
// dispatched variant (SLEEF-ported dd kernel) lives in special_highway.cpp
// and matches this file on every input outside the vectorised range.

#include <keel/memory.h>
#include <keel/math.h>
#include <numkit/builtin/specfun.hpp>

#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>


namespace numkit::builtin {

Value erf(const Value &x, keel::Allocator *mr)
{
    return unaryDouble(x, [](keel::f64 v) { return keel::erf(v); }, mr);
}

} // namespace numkit::builtin
