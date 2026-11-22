// toolboxes/.../int_math_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by int_math.cpp + int_math_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/math.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <utility>


namespace numkit::builtin {

namespace {

inline keel::i64 toInt64(keel::f64 v)
{
    if (keel::is_nan(v) || keel::is_inf(v)) return 0;
    return keel::as<keel::i64>(v);
}

inline keel::i64 gcdInt(keel::i64 a, keel::i64 b)
{
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b != 0) {
        const keel::i64 t = a % b;
        a = b;
        b = t;
    }
    return a;
}

} // namespace

} // namespace numkit::builtin
