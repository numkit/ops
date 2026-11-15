/// @file options.hpp
/// @ingroup group_ode
// toolboxes/ode/include/numkit/ode/options.hpp
//
// odeset / odeget â€” assembled options struct passed to the ode* solvers.
//
// Reference:
//   â€¢ Shampine & Reichelt, "The MATLAB ODE Suite", SIAM J. Sci. Comput.
//     18(1), 1997.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/value/value.hpp>

namespace numkit::ode {

/// @addtogroup group_ode
/// @{


/// @brief Build / extend an ODE options struct (`odeset(...)`).
///
/// MATLAB signatures supported:
///   * `opts = odeset()` â€” return defaults.
///   * `opts = odeset('Name', Value, ...)` â€” name-value pairs.
///   * `opts = odeset(old, 'Name', Value, ...)` â€” extend an existing struct.
///   * `opts = odeset(old, new)` â€” merge two structs (new wins).
///
/// Recognised names (case-insensitive, MATLAB convention):
///   RelTol, AbsTol, NormControl, MaxStep, InitialStep, NonNegative,
///   Refine, Stats, OutputFcn, OutputSel, Mass, MStateDependence,
///   MvPattern, MassSingular, InitialSlope, Vectorized, Events,
///   BDF, MaxOrder, Jacobian, JPattern.
///
/// Unknown names throw `m:odeset:badName`. Unset names default to `[]`.
Value odeset(const Value *args, keel::usize nargs,
             keel::Allocator *mr = nullptr);

/// @brief Read a single field from an ODE options struct
/// (`val = odeget(opts, 'Name'[, default])`).
///
/// Returns the value of `Name` from `opts`, or `default` (or `[]`) if
/// unset. Case-insensitive name lookup.
Value odeget(const Value &opts, const Value &name,
             const Value &default_v, keel::Allocator *mr = nullptr);


/// @}
} // namespace numkit::ode
