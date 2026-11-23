// toolboxes/ode/src/ode45.cpp
//
// Explicit Dormand-Prince 5(4) Runge-Kutta integrator — thin Value/FnHandle
// adapter delegating numerical integration to keel::ode::ode45.

#include <keel/tuple.h>
#include <numkit/ode/options.hpp>
#include <numkit/ode/solvers.hpp>

#include "ode_detail.hpp"

namespace numkit::ode {

keel::Tuple<Value, Value>
ode45(FnHandle rhs, const Value &tspan, const Value &y0,
      const Value &opts, keel::Allocator *mr)
{
    return runKeelOde(&keel::ode::ode45, "ode45", /*refine_default=*/4, rhs, tspan, y0, opts, mr);
}

} // namespace numkit::ode
