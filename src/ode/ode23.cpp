// toolboxes/ode/src/ode23.cpp
//
// Explicit Bogacki-Shampine 3(2) Runge-Kutta integrator — thin Value/FnHandle
// adapter delegating numerical integration to keel::ode::ode23.

#include <numkit/ode/options.hpp>
#include <numkit/ode/solvers.hpp>

#include "ode_detail.hpp"

namespace numkit::ode {

std::tuple<Value, Value>
ode23(FnHandle rhs, const Value &tspan, const Value &y0,
      const Value &opts, keel::Allocator *mr)
{
    return runKeelOde(&keel::ode::ode23, "ode23", /*refine_default=*/1, rhs, tspan, y0, opts, mr);
}

} // namespace numkit::ode
