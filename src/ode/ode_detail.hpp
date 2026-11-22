// toolboxes/ode/src/ode_detail.hpp
//
// Private (src-only) helpers shared by the explicit Runge-Kutta integrators
// ode45 (Dormand-Prince 5(4)) and ode23 (Bogacki-Shampine 3(2)): the options
// struct, its parser, and the FnHandle <-> keel::ode adapter.
#pragma once

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ode.h>
#include <keel/ref.h>
#include <keel/static_array.h>
#include <numkit/value/error.hpp>
#include <numkit/value/fn_handle.hpp>
#include <numkit/value/value.hpp>

#include <tuple>
#include <utility>

namespace numkit::ode {

// Solver options shared by ode45 / ode23. All defaults are common EXCEPT
// `refine` (ode45 -> 4, ode23 -> 1), which the caller supplies to read_opts.
struct OdeOpts {
    keel::f64              rel_tol      = 1e-3;
    keel::Array<keel::f64> abs_tol      = keel::Array<keel::f64>(1, 1e-6, nullptr);
    keel::f64              max_step     = keel::Limits<keel::f64>::infinity();
    keel::f64              initial_step = 0.0;
    keel::i32                 refine       = 4;
    bool                norm_control = false;
    bool                stats        = false;
};

// Parse the options struct. `refine_default` is the solver's default Refine
// (ode45 -> 4, ode23 -> 1); an explicit opts.Refine field overrides it.
inline OdeOpts read_opts(const Value &opts, keel::i32 refine_default)
{
    OdeOpts o;
    o.refine = refine_default;
    if (opts.isEmpty() || !opts.isStruct()) return o;
    auto get_double = [&](const char *name, keel::f64 def) -> keel::f64 {
        if (opts.hasField(name)) {
            const Value &v = opts.field(name);
            if (!v.isEmpty()) return v.toScalar();
        }
        return def;
    };
    o.rel_tol      = get_double("RelTol", 1e-3);
    o.max_step     = get_double("MaxStep", keel::Limits<keel::f64>::infinity());
    o.initial_step = get_double("InitialStep", 0.0);
    if (opts.hasField("Refine")) {
        const Value &v = opts.field("Refine");
        if (!v.isEmpty()) {
            const keel::f64 r = v.toScalar();
            o.refine = (r >= 1.0) ? keel::as<keel::i32>(r) : 1;
        }
    }
    if (opts.hasField("NormControl")) {
        const Value &v = opts.field("NormControl");
        if (!v.isEmpty()) {
            const auto s = v.toString();
            o.norm_control = (!s.empty() && (s[0] == 'o' || s[0] == 'O'));
        }
    }
    if (opts.hasField("AbsTol")) {
        const Value &v = opts.field("AbsTol");
        if (!v.isEmpty()) {
            const keel::usize n = v.numel();
            o.abs_tol.resize(n);
            for (keel::usize i = 0; i < n; ++i) o.abs_tol[i] = v.elemAsDouble(i);
        }
    }
    if (opts.hasField("Stats")) {
        const Value &v = opts.field("Stats");
        if (!v.isEmpty()) {
            const auto s = v.toString();
            o.stats = (!s.empty() && (s[0] == 'o' || s[0] == 'O'));
        }
    }
    return o;
}

using KeelOdeSolverFn = keel::ode::Solution (*)(keel::ode::RhsFn, keel::ArrayRef<keel::f64>,
                                                keel::ArrayRef<keel::f64>, const keel::ode::Options &,
                                                keel::Allocator *);

inline std::tuple<Value, Value>
runKeelOde(KeelOdeSolverFn solver_fn, const char *solver, keel::i32 refine_default,
           FnHandle rhs, const Value &tspan, const Value &y0,
           const Value &opts, keel::Allocator *mr)
{
    const keel::usize nspan = tspan.numel();
    keel::Array<keel::f64> ts(nspan, mr);
    for (keel::usize i = 0; i < nspan; ++i) ts[i] = tspan.elemAsDouble(i);

    const keel::usize d = y0.numel();
    keel::Array<keel::f64> y_init(d, mr);
    for (keel::usize i = 0; i < d; ++i) y_init[i] = y0.elemAsDouble(i);

    OdeOpts O = read_opts(opts, refine_default);
    keel::ode::Options kopts;
    kopts.rel_tol = O.rel_tol;
    kopts.abs_tol = keel::ArrayRef<keel::f64>(O.abs_tol.data(), O.abs_tol.size());
    kopts.max_step = O.max_step;
    kopts.initial_step = O.initial_step;
    kopts.refine = O.refine;
    kopts.max_steps = 100000;
    kopts.max_consecutive_failures = 10;

    auto rhs_bridge = [&](keel::f64 t, keel::ArrayRef<keel::f64> y,
                          keel::MutArrayRef<keel::f64> dydt) -> bool {
        const keel::usize dim = y.size();
        Value tv = Value::scalar(t, mr);
        Value yv = Value::matrix(dim, 1, ValueType::DOUBLE, mr);
        if (dim > 0) keel::copy_n(yv.doubleDataMut(), y.data(), dim);
        keel::StaticArray<Value, 2> args_buf{tv, yv};
        Value out_buf;
        rhs(keel::ArrayRef<Value>(args_buf.data(), 2), keel::MutArrayRef<Value>(&out_buf, 1), mr);
        if (out_buf.numel() != dim)
            throw Error(keel::format("{}: RHS returned {} values but expected {}",
                                     solver, out_buf.numel(), dim),
                        0, 0, solver, "", keel::format("numkit:{}:badRhsSize", solver));
        for (keel::usize i = 0; i < dim; ++i) dydt[i] = out_buf.elemAsDouble(i);
        return true;
    };

    keel::ode::Solution sol = solver_fn(
        rhs_bridge,
        keel::ArrayRef<keel::f64>(ts.data(), ts.size()),
        keel::ArrayRef<keel::f64>(y_init.data(), y_init.size()),
        kopts, mr);

    switch (sol.status) {
    case keel::ode::Status::Ok:
        break;
    case keel::ode::Status::InvalidTspan:
        throw Error(keel::format("{}: tspan must have at least 2 elements", solver),
                    0, 0, solver, "", keel::format("numkit:{}:tspanShort", solver));
    case keel::ode::Status::DegenerateTspan:
        throw Error(keel::format("{}: tspan(end) must differ from tspan(1)", solver),
                    0, 0, solver, "", keel::format("numkit:{}:tspanDegenerate", solver));
    case keel::ode::Status::NonMonotonicTspan:
        throw Error(keel::format("{}: tspan must be strictly monotonic", solver),
                    0, 0, solver, "", keel::format("numkit:{}:tspanMono", solver));
    case keel::ode::Status::InvalidAbsTol:
        throw Error(keel::format("{}: AbsTol must be scalar or match length(y0)", solver),
                    0, 0, solver, "", keel::format("numkit:{}:absTolSize", solver));
    case keel::ode::Status::TooManyFailures:
        throw Error(keel::format("{}: 10 consecutive step rejections; "
                                 "tolerances may be too tight or RHS too stiff", solver),
                    0, 0, solver, "", keel::format("numkit:{}:tooManyFailures", solver));
    case keel::ode::Status::StepUnderflow:
        throw Error(keel::format("{}: step size underflow", solver),
                    0, 0, solver, "", keel::format("numkit:{}:stepUnderflow", solver));
    case keel::ode::Status::TooManySteps:
        throw Error(keel::format("{}: exceeded {} integration steps", solver, kopts.max_steps),
                    0, 0, solver, "", keel::format("numkit:{}:tooManySteps", solver));
    case keel::ode::Status::EmptyState:
    case keel::ode::Status::InvalidRhs:
    case keel::ode::Status::RhsError:
        throw Error(keel::format("{}: integration failed", solver),
                    0, 0, solver, "", keel::format("numkit:{}:failed", solver));
    }

    const keel::usize m = sol.num_points();
    Value t_val = Value::matrix(m, 1, ValueType::DOUBLE, mr);
    if (m > 0) keel::copy_n(t_val.doubleDataMut(), sol.t.data(), m);
    Value y_val = Value::matrix(m, d, ValueType::DOUBLE, mr);
    if (m * d > 0) keel::copy_n(y_val.doubleDataMut(), sol.y_colmajor.data(), m * d);
    return {std::move(t_val), std::move(y_val)};
}

} // namespace numkit::ode
