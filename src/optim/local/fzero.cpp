// toolboxes/optim/src/local/fzero.cpp
//
// fzero / fminbnd / fminsearch â€” the Value-level public API. The numerical
// solver kernels (Brent root / outward bracket / Brent-min / Nelder-Mead) now
// live in ops (numkit::ops::{findBracket,brent,brentMin,nelderMead},
// ops/root_solve.hpp) per LAYERING_TARGET_ARCHITECTURE Â§2/Â§3/Â§9 â€” these wrappers
// just marshal Value<->double and call them. All three accept a numkit::FnHandle
// objective (no Engine dependency); the engine adapters live in bundle.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <numkit/optim/local/fzero.hpp>

#include <numkit/ops/root_solve.hpp>
#include <numkit/value/error.hpp>

#include <cmath>
#include <utility>

namespace numkit::optim {

Value fzero(FnHandle fn, double x0, keel::Allocator *mr)
{
    if (!std::isfinite(x0))
        throw Error("fzero: x0 must be finite", 0, 0, "fzero", "", "numkit:fzero:badX0");
    auto [a, b] = ops::findBracket(fn, x0, mr);
    if (a == b) return Value::scalar(a, mr);
    if (a > b) std::swap(a, b);
    return Value::scalar(ops::brent(fn, a, b, mr), mr);
}

Value fzero(FnHandle fn, double a, double b, keel::Allocator *mr)
{
    if (!std::isfinite(a) || !std::isfinite(b) || a >= b)
        throw Error("fzero: interval [a, b] must satisfy a < b and be finite", 0, 0, "fzero", "",
                    "numkit:fzero:badInterval");
    return Value::scalar(ops::brent(fn, a, b, mr), mr);
}

Value fminbnd(FnHandle fn, double lo, double hi, double tol, keel::Allocator *mr)
{
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo >= hi)
        throw Error("fminbnd: lo < hi must be finite", 0, 0, "fminbnd", "",
                    "numkit:fminbnd:badRange");
    if (!(tol > 0)) tol = 1e-6;
    return Value::scalar(ops::brentMin(fn, lo, hi, tol, mr), mr);
}

Value fminsearch(FnHandle fn, keel::ArrayRef<double> x0, double tol, keel::Allocator *mr)
{
    const keel::usize n = x0.size();
    if (n == 0)
        throw Error("fminsearch: x0 must be non-empty", 0, 0, "fminsearch", "",
                    "numkit:fminsearch:badX0");
    if (!(tol > 0)) tol = 1e-4;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto         best = ops::nelderMead(fn, x0.data(), n, tol, scratch.allocator());
    Value        r    = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < n; ++i) r.doubleDataMut()[i] = best[i];
    return r;
}

} // namespace numkit::optim
