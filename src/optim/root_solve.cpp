// ops/src/optim/root_solve.cpp
//
// Iterative solver kernels (Brent root / bracket / Brent-min / Nelder-Mead) —
// thin FnHandle adapters delegating numerical optimization to keel::optim.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/optim.h>
#include <keel/ref.h>
#include <numkit/ops/root_solve.hpp>

#include <numkit/ops/callback_eval.hpp>
#include <numkit/value/error.hpp>

#include <utility>

namespace numkit::ops {

std::pair<double, double> findBracket(FnHandle fn, double x0, keel::Allocator *mr)
{
    auto eval = [&](keel::f64 x) -> keel::f64 { return evalScalar(fn, x, mr); };
    const auto res = keel::optim::find_bracket(eval, x0, 60);
    if (!res.ok())
        throw Error("fzero: failed to find a bracket containing a sign change near x0", 0, 0, "fzero",
                    "", "numkit:fzero:noBracket");
    return {res.a, res.b};
}

double brent(FnHandle fn, double a, double b, keel::Allocator *mr)
{
    auto eval = [&](keel::f64 x) -> keel::f64 { return evalScalar(fn, x, mr); };
    const auto res = keel::optim::brent_root(eval, a, b, 1e-15, 200);
    if (res.status == keel::optim::Status::NoSignChange)
        throw Error("fzero: f(a) and f(b) must have opposite signs "
                    "(no sign change in the supplied interval)",
                    0, 0, "fzero", "", "numkit:fzero:noSignChange");
    if (!res.ok())
        throw Error("fzero: failed to converge within iteration limit", 0, 0, "fzero", "",
                    "numkit:fzero:noConverge");
    return res.x;
}

double brentMin(FnHandle fn, double a, double b, double tol, keel::Allocator *mr)
{
    auto eval = [&](keel::f64 x) -> keel::f64 { return evalScalar(fn, x, mr); };
    const auto res = keel::optim::brent_min(eval, a, b, tol, 200);
    return res.x;
}

keel::Array<double> nelderMead(FnHandle fn, const double *x0, keel::usize n, double tol,
                               keel::Allocator *mr)
{
    auto eval = [&](keel::ArrayRef<keel::f64> x) -> keel::f64 {
        return evalVecToScalar(fn, x.data(), x.size(), mr);
    };
    auto res = keel::optim::nelder_mead(eval, keel::ArrayRef<keel::f64>(x0, n), tol, 500, mr);
    return std::move(res.x);
}

} // namespace numkit::ops
