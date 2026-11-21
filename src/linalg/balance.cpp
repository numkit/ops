// toolboxes/linalg/src/balance.cpp
//
// Diagonal-similarity scaling and permutation for eigenvalue computations.
// Parlett-Reinsch (1969) algorithm via keel::linalg::balance_matrix.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/linalg.h>
#include <numkit/linalg/balance.hpp>

#include "linalg_detail.hpp"

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>

namespace numkit::linalg {

BalanceResult
balance_impl(const Value &A, bool noperm, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "balance");
    if (A.dims().is3D())
        throw Error("balance: input must be 2D",
                    0, 0, "balance", "", "numkit:balance:Not2D");
    const keel::usize n = A.dims().rows();
    if (A.dims().cols() != n)
        throw Error("balance: matrix must be square",
                    0, 0, "balance", "", "numkit:balance:NotSquare");

    const bool wantSingle = (A.type() == ValueType::SINGLE);
    const Value Aw = detail::widenSingleToDouble(A, mr);

    BalanceResult R;
    R.B = Value::matrix(n, n, Aw.type(), mr);
    R.d_col = Value::matrix(n == 0 ? 0 : n, n == 0 ? 0 : 1, ValueType::DOUBLE, mr);
    R.perm_col = Value::matrix(n == 0 ? 0 : n, n == 0 ? 0 : 1, ValueType::DOUBLE, mr);

    if (n == 0) {
        R.B = detail::narrowDoubleToSingleIf(R.B, wantSingle, mr);
        R.d_col = detail::narrowDoubleToSingleIf(R.d_col, wantSingle, mr);
        return R;
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> d(n, scratch.allocator());
    keel::Array<keel::usize> perm(n, scratch.allocator());

    if (Aw.isComplex()) {
        std::copy(Aw.complexData(), Aw.complexData() + n * n, R.B.complexDataMut());
        keel::linalg::balance_matrix(R.B.complexDataMut(), n, !noperm, d.data(), perm.data());
    } else {
        std::copy(Aw.doubleData(), Aw.doubleData() + n * n, R.B.doubleDataMut());
        keel::linalg::balance_matrix(R.B.doubleDataMut(), n, !noperm, d.data(), perm.data());
    }

    keel::f64 *pd = R.perm_col.doubleDataMut();
    keel::f64 *dd = R.d_col.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) {
        pd[i] = keel::as<keel::f64>(perm[i] + 1);
        dd[i] = d[i];
    }

    R.B = detail::narrowDoubleToSingleIf(R.B, wantSingle, mr);
    R.d_col = detail::narrowDoubleToSingleIf(R.d_col, wantSingle, mr);
    return R;
}

} // namespace numkit::linalg
