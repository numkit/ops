// toolboxes/linalg/src/balance.cpp
//
// MATLAB balance: diagonal-similarity scaling and permutation for eigenvalue computations.
// Parlett-Reinsch (1969) algorithm; same as EISPACK `balanc`.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/math.h>
#include <numkit/linalg/balance.hpp>

#include "linalg_detail.hpp"

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace numkit::linalg {

namespace {

// Parlett-Reinsch permutation phase: isolate rows/columns with zero off-diagonal entries
template <typename T>
void balancePermute(T *B, keel::usize n, keel::usize &low, keel::usize &high, double *p)
{
    for (keel::usize i = 0; i < n; ++i) p[i] = static_cast<double>(i + 1);
    if (n <= 1) { low = 0; high = (n == 0) ? 0 : n - 1; return; }

    low = 0;
    high = n - 1;

    bool found = true;
    while (found) {
        found = false;

        // Search for isolated rows in [low, high]
        for (std::intptr_t i = static_cast<std::intptr_t>(high); i >= static_cast<std::intptr_t>(low); --i) {
            keel::usize ui = static_cast<keel::usize>(i);
            bool isolated = true;
            for (keel::usize j = low; j <= high; ++j) {
                if (j == ui) continue;
                if (B[ui + j * n] != T(0.0)) { isolated = false; break; }
            }
            if (isolated) {
                p[high] = static_cast<double>(ui + 1);
                if (ui != high) {
                    for (keel::usize k = 0; k < n; ++k) std::swap(B[ui + k * n], B[high + k * n]);
                    for (keel::usize k = 0; k < n; ++k) std::swap(B[k + ui * n], B[k + high * n]);
                }
                if (high == 0) break;
                high--;
                found = true;
                break;
            }
        }
        if (found) continue;

        // Search for isolated columns in [low, high]
        for (keel::usize j = low; j <= high; ++j) {
            bool isolated = true;
            for (keel::usize i = low; i <= high; ++i) {
                if (i == j) continue;
                if (B[i + j * n] != T(0.0)) { isolated = false; break; }
            }
            if (isolated) {
                p[low] = static_cast<double>(j + 1);
                if (j != low) {
                    for (keel::usize k = 0; k < n; ++k) std::swap(B[j + k * n], B[low + k * n]);
                    for (keel::usize k = 0; k < n; ++k) std::swap(B[k + j * n], B[k + low * n]);
                }
                low++;
                found = true;
                break;
            }
        }
    }
}

// Parlett-Reinsch scaling phase on submatrix B[low..high, low..high]
template <typename T>
void balanceScale(T *B, keel::usize n, double *d, keel::usize low, keel::usize high)
{
    constexpr double radix = 2.0;
    constexpr double sqrdx = radix * radix;
    constexpr double threshold = 0.95;

    for (keel::usize i = 0; i < n; ++i) d[i] = 1.0;
    if (n <= 1 || low >= high) return;

    bool done = false;
    while (!done) {
        done = true;
        for (keel::usize i = low; i <= high; ++i) {
            double r = 0.0;
            double c = 0.0;
            for (keel::usize j = low; j <= high; ++j) {
                if (j == i) continue;
                r += keel::abs(B[i + j * n]);
                c += keel::abs(B[j + i * n]);
            }
            if (r == 0.0 || c == 0.0) continue;

            double g = r / radix;
            double f = 1.0;
            const double s = c + r;
            while (c < g) {
                f *= radix;
                c *= sqrdx;
            }
            g = r * radix;
            while (c >= g) {
                f /= radix;
                c /= sqrdx;
            }
            if ((c + r) / f < threshold * s) {
                done = false;
                d[i] *= f;
                for (keel::usize j = low; j <= high; ++j) B[i + j * n] /= f;
                for (keel::usize j = low; j <= high; ++j) B[j + i * n] *= f;
            }
        }
    }
}

} // namespace

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
    keel::Array<double> d(n, scratch.allocator());

    double *pd = R.perm_col.doubleDataMut();
    keel::usize low = 0, high = (n > 0) ? n - 1 : 0;

    if (Aw.isComplex()) {
        std::copy(Aw.complexData(), Aw.complexData() + n * n, R.B.complexDataMut());
        if (!noperm) {
            balancePermute(R.B.complexDataMut(), n, low, high, pd);
        } else {
            for (keel::usize i = 0; i < n; ++i) pd[i] = static_cast<double>(i + 1);
        }
        balanceScale(R.B.complexDataMut(), n, d.data(), low, high);
    } else {
        std::copy(Aw.doubleData(), Aw.doubleData() + n * n, R.B.doubleDataMut());
        if (!noperm) {
            balancePermute(R.B.doubleDataMut(), n, low, high, pd);
        } else {
            for (keel::usize i = 0; i < n; ++i) pd[i] = static_cast<double>(i + 1);
        }
        balanceScale(R.B.doubleDataMut(), n, d.data(), low, high);
    }

    double *dd = R.d_col.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) dd[i] = d[i];

    R.B = detail::narrowDoubleToSingleIf(R.B, wantSingle, mr);
    R.d_col = detail::narrowDoubleToSingleIf(R.d_col, wantSingle, mr);
    return R;
}

} // namespace numkit::linalg
