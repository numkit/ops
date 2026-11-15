// toolboxes/builtin/src/lang/arrays/matrix.cpp

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <keel/string.h>
#include <numkit/builtin/elmat.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/ops/reductions.hpp>
#include <numkit/ops/rows_helpers.hpp>
#include <numkit/ops/binary_ops.hpp>
#include <numkit/ops/la_solve.hpp>
#include "datafun/cumsum.hpp"
#include <numkit/builtin/polyfun.hpp>

#include <numkit/builtin/elmat.hpp>     // flip()
#include <numkit/builtin/ops.hpp>  // transposeNC()

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "matrix_detail.hpp"

namespace numkit::builtin {

// ════════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════════

// ── Constructors ──────────────────────────────────────────────────────
Value zeros(keel::usize rows, keel::usize cols, keel::usize pages, keel::Allocator *mr)
{
    return createMatrix({rows, cols, pages}, ValueType::DOUBLE, mr);
}

Value ones(keel::usize rows, keel::usize cols, keel::usize pages, keel::Allocator *mr)
{
    auto m = createMatrix({rows, cols, pages}, ValueType::DOUBLE, mr);
    double *p = m.doubleDataMut();
    for (keel::usize i = 0; i < m.numel(); ++i)
        p[i] = 1.0;
    return m;
}

// ND overloads: caller passes a flat dim list. For nd <= 3 these just
// route to the legacy 2D/3D ctors via createMatrixND; nd > 3 hits the
// Value::matrixND ctor and the SBO Dims storage.
Value zerosND(const keel::usize *dims, keel::usize nDims, keel::Allocator *mr)
{
    return createMatrixND(dims, nDims, ValueType::DOUBLE, mr);
}

Value onesND(const keel::usize *dims, keel::usize nDims, keel::Allocator *mr)
{
    auto m = createMatrixND(dims, nDims, ValueType::DOUBLE, mr);
    double *p = m.doubleDataMut();
    for (keel::usize i = 0; i < m.numel(); ++i)
        p[i] = 1.0;
    return m;
}

Value eye(keel::usize rows, keel::usize cols, keel::Allocator *mr)
{
    auto m = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < std::min(rows, cols); ++i)
        m.elem(i, i) = 1.0;
    return m;
}


// NOTE: inv / linsolve migrated to toolboxes/linalg (properties.cpp, solvers.cpp).
//       pageinv migrated to toolboxes/linalg (page_ops.cpp).
//       The internal helpers laSolveWrap / fillIdentity went with them.

// NOTE: SVD (one-sided Jacobi) migrated to toolboxes/linalg/src/decompositions.cpp.
// Block below kept disabled until removal.

// NOTE: eig family, hess, schur_sym, sylvester_sym, expm, logm_sym,
//       sqrtm_sym migrated to toolboxes/linalg/src/{eig,matrix_functions}.cpp.
//
// poly_of_matrix below is the ONE exception that stays: builtin's
// polynomials::poly_reg dispatches matrix-input to characteristic
// polynomial via this helper, so it has to live in builtin. linalg::
// eig_general_values has its own copy in linalg/src/eig.cpp — a small
// DRY violation tolerated because it's a leaf helper (no other state),
// and the alternative (publishing it from builtin to linalg via a
// public header) trades one tiny duplication for inter-lib include
// surface. TODO: consider promoting to toolboxes/builtin/internal/ later.
Value poly_of_matrix(const Value &A, keel::Allocator *mr)
{
    if (A.dims().ndim() != 2)
        throw Error("poly: input must be a 2D matrix",
                    0, 0, "poly", "", "numkit:poly:notMatrix");
    const keel::usize m = static_cast<keel::usize>(A.dims().dim(0));
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(1));
    if (m != n)
        throw Error("poly: matrix must be square (use poly(roots) for vector input)",
                    0, 0, "poly", "", "numkit:poly:notSquare");
    if (n == 0) {
        auto out = Value::matrix(1, 1, ValueType::DOUBLE, mr);
        out.doubleDataMut()[0] = 1.0;
        return out;
    }

    // Souriau-Faddeev-LeVerrier: char poly p(λ) = λ^n + c[1]*λ^{n-1} + ... + c[n].
    //   M = I, c[0] = 1
    //   for k = 1..n:
    //     M = A * M + c[k-1] * I       (NOT this form; see corrected below)
    //
    // Corrected (standard form, e.g. Faddeev 1959):
    //   M_0 = 0  ;  c[0] = 1
    //   for k = 1..n:
    //     M_k = A * (M_{k-1} + c[k-1] * I)
    //          = A * M_{k-1} + c[k-1] * A
    //     c[k] = -trace(M_k) / k
    // After the loop, c[1..n] are the coefficients (after the leading 1).

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> M(n * n, 0.0, scratch.allocator());
    keel::Array<double> Mnext(n * n, scratch.allocator());

    auto out = Value::matrix(1, n + 1, ValueType::DOUBLE, mr);
    double *c = out.doubleDataMut();
    c[0] = 1.0;

    const double *Adata = A.doubleData();

    for (keel::usize k = 1; k <= n; ++k) {
        // Mnext = A * (M + c[k-1] * I)
        //       = A * M + c[k-1] * A
        // First: Mnext = A * M
        std::fill(Mnext.begin(), Mnext.end(), 0.0);
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize kk = 0; kk < n; ++kk) {
                const double mkj = M[kk + j * n];
                if (mkj == 0.0) continue;
                for (keel::usize i = 0; i < n; ++i)
                    Mnext[i + j * n] += Adata[i + kk * n] * mkj;
            }
        // Add c[k-1] * A
        const double cprev = c[k - 1];
        for (keel::usize i = 0; i < n * n; ++i)
            Mnext[i] += cprev * Adata[i];

        // c[k] = -trace(Mnext) / k
        double tr = 0.0;
        for (keel::usize i = 0; i < n; ++i) tr += Mnext[i + i * n];
        c[k] = -tr / static_cast<double>(k);

        std::swap(M, Mnext);
    }
    return out;
}

// Below: dead code, kept until cleanup. All migrated to toolboxes/linalg.

// NOTE: pinv / orth / null_basis migrated to toolboxes/linalg/src/
// pseudo_subspace.cpp (group 4 of the toolboxes/linalg extraction).
// Block below kept disabled until cleanup pass.

// Extended topkrows. `cols` is the 0-indexed list of columns to sort by
// (priority order — first column is primary key, etc.). If `cols` is
// empty, sort by every column in order. `desc[j]` selects descending
// (true) vs ascending (false) for `cols[j]`. `out_idx` (optional, may
// be nullptr) receives the 0-indexed selected rows for the 2-output
// form. Stable on full ties.
Value topkrows_full(const Value &A, keel::usize k,
                    keel::ArrayRef<keel::usize> colsIn,
                    keel::ArrayRef<keel::u8> descIn,
                    keel::Array<keel::usize> *out_idx,
                    keel::Allocator *mr)
{
    if (A.dims().ndim() != 2)
        throw Error("topkrows: input must be a 2D matrix",
                    0, 0, "topkrows", "", "numkit:topkrows:notMatrix");
    const keel::usize m = static_cast<keel::usize>(A.dims().dim(0));
    const keel::usize n = static_cast<keel::usize>(A.dims().dim(1));
    const keel::usize kk = std::min(k, m);

    keel::ArenaTemp scratch = keel::get_scratch(mr);

    // Resolve cols / desc defaults.
    keel::Array<keel::usize> cols(scratch.allocator());
    keel::Array<keel::u8> desc(scratch.allocator());
    if (colsIn.empty()) {
        cols.reserve(n);
        for (keel::usize j = 0; j < n; ++j) cols.push_back(j);
    } else {
        cols.assign(colsIn);
    }
    if (descIn.size() == 1) {
        desc.assign(cols.size(), descIn[0]);
    } else if (descIn.empty()) {
        desc.assign(cols.size(), 1);  // default descend
    } else if (descIn.size() == cols.size()) {
        desc.assign(descIn);
    } else {
        throw Error("topkrows: direction must be a scalar or match the "
                    "length of col",
                    0, 0, "topkrows", "", "numkit:topkrows:dirSize");
    }
    for (keel::usize j : cols)
        if (j >= n)
            throw Error("topkrows: column index out of range",
                        0, 0, "topkrows", "", "numkit:topkrows:badCol");

    keel::Array<keel::usize> idx(m, scratch.allocator());
    for (keel::usize i = 0; i < m; ++i) idx[i] = i;

    // Use elemAsDouble for any-class input.
    auto elem = [&](keel::usize r, keel::usize c) {
        return A.elemAsDouble(r + c * m);
    };
    std::stable_sort(idx.begin(), idx.end(),
        [&](keel::usize a, keel::usize b) {
            for (keel::usize k2 = 0; k2 < cols.size(); ++k2) {
                const keel::usize j = cols[k2];
                const double va = elem(a, j);
                const double vb = elem(b, j);
                if (va == vb) continue;
                const bool a_first = desc[k2] ? (va > vb) : (va < vb);
                return a_first;
            }
            return false;  // total tie → stable order
        });

    // Preserve element class.
    auto out = (n == 0) ? Value::matrix(kk, 0, A.type(), mr)
                        : Value::matrix(kk, n, A.type(), mr);
    for (keel::usize i = 0; i < kk; ++i) {
        const keel::usize r = idx[i];
        for (keel::usize j = 0; j < n; ++j) {
            const keel::usize src_lin = r + j * m;
            const keel::usize dst_lin = i + j * kk;
            // Class-preserving copy.
            switch (A.type()) {
                case ValueType::DOUBLE: out.doubleDataMut()[dst_lin] = A.doubleData()[src_lin]; break;
                case ValueType::SINGLE: out.singleDataMut()[dst_lin] = A.singleData()[src_lin]; break;
                case ValueType::UINT8:  out.uint8DataMut()[dst_lin]  = A.uint8Data()[src_lin];  break;
                case ValueType::UINT16: out.uint16DataMut()[dst_lin] = A.uint16Data()[src_lin]; break;
                case ValueType::UINT32: out.uint32DataMut()[dst_lin] = A.uint32Data()[src_lin]; break;
                case ValueType::UINT64: out.uint64DataMut()[dst_lin] = A.uint64Data()[src_lin]; break;
                case ValueType::INT8:   out.int8DataMut()[dst_lin]   = A.int8Data()[src_lin];   break;
                case ValueType::INT16:  out.int16DataMut()[dst_lin]  = A.int16Data()[src_lin];  break;
                case ValueType::INT32:  out.int32DataMut()[dst_lin]  = A.int32Data()[src_lin];  break;
                case ValueType::INT64:  out.int64DataMut()[dst_lin]  = A.int64Data()[src_lin];  break;
                case ValueType::LOGICAL: out.logicalDataMut()[dst_lin] = A.logicalData()[src_lin]; break;
                case ValueType::COMPLEX: out.complexDataMut()[dst_lin] = A.complexData()[src_lin]; break;
                case ValueType::CHAR:    out.charDataMut()[dst_lin]    = A.charData()[src_lin];    break;
                default:
                    out.doubleDataMut()[dst_lin] = elem(r, j);
                    break;
            }
        }
    }
    if (out_idx) {
        out_idx->resize(kk);
        for (keel::usize i = 0; i < kk; ++i) (*out_idx)[i] = idx[i];
    }
    return out;
}

Value topkrows(const Value &A, keel::usize k, keel::Allocator *mr)
{
    return topkrows_full(A, k, keel::ArrayRef<keel::usize>(), keel::ArrayRef<keel::u8>(), nullptr, mr);
}

// ── Toeplitz / Hankel / Vandermonde / Companion ─────────────────────


Value toeplitz(const Value &cV, const Value &rV, keel::Allocator *mr)
{
    const bool single1 = rV.isEmpty();
    const bool anyCplx  = cV.isComplex() || (!single1 && rV.isComplex());
    const bool anySingle = !anyCplx &&
        (cV.type() == ValueType::SINGLE ||
         (!single1 && rV.type() == ValueType::SINGLE));

    const keel::usize m = cV.numel();
    const keel::usize n = single1 ? m : rV.numel();
    if (m == 0 || n == 0)
        throw Error("toeplitz: inputs must be non-empty",
                    0, 0, "toeplitz", "", "numkit:toeplitz:empty");

    // COMPLEX: gather per-element so the imaginary part is preserved.
    // Two-arg form is a plain gather; the single-arg complex form is a
    // Hermitian Toeplitz matrix (MATLAB conjugates the strictly-lower
    // triangle, T(i,j) = conj(c(i-j)) for i>j, c(j-i) otherwise).
    if (anyCplx) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<keel::c64> c(m, scratch.allocator());
        for (keel::usize k = 0; k < m; ++k)
            c[k] = cV.isComplex() ? cV.complexData()[k]
                                  : keel::c64(cV.elemAsDouble(k), 0.0);
        keel::Array<keel::c64> r(n, scratch.allocator());
        if (!single1)
            for (keel::usize k = 0; k < n; ++k)
                r[k] = rV.isComplex() ? rV.complexData()[k]
                                      : keel::c64(rV.elemAsDouble(k), 0.0);
        auto M = Value::matrix(m, n, ValueType::COMPLEX, mr);
        keel::c64 *dst = M.complexDataMut();
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < m; ++i) {
                keel::c64 v;
                if (single1) v = (i > j) ? keel::conj(c[i - j]) : c[j - i];
                else         v = (i >= j) ? c[i - j] : r[j - i];
                dst[j * m + i] = v;
            }
        return M;
    }

    // Real path (DOUBLE or SINGLE output; class preserved).
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto c = valueToScratchDoubles(cV, scratch);
    keel::Array<double> r(scratch.allocator());
    if (single1) r.assign(c);
    else         r = valueToScratchDoubles(rV, scratch);

    // T[i,j] = c[i-j] (i>=j) else r[j-i]. MATLAB silently overrides r[0]
    // with c[0] when both are given (r[0] is never read here).
    const ValueType ot = anySingle ? ValueType::SINGLE : ValueType::DOUBLE;
    auto M = Value::matrix(m, n, ot, mr);
    if (ot == ValueType::SINGLE) {
        float *dst = static_cast<float *>(M.rawDataMut());
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < m; ++i)
                dst[j * m + i] = static_cast<float>((i >= j) ? c[i - j] : r[j - i]);
    } else {
        double *dst = M.doubleDataMut();
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < m; ++i)
                dst[j * m + i] = (i >= j) ? c[i - j] : r[j - i];
    }
    return M;
}

Value hankel(const Value &cV, const Value &rV, keel::Allocator *mr)
{
    const bool single1 = rV.isEmpty();
    const bool anyCplx  = cV.isComplex() || (!single1 && rV.isComplex());
    const bool anySingle = !anyCplx &&
        (cV.type() == ValueType::SINGLE ||
         (!single1 && rV.type() == ValueType::SINGLE));

    const keel::usize m = cV.numel();
    const keel::usize n = single1 ? m : rV.numel();
    if (m == 0 || n == 0)
        throw Error("hankel: inputs must be non-empty",
                    0, 0, "hankel", "", "numkit:hankel:empty");

    // H[i,j] = c[i+j] (i+j < m) else r[i+j-m+1]. hankel never conjugates.
    // Single-arg: r is all zeros (anti-triangular Hankel).
    if (anyCplx) {
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<keel::c64> c(m, scratch.allocator());
        for (keel::usize k = 0; k < m; ++k)
            c[k] = cV.isComplex() ? cV.complexData()[k]
                                  : keel::c64(cV.elemAsDouble(k), 0.0);
        keel::Array<keel::c64> r(n, scratch.allocator());
        for (keel::usize k = 0; k < n; ++k)
            r[k] = (!single1 && rV.isComplex()) ? rV.complexData()[k]
                 : (!single1)                   ? keel::c64(rV.elemAsDouble(k), 0.0)
                                                : keel::c64(0.0, 0.0);
        auto M = Value::matrix(m, n, ValueType::COMPLEX, mr);
        keel::c64 *dst = M.complexDataMut();
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < m; ++i) {
                const keel::usize s = i + j;
                dst[j * m + i] = (s < m) ? c[s] : r[s - m + 1];
            }
        return M;
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto c = valueToScratchDoubles(cV, scratch);
    keel::Array<double> r(scratch.allocator());
    if (single1) r.assign(m, 0.0);
    else         r = valueToScratchDoubles(rV, scratch);

    const ValueType ot = anySingle ? ValueType::SINGLE : ValueType::DOUBLE;
    auto M = Value::matrix(m, n, ot, mr);
    if (ot == ValueType::SINGLE) {
        float *dst = static_cast<float *>(M.rawDataMut());
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < m; ++i) {
                const keel::usize s = i + j;
                dst[j * m + i] = static_cast<float>((s < m) ? c[s] : r[s - m + 1]);
            }
    } else {
        double *dst = M.doubleDataMut();
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < m; ++i) {
                const keel::usize s = i + j;
                dst[j * m + i] = (s < m) ? c[s] : r[s - m + 1];
            }
    }
    return M;
}

Value vander(const Value &vV, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto v = valueToScratchDoubles(vV, scratch);
    const keel::usize n = v.size();
    if (n == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto M = Value::matrix(n, n, ValueType::DOUBLE, mr);
    // V[i, j] = v[i] ^ (n - 1 - j) -- highest power on the LEFT.
    for (keel::usize i = 0; i < n; ++i) {
        const double x = v[i];
        M.elem(i, n - 1) = 1.0;
        for (keel::usize k = 1; k < n; ++k)
            M.elem(i, n - 1 - k) = M.elem(i, n - k) * x;
    }
    return M;
}

Value compan(const Value &pV, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto p = valueToScratchDoubles(pV, scratch);
    const keel::usize pn = p.size();
    if (pn < 2)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if (p[0] == 0.0)
        throw Error("compan: leading coefficient must be non-zero",
                    0, 0, "compan", "", "numkit:compan:zeroLead");

    const keel::usize n = pn - 1;
    auto M = Value::matrix(n, n, ValueType::DOUBLE, mr);
    const double inv = 1.0 / p[0];
    for (keel::usize j = 0; j < n; ++j)
        M.elem(0, j) = -p[j + 1] * inv;
    for (keel::usize i = 1; i < n; ++i)
        M.elem(i, i - 1) = 1.0;
    return M;
}

// ── Pascal / Hilbert / Wilkinson / Hadamard / Rosser ────────────────

Value pascal(keel::usize n, keel::Allocator *mr)
{
    if (n == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto M = Value::matrix(n, n, ValueType::DOUBLE, mr);
    // Default symmetric form: P[i, j] = C(i+j, i). Build via the
    // recurrence P[i,j] = P[i-1,j] + P[i,j-1] with P[0,*]=P[*,0]=1.
    for (keel::usize i = 0; i < n; ++i) {
        M.elem(i, 0) = 1.0;
        M.elem(0, i) = 1.0;
    }
    for (keel::usize i = 1; i < n; ++i)
        for (keel::usize j = 1; j < n; ++j)
            M.elem(i, j) = M.elem(i - 1, j) + M.elem(i, j - 1);
    return M;
}

Value hilb(keel::usize n, keel::Allocator *mr)
{
    if (n == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto M = Value::matrix(n, n, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < n; ++i)
        for (keel::usize j = 0; j < n; ++j)
            M.elem(i, j) = 1.0 / static_cast<double>(i + j + 1);
    return M;
}

Value invhilb(keel::usize n, keel::Allocator *mr)
{
    if (n == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto M = Value::matrix(n, n, ValueType::DOUBLE, mr);
    // Closed-form (1-indexed): H⁻¹[i,j] =
    //   (-1)^(i+j) * (i+j-1) * C(n+i-1, n-j) * C(n+j-1, n-i) * C(i+j-2, i-1)²
    // Computed via long-double to delay overflow on n ≈ 13.
    auto binom = [](long n_, long k_) -> long double {
        if (k_ < 0 || k_ > n_) return 0.0L;
        if (k_ > n_ - k_) k_ = n_ - k_;
        long double r = 1.0L;
        for (long t = 1; t <= k_; ++t)
            r = r * static_cast<long double>(n_ - t + 1) / static_cast<long double>(t);
        return r;
    };
    for (keel::usize i0 = 0; i0 < n; ++i0)
        for (keel::usize j0 = 0; j0 < n; ++j0) {
            const long i = static_cast<long>(i0 + 1);
            const long j = static_cast<long>(j0 + 1);
            const long N = static_cast<long>(n);
            const long sgn = ((i + j) % 2 == 0) ? 1 : -1;
            const long double v =
                static_cast<long double>(sgn) *
                static_cast<long double>(i + j - 1) *
                binom(N + i - 1, N - j) *
                binom(N + j - 1, N - i) *
                binom(i + j - 2, i - 1) *
                binom(i + j - 2, i - 1);
            M.elem(i0, j0) = static_cast<double>(v);
        }
    return M;
}

Value wilkinson(keel::usize n, keel::Allocator *mr)
{
    if (n == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto M = Value::matrix(n, n, ValueType::DOUBLE, mr);
    // Diagonal: |(1:n) - (n+1)/2|; subdiagonal/superdiagonal: ones.
    const double mid = (static_cast<double>(n) + 1.0) / 2.0;
    for (keel::usize i = 0; i < n; ++i) {
        M.elem(i, i) = keel::abs(static_cast<double>(i + 1) - mid);
        if (i > 0) {
            M.elem(i, i - 1) = 1.0;
            M.elem(i - 1, i) = 1.0;
        }
    }
    return M;
}

Value hadamard(keel::usize n, keel::Allocator *mr)
{
    if (n == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if (n == 1) {
        auto M = Value::matrix(1, 1, ValueType::DOUBLE, mr);
        M.elem(0, 0) = 1.0;
        return M;
    }
    keel::usize base = 0;
    if ((n & (n - 1)) == 0) {
        base = 1;
    } else if (n % 12 == 0 && ((n / 12) & ((n / 12) - 1)) == 0) {
        base = 12;
    } else if (n % 20 == 0 && ((n / 20) & ((n / 20) - 1)) == 0) {
        base = 20;
    } else {
        throw Error("hadamard: n, n/12, or n/20 must be a power of 2",
                    0, 0, "hadamard", "", "numkit:hadamard:badN");
    }

    auto M = Value::matrix(n, n, ValueType::DOUBLE, mr);
    if (base == 1) {
        M.elem(0, 0) = 1.0;
    } else if (base == 12) {
        static const double r12[11] = {-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1};
        for (keel::usize i = 0; i < 12; ++i) {
            M.elem(0, i) = 1.0;
            M.elem(i, 0) = 1.0;
        }
        for (keel::usize i = 0; i < 11; ++i)
            for (keel::usize j = 0; j < 11; ++j)
                M.elem(i + 1, j + 1) = r12[(j + 11 - i) % 11];
    } else {
        static const double v20[19] = {-1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1};
        for (keel::usize i = 0; i < 20; ++i) {
            M.elem(0, i) = 1.0;
            M.elem(i, 0) = 1.0;
        }
        for (keel::usize i = 0; i < 19; ++i)
            for (keel::usize j = 0; j < 19; ++j)
                M.elem(i + 1, j + 1) = v20[(i + j) % 19];
    }

    // Sylvester recursion: H_{2k} = [Hk Hk; Hk -Hk].
    for (keel::usize k = base; k < n; k <<= 1) {
        // Quadrant copies for size doubling from k → 2k.
        for (keel::usize i = 0; i < k; ++i)
            for (keel::usize j = 0; j < k; ++j) {
                const double v = M.elem(i, j);
                M.elem(i,     j + k) =  v;
                M.elem(i + k, j)     =  v;
                M.elem(i + k, j + k) = -v;
            }
    }
    return M;
}

Value rosser(keel::Allocator *mr)
{
    // Hardcoded 8×8 Rosser test matrix (MATLAB R2025b: rosser()).
    static constexpr double R[64] = {
         611,   196, -192,  407,   -8,  -52,  -49,   29,
         196,   899,  113, -192,  -71,  -43,   -8,  -44,
        -192,   113,  899,  196,   61,   49,    8,   52,
         407,  -192,  196,  611,    8,   44,   59,  -23,
          -8,   -71,   61,    8,  411, -599,  208,  208,
         -52,   -43,   49,   44, -599,  411,  208,  208,
         -49,    -8,    8,   59,  208,  208,   99, -911,
          29,   -44,   52,  -23,  208,  208, -911,   99
    };
    auto M = Value::matrix(8, 8, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < 8; ++i)
        for (keel::usize j = 0; j < 8; ++j)
            M.elem(i, j) = R[i * 8 + j];
    return M;
}

Value magic(keel::usize N, keel::Allocator *mr)
{
    if (N == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    if (N == 1) {
        auto m = Value::matrix(1, 1, ValueType::DOUBLE, mr);
        m.doubleDataMut()[0] = 1.0;
        return m;
    }
    if (N == 2) {
        // MATLAB's magic(2) is conventional [1 3; 4 2] (not strictly magic).
        auto m = Value::matrix(2, 2, ValueType::DOUBLE, mr);
        double *p = m.doubleDataMut();
        // column-major layout: column 0 = {1, 4}, column 1 = {3, 2}
        p[0] = 1.0; p[1] = 4.0; p[2] = 3.0; p[3] = 2.0;
        return m;
    }

    auto m = Value::matrix(N, N, ValueType::DOUBLE, mr);
    // Build into a row-major scratch then transpose into column-major
    // storage. Algorithms above are written in row-major for clarity.
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> buf(N * N, 0.0, scratch.allocator());
    if (N % 2 == 1) {
        magicOdd(buf.data(), N);
    } else if (N % 4 == 0) {
        magicDoublyEven(buf.data(), N);
    } else {
        magicSinglyEven(buf.data(), N);
    }
    // Row-major buf[i*N + j] → column-major out[j*N + i]
    double *out = m.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i)
        for (keel::usize j = 0; j < N; ++j)
            out[j * N + i] = buf[i * N + j];
    return m;
}

// ── Shape queries ────────────────────────────────────────────────────
Value size(const Value &x, keel::Allocator *mr)
{
    const auto &dims = x.dims();
    // Output ndim: at least 2 (MATLAB convention — a row vector reports
    // [1, n], not [n]). Otherwise the actual rank, including any extra
    // dims past 3.
    const int n = std::max(2, dims.ndim());
    auto sv = Value::matrix(1, n, ValueType::DOUBLE, mr);
    double *out = sv.doubleDataMut();
    for (int i = 0; i < n; ++i)
        out[i] = static_cast<double>(dims.dim(i));
    return sv;
}

Value size(const Value &x, int dim, keel::Allocator *mr)
{
    return Value::scalar(static_cast<double>(x.dims().dimSize(dim - 1)), mr);
}

std::tuple<Value, Value> sizePair(const Value &x, keel::Allocator *mr)
{
    const auto &dims = x.dims();
    return std::make_tuple(
        Value::scalar(static_cast<double>(dims.rows()), mr),
        Value::scalar(static_cast<double>(dims.cols()), mr));
}

Value length(const Value &x, keel::Allocator *mr)
{
    if (x.isEmpty() || x.numel() == 0)
        return Value::scalar(0.0, mr);
    const auto &dims = x.dims();
    const double len = static_cast<double>(std::max({dims.rows(), dims.cols(), dims.pages()}));
    return Value::scalar(len, mr);
}

Value numel(const Value &x, keel::Allocator *mr)
{
    return Value::scalar(static_cast<double>(x.numel()), mr);
}

Value ndims(const Value &x, keel::Allocator *mr)
{
    return Value::scalar(static_cast<double>(x.dims().ndims()), mr);
}

// ── Shape transformations ────────────────────────────────────────────
namespace {
// reshape's compute-side contract — the one violation a size_t-taking
// C++ caller can still make: the element count must not change. First
// line of every reshape entry (bugs/closed/lang/
// reshape-size-arg-validation.md). `dimsDesc` names the requested shape
// ("5x5") so the message carries both element counts.
void reshape_argcheck(const Value &x, keel::ArrayRef<keel::usize> dims)
{
    keel::usize newNumel = 1;
    keel::String dimsDesc;
    for (keel::usize i = 0; i < dims.size(); ++i) {
        newNumel *= dims[i];
        if (i) dimsDesc.push_back('x');
        keel::format_into(dimsDesc, "{}", dims[i]);
    }
    if (newNumel != x.numel())
        throw Error(keel::format("reshape: cannot reshape {} elements into {} ({} elements): the "
                                 "element count must not change — use [] to infer "
                                 "one dimension.",
                                 x.numel(), dimsDesc, newNumel),
                    0, 0, "reshape", "", "numkit:reshape:elementCountMismatch");
}
} // namespace

Value reshape(const Value &x, keel::usize rows, keel::usize cols, keel::usize pages, keel::Allocator *mr)
{
    const keel::usize shape[3] = {rows, cols, pages};
    reshape_argcheck(x, keel::ArrayRef<keel::usize>(shape, pages > 0 ? 3 : 2));

    DimsArg d{rows, cols, pages};

    // OBJECT arrays store per-element state, column-major — reshape just
    // changes the shape (element order preserved), like CELL.
    if (x.isObject()) {
        Dims nd = (d.pages > 0) ? Dims(d.rows, d.cols, d.pages) : Dims(d.rows, d.cols);
        return x.objectReshape(nd, mr);
    }

    // CELL and STRING store element-wise, not in the raw buffer — memcpy
    // wouldn't copy Value members.
    if (x.type() == ValueType::CELL || x.type() == ValueType::STRING) {
        const bool is3D = d.pages > 0;
        Value r = (x.type() == ValueType::CELL)
            ? (is3D ? Value::cell3D(d.rows, d.cols, d.pages)
                    : Value::cell(d.rows, d.cols))
            : (is3D ? Value::stringArray3D(d.rows, d.cols, d.pages)
                    : Value::stringArray(d.rows, d.cols));
        auto &src = x.cellDataVec();
        auto &dst = r.cellDataVec();
        for (keel::usize i = 0; i < src.size() && i < dst.size(); ++i)
            dst[i] = src[i];
        return r;
    }

    auto r = createMatrix(d, x.type(), mr);
    if (x.rawBytes() > 0)
        std::memcpy(r.rawDataMut(), x.rawData(), x.rawBytes());
    return r;
}

// ND reshape. Same elem-count check, then route to matrixND for nd > 3.
// CELL/STRING ND not supported yet (matches the 2D/3D behaviour: only
// CELL/STRING currently handles 2D and 3D shapes via cell3D/stringArray3D).
Value reshapeND(const Value &x, keel::ArrayRef<keel::usize> dims, keel::Allocator *mr)
{
    const keel::usize nDims = dims.size();
    reshape_argcheck(x, dims);

    if (x.isObject())
        return x.objectReshape(Dims(dims.data(), static_cast<int>(nDims)), mr);

    if (x.type() == ValueType::CELL || x.type() == ValueType::STRING) {
        if (nDims <= 3) {
            const keel::usize r = nDims > 0 ? dims[0] : 1;
            const keel::usize c = nDims > 1 ? dims[1] : 1;
            const keel::usize p = nDims > 2 ? dims[2] : 0;
            return reshape(x, r, c, p, mr);
        }
        Value r = (x.type() == ValueType::CELL)
            ? Value::cellND(dims.data(), static_cast<int>(nDims), mr)
            : Value::stringArrayND(dims.data(), static_cast<int>(nDims), mr);
        auto &src = x.cellDataVec();
        auto &dst = r.cellDataVec();
        for (keel::usize i = 0; i < src.size() && i < dst.size(); ++i)
            dst[i] = src[i];
        return r;
    }

    auto r = createMatrixND(dims.data(), nDims, x.type(), mr);
    if (x.rawBytes() > 0)
        std::memcpy(r.rawDataMut(), x.rawData(), x.rawBytes());
    return r;
}

Value reshape(const Value &x, keel::ArrayRef<keel::usize> dims, keel::Allocator *mr)
{
    return reshapeND(x, dims, mr);
}

// ── pagetranspose / pagectranspose ───────────────────────────────────

Value pagetranspose(const Value &x, keel::Allocator *mr)
{
    return pageTransposeAny(x, /*conjugate=*/false, mr);
}

Value pagectranspose(const Value &x, keel::Allocator *mr)
{
    return pageTransposeAny(x, /*conjugate=*/true, mr);
}

// ── sphere / cylinder / ellipsoid ───────────────────────────────────
Surface3 sphere(keel::usize n, keel::Allocator *mr)
{
    // Match MATLAB's parametrisation exactly:
    //   theta = (-n:2:n)/n * pi          → linspace(-π,  π,  n+1)
    //   phi   = (-n:2:n)/n * pi/2        → linspace(-π/2, π/2, n+1)
    //   cosphi(end-points) := 0          (clamp at the poles)
    //   sintheta(end-points) := 0        (clamp θ at ±π)
    //   X = cos(phi) * cos(theta)
    //   Y = cos(phi) * sin(theta)
    //   Z = sin(phi) * ones(1, n+1)
    constexpr double kPi = 3.14159265358979323846;
    const keel::usize m = n + 1;
    auto X = Value::matrix(m, m, ValueType::DOUBLE, mr);
    auto Y = Value::matrix(m, m, ValueType::DOUBLE, mr);
    auto Z = Value::matrix(m, m, ValueType::DOUBLE, mr);
    double *xd = X.doubleDataMut();
    double *yd = Y.doubleDataMut();
    double *zd = Z.doubleDataMut();

    if (n == 0) return { std::move(X), std::move(Y), std::move(Z) };

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<double> cosPhi(m, scratch.allocator());
    keel::Array<double> sinPhi(m, scratch.allocator());
    for (keel::usize i = 0; i < m; ++i) {
        const double phi = (2.0 * static_cast<double>(i) - static_cast<double>(n))
                              / static_cast<double>(n) * (kPi / 2.0);
        cosPhi[i] = std::cos(phi);
        sinPhi[i] = std::sin(phi);
    }
    cosPhi[0] = 0.0;
    cosPhi[m - 1] = 0.0;

    keel::Array<double> cosTh(m, scratch.allocator());
    keel::Array<double> sinTh(m, scratch.allocator());
    for (keel::usize j = 0; j < m; ++j) {
        const double theta = (2.0 * static_cast<double>(j) - static_cast<double>(n))
                                / static_cast<double>(n) * kPi;
        cosTh[j] = std::cos(theta);
        sinTh[j] = std::sin(theta);
    }
    sinTh[0] = 0.0;
    sinTh[m - 1] = 0.0;

    for (keel::usize j = 0; j < m; ++j) {
        for (keel::usize i = 0; i < m; ++i) {
            const keel::usize k = j * m + i;
            xd[k] = cosPhi[i] * cosTh[j] + 0.0;
            yd[k] = cosPhi[i] * sinTh[j] + 0.0;
            zd[k] = sinPhi[i] + 0.0;
        }
    }
    return { std::move(X), std::move(Y), std::move(Z) };
}

Surface3 cylinder(const Value &R, keel::usize n, keel::Allocator *mr)
{
    // Profile R is a 1-D vector of radii along z. Output is length(R) ×
    // (n+1). z is linspace(0, 1, length(R)) repeated across columns.
    constexpr double kPi = 3.14159265358979323846;
    const keel::usize rows = R.numel();
    if (rows == 0) {
        auto Z = Value::matrix(0, 0, ValueType::DOUBLE, mr);
        return { Z, Z, Z };
    }

    const keel::usize cols = n + 1;
    auto X = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    auto Y = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    auto Z = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    double *xd = X.doubleDataMut();
    double *yd = Y.doubleDataMut();
    double *zd = Z.doubleDataMut();

    const double *Rd = R.doubleData();
    for (keel::usize j = 0; j < cols; ++j) {
        const double theta = 2.0 * kPi * static_cast<double>(j) / static_cast<double>(n);
        const double ct = std::cos(theta), st = std::sin(theta);
        for (keel::usize i = 0; i < rows; ++i) {
            const keel::usize k = j * rows + i;
            xd[k] = Rd[i] * ct;
            yd[k] = Rd[i] * st;
            zd[k] = (rows == 1) ? 0.0
                                 : static_cast<double>(i) / static_cast<double>(rows - 1);
        }
    }
    return { std::move(X), std::move(Y), std::move(Z) };
}

Surface3 ellipsoid(double xc, double yc, double zc, double xr, double yr, double zr, keel::usize n, keel::Allocator *mr)
{
    // Same parametrisation as sphere, scaled by (xr, yr, zr) and shifted.
    auto sph = sphere(n, mr);
    const keel::usize total = sph.X.numel();
    double *xd = sph.X.doubleDataMut();
    double *yd = sph.Y.doubleDataMut();
    double *zd = sph.Z.doubleDataMut();
    for (keel::usize k = 0; k < total; ++k) {
        xd[k] = xc + xr * xd[k];
        yd[k] = yc + yr * yd[k];
        zd[k] = zc + zr * zd[k];
    }
    return sph;
}

// ── peaks ────────────────────────────────────────────────────────────
Value peaks(keel::usize n, keel::Allocator *mr)
{
    if (n == 0) return Value::matrix(0, 0, ValueType::DOUBLE, mr);

    auto Z = Value::matrix(n, n, ValueType::DOUBLE, mr);
    double *zd = Z.doubleDataMut();

    // x, y = linspace(-3, 3, n) on a meshgrid: X varies along columns,
    // Y along rows (so row i → y, col j → x — matches MATLAB peaks).
    const double step = (n > 1) ? 6.0 / static_cast<double>(n - 1) : 0.0;
    for (keel::usize j = 0; j < n; ++j) {  // column index → x axis
        const double x = (n > 1) ? -3.0 + step * static_cast<double>(j) : 0.0;
        for (keel::usize i = 0; i < n; ++i) {  // row index → y axis
            const double y = (n > 1) ? -3.0 + step * static_cast<double>(i) : 0.0;
            // MATLAB peaks formula (Press et al., adapted from MATLAB source).
            const double xm1   = 1.0 - x;
            const double yp1   = y + 1.0;
            const double term1 = 3.0 * xm1 * xm1
                                  * std::exp(-(x * x) - yp1 * yp1);
            const double term2 = -10.0 * (x / 5.0 - x * x * x - y * y * y * y * y)
                                  * std::exp(-(x * x) - (y * y));
            const double xp1   = x + 1.0;
            const double term3 = -std::exp(-(xp1 * xp1) - (y * y)) / 3.0;
            zd[j * n + i] = term1 + term2 + term3;
        }
    }
    return Z;
}

// ── pagemtimes: page-wise matrix multiply ──────────────────────────────
//
// MATLAB R2020b+ batched matmul. Treats axes 1-2 of each operand as the
// matrix (M×K, K×N) and axes ≥3 as a batch index. Output batch shape is
// the NumPy broadcast of the two batch shapes. Supports DOUBLE and
// SINGLE (mixed → SINGLE, matching MATLAB's promotion rule).
//
//   Z = pagemtimes(X, Y)                     // tx = ty = None
//   Z = pagemtimes(X, "transpose", Y, "none")
//
// 'transpose' / 'ctranspose' transpose each X (or Y) page before
// multiply. For real input the two flags are identical (no imaginary
// component to conjugate). 2D × 2D collapses to ordinary matmul. One
// operand may be 2D (broadcast across the other's batch dims).


Value pagemtimes(const Value &x, const Value &y, keel::Allocator *mr)
{
    return pagemtimes(x, TranspOp::None, y, TranspOp::None, mr);
}

Value pagemtimes(const Value &x, TranspOp tx, const Value &y, TranspOp ty, keel::Allocator *mr)
{
    // MATLAB type promotion: COMPLEX wins over real; SINGLE wins over
    // DOUBLE. Integer/logical/char inputs are rejected — pagemtimes
    // requires floating or complex inputs.
    auto isFloatLike = [](ValueType t) {
        return t == ValueType::DOUBLE || t == ValueType::SINGLE || t == ValueType::COMPLEX;
    };
    if (!isFloatLike(x.type()) || !isFloatLike(y.type()))
        throw Error("pagemtimes: inputs must be 'single', 'double', or complex",
                     0, 0, "pagemtimes", "", "numkit:pagemtimes:type");
    if (x.isComplex() || y.isComplex())
        return pagemtimesImpl<keel::c64>(x, tx, y, ty, mr);
    if (x.type() == ValueType::SINGLE || y.type() == ValueType::SINGLE)
        return pagemtimesImpl<float  >(x, tx, y, ty, mr);
    return     pagemtimesImpl<double >(x, tx, y, ty, mr);
}

Value diag(const Value &x, keel::Allocator *mr)
{
    return diag(x, 0, mr);
}

Value diag(const Value &x, int k, keel::Allocator *mr)
{
    const ValueType t = x.type();
    // MATLAB: "Inputs must be numeric, char, or logical." Cells / strings
    // / structs / function handles are unsupported.
    if (t == ValueType::CELL || t == ValueType::STRING ||
        t == ValueType::STRUCT || t == ValueType::FUNC_HANDLE)
        throw Error("diag: inputs must be numeric, char, or logical",
                     0, 0, "diag", "", "numkit:diag:badType");

    const keel::usize es = elementSize(t);
    const keel::usize ak = static_cast<keel::usize>(k < 0 ? -k : k);

    if (x.dims().isVector()) {
        // Build a square matrix with x on the k-th diagonal. The output
        // matrix is zero-filled (= 0 / char(0) / false / 0+0i), so only
        // the diagonal entries need to be written.
        const keel::usize n = x.numel();
        const keel::usize m = n + ak;
        auto r = Value::matrix(m, m, t, mr);
        const char *src = static_cast<const char *>(x.rawData());
        char *dst = static_cast<char *>(r.rawDataMut());
        for (keel::usize i = 0; i < n; ++i) {
            const keel::usize row = (k >= 0) ? i : i + ak;
            const keel::usize col = (k >= 0) ? i + ak : i;
            std::memcpy(dst + (col * m + row) * es, src + i * es, es);  // col-major
        }
        return r;
    }

    // Matrix input: extract the k-th diagonal as a column vector.
    const keel::usize R = x.dims().rows(), C = x.dims().cols();
    const keel::usize row0 = (k >= 0) ? 0 : ak;
    const keel::usize col0 = (k >= 0) ? ak : 0;
    keel::usize len = 0;
    if (row0 < R && col0 < C)
        len = std::min(R - row0, C - col0);
    auto r = Value::matrix(len, 1, t, mr);
    const char *src = static_cast<const char *>(x.rawData());
    char *dst = static_cast<char *>(r.rawDataMut());
    for (keel::usize i = 0; i < len; ++i) {
        const keel::usize row = row0 + i, col = col0 + i;
        std::memcpy(dst + i * es, src + (col * R + row) * es, es);  // x col-major
    }
    return r;
}

// ── Sort / find ──────────────────────────────────────────────────────
// Complex sort (MATLAB rule): order by magnitude |z| ascending; ties broken
// by phase angle arg(z) in (-pi, pi] ascending. A NaN component (|z| = NaN)
// sorts LAST for ascending, FIRST for descending. 'descend' fully reverses
// both keys. Unlike min/max there is NO all-real fast path — a COMPLEX-typed
// all-real input still sorts by |z|+angle (so sort([2 -2]+0i) = [2 -2]).
std::tuple<Value, Value> sortComplex(const Value &x, int dim, bool descend,
                                     NanPlace nanPlace,
                                     keel::Allocator *mr)
{
    // NaN side: Auto = first for descending / last for ascending; First/Last
    // force the side regardless of direction (MATLAB 'MissingPlacement').
    const bool nanFirst = (nanPlace == NanPlace::Auto) ? descend
                                                       : (nanPlace == NanPlace::First);
    const keel::usize R = x.dims().rows(), C = x.dims().cols();
    const keel::usize P = x.dims().is3D() ? x.dims().pages() : 1;
    const int sortDim = (dim >= 1) ? std::min(dim - 1, 2)
                                   : ((R > 1) ? 0 : (C > 1) ? 1 : 2);
    const keel::usize N = (sortDim == 0) ? R : (sortDim == 1) ? C : P;

    auto r = x.dims().is3D() ? Value::matrix3d(R, C, P, ValueType::COMPLEX, mr)
                             : Value::matrix(R, C, ValueType::COMPLEX, mr);
    auto idx = x.dims().is3D() ? Value::matrix3d(R, C, P, ValueType::DOUBLE, mr)
                               : Value::matrix(R, C, ValueType::DOUBLE, mr);

    const keel::usize slice0 = (sortDim == 0) ? 1 : R;
    const keel::usize slice1 = (sortDim == 1) ? 1 : C;
    const keel::usize slice2 = (sortDim == 2) ? 1 : P;
    const keel::c64 *src = x.complexData();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<std::pair<keel::c64, keel::usize>> buf(N, scratch.allocator());

    for (keel::usize pp = 0; pp < slice2; ++pp)
        for (keel::usize c = 0; c < slice1; ++c)
            for (keel::usize rr = 0; rr < slice0; ++rr) {
                for (keel::usize k = 0; k < N; ++k) {
                    const keel::usize rIdx = (sortDim == 0) ? k : rr;
                    const keel::usize cIdx = (sortDim == 1) ? k : c;
                    const keel::usize pIdx = (sortDim == 2) ? k : pp;
                    buf[k] = {src[pIdx * R * C + cIdx * R + rIdx], k};
                }
                std::stable_sort(buf.begin(), buf.end(),
                          [descend, nanFirst](const auto &a, const auto &b) {
                              const double am = keel::abs(a.first), bm = keel::abs(b.first);
                              const bool an = std::isnan(am), bn = std::isnan(bm);
                              if (an || bn) {
                                  if (an && bn) return false;
                                  return nanFirst ? an : bn;
                              }
                              if (am != bm) return descend ? (am > bm) : (am < bm);
                              const double aa = keel::arg(a.first), ba = keel::arg(b.first);
                              if (aa != ba) return descend ? (aa > ba) : (aa < ba);
                              return false;
                          });
                for (keel::usize k = 0; k < N; ++k) {
                    const keel::usize rIdx = (sortDim == 0) ? k : rr;
                    const keel::usize cIdx = (sortDim == 1) ? k : c;
                    const keel::usize pIdx = (sortDim == 2) ? k : pp;
                    const keel::usize lin = pIdx * R * C + cIdx * R + rIdx;
                    r.complexDataMut()[lin] = buf[k].first;
                    idx.doubleDataMut()[lin] = static_cast<double>(buf[k].second + 1);
                }
            }
    return std::make_tuple(std::move(r), std::move(idx));
}

std::tuple<Value, Value> sort(const Value &x, int dim, bool descend,
                              NanPlace nanPlace, keel::Allocator *mr)
{
    if (x.isScalar())
        return std::make_tuple(x, Value::scalar(1.0, mr));
    if (x.type() == ValueType::COMPLEX)
        return sortComplex(x, dim, descend, nanPlace, mr);

    // NaN side: Auto = first for descending / last for ascending; First/Last
    // force the side regardless of direction (MATLAB 'MissingPlacement').
    const bool nanFirst = (nanPlace == NanPlace::Auto) ? descend
                                                       : (nanPlace == NanPlace::First);

    const keel::usize R = x.dims().rows(), C = x.dims().cols();
    const keel::usize P = x.dims().is3D() ? x.dims().pages() : 1;
    // dim<1 -> auto (first non-singleton, MATLAB convention). Explicit dim
    // is 1-based: 1=down columns, 2=along rows, 3=along pages.
    const int sortDim = (dim >= 1) ? std::min(dim - 1, 2)
                                   : ((R > 1) ? 0 : (C > 1) ? 1 : 2);
    const keel::usize N = (sortDim == 0) ? R : (sortDim == 1) ? C : P;

    auto r = x.dims().is3D() ? Value::matrix3d(R, C, P, ValueType::DOUBLE, mr)
                             : Value::matrix(R, C, ValueType::DOUBLE, mr);
    auto idx = x.dims().is3D() ? Value::matrix3d(R, C, P, ValueType::DOUBLE, mr)
                               : Value::matrix(R, C, ValueType::DOUBLE, mr);

    const keel::usize slice0 = (sortDim == 0) ? 1 : R;
    const keel::usize slice1 = (sortDim == 1) ? 1 : C;
    const keel::usize slice2 = (sortDim == 2) ? 1 : P;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<std::pair<double, keel::usize>> buf(N, scratch.allocator());

    for (keel::usize pp = 0; pp < slice2; ++pp)
        for (keel::usize c = 0; c < slice1; ++c)
            for (keel::usize rr = 0; rr < slice0; ++rr) {
                for (keel::usize k = 0; k < N; ++k) {
                    const keel::usize rIdx = (sortDim == 0) ? k : rr;
                    const keel::usize cIdx = (sortDim == 1) ? k : c;
                    const keel::usize pIdx = (sortDim == 2) ? k : pp;
                    buf[k] = {x.doubleData()[pIdx * R * C + cIdx * R + rIdx], k};
                }
                // MATLAB sort: stable; NaN sorts LAST for ascend, FIRST
                // for descend. std::stable_sort keeps the original index
                // order for ties (matches MATLAB's [s,i]).
                std::stable_sort(buf.begin(), buf.end(),
                          [descend, nanFirst](const auto &a, const auto &b) {
                              const double av = a.first, bv = b.first;
                              const bool an = std::isnan(av), bn = std::isnan(bv);
                              if (an || bn) {
                                  if (an && bn) return false;
                                  return nanFirst ? an : bn;
                              }
                              return descend ? (av > bv) : (av < bv);
                          });
                for (keel::usize k = 0; k < N; ++k) {
                    const keel::usize rIdx = (sortDim == 0) ? k : rr;
                    const keel::usize cIdx = (sortDim == 1) ? k : c;
                    const keel::usize pIdx = (sortDim == 2) ? k : pp;
                    const keel::usize lin = pIdx * R * C + cIdx * R + rIdx;
                    r.doubleDataMut()[lin] = buf[k].first;
                    idx.doubleDataMut()[lin] = static_cast<double>(buf[k].second + 1);
                }
            }
    return std::make_tuple(std::move(r), std::move(idx));
}

// ── sortrows ─────────────────────────────────────────────────────────

std::tuple<Value, Value> sortrows(const Value &x, keel::Allocator *mr)
{
    return sortRowsImpl(x, nullptr, 0, mr);
}

std::tuple<Value, Value> sortrows(const Value &x, keel::ArrayRef<int> cols, keel::Allocator *mr)
{
    return sortRowsImpl(x, cols.data(), cols.size(), mr);
}

Value find(const Value &x, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto indices = keel::Array<double>(scratch.allocator());
    if (x.isLogical()) {
        const keel::u8 *ld = x.logicalData();
        for (keel::usize i = 0; i < x.numel(); ++i)
            if (ld[i])
                indices.push_back(static_cast<double>(i + 1));
    } else {
        const double *dd = x.doubleData();
        for (keel::usize i = 0; i < x.numel(); ++i)
            if (dd[i] != 0.0)
                indices.push_back(static_cast<double>(i + 1));
    }
    const bool rowResult = !x.dims().is3D() && x.dims().rows() == 1;
    auto r = rowResult ? Value::matrix(1, indices.size(), ValueType::DOUBLE, mr)
                       : Value::matrix(indices.size(), 1, ValueType::DOUBLE, mr);
    if (!indices.empty())
        std::memcpy(r.doubleDataMut(), indices.data(), indices.size() * sizeof(double));
    return r;
}

// ── nnz / nonzeros ───────────────────────────────────────────────────

Value nnz(const Value &x, keel::Allocator *mr)
{
    if (x.numel() == 0)
        return Value::scalar(0.0, mr);
    keel::usize count = 0;
    forEachNonzero(x, [&](keel::usize) { ++count; });
    return Value::scalar(static_cast<double>(count), mr);
}

Value nonzeros(const Value &x, keel::Allocator *mr)
{
    if (x.numel() == 0) {
        // Empty input → 0×1 column of the source type (DOUBLE if unknown).
        const ValueType outT = (x.type() == ValueType::EMPTY) ? ValueType::DOUBLE : x.type();
        return Value::matrix(0, 1, outT, mr);
    }
    switch (x.type()) {
    case ValueType::LOGICAL: {
        const keel::u8 *p = x.logicalData();
        return collectTypedNonzeros<keel::u8>(x, ValueType::LOGICAL, [&](keel::usize i) -> keel::u8 { return p[i]; }, mr);
    }
    case ValueType::DOUBLE: {
        const double *p = x.doubleData();
        return collectTypedNonzeros<double>(x, ValueType::DOUBLE, [&](keel::usize i) -> double { return p[i]; }, mr);
    }
    case ValueType::SINGLE: {
        const float *p = x.singleData();
        return collectTypedNonzeros<float>(x, ValueType::SINGLE, [&](keel::usize i) -> float { return p[i]; }, mr);
    }
    case ValueType::COMPLEX: {
        const keel::c64 *p = x.complexData();
        return collectTypedNonzeros<keel::c64>(x, ValueType::COMPLEX, [&](keel::usize i) -> keel::c64 { return p[i]; }, mr);
    }
    case ValueType::INT8: {
        const keel::i8 *p = x.int8Data();
        return collectTypedNonzeros<keel::i8>(x, ValueType::INT8, [&](keel::usize i) -> keel::i8 { return p[i]; }, mr);
    }
    case ValueType::INT16: {
        const keel::i16 *p = x.int16Data();
        return collectTypedNonzeros<keel::i16>(x, ValueType::INT16, [&](keel::usize i) -> keel::i16 { return p[i]; }, mr);
    }
    case ValueType::INT32: {
        const keel::i32 *p = x.int32Data();
        return collectTypedNonzeros<keel::i32>(x, ValueType::INT32, [&](keel::usize i) -> keel::i32 { return p[i]; }, mr);
    }
    case ValueType::INT64: {
        const keel::i64 *p = x.int64Data();
        return collectTypedNonzeros<keel::i64>(x, ValueType::INT64, [&](keel::usize i) -> keel::i64 { return p[i]; }, mr);
    }
    case ValueType::UINT8: {
        const keel::u8 *p = x.uint8Data();
        return collectTypedNonzeros<keel::u8>(x, ValueType::UINT8, [&](keel::usize i) -> keel::u8 { return p[i]; }, mr);
    }
    case ValueType::UINT16: {
        const keel::u16 *p = x.uint16Data();
        return collectTypedNonzeros<keel::u16>(x, ValueType::UINT16, [&](keel::usize i) -> keel::u16 { return p[i]; }, mr);
    }
    case ValueType::UINT32: {
        const keel::u32 *p = x.uint32Data();
        return collectTypedNonzeros<keel::u32>(x, ValueType::UINT32, [&](keel::usize i) -> keel::u32 { return p[i]; }, mr);
    }
    case ValueType::UINT64: {
        const keel::u64 *p = x.uint64Data();
        return collectTypedNonzeros<keel::u64>(x, ValueType::UINT64, [&](keel::usize i) -> keel::u64 { return p[i]; }, mr);
    }
    default:
        throw Error("nonzeros: unsupported element type",
                     0, 0, "nonzeros", "", "numkit:nonzeros:badType");
    }
}

// ── Concatenation ────────────────────────────────────────────────────
Value horzcat(keel::ArrayRef<Value> values, keel::Allocator *mr)
{
    if (values.empty())
        return Value();
    return Value::horzcat(values.data(), values.size(), mr);
}

Value vertcat(keel::ArrayRef<Value> values, keel::Allocator *mr)
{
    if (values.empty())
        return Value();
    return Value::vertcat(values.data(), values.size(), mr);
}

// ── Grids ────────────────────────────────────────────────────────────
std::tuple<Value, Value> meshgrid(const Value &x, const Value &y, keel::Allocator *mr)
{
    const keel::usize nx = x.numel(), ny = y.numel();
    auto X = Value::matrix(ny, nx, ValueType::DOUBLE, mr);
    auto Y = Value::matrix(ny, nx, ValueType::DOUBLE, mr);
    for (keel::usize r = 0; r < ny; ++r)
        for (keel::usize c = 0; c < nx; ++c) {
            X.elem(r, c) = x.doubleData()[c];
            Y.elem(r, c) = y.doubleData()[r];
        }
    return std::make_tuple(std::move(X), std::move(Y));
}

// 3-arg meshgrid: returns three [ny, nx, nz] 3-D arrays. See BUGS.md #23.
std::tuple<Value, Value, Value>
meshgrid(const Value &x, const Value &y, const Value &z, keel::Allocator *mr)
{
    const keel::usize nx = x.numel(), ny = y.numel(), nz = z.numel();
    auto X = Value::matrix3d(ny, nx, nz, ValueType::DOUBLE, mr);
    auto Y = Value::matrix3d(ny, nx, nz, ValueType::DOUBLE, mr);
    auto Z = Value::matrix3d(ny, nx, nz, ValueType::DOUBLE, mr);
    double *xd = X.doubleDataMut();
    double *yd = Y.doubleDataMut();
    double *zd = Z.doubleDataMut();
    for (keel::usize p = 0; p < nz; ++p) {
        const double zp = z.elemAsDouble(p);
        for (keel::usize c = 0; c < nx; ++c) {
            const double xc = x.elemAsDouble(c);
            for (keel::usize r = 0; r < ny; ++r) {
                const keel::usize idx = r + c * ny + p * (nx * ny);
                xd[idx] = xc;
                yd[idx] = y.elemAsDouble(r);
                zd[idx] = zp;
            }
        }
    }
    return std::make_tuple(std::move(X), std::move(Y), std::move(Z));
}

// ── ndgrid ──────────────────────────────────────────────────────────
std::tuple<Value, Value>
ndgrid(const Value &x, const Value &y, keel::Allocator *mr)
{
    const keel::usize nx = x.numel(), ny = y.numel();
    // Output shape: [nx, ny] — first arg is row dim (axes-major).
    auto X = Value::matrix(nx, ny, ValueType::DOUBLE, mr);
    auto Y = Value::matrix(nx, ny, ValueType::DOUBLE, mr);
    for (keel::usize r = 0; r < nx; ++r)
        for (keel::usize c = 0; c < ny; ++c) {
            X.elem(r, c) = x.elemAsDouble(r);
            Y.elem(r, c) = y.elemAsDouble(c);
        }
    return std::make_tuple(std::move(X), std::move(Y));
}

std::tuple<Value, Value, Value>
ndgrid(const Value &x, const Value &y, const Value &z, keel::Allocator *mr)
{
    const keel::usize nx = x.numel(), ny = y.numel(), nz = z.numel();
    auto X = Value::matrix3d(nx, ny, nz, ValueType::DOUBLE, mr);
    auto Y = Value::matrix3d(nx, ny, nz, ValueType::DOUBLE, mr);
    auto Z = Value::matrix3d(nx, ny, nz, ValueType::DOUBLE, mr);
    for (keel::usize p = 0; p < nz; ++p)
        for (keel::usize c = 0; c < ny; ++c)
            for (keel::usize r = 0; r < nx; ++r) {
                X.elem(r, c, p) = x.elemAsDouble(r);
                Y.elem(r, c, p) = y.elemAsDouble(c);
                Z.elem(r, c, p) = z.elemAsDouble(p);
            }
    return std::make_tuple(std::move(X), std::move(Y), std::move(Z));
}

// ── kron ────────────────────────────────────────────────────────────
// NOTE: kron / cross / dot migrated to toolboxes/linalg/src/vector_ops.cpp.

// ── Reductions and products ──────────────────────────────────────────


Value cumsum(const Value &x, keel::Allocator *mr)
{
    if (isIntegerType(x.type()))
        return cumIntegerNative(x, 0, /*isProd=*/false, mr);
    if (x.type() == ValueType::COMPLEX)
        return cumComplexAlongDim(x, firstNonSingletonDim(x), /*isProd=*/false, mr);
    // MATLAB promotes logical → double for cumsum (char/string error out and
    // keep doing so via the doubleData() path below — only logical is valid).
    if (x.isLogical())
        return cumsum(toDoubleValue(x, mr), mr);
    if (x.isScalar()) {
        auto r = Value::matrix(x.dims().rows(), x.dims().cols(), ValueType::DOUBLE, mr);
        r.doubleDataMut()[0] = x.toScalar();
        return r;
    }
    if (x.dims().isVector()) {
        auto r = Value::matrix(x.dims().rows(), x.dims().cols(), ValueType::DOUBLE, mr);
        numkit::builtin::cumsumScan(x.doubleData(), r.doubleDataMut(), x.numel());
        return r;
    }
    const keel::usize R = x.dims().rows(), C = x.dims().cols();
    auto r = Value::matrix(R, C, ValueType::DOUBLE, mr);
    const double *src = x.doubleData();
    double *dst = r.doubleDataMut();
    // Per-column inclusive scan — column data is contiguous.
    for (keel::usize c = 0; c < C; ++c)
        numkit::builtin::cumsumScan(src + c * R, dst + c * R, R);
    return r;
}

// cumsum along an explicit dim. Output shape equals input shape (this is
// not a reduction). Vector / scalar input ignores dim and walks linearly.
Value cumsum(const Value &x, int dim, keel::Allocator *mr)
{
    if (isIntegerType(x.type()))
        return cumIntegerNative(x, dim, /*isProd=*/false, mr);
    if (x.type() == ValueType::COMPLEX)
        return cumComplexAlongDim(x, dim <= 0 ? firstNonSingletonDim(x) : dim,
                                  /*isProd=*/false, mr);
    if (x.isLogical())
        return cumsum(toDoubleValue(x, mr), dim, mr);
    if (dim <= 0) return cumsum(x, mr);
    if (x.dims().isVector() || x.isScalar()) return cumsum(x, mr);

    const int d = numkit::ops::resolveDim(x, dim, "cumsum");
    const auto &dd = x.dims();

    // ND fallback for rank ≥ 4: per-slice scan along axis d-1. Inner
    // block size B = prod(dim[0..d-2]); outer count O = prod(dim[d..]).
    if (dd.ndim() >= 4) {
        constexpr int kMaxNd = Dims::kMaxRank;
        if (dd.ndim() > kMaxNd)
            throw Error("cumsum: rank exceeds 32",
                         0, 0, "cumsum", "", "numkit:cumsum:tooManyDims");
        keel::usize outDims[kMaxNd];
        for (int i = 0; i < dd.ndim(); ++i) outDims[i] = dd.dim(i);
        auto r = Value::matrixND(outDims, dd.ndim(), ValueType::DOUBLE, mr);
        const keel::usize sliceLen = dd.dim(d - 1);
        keel::usize B = 1;
        for (int i = 0; i < d - 1; ++i) B *= dd.dim(i);
        keel::usize O = 1;
        for (int i = d; i < dd.ndim(); ++i) O *= dd.dim(i);
        const double *src = x.doubleData();
        double *dst = r.doubleDataMut();
        if (B == 1) {
            for (keel::usize o = 0; o < O; ++o) {
                const keel::usize base = o * sliceLen;
                numkit::builtin::cumsumScan(src + base, dst + base, sliceLen);
            }
        } else {
            for (keel::usize o = 0; o < O; ++o)
                for (keel::usize b = 0; b < B; ++b) {
                    const keel::usize base = o * sliceLen * B + b;
                    if (sliceLen == 0) continue;
                    double acc = src[base];
                    dst[base] = acc;
                    for (keel::usize k = 1; k < sliceLen; ++k) {
                        acc += src[base + k * B];
                        dst[base + k * B] = acc;
                    }
                }
        }
        return r;
    }

    const keel::usize R = dd.rows(), C = dd.cols();
    const keel::usize P = dd.is3D() ? dd.pages() : 1;
    auto r = dd.is3D() ? Value::matrix3d(R, C, P, ValueType::DOUBLE, mr)
                       : Value::matrix(R, C, ValueType::DOUBLE, mr);
    const double *src = x.doubleData();
    double *dst = r.doubleDataMut();

    if (d == 1) {
        // dim=1: scan down each column. Column data is contiguous so
        // route through the SIMD prefix-sum kernel.
        for (keel::usize pp = 0; pp < P; ++pp)
            for (keel::usize c = 0; c < C; ++c) {
                const keel::usize base = pp * R * C + c * R;
                numkit::builtin::cumsumScan(src + base, dst + base, R);
            }
    } else if (d == 2) {
        // Walk across columns for each (row, page). Stride = R.
        for (keel::usize pp = 0; pp < P; ++pp)
            for (keel::usize rr = 0; rr < R; ++rr) {
                double s = 0;
                const keel::usize pageBase = pp * R * C;
                for (keel::usize c = 0; c < C; ++c) {
                    s += src[pageBase + c * R + rr];
                    dst[pageBase + c * R + rr] = s;
                }
            }
    } else if (d == 3) {
        // Walk through pages for each (row, col). Stride = R*C.
        for (keel::usize c = 0; c < C; ++c)
            for (keel::usize rr = 0; rr < R; ++rr) {
                double s = 0;
                for (keel::usize pp = 0; pp < P; ++pp) {
                    s += src[pp * R * C + c * R + rr];
                    dst[pp * R * C + c * R + rr] = s;
                }
            }
    }
    return r;
}

// ── Generic cumulative kernel for cumprod / cummax / cummin ─────────
//
// Op is a binary functor (double, double) -> double. Init is the value
// the running accumulator starts at; for cumulative ops we instead seed
// with the first element of the slice, but seeding behavior is still
// captured by Op (e.g. cumprod could just use init=1.0 multiplicative).
// We use the seed-with-first style so cummax / cummin work without
// needing to know +/- infinity for the initial value.

// cumprod / cummax / cummin: SIMD prefix-op kernels in
// backends/MStdCumSum_{simd,portable}.cpp handle vector input and the
// dim=1 (column) path where access is contiguous. For dim=2/3 the
// strided access pattern doesn't benefit from SIMD; cumImpl's scalar
// cumKernel still handles those (with the same Op as before).

Value cumprod(const Value &x, int dim, keel::Allocator *mr)
{
    if (isIntegerType(x.type()))
        return cumIntegerNative(x, dim, /*isProd=*/true, mr);
    if (x.type() == ValueType::COMPLEX)
        return cumComplexAlongDim(x, dim <= 0 ? firstNonSingletonDim(x) : dim,
                                  /*isProd=*/true, mr);
    if (x.isLogical())
        return cumprod(toDoubleValue(x, mr), dim, mr);
    return cumScanDispatch(x, dim, numkit::builtin::cumprodScan, [](double a, double b) { return a * b; }, "cumprod", mr);
}


Value cummax(const Value &x, int dim, keel::Allocator *mr)
{
    if (x.isLogical())
        return logicalizeCumResult(cummax(toDoubleValue(x, mr), dim, mr), mr);
    // Integer: MATLAB cummax/cummin PRESERVE the class (order statistics, so
    // the result is a subset of the input — the double round-trip is exact).
    if (isIntegerType(x.type()))
        return doubleToIntegerExact(cummax(toDoubleValue(x, mr), dim, mr), x.type(), mr);
    if (x.type() == ValueType::COMPLEX)
        return cumComplexMinMaxAlongDim(x, dim <= 0 ? firstNonSingletonDim(x) : dim,
                                        /*isMax=*/true, mr);
    // NaN propagation: MATLAB cummax skips NaN if 'omitnan' is passed
    // and propagates otherwise. Default = 'omitnan' since R2018a; we
    // skip NaN here, treating them as identity.
    return cumScanDispatch(x, dim, numkit::builtin::cummaxScan, [](double a, double b) {
                               if (std::isnan(b)) return a;
                               if (std::isnan(a)) return b;
                               return std::max(a, b);
                           }, "cummax", mr);
}

Value cummin(const Value &x, int dim, keel::Allocator *mr)
{
    if (x.isLogical())
        return logicalizeCumResult(cummin(toDoubleValue(x, mr), dim, mr), mr);
    if (isIntegerType(x.type()))
        return doubleToIntegerExact(cummin(toDoubleValue(x, mr), dim, mr), x.type(), mr);
    if (x.type() == ValueType::COMPLEX)
        return cumComplexMinMaxAlongDim(x, dim <= 0 ? firstNonSingletonDim(x) : dim,
                                        /*isMax=*/false, mr);
    return cumScanDispatch(x, dim, numkit::builtin::cumminScan, [](double a, double b) {
                               if (std::isnan(b)) return a;
                               if (std::isnan(a)) return b;
                               return std::min(a, b);
                           }, "cummin", mr);
}

// ── diff: discrete difference ────────────────────────────────────────

Value diff(const Value &x, int n, int dim, keel::Allocator *mr)
{
    // MATLAB: the difference order N must be a positive integer scalar; n == 0
    // (identity) and negative orders are errors. The reg validates the
    // user-facing path; this guards the C++ primitive too (no internal caller
    // passes 0).
    if (n < 1)
        throw Error("diff: Difference order N must be a positive integer scalar",
                     0, 0, "diff", "", "numkit:diff:badOrder");

    const bool isInt = isIntegerType(x.type());

    // Scalar: MATLAB returns 1×0 empty (class preserved for integer input).
    if (x.isScalar())
        return Value::matrix(1, 0, isInt ? x.type() : ValueType::DOUBLE, mr);

    const int d = numkit::ops::resolveDim(x, dim, "diff");
    const auto &dd = x.dims();
    const keel::usize sliceLen = (d >= 1 && d <= dd.ndim()) ? dd.dim(d - 1) : 1;

    // If n collapses or exceeds the dim, return correctly-shaped empty
    // (preserving the integer class for integer input).
    if (sliceLen <= static_cast<keel::usize>(n)) {
        if (isInt) {
            const int nd = dd.ndim();
            keel::usize outDims[Dims::kMaxRank];
            for (int i = 0; i < nd; ++i) outDims[i] = dd.dim(i);
            outDims[d - 1] = 0;
            return Value::matrixND(outDims, nd, x.type(), mr);
        }
        return makeDiffOutput(dd, d, sliceLen, ValueType::DOUBLE, mr);
    }

    // Integer types: keep the class and saturate at each pass (MATLAB).
    if (isInt)
        return diffInteger(x, n, d, mr);

    // Complex: difference real and imaginary parts together (MATLAB).
    if (x.type() == ValueType::COMPLEX) {
        Value cur = copyComplexSameShape(x, mr);
        for (int pass = 0; pass < n; ++pass) {
            const auto &curDims = cur.dims();
            auto out = makeDiffOutput(curDims, d, 1, ValueType::COMPLEX, mr);
            diffOnceComplex(cur.complexData(), out.complexDataMut(), curDims, d);
            cur = std::move(out);
        }
        return cur;
    }

    // Promote logical to DOUBLE first.
    Value cur = copyToDouble(x, mr);

    for (int pass = 0; pass < n; ++pass) {
        const auto &curDims = cur.dims();
        auto out = makeDiffOutput(curDims, d, 1, ValueType::DOUBLE, mr);
        diffOnceDouble(cur.doubleData(), out.doubleDataMut(), curDims, d);
        cur = std::move(out);
    }
    return cur;
}

// ── any / all moved to backends/MStdLogicalReductions_{simd,portable}.cpp
//
// MATLAB's any(X) returns true if ANY element is non-zero (NaN counts
// as true since NaN != 0). all(X) returns true if ALL elements are
// non-zero. Empty: any → false, all → true (vacuously). The SIMD
// backend scans LOGICAL bytes and DOUBLE lanes directly with early
// exit (Phase P1 of project_perf_optimization_plan.md).


// ── xor (elementwise logical) ────────────────────────────────────────
Value xorOf(const Value &a, const Value &b, keel::Allocator *mr)
{
    auto ad = promoteToDouble(a, mr);
    auto bd = promoteToDouble(b, mr);
    auto d = elementwiseDouble(ad, bd,
        [](double aa, double bb) {
            return ((aa != 0.0) != (bb != 0.0)) ? 1.0 : 0.0;
        }, mr);
    if (d.isScalar()) return Value::logicalScalar(d.toScalar() != 0.0, mr);
    auto r = createLike(d, ValueType::LOGICAL, mr);
    for (keel::usize i = 0; i < d.numel(); ++i)
        r.logicalDataMut()[i] = (d.doubleData()[i] != 0.0) ? 1 : 0;
    return r;
}

} // namespace numkit::builtin
