// toolboxes/linalg/src/schur_convert.cpp
//
// cdf2rdf â€” complex-diagonal Schur â†’ real-block-diagonal Schur
// rsf2csf â€” real-block-diagonal Schur â†’ complex-diagonal Schur
//
// Both functions convert between the two equivalent representations
// of the (quasi-)triangular Schur factor for real input matrices.
// MATLAB ships them as utilities for users who need to switch
// between the two forms (e.g. interpreting Schur output from one
// algorithm and feeding it to another that expects the other form).
//
// Practical note: nk's eig_general_VD currently throws on
// matrices with complex eigenvalues (Francis QR deferred), so the
// complex-input branches here are exercised only when the user
// constructs (V, D) manually or imports them from another source.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/linalg/schur_convert.hpp>

// Compute-only TU: Value substrate + Error, no engine. The cdf2rdf/rsf2csf
// builtins (CallContext wrappers) live in schur_convert_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>

namespace numkit::linalg {

namespace {

constexpr double kPairTol = 1e-10;

inline bool isApproxConj(const keel::c64 &a, const keel::c64 &b, double tol)
{
    return keel::abs(a.real() - b.real()) <= tol * (1.0 + keel::abs(a.real()))
        && keel::abs(a.imag() + b.imag()) <= tol * (1.0 + keel::abs(a.imag()));
}

} // anonymous namespace

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// cdf2rdf
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

std::tuple<Value, Value>
cdf2rdf(const Value &V, const Value &D, keel::Allocator *mr)
{
    if (V.dims().ndim() != 2 || D.dims().ndim() != 2)
        throw Error("cdf2rdf: V and D must be 2-D matrices",
                    0, 0, "cdf2rdf", "", "numkit:cdf2rdf:notMatrix");
    const keel::usize n = static_cast<keel::usize>(D.dims().dim(0));
    if (n != static_cast<keel::usize>(D.dims().dim(1)))
        throw Error("cdf2rdf: D must be square",
                    0, 0, "cdf2rdf", "", "numkit:cdf2rdf:notSquare");
    if (static_cast<keel::usize>(V.dims().dim(0)) != n
        || static_cast<keel::usize>(V.dims().dim(1)) != n)
        throw Error("cdf2rdf: V and D shape mismatch",
                    0, 0, "cdf2rdf", "", "numkit:cdf2rdf:shapeMismatch");
    if (n == 0)
        return std::make_tuple(
            Value::matrix(0, 0, ValueType::DOUBLE, mr),
            Value::matrix(0, 0, ValueType::DOUBLE, mr));

    // Read eigenvalues off D's diagonal. Walk i = 0..n-1; for each
    // real eigenvalue copy column; for each complex eigenvalue expect
    // its conjugate immediately at i+1 and emit a 2Ã—2 real block.
    auto VRout = Value::matrix(n, n, ValueType::DOUBLE, mr);
    auto DRout = Value::matrix(n, n, ValueType::DOUBLE, mr);
    double *vr = VRout.doubleDataMut();
    double *dr = DRout.doubleDataMut();
    std::fill(vr, vr + n * n, 0.0);
    std::fill(dr, dr + n * n, 0.0);

    auto getD = [&](keel::usize i, keel::usize j) -> keel::c64 {
        if (D.isComplex())
            return D.complexElem(i, j);
        return keel::c64(D.elemAsDouble(i + j * n), 0.0);
    };
    auto getV = [&](keel::usize i, keel::usize j) -> keel::c64 {
        if (V.isComplex())
            return V.complexElem(i, j);
        return keel::c64(V.elemAsDouble(i + j * n), 0.0);
    };

    keel::usize i = 0;
    while (i < n) {
        keel::c64 lam = getD(i, i);
        // Treat anything within tolerance of real as real.
        if (std::fabs(lam.imag()) <= kPairTol * (1.0 + std::fabs(lam.real()))) {
            // Real eigenvalue.
            dr[i + i * n] = lam.real();
            for (keel::usize r = 0; r < n; ++r)
                vr[r + i * n] = getV(r, i).real();
            ++i;
            continue;
        }
        // Expect a conjugate pair at (i, i+1).
        if (i + 1 >= n)
            throw Error("cdf2rdf: dangling complex eigenvalue at end of D",
                        0, 0, "cdf2rdf", "", "numkit:cdf2rdf:badPair");
        keel::c64 lam2 = getD(i + 1, i + 1);
        if (!isApproxConj(lam, lam2, kPairTol))
            throw Error("cdf2rdf: complex eigenvalues must come in conjugate pairs",
                        0, 0, "cdf2rdf", "", "numkit:cdf2rdf:badPair");
        const double a = lam.real();
        const double b = lam.imag();   // assume b > 0; swap if not
        const double bb = (b >= 0.0) ? b : -b;
        // 2Ã—2 real block. MATLAB convention is [[a b]; [-b a]] â€” i.e.
        // the positive-imag eigenvalue sits in the upper-right.
        dr[i + i * n]         =  a;
        dr[i + (i + 1) * n]   =  bb;
        dr[(i + 1) + i * n]   = -bb;
        dr[(i + 1) + (i + 1) * n] = a;
        // Real eigvecs paired with the above block convention:
        // VR(:, k+1) = +Im(v) for positive b. Flips for negative b.
        for (keel::usize r = 0; r < n; ++r) {
            keel::c64 v = getV(r, i);
            vr[r + i * n]       = v.real();
            vr[r + (i + 1) * n] = (b >= 0.0) ? v.imag() : -v.imag();
        }
        i += 2;
    }
    return std::make_tuple(std::move(VRout), std::move(DRout));
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// rsf2csf
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

std::tuple<Value, Value>
rsf2csf(const Value &UR, const Value &TR, keel::Allocator *mr)
{
    if (UR.dims().ndim() != 2 || TR.dims().ndim() != 2)
        throw Error("rsf2csf: UR and TR must be 2-D matrices",
                    0, 0, "rsf2csf", "", "numkit:rsf2csf:notMatrix");
    const keel::usize n = static_cast<keel::usize>(TR.dims().dim(0));
    if (n != static_cast<keel::usize>(TR.dims().dim(1)))
        throw Error("rsf2csf: TR must be square",
                    0, 0, "rsf2csf", "", "numkit:rsf2csf:notSquare");
    if (static_cast<keel::usize>(UR.dims().dim(0)) != n
        || static_cast<keel::usize>(UR.dims().dim(1)) != n)
        throw Error("rsf2csf: UR and TR shape mismatch",
                    0, 0, "rsf2csf", "", "numkit:rsf2csf:shapeMismatch");
    if (n == 0)
        return std::make_tuple(
            Value::matrix(0, 0, ValueType::COMPLEX, mr),
            Value::matrix(0, 0, ValueType::COMPLEX, mr));

    // Working copies promoted to complex.
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> Td(n * n, scratch.allocator());
    keel::Array<keel::c64> Ud(n * n, scratch.allocator());
    for (keel::usize j = 0; j < n; ++j)
        for (keel::usize i = 0; i < n; ++i) {
            Td[i + j * n] = keel::c64(TR.elemAsDouble(i + j * n), 0.0);
            Ud[i + j * n] = keel::c64(UR.elemAsDouble(i + j * n), 0.0);
        }

    // Walk diagonal. For each non-zero sub-diagonal entry TR(i+1, i),
    // the 2Ã—2 block at (i, i+1) has complex eigenvalues. Apply a 2Ã—2
    // unitary similarity (Givens-like with complex c, s) that zeros
    // the sub-diagonal entry and brings the diagonal entries to the
    // eigenvalues of the block.
    keel::usize i = 0;
    while (i + 1 < n) {
        const keel::c64 sub = Td[(i + 1) + i * n];
        if (keel::abs(sub) <= kPairTol) {
            ++i;
            continue;
        }
        // 2Ã—2 block: [[a c]; [d a']]. Compute eigenvalues
        //   Î» = ((a + a') Â± sqrt((a - a')^2 + 4Â·cÂ·d)) / 2
        const keel::c64 a  = Td[i + i * n];
        const keel::c64 ap = Td[(i + 1) + (i + 1) * n];
        const keel::c64 c  = Td[i + (i + 1) * n];
        const keel::c64 d  = sub;
        const keel::c64 disc = keel::sqrt((a - ap) * (a - ap) + keel::c64(4.0, 0.0) * c * d);
        const keel::c64 lam1 = (a + ap + disc) * keel::c64(0.5, 0.0);

        // Build the 2-component eigenvector of the 2Ã—2 block for lam1:
        // (A - lam1Â·I) Â· x = 0. Take x = [c; lam1 - a]; normalise.
        keel::c64 x0 = c;
        keel::c64 x1 = lam1 - a;
        // If c is ~0, fall back to alt: x = [lam1 - a'; d].
        if (keel::abs(x0) < keel::abs(x1) * 1e-300) {
            x0 = lam1 - ap;
            x1 = d;
        }
        const double nrm = std::sqrt(keel::norm(x0) + keel::norm(x1));
        if (nrm == 0.0) {
            ++i;
            continue;
        }
        x0 /= nrm; x1 /= nrm;
        // Build orthonormal complement y = [-conj(x1); conj(x0)]:
        // then U_block = [x y] is unitary; U_block' * block * U_block
        // is upper triangular with eigenvalues lam1, lam2 on diagonal.
        const keel::c64 y0 = -keel::conj(x1);
        const keel::c64 y1 =  keel::conj(x0);

        // Apply U_block from the RIGHT to columns i, i+1 of T:
        //   T_new(:, i)   = T(:, i)*x0 + T(:, i+1)*x1
        //   T_new(:, i+1) = T(:, i)*y0 + T(:, i+1)*y1
        for (keel::usize r = 0; r < n; ++r) {
            const keel::c64 t0 = Td[r + i * n];
            const keel::c64 t1 = Td[r + (i + 1) * n];
            Td[r + i * n]       = t0 * x0 + t1 * x1;
            Td[r + (i + 1) * n] = t0 * y0 + t1 * y1;
        }
        // Apply U_block' from the LEFT to rows i, i+1 of T:
        //   T_new(i,   :) = conj(x0)*T(i, :) + conj(x1)*T(i+1, :)
        //   T_new(i+1, :) = conj(y0)*T(i, :) + conj(y1)*T(i+1, :)
        for (keel::usize k = 0; k < n; ++k) {
            const keel::c64 t0 = Td[i + k * n];
            const keel::c64 t1 = Td[(i + 1) + k * n];
            Td[i + k * n]       = keel::conj(x0) * t0 + keel::conj(x1) * t1;
            Td[(i + 1) + k * n] = keel::conj(y0) * t0 + keel::conj(y1) * t1;
        }
        // Apply U_block to columns i, i+1 of U.
        for (keel::usize r = 0; r < n; ++r) {
            const keel::c64 u0 = Ud[r + i * n];
            const keel::c64 u1 = Ud[r + (i + 1) * n];
            Ud[r + i * n]       = u0 * x0 + u1 * x1;
            Ud[r + (i + 1) * n] = u0 * y0 + u1 * y1;
        }
        // The sub-diagonal entry should now be ~0 â€” clear it for cleanliness.
        Td[(i + 1) + i * n] = keel::c64(0.0, 0.0);

        i += 2;  // skip the freshly-diagonalised pair
    }

    auto Uout = Value::matrix(n, n, ValueType::COMPLEX, mr);
    auto Tout = Value::matrix(n, n, ValueType::COMPLEX, mr);
    keel::c64 *Up = Uout.complexDataMut();
    keel::c64 *Tp = Tout.complexDataMut();
    std::copy(Ud.begin(), Ud.end(), Up);
    std::copy(Td.begin(), Td.end(), Tp);
    return std::make_tuple(std::move(Uout), std::move(Tout));
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Engine adapters
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

} // namespace numkit::linalg
