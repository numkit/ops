// toolboxes/linalg/src/qz.cpp
//
// qz â€” Generalized Schur decomposition [AA, BB, Q, Z] = qz(A, B).

#include <numkit/linalg/qz.hpp>
#include "linalg_detail.hpp"

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace numkit::linalg {

namespace {

using Complex = std::complex<double>;

// Compute Givens rotation G = [c, s; -conj(s), c] such that G * [x; y] = [r; 0]
void makeGivensLeft(Complex x, Complex y, double &c, Complex &s)
{
    double norm_y = std::abs(y);
    if (norm_y == 0.0) {
        c = 1.0;
        s = Complex(0.0, 0.0);
        return;
    }
    double norm_x = std::abs(x);
    double r = std::hypot(norm_x, norm_y);
    c = norm_x / r;
    Complex phase = (norm_x > 0.0) ? (x / norm_x) : Complex(1.0, 0.0);
    s = phase * std::conj(y) / r;
}

// Compute Givens rotation G = [c, s; -conj(s), c] such that [x, y] * G = [0, r]
void makeGivensRight(Complex x, Complex y, double &c, Complex &s)
{
    double norm_x = std::abs(x);
    if (norm_x == 0.0) {
        c = 1.0;
        s = Complex(0.0, 0.0);
        return;
    }
    double norm_y = std::abs(y);
    double r = std::hypot(norm_x, norm_y);
    c = norm_y / r;
    Complex phase = (norm_y > 0.0) ? (y / norm_y) : Complex(1.0, 0.0);
    s = -phase * std::conj(x) / r;
}

void applyGivensLeft(Complex *M, std::size_t n, std::size_t row1, std::size_t row2, double c, Complex s, std::size_t col_start = 0)
{
    Complex G00 = c;
    Complex G01 = s;
    Complex G10 = -std::conj(s);
    Complex G11 = c;

    for (std::size_t k = col_start; k < n; ++k) {
        Complex r0 = M[row1 + k * n];
        Complex r1 = M[row2 + k * n];
        M[row1 + k * n] = G00 * r0 + G01 * r1;
        M[row2 + k * n] = G10 * r0 + G11 * r1;
    }
}

void applyGivensRight(Complex *M, std::size_t n, std::size_t col1, std::size_t col2, double c, Complex s, std::size_t row_end = 0)
{
    if (row_end == 0) row_end = n;
    Complex GH00 = c;
    Complex GH01 = -s;
    Complex GH10 = std::conj(s);
    Complex GH11 = c;

    for (std::size_t i = 0; i < row_end; ++i) {
        Complex c0 = M[i + col1 * n];
        Complex c1 = M[i + col2 * n];
        M[i + col1 * n] = c0 * GH00 + c1 * GH10;
        M[i + col2 * n] = c0 * GH01 + c1 * GH11;
    }
}

} // anonymous namespace

std::tuple<Value, Value, Value, Value> qz(const Value &A, const Value &B,
                                           std::pmr::memory_resource *mr)
{
    detail::requireDoubleInput(A, "qz", "A", 1);
    detail::requireDoubleInput(B, "qz", "B", 2);
    if (A.dims().ndim() != 2 || B.dims().ndim() != 2)
        throw Error("qz: A and B must be 2D matrices", 0, 0, "qz", "", "numkit:qz:notMatrix");
    const std::size_t n = static_cast<std::size_t>(A.dims().dim(0));
    if (n != static_cast<std::size_t>(A.dims().dim(1)) ||
        static_cast<std::size_t>(B.dims().dim(0)) != n ||
        static_cast<std::size_t>(B.dims().dim(1)) != n)
        throw Error("qz: A and B must be square matrices of the same size", 0, 0, "qz", "", "numkit:qz:badDims");

    const bool wantSingle = A.type() == ValueType::SINGLE || B.type() == ValueType::SINGLE;
    const ValueType rtype = wantSingle ? ValueType::SINGLE : ValueType::DOUBLE;
    if (n == 0) {
        return {Value::matrix(0, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr),
                Value::matrix(0, 0, rtype, mr)};
    }

    ScratchArena scratch(mr);
    ScratchVec<Complex> Ac(n * n, &scratch);
    ScratchVec<Complex> Bc(n * n, &scratch);
    ScratchVec<Complex> Qc(n * n, Complex(0.0, 0.0), &scratch);
    ScratchVec<Complex> Zc(n * n, Complex(0.0, 0.0), &scratch);

    for (std::size_t i = 0; i < n; ++i) {
        Qc[i + i * n] = Complex(1.0, 0.0);
        Zc[i + i * n] = Complex(1.0, 0.0);
    }

    if (A.isComplex()) {
        std::copy(A.complexData(), A.complexData() + n * n, Ac.begin());
    } else {
        for (std::size_t i = 0; i < n * n; ++i) Ac[i] = Complex(A.elemAsDouble(i), 0.0);
    }

    if (B.isComplex()) {
        std::copy(B.complexData(), B.complexData() + n * n, Bc.begin());
    } else {
        for (std::size_t i = 0; i < n * n; ++i) Bc[i] = Complex(B.elemAsDouble(i), 0.0);
    }

    // Step 1: Reduce B to upper-triangular using QR and apply to A, Q
    for (std::size_t k = 0; k < n; ++k) {
        for (std::intptr_t i = static_cast<std::intptr_t>(n) - 1; i > static_cast<std::intptr_t>(k); --i) {
            std::size_t ui = static_cast<std::size_t>(i);
            if (Bc[ui + k * n] != Complex(0.0, 0.0)) {
                double c; Complex s;
                makeGivensLeft(Bc[ui - 1 + k * n], Bc[ui + k * n], c, s);
                applyGivensLeft(Bc.data(), n, ui - 1, ui, c, s, k);
                applyGivensLeft(Ac.data(), n, ui - 1, ui, c, s, 0);
                applyGivensLeft(Qc.data(), n, ui - 1, ui, c, s, 0);
                Bc[ui + k * n] = Complex(0.0, 0.0);
            }
        }
    }

    // Step 2: Reduce A to upper Hessenberg while keeping B upper triangular
    for (std::size_t j = 0; j < n; ++j) {
        for (std::intptr_t i = static_cast<std::intptr_t>(n) - 1; i >= static_cast<std::intptr_t>(j) + 2; --i) {
            std::size_t ui = static_cast<std::size_t>(i);
            if (Ac[ui + j * n] != Complex(0.0, 0.0)) {
                // Zero Ac(ui, j) via left Givens on rows ui-1, ui
                double c; Complex s;
                makeGivensLeft(Ac[ui - 1 + j * n], Ac[ui + j * n], c, s);
                applyGivensLeft(Ac.data(), n, ui - 1, ui, c, s, j);
                applyGivensLeft(Bc.data(), n, ui - 1, ui, c, s, ui - 1);
                applyGivensLeft(Qc.data(), n, ui - 1, ui, c, s, 0);
                Ac[ui + j * n] = Complex(0.0, 0.0);

                // Zero fill-in entry Bc(ui, ui-1) via right Givens on cols ui-1, ui
                if (Bc[ui + (ui - 1) * n] != Complex(0.0, 0.0)) {
                    double cr; Complex sr;
                    makeGivensRight(Bc[ui + (ui - 1) * n], Bc[ui + ui * n], cr, sr);
                    applyGivensRight(Bc.data(), n, ui - 1, ui, cr, sr, ui + 1);
                    applyGivensRight(Ac.data(), n, ui - 1, ui, cr, sr, n);
                    applyGivensRight(Zc.data(), n, ui - 1, ui, cr, sr, n);
                    Bc[ui + (ui - 1) * n] = Complex(0.0, 0.0);
                }
            }
        }
    }

    // Step 3: Moler-Stewart single-shift complex QZ bulge-chasing with zero-B deflation
    if (n > 1) {
        double normA = 0.0, normB = 0.0;
        for (std::size_t i = 0; i < n * n; ++i) {
            normA = std::hypot(normA, std::abs(Ac[i]));
            normB = std::hypot(normB, std::abs(Bc[i]));
        }
        const double eps = std::numeric_limits<double>::epsilon();
        const double aZeroTol = 4.0 * eps * std::max(1.0, normA);
        const double bZeroTol = 4.0 * eps * std::max(1.0, normB);

        std::size_t ihi = n - 1;
        std::size_t iter = 0;
        std::size_t localIter = 0;
        const std::size_t maxIter = 200 * n;

        while (ihi > 0 && iter < maxIter) {
            // Deflate converged subdiagonal entries at bottom of active window
            while (ihi > 0) {
                double diag = std::abs(Ac[(ihi - 1) + (ihi - 1) * n]) + std::abs(Ac[ihi + ihi * n]);
                double thresh = std::max(aZeroTol, 4.0 * eps * diag);
                if (std::abs(Ac[ihi + (ihi - 1) * n]) <= thresh) {
                    Ac[ihi + (ihi - 1) * n] = Complex(0.0, 0.0);
                    --ihi;
                    localIter = 0;
                } else {
                    break;
                }
            }
            if (ihi == 0) break;

            // Find start ilo of active unreduced Hessenberg block [ilo .. ihi]
            std::size_t ilo = ihi;
            while (ilo > 0) {
                double diag = std::abs(Ac[(ilo - 1) + (ilo - 1) * n]) + std::abs(Ac[ilo + ilo * n]);
                double thresh = std::max(aZeroTol, 4.0 * eps * diag);
                if (std::abs(Ac[ilo + (ilo - 1) * n]) <= thresh) {
                    Ac[ilo + (ilo - 1) * n] = Complex(0.0, 0.0);
                    break;
                }
                --ilo;
            }

            // Deflate any zero diagonal entry in B within [ilo .. ihi] (Golub & Van Loan 7.7.2)
            bool deflatedZeroB = false;
            for (std::size_t p = ihi + 1; p > ilo; --p) {
                const std::size_t idx = p - 1;
                if (std::abs(Bc[idx + idx * n]) <= bZeroTol) {
                    Bc[idx + idx * n] = Complex(0.0, 0.0);
                    for (std::size_t j = idx; j < ihi; ++j) {
                        double c; Complex s;
                        makeGivensLeft(Bc[j + (j + 1) * n], Bc[(j + 1) + (j + 1) * n], c, s);
                        applyGivensLeft(Bc.data(), n, j, j + 1, c, s, j + 1);
                        applyGivensLeft(Ac.data(), n, j, j + 1, c, s, (j > ilo ? j - 1 : ilo));
                        applyGivensLeft(Qc.data(), n, j, j + 1, c, s, 0);
                        Bc[(j + 1) + (j + 1) * n] = Complex(0.0, 0.0);
                        if (j > ilo) {
                            double cr; Complex sr;
                            makeGivensRight(Ac[(j + 1) + (j - 1) * n], Ac[(j + 1) + j * n], cr, sr);
                            applyGivensRight(Ac.data(), n, j - 1, j, cr, sr, j + 2);
                            applyGivensRight(Bc.data(), n, j - 1, j, cr, sr, j);
                            applyGivensRight(Zc.data(), n, j - 1, j, cr, sr, n);
                            Ac[(j + 1) + (j - 1) * n] = Complex(0.0, 0.0);
                        }
                    }
                    double cr; Complex sr;
                    makeGivensRight(Ac[ihi + (ihi - 1) * n], Ac[ihi + ihi * n], cr, sr);
                    applyGivensRight(Ac.data(), n, ihi - 1, ihi, cr, sr, ihi + 1);
                    applyGivensRight(Bc.data(), n, ihi - 1, ihi, cr, sr, ihi);
                    applyGivensRight(Zc.data(), n, ihi - 1, ihi, cr, sr, n);
                    Ac[ihi + (ihi - 1) * n] = Complex(0.0, 0.0);
                    Bc[ihi + ihi * n] = Complex(0.0, 0.0);
                    --ihi;
                    localIter = 0;
                    deflatedZeroB = true;
                    break;
                }
            }
            if (deflatedZeroB) continue;

            // Wilkinson shift from trailing 2x2 pencil of [ilo .. ihi]
            ++iter;
            ++localIter;
            const std::size_t m1 = ihi - 1;
            const Complex a11 = Ac[m1 + m1 * n] / Bc[m1 + m1 * n];
            const Complex a21 = Ac[ihi + m1 * n] / Bc[m1 + m1 * n];
            const Complex b12_ov_b22 = Bc[m1 + ihi * n] / Bc[ihi + ihi * n];
            const Complex a12 = (Ac[m1 + ihi * n] - a11 * Bc[m1 + ihi * n]) / Bc[ihi + ihi * n];
            const Complex a22 = (Ac[ihi + ihi * n] - a21 * Bc[m1 + ihi * n]) / Bc[ihi + ihi * n];
            (void)b12_ov_b22;

            Complex shift;
            if (localIter % 15 == 0) {
                const double mag = std::abs(a22) + std::abs(a21) + 1.0;
                const double ang = 0.7 * static_cast<double>(localIter / 15);
                shift = a22 + std::polar(0.75 * mag, ang);
            } else {
                const Complex halfDiff = 0.5 * (a11 - a22);
                const Complex disc = std::sqrt(halfDiff * halfDiff + a12 * a21);
                const Complex r1 = a22 + halfDiff + disc;
                const Complex r2 = a22 + halfDiff - disc;
                shift = (std::abs(r1 - a22) < std::abs(r2 - a22)) ? r1 : r2;
            }

            // Introduce bulge at ilo
            {
                const Complex x = Ac[ilo + ilo * n] - shift * Bc[ilo + ilo * n];
                const Complex y = Ac[(ilo + 1) + ilo * n];
                double c; Complex s;
                makeGivensLeft(x, y, c, s);
                applyGivensLeft(Ac.data(), n, ilo, ilo + 1, c, s, ilo);
                applyGivensLeft(Bc.data(), n, ilo, ilo + 1, c, s, ilo);
                applyGivensLeft(Qc.data(), n, ilo, ilo + 1, c, s, 0);

                double cr; Complex sr;
                makeGivensRight(Bc[(ilo + 1) + ilo * n], Bc[(ilo + 1) + (ilo + 1) * n], cr, sr);
                applyGivensRight(Bc.data(), n, ilo, ilo + 1, cr, sr, ilo + 2);
                applyGivensRight(Ac.data(), n, ilo, ilo + 1, cr, sr, std::min(n, ilo + 3));
                applyGivensRight(Zc.data(), n, ilo, ilo + 1, cr, sr, n);
                Bc[(ilo + 1) + ilo * n] = Complex(0.0, 0.0);
            }

            // Chase bulge down to ihi
            for (std::size_t k = ilo + 1; k < ihi; ++k) {
                double c; Complex s;
                makeGivensLeft(Ac[k + (k - 1) * n], Ac[(k + 1) + (k - 1) * n], c, s);
                applyGivensLeft(Ac.data(), n, k, k + 1, c, s, k - 1);
                applyGivensLeft(Bc.data(), n, k, k + 1, c, s, k);
                applyGivensLeft(Qc.data(), n, k, k + 1, c, s, 0);
                Ac[(k + 1) + (k - 1) * n] = Complex(0.0, 0.0);

                double cr; Complex sr;
                makeGivensRight(Bc[(k + 1) + k * n], Bc[(k + 1) + (k + 1) * n], cr, sr);
                applyGivensRight(Bc.data(), n, k, k + 1, cr, sr, k + 2);
                applyGivensRight(Ac.data(), n, k, k + 1, cr, sr, std::min(n, k + 3));
                applyGivensRight(Zc.data(), n, k, k + 1, cr, sr, n);
                Bc[(k + 1) + k * n] = Complex(0.0, 0.0);
            }
        }

        if (ihi > 0) {
            throw Error("qz: QZ iteration failed to converge", 0, 0, "qz", "", "numkit:qz:noConverge");
        }
    }

    // Strict lower triangles are converged to numerical zero
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            Ac[i + j * n] = Complex(0.0, 0.0);
            Bc[i + j * n] = Complex(0.0, 0.0);
        }
    }

    auto AA_out = Value::complexMatrix(n, n, mr);
    auto BB_out = Value::complexMatrix(n, n, mr);
    auto Q_out  = Value::complexMatrix(n, n, mr);
    auto Z_out  = Value::complexMatrix(n, n, mr);

    std::copy(Ac.begin(), Ac.end(), AA_out.complexDataMut());
    std::copy(Bc.begin(), Bc.end(), BB_out.complexDataMut());
    std::copy(Qc.begin(), Qc.end(), Q_out.complexDataMut());
    std::copy(Zc.begin(), Zc.end(), Z_out.complexDataMut());

    return {detail::narrowDoubleToSingleIf(detail::narrow_if_real(AA_out, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(BB_out, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Q_out, mr), wantSingle, mr),
            detail::narrowDoubleToSingleIf(detail::narrow_if_real(Z_out, mr), wantSingle, mr)};
}

} // namespace numkit::linalg
