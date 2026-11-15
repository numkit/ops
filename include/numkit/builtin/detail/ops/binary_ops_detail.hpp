// src/builtin/src/ops/binary_ops_detail.hpp
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/value/value.hpp>
#include <numkit/ops/helpers.hpp>
#include <numkit/ops/reductions.hpp>
#include <numkit/ops/compare.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/shape_ops.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::builtin {

using namespace ::numkit::ops;

namespace {

inline bool sameShapeDoubleFastPath(const numkit::Value &a,
                                    const numkit::Value &b)
{
    return !a.isScalar() && !b.isScalar() && a.dims() == b.dims();
}

inline numkit::Value coerceLogicalToDouble(const numkit::Value &v,
                                           keel::Allocator *mr)
{
    using namespace numkit;
    if (v.type() != ValueType::LOGICAL) return v;
    Value out = numkit::createLike(v, ValueType::DOUBLE, mr);
    const keel::u8 *src = v.logicalData();
    double *dst = out.doubleDataMut();
    const keel::usize n = v.numel();
    for (keel::usize i = 0; i < n; ++i)
        dst[i] = static_cast<double>(src[i]);
    return out;
}

inline numkit::Value coerceCharToDouble(const numkit::Value &v,
                                        keel::Allocator *mr)
{
    using namespace numkit;
    if (v.type() != ValueType::CHAR) return v;
    Value out = numkit::createLike(v, ValueType::DOUBLE, mr);
    const char *src = v.charData();
    double *dst = out.doubleDataMut();
    const keel::usize n = v.numel();
    for (keel::usize i = 0; i < n; ++i)
        dst[i] = static_cast<double>(static_cast<unsigned char>(src[i]));
    return out;
}

void copyColumnMajorDouble(const Value &v, double *dst)
{
    const keel::usize n = v.numel();
    if (v.type() == ValueType::DOUBLE) {
        const double *src = v.doubleData();
        std::copy(src, src + n, dst);
    } else {
        for (keel::usize i = 0; i < n; ++i) dst[i] = v.elemAsDouble(i);
    }
}

inline Value narrowIfReal(const Value &val, keel::Allocator *mr) {
    if (!val.isComplex()) return val;
    const keel::c64 *cd = val.complexData();
    const keel::usize num = val.numel();
    for (keel::usize i = 0; i < num; ++i) {
        if (keel::abs(cd[i].imag()) > 1e-14 * (1.0 + keel::abs(cd[i].real())))
            return val;
    }
    const keel::usize rows = val.dims().rows();
    const keel::usize cols = val.dims().cols();
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (keel::usize i = 0; i < num; ++i) od[i] = cd[i].real();
    return out;
}

// mldivide/mrdivide operand class contract (probed R2025b 2026-09-18):
// char, single and logical operands are solved by coercion (logical is
// coerced by the callers before dispatch); integer-class operands are
// rejected as a separate "integer linear algebra" class in the matrix
// path; cell, string, struct and beyond are rejected outright.
inline void checkSolveOperandClass(const Value &v, const char *opname)
{
    const ValueType t = v.type();
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX
        || t == ValueType::SINGLE || t == ValueType::LOGICAL
        || t == ValueType::CHAR || isIntegerType(t))
        return;
    throw Error(std::string(opname) + ": inputs must be numeric, logical"
                " or char (got a " + mtypeName(t) + " operand).",
                0, 0, opname, "",
                std::string("numkit:") + opname + ":inputType");
}

// Reference engine behavior for an exactly singular square solve A \ B:
// partial-pivoting LU continues past zero pivots (leaving U(j,j) == 0)
// and backward substitution divides by U(j,j) under IEEE-754 arithmetic,
// yielding an Inf/NaN-filled solution instead of throwing.
template <typename T>
void singularSquareSolve(T *A_lu, keel::usize n, T *B_work, keel::usize nrhs, T *X)
{
    for (keel::usize j = 0; j < n; ++j) {
        keel::usize piv = j;
        double maxAbs = keel::abs(A_lu[j + j * n]);
        for (keel::usize i = j + 1; i < n; ++i) {
            double v = keel::abs(A_lu[i + j * n]);
            if (v > maxAbs) {
                maxAbs = v;
                piv = i;
            }
        }
        if (piv != j) {
            for (keel::usize c = 0; c < n; ++c)
                std::swap(A_lu[j + c * n], A_lu[piv + c * n]);
            for (keel::usize r = 0; r < nrhs; ++r)
                std::swap(B_work[j + r * n], B_work[piv + r * n]);
        }
        const T ajj = A_lu[j + j * n];
        if (ajj != T(0)) {
            for (keel::usize i = j + 1; i < n; ++i) {
                A_lu[i + j * n] /= ajj;
                const T lij = A_lu[i + j * n];
                if (lij != T(0)) {
                    for (keel::usize c = j + 1; c < n; ++c)
                        A_lu[i + c * n] -= lij * A_lu[j + c * n];
                }
            }
        }
    }
    for (keel::usize r = 0; r < nrhs; ++r) {
        T *b_col = B_work + r * n;
        for (keel::usize j = 0; j < n; ++j) {
            const T bj = b_col[j];
            if (bj != T(0)) {
                for (keel::usize i = j + 1; i < n; ++i)
                    b_col[i] -= A_lu[i + j * n] * bj;
            }
        }
        T *x_col = X + r * n;
        for (keel::usize i = 0; i < n; ++i)
            x_col[i] = b_col[i];
        for (keel::usize j = n; j-- > 0;) {
            x_col[j] /= A_lu[j + j * n];
            const T xj = x_col[j];
            if (xj != T(0)) {
                for (keel::usize i = 0; i < j; ++i) {
                    const T uij = A_lu[i + j * n];
                    if (uij != T(0))
                        x_col[i] -= uij * xj;
                }
            }
        }
    }
}

Value matrixSolve(const Value &A, const Value &B, const char *opname, keel::Allocator *mr)
{
    // Integer-class matrices reach here only when both operands are
    // integer matrices (scalar expansion was tried first) — the
    // reference rejects that as a separate integer-linear-algebra class.
    for (const Value *op : {&A, &B}) {
        const ValueType t = op->type();
        if (!isIntegerType(t))
            continue;
        const Dims &d = op->dims();
        throw Error(std::string(opname) + ": integer matrices are not"
                    " solved directly — convert with double() (got a "
                        + std::to_string(d.dim(0)) + "x" + std::to_string(d.dim(1))
                        + " " + mtypeName(t) + " array).",
                    0, 0, opname, "",
                    std::string("numkit:") + opname + ":integerUnsupported");
    }
    const keel::usize m  = A.dims().rows();
    const keel::usize n  = A.dims().cols();
    const keel::usize bm = B.dims().rows();
    const keel::usize k  = B.dims().cols();
    if (bm != m)
        throw Error(std::string(opname) + ": matrix dimensions must agree",
                    0, 0, opname, "", std::string("numkit:") + opname + ":dim");
    if (m < n)
        throw Error(std::string(opname)
                    + ": underdetermined (wide A, m<n) not yet supported",
                    0, 0, opname, "",
                    std::string("numkit:") + opname + ":wide");

    keel::ArenaTemp arena = keel::get_scratch(mr);

    if (A.isComplex() || B.isComplex()) {
keel::Array<keel::c64> A_buf(m * n, arena.allocator());
        keel::Array<keel::c64> B_buf(m * k, arena.allocator());
        if (A.isComplex()) {
            std::copy(A.complexData(), A.complexData() + m * n, A_buf.begin());
        } else {
            for (keel::usize i = 0; i < m * n; ++i) A_buf[i] = keel::c64(A.elemAsDouble(i), 0.0);
        }
        if (B.isComplex()) {
            std::copy(B.complexData(), B.complexData() + m * k, B_buf.begin());
        } else {
            for (keel::usize i = 0; i < m * k; ++i) B_buf[i] = keel::c64(B.elemAsDouble(i), 0.0);
        }
        Value X = Value::complexMatrix(n, k, mr);
        if (!numkit::ops::la_solve(A_buf.data(), m, n, B_buf.data(), k, X.complexDataMut(), arena.allocator())) {
            if (m == n) {
                singularSquareSolve(A_buf.data(), n, B_buf.data(), k, X.complexDataMut());
                return narrowIfReal(X, mr);
            }
            throw Error(std::string(opname)
                        + ": matrix is singular or rank-deficient",
                        0, 0, opname, "",
                        std::string("numkit:") + opname + ":singular");
        }
        return narrowIfReal(X, mr);
    }

    keel::Array<double> A_buf(m * n, arena.allocator());
    keel::Array<double> B_buf(m * k, arena.allocator());
    copyColumnMajorDouble(A, A_buf.data());
    copyColumnMajorDouble(B, B_buf.data());

    Value X = Value::matrix(n, k, ValueType::DOUBLE, mr);
    double *Xd = X.doubleDataMut();
    if (!numkit::ops::la_solve(A_buf.data(), m, n, B_buf.data(), k, Xd, arena.allocator())) {
        if (m == n) {
            singularSquareSolve(A_buf.data(), n, B_buf.data(), k, Xd);
            return X;
        }
        throw Error(std::string(opname)
                    + ": matrix is singular or rank-deficient",
                    0, 0, opname, "",
                    std::string("numkit:") + opname + ":singular");
    }
    return X;
}

Value transposeDouble(const Value &A, keel::Allocator *mr)
{
    const keel::usize m = A.dims().rows();
    const keel::usize n = A.dims().cols();
    Value T = Value::matrix(n, m, ValueType::DOUBLE, mr);
    double *Td = T.doubleDataMut();
    if (A.type() == ValueType::DOUBLE) {
        const double *Ad = A.doubleData();
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < m; ++i)
                Td[j + i * n] = Ad[i + j * m];
    } else {
        for (keel::usize j = 0; j < n; ++j)
            for (keel::usize i = 0; i < m; ++i)
                Td[j + i * n] = A.elemAsDouble(i + j * m);
    }
    return T;
}

enum class Cmp { EQ, NE, LT, GT, LE, GE };

inline const char *cmpOpName(Cmp c)
{
    switch (c) {
    case Cmp::EQ: return "==";
    case Cmp::NE: return "~=";
    case Cmp::LT: return "<";
    case Cmp::GT: return ">";
    case Cmp::LE: return "<=";
    case Cmp::GE: return ">=";
    }
    return "";
}

inline bool applyCmp(Cmp c, double x, double y)
{
    switch (c) {
    case Cmp::EQ: return x == y;
    case Cmp::NE: return x != y;
    case Cmp::LT: return x <  y;
    case Cmp::GT: return x >  y;
    case Cmp::LE: return x <= y;
    case Cmp::GE: return x >= y;
    }
    return false;
}

Value compareImpl(Cmp c, const Value &a, const Value &b)
{
    if (a.isChar() && b.isChar()) {
        if (c == Cmp::EQ)
            return Value::logicalScalar(a.toString() == b.toString(), nullptr);
        if (c == Cmp::NE)
            return Value::logicalScalar(a.toString() != b.toString(), nullptr);
    }

    if (a.isString() || b.isString()) {
        auto toStr = [](const Value &v) -> std::string {
            if (v.isString() || v.isChar())
                return std::string(v.toString().as_ref());
            throw Error("Comparison between string and non-string is not supported",
                         0, 0, "compare", "", "numkit:compare:stringType");
        };
        std::string sa = toStr(a), sb = toStr(b);
        switch (c) {
        case Cmp::EQ: return Value::logicalScalar(sa == sb, nullptr);
        case Cmp::NE: return Value::logicalScalar(sa != sb, nullptr);
        case Cmp::LT: return Value::logicalScalar(sa <  sb, nullptr);
        case Cmp::GT: return Value::logicalScalar(sa >  sb, nullptr);
        case Cmp::LE: return Value::logicalScalar(sa <= sb, nullptr);
        case Cmp::GE: return Value::logicalScalar(sa >= sb, nullptr);
        }
    }

    if (a.isComplex() || b.isComplex()) {
        // MATLAB R2025b: EQ/NE use full complex equality; the orderings
        // (< > <= >=) compare REAL parts only — the imaginary part is
        // ignored entirely (verified: (0+1i) < 2 -> 1, 2 < (0+1i) -> 0,
        // (3+4i) < (1+0i) -> 0, elementwise on arrays).
        auto cmpC = [c](keel::c64 x, keel::c64 y) -> bool {
            switch (c) {
            case Cmp::EQ: return x.real() == y.real() && x.imag() == y.imag();
            case Cmp::NE: return !(x.real() == y.real() && x.imag() == y.imag());
            case Cmp::LT: return x.real() <  y.real();
            case Cmp::GT: return x.real() >  y.real();
            case Cmp::LE: return x.real() <= y.real();
            case Cmp::GE: return x.real() >= y.real();
            }
            return false;
        };
        auto getC = [](const Value &v, keel::usize i) -> keel::c64 {
            if (v.isComplex())
                return v.complexData()[i];
            return keel::c64(v.type() == ValueType::LOGICAL
                                ? static_cast<double>(v.logicalData()[i])
                                : v.doubleData()[i],
                           0.0);
        };
        if (a.isScalar() && b.isScalar()) {
            keel::c64 ca = a.isComplex() ? a.toComplex() : keel::c64(a.toScalar(), 0.0);
            keel::c64 cb = b.isComplex() ? b.toComplex() : keel::c64(b.toScalar(), 0.0);
            return Value::logicalScalar(cmpC(ca, cb), nullptr);
        }
        if (a.isScalar()) {
            keel::c64 ca = a.isComplex() ? a.toComplex() : keel::c64(a.toScalar(), 0.0);
            auto r = createLike(b, ValueType::LOGICAL, nullptr);
            for (keel::usize i = 0; i < b.numel(); ++i)
                r.logicalDataMut()[i] = cmpC(ca, getC(b, i)) ? 1 : 0;
            return r;
        }
        if (b.isScalar()) {
            keel::c64 cb = b.isComplex() ? b.toComplex() : keel::c64(b.toScalar(), 0.0);
            auto r = createLike(a, ValueType::LOGICAL, nullptr);
            for (keel::usize i = 0; i < a.numel(); ++i)
                r.logicalDataMut()[i] = cmpC(getC(a, i), cb) ? 1 : 0;
            return r;
        }
        if (a.dims() != b.dims())
            throw Error("Matrix dimensions must agree for comparison. a=[" + std::to_string(a.dims().rows()) + "x" + std::to_string(a.dims().cols()) + "], b=[" + std::to_string(b.dims().rows()) + "x" + std::to_string(b.dims().cols()) + "]",
                        0, 0, "compareImpl", "", "numkit:compare:dimMismatch");
        auto r = createLike(a, ValueType::LOGICAL, nullptr);
        for (keel::usize i = 0; i < a.numel(); ++i)
            r.logicalDataMut()[i] = cmpC(getC(a, i), getC(b, i)) ? 1 : 0;
        return r;
    }

    {
        Value r;
        switch (c) {
        case Cmp::EQ: r = ops::eqFast(a, b); break;
        case Cmp::NE: r = ops::neFast(a, b); break;
        case Cmp::LT: r = ops::ltFast(a, b); break;
        case Cmp::GT: r = ops::gtFast(a, b); break;
        case Cmp::LE: r = ops::leFast(a, b); break;
        case Cmp::GE: r = ops::geFast(a, b); break;
        }
        if (!r.isUnset()) return r;
    }

    auto getD = [](const Value &v, keel::usize r, keel::usize col) -> double {
        keel::usize idx = col * v.dims().rows() + r;
        if (v.isLogical()) return static_cast<double>(v.logicalData()[idx]);
        if (v.type() == ValueType::DOUBLE) return v.doubleData()[idx];
        return v.elemAsDouble(idx);
    };
    auto getDScalar = [](const Value &v) -> double {
        if (v.isLogical()) return v.toBool() ? 1.0 : 0.0;
        if (v.type() == ValueType::DOUBLE) return v.toScalar();
        return v.elemAsDouble(0);
    };

    if (a.isScalar() && b.isScalar())
        return Value::logicalScalar(applyCmp(c, getDScalar(a), getDScalar(b)), nullptr);

    auto elemD = [](const Value &v, keel::usize i) -> double {
        if (v.isLogical()) return v.logicalData()[i];
        if (v.type() == ValueType::DOUBLE) return v.doubleData()[i];
        return v.elemAsDouble(i);
    };

    if (a.dims().ndim() >= 4 || b.dims().ndim() >= 4) {
        if (a.isScalar()) {
            auto r = createLike(b, ValueType::LOGICAL, nullptr);
            double s = getDScalar(a);
            keel::u8 *dst = r.logicalDataMut();
            for (keel::usize i = 0; i < b.numel(); ++i)
                dst[i] = applyCmp(c, s, elemD(b, i)) ? 1 : 0;
            return r;
        }
        if (b.isScalar()) {
            auto r = createLike(a, ValueType::LOGICAL, nullptr);
            double s = getDScalar(b);
            keel::u8 *dst = r.logicalDataMut();
            for (keel::usize i = 0; i < a.numel(); ++i)
                dst[i] = applyCmp(c, elemD(a, i), s) ? 1 : 0;
            return r;
        }
        Dims outD;
        if (!broadcastDimsND(a.dims(), b.dims(), outD))
            throw Error("ND dimensions must broadcast for comparison: each axis must match or be 1",
                         0, 0, "compare", "", "numkit:dimagree");
        auto r = createForDims(outD, ValueType::LOGICAL, nullptr);
        keel::u8 *dst = r.logicalDataMut();
        forEachNDPair(a.dims(), b.dims(), outD,
            [&](keel::usize outIdx, keel::usize aOff, keel::usize bOff) {
                dst[outIdx] = applyCmp(c, elemD(a, aOff), elemD(b, bOff)) ? 1 : 0;
            });
        return r;
    }

    if (a.dims().is3D() || b.dims().is3D()) {
        if (a.isScalar()) {
            auto r = createLike(b, ValueType::LOGICAL, nullptr);
            double s = getDScalar(a);
            for (keel::usize i = 0; i < b.numel(); ++i)
                r.logicalDataMut()[i] = applyCmp(c, s, elemD(b, i)) ? 1 : 0;
            return r;
        }
        if (b.isScalar()) {
            auto r = createLike(a, ValueType::LOGICAL, nullptr);
            double s = getDScalar(b);
            for (keel::usize i = 0; i < a.numel(); ++i)
                r.logicalDataMut()[i] = applyCmp(c, elemD(a, i), s) ? 1 : 0;
            return r;
        }
        const keel::usize aR = a.dims().rows(), aC = a.dims().cols();
        const keel::usize aP = a.dims().is3D() ? a.dims().pages() : 1;
        const keel::usize bR = b.dims().rows(), bC = b.dims().cols();
        const keel::usize bP = b.dims().is3D() ? b.dims().pages() : 1;
        keel::usize outR, outC, outP;
        if (!broadcastDims3D(aR, aC, aP, bR, bC, bP, outR, outC, outP))
            throw Error("3D dimensions must broadcast for comparison: each axis must match or be 1",
                         0, 0, "compare", "", "numkit:dimagree");

        if (aR == bR && aC == bC && aP == bP) {
            auto r = createLike(a, ValueType::LOGICAL, nullptr);
            for (keel::usize i = 0; i < a.numel(); ++i)
                r.logicalDataMut()[i] = applyCmp(c, elemD(a, i), elemD(b, i)) ? 1 : 0;
            return r;
        }
        auto r = (outP > 1) ? Value::matrix3d(outR, outC, outP, ValueType::LOGICAL, nullptr)
                            : Value::matrix(outR, outC, ValueType::LOGICAL, nullptr);
        keel::u8 *dst = r.logicalDataMut();
        for (keel::usize pp = 0; pp < outP; ++pp)
            for (keel::usize cc = 0; cc < outC; ++cc)
                for (keel::usize rr = 0; rr < outR; ++rr) {
                    const keel::usize aOff = broadcastOffset3D(rr, cc, pp, aR, aC, aP);
                    const keel::usize bOff = broadcastOffset3D(rr, cc, pp, bR, bC, bP);
                    dst[pp * outR * outC + cc * outR + rr] =
                        applyCmp(c, elemD(a, aOff), elemD(b, bOff)) ? 1 : 0;
                }
        return r;
    }

    keel::usize ar = a.dims().rows(), ac = a.dims().cols();
    keel::usize br = b.dims().rows(), bc = b.dims().cols();
    keel::usize outR, outC;
    if (!broadcastDims(ar, ac, br, bc, outR, outC))
        throw Error("Matrix dimensions must agree for comparison. a=[" + std::to_string(ar) + "x" + std::to_string(ac) + "], b=[" + std::to_string(br) + "x" + std::to_string(bc) + "]",
                    0, 0, "compareString", "", "numkit:compare:dimMismatch");

    auto r = Value::matrix(outR, outC, ValueType::LOGICAL, nullptr);
    keel::u8 *dst = r.logicalDataMut();
    for (keel::usize col = 0; col < outC; ++col) {
        keel::usize ca = (ac == 1) ? 0 : col;
        keel::usize cb = (bc == 1) ? 0 : col;
        for (keel::usize row = 0; row < outR; ++row) {
            keel::usize ra = (ar == 1) ? 0 : row;
            keel::usize rb = (br == 1) ? 0 : row;
            dst[col * outR + row] =
                applyCmp(c, getD(a, ra, ca), getD(b, rb, cb)) ? 1 : 0;
        }
    }
    return r;
}

keel::Array<keel::u8> toBoolArray(const Value &v, keel::Allocator *mr)
{
    auto nanErr = [] {
        throw std::runtime_error("NaN values cannot be converted to logicals.");
    };
    keel::Array<keel::u8> r(v.numel(), mr);
    if (v.isLogical()) {
        const keel::u8 *d = v.logicalData();
        for (keel::usize i = 0; i < v.numel(); ++i)
            r[i] = d[i] ? 1 : 0;
    } else if (v.type() == ValueType::DOUBLE) {
        const double *d = v.doubleData();
        for (keel::usize i = 0; i < v.numel(); ++i) {
            if (std::isnan(d[i]))
                nanErr();
            r[i] = (d[i] != 0.0) ? 1 : 0;
        }
    } else if (v.isComplex()) {
        const keel::c64 *d = v.complexData();
        for (keel::usize i = 0; i < v.numel(); ++i) {
            if (std::isnan(d[i].real()) || std::isnan(d[i].imag()))
                nanErr();
            r[i] = (d[i].real() != 0.0 || d[i].imag() != 0.0) ? 1 : 0;
        }
    } else {
        for (keel::usize i = 0; i < v.numel(); ++i) {
            double x = v.elemAsDouble(i);
            if (std::isnan(x))
                nanErr();
            r[i] = (x != 0.0) ? 1 : 0;
        }
    }
    return r;
}

bool scalarTruth(const Value &v)
{
    if (v.isComplex()) {
        keel::c64 c = v.toComplex();
        if (std::isnan(c.real()) || std::isnan(c.imag()))
            throw std::runtime_error("NaN values cannot be converted to logicals.");
        return c.real() != 0.0 || c.imag() != 0.0;
    }
    double x = v.elemAsDouble(0);
    if (std::isnan(x))
        throw std::runtime_error("NaN values cannot be converted to logicals.");
    return x != 0.0;
}

template <typename Op>
Value logicalBinary(const char *opName, Op op,
                     keel::Allocator *mr, const Value &a, const Value &b)
{
    if (a.isScalar() && b.isScalar())
        return Value::logicalScalar(op(scalarTruth(a), scalarTruth(b)), mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (a.isScalar()) {
        bool av = scalarTruth(a);
        auto bb = toBoolArray(b, scratch.allocator());
        auto r = createLike(b, ValueType::LOGICAL, mr);
        keel::u8 *dst = r.logicalDataMut();
        for (keel::usize i = 0; i < bb.size(); ++i)
            dst[i] = op(av, static_cast<bool>(bb[i])) ? 1 : 0;
        return r;
    }
    if (b.isScalar()) {
        bool bv = scalarTruth(b);
        auto aa = toBoolArray(a, scratch.allocator());
        auto r = createLike(a, ValueType::LOGICAL, mr);
        keel::u8 *dst = r.logicalDataMut();
        for (keel::usize i = 0; i < aa.size(); ++i)
            dst[i] = op(static_cast<bool>(aa[i]), bv) ? 1 : 0;
        return r;
    }
    if (a.numel() != b.numel())
        throw Error(std::string("Matrix dimensions must agree for ") + opName + ". a=[" + std::to_string(a.dims().rows()) + "x" + std::to_string(a.dims().cols()) + "], b=[" + std::to_string(b.dims().rows()) + "x" + std::to_string(b.dims().cols()) + "]",
                    0, 0, opName, "", "numkit:binary_ops:dimMismatch");
    auto aa = toBoolArray(a, scratch.allocator());
    auto bb = toBoolArray(b, scratch.allocator());
    auto r = createLike(a, ValueType::LOGICAL, mr);
    keel::u8 *dst = r.logicalDataMut();
    for (keel::usize i = 0; i < aa.size(); ++i)
        dst[i] = op(static_cast<bool>(aa[i]), static_cast<bool>(bb[i])) ? 1 : 0;
    return r;
}

template <typename T>
inline void runPageMatmul(const T *, const T *, T *,
                          keel::usize, keel::usize, keel::usize);

template <>
inline void runPageMatmul<double>(const double *a, const double *b, double *c,
                                  keel::usize M, keel::usize N, keel::usize K)
{
    ops::detail::matmulDoubleLoop(a, b, c, M, N, K);
}

template <>
inline void runPageMatmul<float>(const float *a, const float *b, float *c,
                                 keel::usize M, keel::usize N, keel::usize K)
{
    for (keel::usize j = 0; j < N; ++j) {
        float *cj = c + j * M;
        for (keel::usize i = 0; i < M; ++i) cj[i] = 0.0f;
        for (keel::usize k = 0; k < K; ++k) {
            const float bkj = b[j * K + k];
            const float *ak = a + k * M;
            for (keel::usize i = 0; i < M; ++i)
                cj[i] += ak[i] * bkj;
        }
    }
}

template <>
inline void runPageMatmul<keel::c64>(const keel::c64 *a, const keel::c64 *b, keel::c64 *c,
                                   keel::usize M, keel::usize N, keel::usize K)
{
    for (keel::usize j = 0; j < N; ++j) {
        keel::c64 *cj = c + j * M;
        for (keel::usize i = 0; i < M; ++i) cj[i] = keel::c64(0.0, 0.0);
        for (keel::usize k = 0; k < K; ++k) {
            const keel::c64 bkj = b[j * K + k];
            const keel::c64 *ak = a + k * M;
            for (keel::usize i = 0; i < M; ++i)
                cj[i] += ak[i] * bkj;
        }
    }
}

template <typename T> constexpr ValueType pagemtimesElemMType();
template <> constexpr ValueType pagemtimesElemMType<double >() { return ValueType::DOUBLE;  }
template <> constexpr ValueType pagemtimesElemMType<float  >() { return ValueType::SINGLE;  }
template <> constexpr ValueType pagemtimesElemMType<keel::c64>() { return ValueType::COMPLEX; }

template <typename T>
inline T readElemAsT(const Value &src, keel::usize i, bool typeMatches)
{
    if constexpr (std::is_same_v<T, keel::c64>) {
        if (typeMatches) return src.complexData()[i];
        return keel::c64(src.elemAsDouble(i), 0.0);
    } else {
        if (typeMatches) return static_cast<const T *>(src.rawData())[i];
        return static_cast<T>(src.elemAsDouble(i));
    }
}

template <typename T>
inline T conjIfComplex(T v)
{
    if constexpr (std::is_same_v<T, keel::c64>) return keel::conj(v);
    else return v;
}

template <typename T>
void materialisePage(T *dst, const Value &src, keel::usize pageOff,
                     keel::usize rowDim, keel::usize colDim, TranspOp tr)
{
    const keel::usize pageElems = rowDim * colDim;
    const keel::usize base = pageOff * pageElems;
    const bool typeMatches = (src.type() == pagemtimesElemMType<T>());

    if (tr == TranspOp::None) {
        if (typeMatches) {
            std::memcpy(dst, static_cast<const T *>(src.rawData()) + base,
                        pageElems * sizeof(T));
        } else {
            for (keel::usize i = 0; i < pageElems; ++i)
                dst[i] = readElemAsT<T>(src, base + i, false);
        }
        return;
    }
    const bool needsConj = (tr == TranspOp::CTranspose);
    for (keel::usize r = 0; r < rowDim; ++r) {
        for (keel::usize c = 0; c < colDim; ++c) {
            const keel::usize srcOff = base + c * rowDim + r;
            T v = readElemAsT<T>(src, srcOff, typeMatches);
            if (needsConj) v = conjIfComplex<T>(v);
            dst[r * colDim + c] = v;
        }
    }
}

template <typename T>
Value pagemtimesImpl(const Value &x, TranspOp tx, const Value &y, TranspOp ty, keel::Allocator *mr)
{
    const auto &xd = x.dims();
    const auto &yd = y.dims();
    const int xnd = xd.ndim();
    const int ynd = yd.ndim();
    if (xnd < 2 || ynd < 2)
        throw Error("pagemtimes: each input must have at least 2 dimensions",
                     0, 0, "pagemtimes", "", "numkit:pagemtimes:rank");

    const keel::usize xRowDim = xd.dim(0), xColDim = xd.dim(1);
    const keel::usize yRowDim = yd.dim(0), yColDim = yd.dim(1);

    const keel::usize M  = (tx == TranspOp::None) ? xRowDim : xColDim;
    const keel::usize Kx = (tx == TranspOp::None) ? xColDim : xRowDim;
    const keel::usize Ky = (ty == TranspOp::None) ? yRowDim : yColDim;
    const keel::usize N  = (ty == TranspOp::None) ? yColDim : yRowDim;
    if (Kx != Ky)
        throw Error("pagemtimes: inner matrix dimensions must agree",
                     0, 0, "pagemtimes", "", "numkit:pagemtimes:innerdim");
    const keel::usize K = Kx;

    constexpr int kMaxNd = Dims::kMaxRank;
    const int xb = std::max(0, xnd - 2);
    const int yb = std::max(0, ynd - 2);
    const int outBatchNd = std::max(xb, yb);
    keel::usize xBatch[kMaxNd], yBatch[kMaxNd], outBatch[kMaxNd];
    for (int i = 0; i < outBatchNd; ++i) {
        xBatch[i] = (i < xb) ? xd.dim(2 + i) : 1;
        yBatch[i] = (i < yb) ? yd.dim(2 + i) : 1;
        if (xBatch[i] != yBatch[i] && xBatch[i] != 1 && yBatch[i] != 1)
            throw Error("pagemtimes: batch dimensions must broadcast "
                         "(each axis must match or be 1)",
                         0, 0, "pagemtimes", "", "numkit:pagemtimes:dimagree");
        outBatch[i] = std::max(xBatch[i], yBatch[i]);
    }

    keel::usize batchN = 1;
    for (int i = 0; i < outBatchNd; ++i) batchN *= outBatch[i];

    const int outNd = 2 + outBatchNd;
    keel::usize outDimArr[kMaxNd];
    outDimArr[0] = M;
    outDimArr[1] = N;
    for (int i = 0; i < outBatchNd; ++i) outDimArr[2 + i] = outBatch[i];
    auto z = createForDims(Dims(outDimArr, outNd), pagemtimesElemMType<T>(), mr);
    if (M == 0 || N == 0 || batchN == 0)
        return z;

    T *zData = static_cast<T *>(z.rawDataMut());
    const keel::usize xPageStride = xRowDim * xColDim;
    const keel::usize yPageStride = yRowDim * yColDim;
    const keel::usize zPageStride = M * N;

    const bool xDirect = (x.type() == pagemtimesElemMType<T>()) && (tx == TranspOp::None);
    const bool yDirect = (y.type() == pagemtimesElemMType<T>()) && (ty == TranspOp::None);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<T> scratchX(scratch.allocator()), scratchY(scratch.allocator());
    if (!xDirect) scratchX.resize(xPageStride);
    if (!yDirect) scratchY.resize(yPageStride);

    auto getXPage = [&](keel::usize pageOff) -> const T * {
        if (xDirect)
            return static_cast<const T *>(x.rawData()) + pageOff * xPageStride;
        materialisePage(scratchX.data(), x, pageOff, xRowDim, xColDim, tx);
        return scratchX.data();
    };
    auto getYPage = [&](keel::usize pageOff) -> const T * {
        if (yDirect)
            return static_cast<const T *>(y.rawData()) + pageOff * yPageStride;
        materialisePage(scratchY.data(), y, pageOff, yRowDim, yColDim, ty);
        return scratchY.data();
    };

    if (outBatchNd == 0) {
        runPageMatmul<T>(getXPage(0), getYPage(0), zData, M, N, K);
        return z;
    }

    keel::usize xBatchStride[kMaxNd], yBatchStride[kMaxNd];
    {
        keel::usize sx = 1, sy = 1;
        for (int i = 0; i < outBatchNd; ++i) {
            xBatchStride[i] = sx;
            yBatchStride[i] = sy;
            sx *= xBatch[i];
            sy *= yBatch[i];
        }
    }

    keel::usize coords[kMaxNd] = {0};
    Dims outBatchDims(outBatch, outBatchNd);
    keel::usize pageIdx = 0;
    do {
        keel::usize xOff = 0, yOff = 0;
        for (int i = 0; i < outBatchNd; ++i) {
            const keel::usize xc = (xBatch[i] == 1) ? 0 : coords[i];
            const keel::usize yc = (yBatch[i] == 1) ? 0 : coords[i];
            xOff += xc * xBatchStride[i];
            yOff += yc * yBatchStride[i];
        }
        runPageMatmul<T>(getXPage(xOff), getYPage(yOff),
                         zData + pageIdx * zPageStride,
                         M, N, K);
        ++pageIdx;
    } while (incrementCoords(coords, outBatchDims));

    return z;
}

} // namespace

} // namespace numkit::builtin
