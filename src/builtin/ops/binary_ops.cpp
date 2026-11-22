// src/builtin/src/ops/binary_ops.cpp

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/builtin/ops.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/ops/blas3.hpp>
#include <numkit/ops/la_solve.hpp>
#include <numkit/ops/binary_ops.hpp>
#include <numkit/ops/compare.hpp>
#include <numkit/linalg/eig.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "binary_ops_detail.hpp"

namespace numkit::builtin {

// ════════════════════════════════════════════════════════════════════════
// Public API — binary operators
// ════════════════════════════════════════════════════════════════════════

// ── Arithmetic ──────────────────────────────────────────────────────────

Value plus(const Value &a, const Value &b, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (a.isEmpty() || b.isEmpty())
        return emptyArithResult(a, b, p);
    // MATLAB auto-coerces logical to double for arithmetic. See BUGS.md #24.
    if (a.type() == ValueType::LOGICAL || b.type() == ValueType::LOGICAL)
        return plus(coerceLogicalToDouble(a, p), coerceLogicalToDouble(b, p), p);
    if (a.isComplex() || b.isComplex())
        return elementwiseComplex(a, b, [](keel::c64 x, keel::c64 y) { return x + y; }, p);
    if (a.type() == ValueType::DOUBLE && b.type() == ValueType::DOUBLE) {
        if (sameShapeDoubleFastPath(a, b)) {
            auto r = createLike(a, ValueType::DOUBLE, p);
            ops::detail::plusLoop(a.doubleData(), b.doubleData(), r.doubleDataMut(), a.numel());
            return r;
        }
        return elementwiseDouble(a, b, [](keel::f64 x, keel::f64 y) { return x + y; }, p);
    }
    if (a.isChar() && b.isChar())
        return Value::fromString(keel::format("{}{}", a.toString(), b.toString()), p);
    if (a.isChar() && b.type() == ValueType::DOUBLE) {
        auto ca = createLike(a, ValueType::DOUBLE, p);
        const char *cd = a.charData();
        keel::f64 *dd = ca.doubleDataMut();
        for (keel::usize i = 0; i < a.numel(); ++i)
            dd[i] = keel::as<keel::f64>(keel::as<keel::u8>(cd[i]));
        return elementwiseDouble(ca, b, [](keel::f64 x, keel::f64 y) { return x + y; }, p);
    }
    if (a.type() == ValueType::DOUBLE && b.isChar()) {
        auto cb = createLike(b, ValueType::DOUBLE, p);
        const char *cd = b.charData();
        keel::f64 *dd = cb.doubleDataMut();
        for (keel::usize i = 0; i < b.numel(); ++i)
            dd[i] = keel::as<keel::f64>(keel::as<keel::u8>(cd[i]));
        return elementwiseDouble(a, cb, [](keel::f64 x, keel::f64 y) { return x + y; }, p);
    }
    if (a.isString() && b.isString())
        return Value::stringScalar(keel::format("{}{}", a.toString(), b.toString()), p);
    if (a.isString() && b.isChar())
        return Value::stringScalar(keel::format("{}{}", a.toString(), b.toString()), p);
    if (a.isChar() && b.isString())
        return Value::stringScalar(keel::format("{}{}", a.toString(), b.toString()), p);
    if (a.isString() && b.isNumeric())
        return Value::stringScalar(keel::format("{}{}", a.toString(), b.toScalar()), p);
    if (a.isNumeric() && b.isString())
        return Value::stringScalar(keel::format("{}{}", a.toScalar(), b.toString()), p);
    if (a.type() == ValueType::CHAR || b.type() == ValueType::CHAR)
        return plus(coerceCharToDouble(a, p), coerceCharToDouble(b, p), p);
    {
        auto r = dispatchIntegerBinaryOp(a, b, [](auto x, auto y) { return saturateAdd(x, y); }, p);
        if (!r.isUnset()) return r;
    }
    throw Error("Unsupported types for +", 0, 0, "plus", "", "numkit:plus:unsupportedTypes");
}

Value minus(const Value &a, const Value &b, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (a.isEmpty() || b.isEmpty())
        return emptyArithResult(a, b, p);
    if (a.type() == ValueType::LOGICAL || b.type() == ValueType::LOGICAL)
        return minus(coerceLogicalToDouble(a, p), coerceLogicalToDouble(b, p), p);
    if (a.type() == ValueType::CHAR || b.type() == ValueType::CHAR)
        return minus(coerceCharToDouble(a, p), coerceCharToDouble(b, p), p);
    if (a.isComplex() || b.isComplex())
        return elementwiseComplex(a, b, [](keel::c64 x, keel::c64 y) { return x - y; }, p);
    if (a.type() == ValueType::DOUBLE && b.type() == ValueType::DOUBLE) {
        if (sameShapeDoubleFastPath(a, b)) {
            auto r = createLike(a, ValueType::DOUBLE, p);
            ops::detail::minusLoop(a.doubleData(), b.doubleData(), r.doubleDataMut(), a.numel());
            return r;
        }
        return elementwiseDouble(a, b, [](keel::f64 x, keel::f64 y) { return x - y; }, p);
    }
    {
        auto r = dispatchIntegerBinaryOp(a, b, [](auto x, auto y) { return saturateSub(x, y); }, p);
        if (!r.isUnset()) return r;
    }
    throw Error("Unsupported types for -", 0, 0, "minus", "", "numkit:minus:unsupportedTypes");
}

Value times(const Value &a, const Value &b, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (a.isEmpty() || b.isEmpty())
        return emptyArithResult(a, b, p);
    if (a.type() == ValueType::LOGICAL || b.type() == ValueType::LOGICAL)
        return times(coerceLogicalToDouble(a, p), coerceLogicalToDouble(b, p), p);
    if (a.type() == ValueType::CHAR || b.type() == ValueType::CHAR)
        return times(coerceCharToDouble(a, p), coerceCharToDouble(b, p), p);
    if (a.isComplex() || b.isComplex())
        return elementwiseComplex(a, b, [](keel::c64 x, keel::c64 y) { return x * y; }, p);
    if (a.type() == ValueType::DOUBLE && b.type() == ValueType::DOUBLE) {
        if (sameShapeDoubleFastPath(a, b)) {
            auto r = createLike(a, ValueType::DOUBLE, p);
            ops::detail::timesLoop(a.doubleData(), b.doubleData(), r.doubleDataMut(), a.numel());
            return r;
        }
        return elementwiseDouble(a, b, [](keel::f64 x, keel::f64 y) { return x * y; }, p);
    }
    {
        auto r = dispatchIntegerBinaryOp(a, b, [](auto x, auto y) { return saturateMul(x, y); }, p);
        if (!r.isUnset()) return r;
    }
    throw Error("Unsupported types for .*", 0, 0, "times", "", "numkit:times:unsupportedTypes");
}

Value mtimes(const Value &a, const Value &b, keel::Allocator *mr)
{
    keel::Allocator *p = mr;

    if (a.type() == ValueType::LOGICAL || b.type() == ValueType::LOGICAL)
        return mtimes(coerceLogicalToDouble(a, p), coerceLogicalToDouble(b, p), p);
    if (a.type() == ValueType::CHAR || b.type() == ValueType::CHAR)
        return mtimes(coerceCharToDouble(a, p), coerceCharToDouble(b, p), p);

    if ((a.dims().is3D() || b.dims().is3D()) && !a.isScalar() && !b.isScalar())
        throw Error("MTIMES is not supported for N-D arrays",
                     0, 0, "mtimes", "", "numkit:mtimes:notSupportedND");

    if (a.isComplex() || b.isComplex()) {
        auto [ca, cb] = promoteToComplex(a, b, p);
        if (ca.isScalar() || cb.isScalar())
            return elementwiseComplex(a, b, [](keel::c64 x, keel::c64 y) { return x * y; }, p);
        keel::usize M = ca.dims().rows(), K = ca.dims().cols(), N = cb.dims().cols();
        if (K != cb.dims().rows())
            throw Error("Inner matrix dimensions must agree", 0, 0, "mtimes", "",
                         "numkit:innerdim");
        auto r = Value::complexMatrix(M, N, p);
        if (ca.isSingleComplex() || cb.isSingleComplex())
            r.setSingleComplex(true);
        if (M > 0 && N > 0) {
            ops::gemm(ops::MatrixTranspose::NoTrans, ops::MatrixTranspose::NoTrans,
                      M, N, K,
                      keel::c64(1.0, 0.0), ca.complexData(), M,
                      cb.complexData(), K,
                      keel::c64(0.0, 0.0), r.complexDataMut(), M);
        }
        return narrowComplex(r, p);
    }

    if (a.isScalar() || b.isScalar()) {
        auto r = dispatchIntegerBinaryOp(a, b, [](auto x, auto y) { return saturateMul(x, y); }, p);
        if (!r.isUnset()) return r;
        return elementwiseDouble(a, b, [](keel::f64 x, keel::f64 y) { return x * y; }, p);
    }
    if ((a.type() == ValueType::SINGLE || b.type() == ValueType::SINGLE) &&
        (a.type() == ValueType::SINGLE || a.type() == ValueType::DOUBLE) &&
        (b.type() == ValueType::SINGLE || b.type() == ValueType::DOUBLE)) {
        keel::usize M = a.dims().rows(), K = a.dims().cols(), N = b.dims().cols();
        if (K != b.dims().rows())
            throw Error("Inner matrix dimensions must agree", 0, 0, "mtimes", "",
                         "numkit:innerdim");
        auto r = Value::matrix(M, N, ValueType::SINGLE, p);
        keel::f32 *rd = static_cast<keel::f32 *>(r.rawDataMut());
        if (M == 0 || N == 0)
            return r;
        if (K == 0) {
            std::fill(rd, rd + M * N, 0.0f);
            return r;
        }
        keel::ArenaTemp scratch = keel::get_scratch(p);
        keel::Array<keel::f64> cd(M * N, scratch.allocator());
        const keel::f64 *adPtr = nullptr;
        keel::Array<keel::f64> adScratch(scratch.allocator());
        if (a.type() == ValueType::DOUBLE) {
            adPtr = a.doubleData();
        } else {
            adScratch.resize(M * K);
            const keel::f32 *af = static_cast<const keel::f32 *>(a.rawData());
            for (keel::usize i = 0; i < M * K; ++i) adScratch[i] = keel::as<keel::f64>(af[i]);
            adPtr = adScratch.data();
        }
        const keel::f64 *bdPtr = nullptr;
        keel::Array<keel::f64> bdScratch(scratch.allocator());
        if (b.type() == ValueType::DOUBLE) {
            bdPtr = b.doubleData();
        } else {
            bdScratch.resize(K * N);
            const keel::f32 *bf = static_cast<const keel::f32 *>(b.rawData());
            for (keel::usize i = 0; i < K * N; ++i) bdScratch[i] = keel::as<keel::f64>(bf[i]);
            bdPtr = bdScratch.data();
        }
        ops::detail::matmulDoubleLoop(adPtr, bdPtr, cd.data(), M, N, K);
        for (keel::usize i = 0; i < M * N; ++i)
            rd[i] = keel::as<keel::f32>(cd[i]);
        return r;
    }
    if (a.type() == ValueType::DOUBLE && b.type() == ValueType::DOUBLE) {
        keel::usize M = a.dims().rows(), K = a.dims().cols(), N = b.dims().cols();
        if (K != b.dims().rows())
            throw Error("Inner matrix dimensions must agree", 0, 0, "mtimes", "",
                         "numkit:innerdim");
        auto r = Value::matrix(M, N, ValueType::DOUBLE, p);
        ops::detail::matmulDoubleLoop(a.doubleData(), b.doubleData(), r.doubleDataMut(),
                                 M, N, K);
        return r;
    }
    throw Error("Unsupported types for *", 0, 0, "mtimes", "", "numkit:mtimes:unsupportedTypes");
}

Value rdivide(const Value &a, const Value &b, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (a.isEmpty() || b.isEmpty())
        return emptyArithResult(a, b, p);
    if (a.type() == ValueType::LOGICAL || b.type() == ValueType::LOGICAL)
        return rdivide(coerceLogicalToDouble(a, p), coerceLogicalToDouble(b, p), p);
    if (a.type() == ValueType::CHAR || b.type() == ValueType::CHAR)
        return rdivide(coerceCharToDouble(a, p), coerceCharToDouble(b, p), p);
    if (a.isComplex() || b.isComplex())
        return elementwiseComplex(a, b, [](keel::c64 x, keel::c64 y) { return x / y; }, p);
    if (a.type() == ValueType::DOUBLE && b.type() == ValueType::DOUBLE) {
        if (sameShapeDoubleFastPath(a, b)) {
            auto r = createLike(a, ValueType::DOUBLE, p);
            ops::detail::rdivideLoop(a.doubleData(), b.doubleData(), r.doubleDataMut(), a.numel());
            return r;
        }
        return elementwiseDouble(a, b, [](keel::f64 x, keel::f64 y) { return x / y; }, p);
    }
    {
        auto r = dispatchIntegerBinaryOp(a, b, [](auto x, auto y) { return saturateDiv(x, y); }, p);
        if (!r.isUnset()) return r;
    }
    throw Error("Unsupported types for ./", 0, 0, "rdivide", "",
                 "numkit:rdivide:unsupportedTypes");
}

Value ldivide(const Value &a, const Value &b, keel::Allocator *mr)
{
    return rdivide(b, a, mr);
}

Value mrdivide(const Value &a, const Value &b, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (a.isEmpty() || b.isEmpty())
        return emptyArithResult(a, b, p);
    // Non-numeric operands (string/cell/struct scalars included) are
    // rejected before any fast path — the reference's mustBeNumeric-
    // CharOrLogical class (probed R2025b).
    checkSolveOperandClass(a, "mrdivide");
    checkSolveOperandClass(b, "mrdivide");
    if (a.type() == ValueType::LOGICAL || b.type() == ValueType::LOGICAL)
        return mrdivide(coerceLogicalToDouble(a, p), coerceLogicalToDouble(b, p), p);
    if (a.type() == ValueType::CHAR || b.type() == ValueType::CHAR)
        return mrdivide(coerceCharToDouble(a, p), coerceCharToDouble(b, p), p);
    // Integer operands divide elementwise only with scalar expansion;
    // matrix/matrix integer division is matrix solve, which the reference
    // rejects as a separate integer-linear-algebra class — fall through
    // to matrixSolve's gate instead of silently dividing elementwise.
    if (a.isScalar() || b.isScalar()) {
        auto r = dispatchIntegerBinaryOp(a, b, [](auto x, auto y) { return saturateDiv(x, y); }, p);
        if (!r.isUnset()) return r;
    }
    if (b.isScalar()) {
        if (a.isComplex() || b.isComplex())
            return elementwiseComplex(a, b, [](keel::c64 x, keel::c64 y) { return x / y; }, p);
        return elementwiseDouble(a, b, [](keel::f64 x, keel::f64 y) { return x / y; }, p);
    }
    if (a.isScalar() && !b.isScalar()) {
        if (b.dims().ndim() > 2)
            return rdivide(a, b, p);
        if (b.dims().cols() != 1) {
            throw Error("mrdivide: matrix dimensions must agree",
                        0, 0, "mrdivide", "", "numkit:mrdivide:dim");
        }
    }
    Value Bt = ctranspose(b, p);
    Value At = ctranspose(a, p);
    // Solve X = (B'\A')' directly with mrdivide's own voice (operands
    // are non-scalar matrices here, so this is exactly mldivide's tail).
    Value Y  = matrixSolve(Bt, At, "mrdivide", p);
    return ctranspose(Y, p);
}

Value mldivide(const Value &a, const Value &b, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (a.isEmpty() || b.isEmpty())
        return emptyArithResult(a, b, p);
    // Non-numeric operands (string/cell/struct scalars included) are
    // rejected before any fast path — the reference's mustBeNumeric-
    // CharOrLogical class (probed R2025b).
    checkSolveOperandClass(a, "mldivide");
    checkSolveOperandClass(b, "mldivide");
    if (a.type() == ValueType::LOGICAL || b.type() == ValueType::LOGICAL)
        return mldivide(coerceLogicalToDouble(a, p), coerceLogicalToDouble(b, p), p);
    if (a.type() == ValueType::CHAR || b.type() == ValueType::CHAR)
        return mldivide(coerceCharToDouble(a, p), coerceCharToDouble(b, p), p);
    if (a.isScalar() && b.isScalar()) {
        if (a.isComplex() || b.isComplex()) {
            keel::c64 ca = a.isComplex() ? a.complexData()[0] : keel::c64(a.doubleData()[0], 0.0);
            keel::c64 cb = b.isComplex() ? b.complexData()[0] : keel::c64(b.doubleData()[0], 0.0);
            keel::c64 res = cb / ca;
            return narrowIfReal(Value::complexScalar(res.real(), res.imag(), p), p);
        }
        return Value::scalar(b.toScalar() / a.toScalar(), p);
    }
    if (a.isScalar() && !b.isScalar()) {
        if (a.isComplex() || b.isComplex())
            return elementwiseComplex(b, a, [](keel::c64 x, keel::c64 y) { return x / y; }, p);
        return elementwiseDouble(b, a, [](keel::f64 x, keel::f64 y) { return x / y; }, p);
    }
    return matrixSolve(a, b, "mldivide", p);
}

static bool powNeedsComplex(const Value &a, const Value &b)
{
    if (a.type() != ValueType::DOUBLE || b.type() != ValueType::DOUBLE)
        return false;
    auto negBase = [](keel::f64 x) { return x < 0.0; };
    auto fracExp = [](keel::f64 y) { return y != std::floor(y); };
    if (a.isScalar() && b.isScalar())
        return negBase(a.toScalar()) && fracExp(b.toScalar());
    if (b.isScalar()) {
        if (!fracExp(b.toScalar())) return false;
        const keel::f64 *da = a.doubleData();
        for (keel::usize i = 0; i < a.numel(); ++i)
            if (negBase(da[i])) return true;
        return false;
    }
    if (a.isScalar()) {
        if (!negBase(a.toScalar())) return false;
        const keel::f64 *db = b.doubleData();
        for (keel::usize i = 0; i < b.numel(); ++i)
            if (fracExp(db[i])) return true;
        return false;
    }
    if (a.dims() == b.dims()) {
        const keel::f64 *da = a.doubleData(), *db = b.doubleData();
        for (keel::usize i = 0; i < a.numel(); ++i)
            if (negBase(da[i]) && fracExp(db[i])) return true;
    }
    return false;
}

Value power(const Value &a, const Value &b, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (a.isEmpty() || b.isEmpty())
        return emptyArithResult(a, b, p);
    if (a.type() == ValueType::LOGICAL || b.type() == ValueType::LOGICAL)
        return power(coerceLogicalToDouble(a, p), coerceLogicalToDouble(b, p), p);
    if (a.type() == ValueType::CHAR || b.type() == ValueType::CHAR)
        return power(coerceCharToDouble(a, p), coerceCharToDouble(b, p), p);
    if (a.isScalar() && b.isScalar()) {
        if (a.isComplex() || b.isComplex()) {
            auto [ca, cb] = promoteToComplex(a, b, p);
            return Value::complexScalar(keel::pow(ca.toComplex(), cb.toComplex()), p);
        }
        const keel::f64 base = a.toScalar(), e = b.toScalar();
        if (base < 0.0 && e != std::floor(e))
            return Value::complexScalar(keel::pow(keel::c64(base, 0.0), e), p);
        return Value::scalar(std::pow(base, e), p);
    }
    if (b.isScalar() && !a.isScalar()
        && a.dims().rows() == a.dims().cols()
        && a.dims().ndim() == 2) {
        const keel::usize R = a.dims().rows();
        const bool bCplx = b.isComplex();
        const keel::f64 bs = bCplx ? b.toComplex().real() : b.toScalar();
        const bool bRealInt = (!bCplx || b.toComplex().imag() == 0.0)
                              && std::isfinite(bs) && std::floor(bs) == bs
                              && keel::abs(bs) <= 1e9;
        if (bRealInt) {
            const keel::i64 n = keel::as<keel::i64>(bs);
            auto makeEye = [&]() {
                if (a.type() == ValueType::SINGLE && !a.isComplex()) {
                    auto I = Value::matrix(R, R, ValueType::SINGLE, p);
                    keel::f32 *id = static_cast<keel::f32 *>(I.rawDataMut());
                    std::fill(id, id + R * R, 0.0f);
                    for (keel::usize i = 0; i < R; ++i) id[i * R + i] = 1.0f;
                    return I;
                }
                auto I = Value::matrix(R, R, ValueType::DOUBLE, p);
                keel::f64 *id = I.doubleDataMut();
                std::memset(id, 0, sizeof(keel::f64) * R * R);
                for (keel::usize i = 0; i < R; ++i) id[i * R + i] = 1.0;
                return I;
            };
            if (n == 0) return makeEye();
            keel::u64 expAbs = keel::as<keel::u64>(n > 0 ? n : -n);
            Value base = a;
            Value acc;
            bool haveAcc = false;
            while (expAbs > 0) {
                if ((expAbs & 1u) != 0u) {
                    acc = haveAcc ? mtimes(acc, base, p) : base;
                    haveAcc = true;
                }
                expAbs >>= 1u;
                if (expAbs > 0)
                    base = mtimes(base, base, p);
            }
            if (n < 0) return mldivide(acc, makeEye(), p);
            return acc;
        }
        // Fractional or complex scalar exponent on square matrix A (R x R)
        const keel::c64 expC = bCplx ? b.toComplex() : keel::c64(bs, 0.0);
        const bool aCplx = a.isComplex();
        // Fast path for diagonal matrices (e.g. D^(-0.5))
        bool isDiag = true;
        for (keel::usize j = 0; j < R && isDiag; ++j) {
            for (keel::usize i = 0; i < R; ++i) {
                if (i == j) continue;
                keel::c64 v = aCplx ? a.complexElem(i + j * R) : keel::c64(a.elemAsDouble(i + j * R), 0.0);
                if (v.real() != 0.0 || v.imag() != 0.0) {
                    isDiag = false;
                    break;
                }
            }
        }
        if (isDiag) {
            Value outC = Value::matrix(R, R, ValueType::COMPLEX, p);
            keel::c64 *od = outC.complexDataMut();
            std::fill(od, od + R * R, keel::c64(0.0, 0.0));
            bool anyImag = false;
            for (keel::usize i = 0; i < R; ++i) {
                keel::c64 d = aCplx ? a.complexElem(i + i * R) : keel::c64(a.elemAsDouble(i + i * R), 0.0);
                keel::c64 dp = (!aCplx && !bCplx && d.real() >= 0.0)
                                 ? keel::c64(std::pow(d.real(), bs), 0.0)
                                 : keel::pow(d, expC);
                od[i + i * R] = dp;
                if (dp.imag() != 0.0) anyImag = true;
            }
            if (!anyImag && !aCplx && !bCplx) {
                Value outR = Value::matrix(R, R, ValueType::DOUBLE, p);
                keel::f64 *rd = outR.doubleDataMut();
                std::memset(rd, 0, sizeof(keel::f64) * R * R);
                for (keel::usize i = 0; i < R; ++i) rd[i + i * R] = od[i + i * R].real();
                return outR;
            }
            return outC;
        }
        // General square matrix: eigendecomposition A = V * D * V^-1 => A^b = (V * D^b) / V
        auto [V, D] = ::numkit::linalg::eig_general_VD(a, p);
        Value Vc = Value::matrix(R, R, ValueType::COMPLEX, p);
        keel::c64 *vcd = Vc.complexDataMut();
        const bool vIsCplx = V.isComplex();
        const bool dIsCplx = D.isComplex();
        for (keel::usize j = 0; j < R; ++j) {
            keel::c64 dj = dIsCplx ? D.complexElem(j + j * R) : keel::c64(D.elemAsDouble(j + j * R), 0.0);
            keel::c64 dp = (!dIsCplx && !bCplx && dj.real() >= 0.0)
                             ? keel::c64(std::pow(dj.real(), bs), 0.0)
                             : keel::pow(dj, expC);
            for (keel::usize i = 0; i < R; ++i) {
                keel::c64 vij = vIsCplx ? V.complexElem(i + j * R) : keel::c64(V.elemAsDouble(i + j * R), 0.0);
                vcd[i + j * R] = vij * dp;
            }
        }
        Value res = mrdivide(Vc, V, p);
        if (!aCplx && !bCplx && res.isComplex()) {
            const keel::c64 *rc = res.complexData();
            keel::f64 maxRe = 0.0, maxIm = 0.0;
            for (keel::usize k = 0; k < R * R; ++k) {
                maxRe = std::max(maxRe, keel::abs(rc[k].real()));
                maxIm = std::max(maxIm, keel::abs(rc[k].imag()));
            }
            if (maxIm <= 1e-12 * std::max(1.0, maxRe)) {
                Value outR = Value::matrix(R, R, ValueType::DOUBLE, p);
                keel::f64 *rd = outR.doubleDataMut();
                for (keel::usize k = 0; k < R * R; ++k) rd[k] = rc[k].real();
                return outR;
            }
        }
        return res;
    }
    throw Error("Matrix power not implemented", 0, 0, "power", "",
                 "numkit:power:notImplemented");
}

Value elementPower(const Value &a, const Value &b, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (a.isEmpty() || b.isEmpty())
        return emptyArithResult(a, b, p);
    if (a.type() == ValueType::LOGICAL || b.type() == ValueType::LOGICAL)
        return elementPower(coerceLogicalToDouble(a, p), coerceLogicalToDouble(b, p), p);
    if (a.type() == ValueType::CHAR || b.type() == ValueType::CHAR)
        return elementPower(coerceCharToDouble(a, p), coerceCharToDouble(b, p), p);
    if (a.isComplex() || b.isComplex()) {
        return elementwiseComplex(
            a, b, [](const keel::c64 &x, const keel::c64 &y) { return keel::pow(x, y); }, p);
    }
    if (a.type() == ValueType::DOUBLE && b.type() == ValueType::DOUBLE) {
        if (b.isScalar() && b.toScalar() == 2.0)
            return times(a, a, p);
        if (powNeedsComplex(a, b)) {
            Value ca = a, cb = b;
            ca.promoteToComplex(p);
            cb.promoteToComplex(p);
            return elementwiseComplex(
                ca, cb, [](const keel::c64 &x, const keel::c64 &y) { return keel::pow(x, y); }, p);
        }
        return elementwiseDouble(a, b, [](keel::f64 x, keel::f64 y) { return std::pow(x, y); }, p);
    }
    {
        auto r = dispatchIntegerBinaryOp(a, b, [](auto x, auto y) -> decltype(x) {
            keel::f64 r = std::pow(keel::as<keel::f64>(x), keel::as<keel::f64>(y));
            if constexpr (std::is_integral_v<decltype(x)>) {
                r = std::round(r);
                if (r > keel::as<keel::f64>(std::numeric_limits<decltype(x)>::max()))
                    return std::numeric_limits<decltype(x)>::max();
                if (r < keel::as<keel::f64>(std::numeric_limits<decltype(x)>::min()))
                    return std::numeric_limits<decltype(x)>::min();
                return static_cast<decltype(x)>(r);
            } else {
                return static_cast<decltype(x)>(r);
            }
        }, p);
        if (!r.isUnset()) return r;
    }
    throw Error("Unsupported types for .^", 0, 0, "elementPower", "",
                 "numkit:elementPower:unsupportedTypes");
}

Value mpower(const Value &a, const Value &b, keel::Allocator *mr)
{
    return power(a, b, mr);
}

// ── Comparisons ──────────────────────────────────────────────────────────

Value eq(const Value &a, const Value &b, keel::Allocator *) { return compareImpl(Cmp::EQ, a, b); }
Value ne(const Value &a, const Value &b, keel::Allocator *) { return compareImpl(Cmp::NE, a, b); }
Value lt(const Value &a, const Value &b, keel::Allocator *) { return compareImpl(Cmp::LT, a, b); }
Value gt(const Value &a, const Value &b, keel::Allocator *) { return compareImpl(Cmp::GT, a, b); }
Value le(const Value &a, const Value &b, keel::Allocator *) { return compareImpl(Cmp::LE, a, b); }
Value ge(const Value &a, const Value &b, keel::Allocator *) { return compareImpl(Cmp::GE, a, b); }

// ── Logical ──────────────────────────────────────────────────────────────

Value logical_and(const Value &a, const Value &b, keel::Allocator *mr)
{
    return logicalBinary("&", [](bool x, bool y) { return x && y; }, mr, a, b);
}

Value logical_or(const Value &a, const Value &b, keel::Allocator *mr)
{
    return logicalBinary("|", [](bool x, bool y) { return x || y; }, mr, a, b);
}

Value logical_xor(const Value &a, const Value &b, keel::Allocator *mr)
{
    return logicalBinary("xor", [](bool x, bool y) { return (x || y) && !(x && y); }, mr, a, b);
}

} // namespace numkit::builtin
