// src/builtin/src/ops/binary_ops.cpp

#include <numkit/builtin/ops.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>
#include <numkit/ops/la_solve.hpp>
#include <numkit/ops/binary_ops.hpp>
#include <numkit/ops/compare.hpp>
#include <numkit/linalg/eig.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory_resource>
#include <string>

#include "binary_ops_detail.hpp"

namespace numkit::builtin {

// ════════════════════════════════════════════════════════════════════════
// Public API — binary operators
// ════════════════════════════════════════════════════════════════════════

// ── Arithmetic ──────────────────────────────────────────────────────────

namespace {

bool isTfStruct(const Value &v)
{
    return v.isStruct() && v.numel() == 1 && v.hasField("kind")
           && (v.field("kind").isChar() || v.field("kind").isString())
           && v.field("kind").toString() == "tf"
           && v.hasField("num") && v.hasField("den");
}

bool isTfOrScalarGain(const Value &v)
{
    return isTfStruct(v) || (v.isNumeric() && !v.isComplex() && v.numel() == 1);
}

struct TfPolyPair {
    std::vector<double> num;
    std::vector<double> den;
    double Ts = -1.0;
};

TfPolyPair extractTfPoly(const Value &v)
{
    if (isTfStruct(v)) {
        const Value &nv = v.field("num");
        const Value &dv = v.field("den");
        TfPolyPair out;
        out.num.resize(nv.numel());
        out.den.resize(dv.numel());
        for (size_t i = 0; i < nv.numel(); ++i) out.num[i] = nv.elemAsDouble(i);
        for (size_t i = 0; i < dv.numel(); ++i) out.den[i] = dv.elemAsDouble(i);
        out.Ts = v.hasField("Ts") ? v.field("Ts").toScalar() : 0.0;
        return out;
    }
    return {{v.toScalar()}, {1.0}, -1.0};
}

double combineTfTs(double Ts1, double Ts2, const char *op)
{
    if (Ts1 == 0.0 && Ts2 == 0.0) return 0.0;
    if (Ts1 == Ts2) return Ts1;
    if (Ts1 == -1.0) return Ts2;
    if (Ts2 == -1.0) return Ts1;
    throw Error(std::string(op) + ": transfer function sample times must match",
                0, 0, op, "", std::string("numkit:") + op + ":TsMismatch");
}

std::vector<double> polyConv(const std::vector<double> &a, const std::vector<double> &b)
{
    if (a.empty() || b.empty()) return {};
    std::vector<double> y(a.size() + b.size() - 1, 0.0);
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            y[i + j] += a[i] * b[j];
    return y;
}

std::vector<double> polyAddRightAligned(const std::vector<double> &a,
                                        const std::vector<double> &b,
                                        double bSign)
{
    const size_t na = a.size(), nb = b.size();
    const size_t n = std::max(na, nb);
    std::vector<double> y(n, 0.0);
    for (size_t i = 0; i < na; ++i) y[n - na + i] += a[i];
    for (size_t i = 0; i < nb; ++i) y[n - nb + i] += bSign * b[i];
    return y;
}

Value buildTfValue(const std::vector<double> &num, const std::vector<double> &den,
                   double Ts, std::pmr::memory_resource *mr)
{
    Value nv = Value::matrix(1, num.size(), ValueType::DOUBLE, mr);
    if (!num.empty()) std::copy(num.begin(), num.end(), nv.doubleDataMut());
    Value dv = Value::matrix(1, den.size(), ValueType::DOUBLE, mr);
    if (!den.empty()) std::copy(den.begin(), den.end(), dv.doubleDataMut());
    Value s = Value::structure(mr);
    s.field("kind") = Value::fromString("tf", mr);
    s.field("Ts") = Value::scalar(Ts, mr);
    s.field("num") = std::move(nv);
    s.field("den") = std::move(dv);
    return s;
}

} // namespace

Value plus(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    std::pmr::memory_resource *p = mr;
    if ((isTfStruct(a) || isTfStruct(b)) && isTfOrScalarGain(a) && isTfOrScalarGain(b)) {
        auto ta = extractTfPoly(a);
        auto tb = extractTfPoly(b);
        auto num = polyAddRightAligned(polyConv(ta.num, tb.den), polyConv(tb.num, ta.den), +1.0);
        auto den = polyConv(ta.den, tb.den);
        return buildTfValue(num, den, combineTfTs(ta.Ts, tb.Ts, "plus"), p);
    }
    if (a.isEmpty() || b.isEmpty())
        return emptyArithResult(a, b, p);
    // MATLAB auto-coerces logical to double for arithmetic. See BUGS.md #24.
    if (a.type() == ValueType::LOGICAL || b.type() == ValueType::LOGICAL)
        return plus(coerceLogicalToDouble(a, p), coerceLogicalToDouble(b, p), p);
    if (a.isComplex() || b.isComplex())
        return elementwiseComplex(a, b, std::plus<Complex>{}, p);
    if (a.type() == ValueType::DOUBLE && b.type() == ValueType::DOUBLE) {
        if (sameShapeDoubleFastPath(a, b)) {
            auto r = createLike(a, ValueType::DOUBLE, p);
            ops::detail::plusLoop(a.doubleData(), b.doubleData(), r.doubleDataMut(), a.numel());
            return r;
        }
        return elementwiseDouble(a, b, std::plus<double>{}, p);
    }
    if (a.isChar() && b.isChar())
        return Value::fromString(a.toString() + b.toString(), p);
    if (a.isChar() && b.type() == ValueType::DOUBLE) {
        auto ca = createLike(a, ValueType::DOUBLE, p);
        const char *cd = a.charData();
        double *dd = ca.doubleDataMut();
        for (size_t i = 0; i < a.numel(); ++i)
            dd[i] = static_cast<double>(static_cast<unsigned char>(cd[i]));
        return elementwiseDouble(ca, b, std::plus<double>{}, p);
    }
    if (a.type() == ValueType::DOUBLE && b.isChar()) {
        auto cb = createLike(b, ValueType::DOUBLE, p);
        const char *cd = b.charData();
        double *dd = cb.doubleDataMut();
        for (size_t i = 0; i < b.numel(); ++i)
            dd[i] = static_cast<double>(static_cast<unsigned char>(cd[i]));
        return elementwiseDouble(a, cb, std::plus<double>{}, p);
    }
    if (a.isString() && b.isString())
        return Value::stringScalar(a.toString() + b.toString(), p);
    if (a.isString() && b.isChar())
        return Value::stringScalar(a.toString() + b.toString(), p);
    if (a.isChar() && b.isString())
        return Value::stringScalar(a.toString() + b.toString(), p);
    if (a.isString() && b.isNumeric())
        return Value::stringScalar(a.toString() + std::to_string(b.toScalar()), p);
    if (a.isNumeric() && b.isString())
        return Value::stringScalar(std::to_string(a.toScalar()) + b.toString(), p);
    if (a.type() == ValueType::CHAR || b.type() == ValueType::CHAR)
        return plus(coerceCharToDouble(a, p), coerceCharToDouble(b, p), p);
    {
        auto r = dispatchIntegerBinaryOp(a, b, [](auto x, auto y) { return saturateAdd(x, y); }, p);
        if (!r.isUnset()) return r;
    }
    throw Error("Unsupported types for +", 0, 0, "plus", "", "numkit:plus:unsupportedTypes");
}

Value minus(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    std::pmr::memory_resource *p = mr;
    if ((isTfStruct(a) || isTfStruct(b)) && isTfOrScalarGain(a) && isTfOrScalarGain(b)) {
        auto ta = extractTfPoly(a);
        auto tb = extractTfPoly(b);
        auto num = polyAddRightAligned(polyConv(ta.num, tb.den), polyConv(tb.num, ta.den), -1.0);
        auto den = polyConv(ta.den, tb.den);
        return buildTfValue(num, den, combineTfTs(ta.Ts, tb.Ts, "minus"), p);
    }
    if (a.isEmpty() || b.isEmpty())
        return emptyArithResult(a, b, p);
    if (a.type() == ValueType::LOGICAL || b.type() == ValueType::LOGICAL)
        return minus(coerceLogicalToDouble(a, p), coerceLogicalToDouble(b, p), p);
    if (a.type() == ValueType::CHAR || b.type() == ValueType::CHAR)
        return minus(coerceCharToDouble(a, p), coerceCharToDouble(b, p), p);
    if (a.isComplex() || b.isComplex())
        return elementwiseComplex(a, b, std::minus<Complex>{}, p);
    if (a.type() == ValueType::DOUBLE && b.type() == ValueType::DOUBLE) {
        if (sameShapeDoubleFastPath(a, b)) {
            auto r = createLike(a, ValueType::DOUBLE, p);
            ops::detail::minusLoop(a.doubleData(), b.doubleData(), r.doubleDataMut(), a.numel());
            return r;
        }
        return elementwiseDouble(a, b, std::minus<double>{}, p);
    }
    {
        auto r = dispatchIntegerBinaryOp(a, b, [](auto x, auto y) { return saturateSub(x, y); }, p);
        if (!r.isUnset()) return r;
    }
    throw Error("Unsupported types for -", 0, 0, "minus", "", "numkit:minus:unsupportedTypes");
}

Value times(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    std::pmr::memory_resource *p = mr;
    if (a.isEmpty() || b.isEmpty())
        return emptyArithResult(a, b, p);
    if (a.type() == ValueType::LOGICAL || b.type() == ValueType::LOGICAL)
        return times(coerceLogicalToDouble(a, p), coerceLogicalToDouble(b, p), p);
    if (a.type() == ValueType::CHAR || b.type() == ValueType::CHAR)
        return times(coerceCharToDouble(a, p), coerceCharToDouble(b, p), p);
    if (a.isComplex() || b.isComplex())
        return elementwiseComplex(a, b, std::multiplies<Complex>{}, p);
    if (a.type() == ValueType::DOUBLE && b.type() == ValueType::DOUBLE) {
        if (sameShapeDoubleFastPath(a, b)) {
            auto r = createLike(a, ValueType::DOUBLE, p);
            ops::detail::timesLoop(a.doubleData(), b.doubleData(), r.doubleDataMut(), a.numel());
            return r;
        }
        return elementwiseDouble(a, b, std::multiplies<double>{}, p);
    }
    {
        auto r = dispatchIntegerBinaryOp(a, b, [](auto x, auto y) { return saturateMul(x, y); }, p);
        if (!r.isUnset()) return r;
    }
    throw Error("Unsupported types for .*", 0, 0, "times", "", "numkit:times:unsupportedTypes");
}

Value mtimes(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    std::pmr::memory_resource *p = mr;

    if ((isTfStruct(a) || isTfStruct(b)) && isTfOrScalarGain(a) && isTfOrScalarGain(b)) {
        auto ta = extractTfPoly(a);
        auto tb = extractTfPoly(b);
        auto num = polyConv(ta.num, tb.num);
        auto den = polyConv(ta.den, tb.den);
        return buildTfValue(num, den, combineTfTs(ta.Ts, tb.Ts, "mtimes"), p);
    }

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
            return elementwiseComplex(a, b, std::multiplies<Complex>{}, p);
        size_t M = ca.dims().rows(), K = ca.dims().cols(), N = cb.dims().cols();
        if (K != cb.dims().rows())
            throw Error("Inner matrix dimensions must agree", 0, 0, "mtimes", "",
                         "numkit:innerdim");
        auto r = Value::complexMatrix(M, N, p);
        if (ca.isSingleComplex() || cb.isSingleComplex())
            r.setSingleComplex(true);
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j) {
                Complex s(0, 0);
                for (size_t k = 0; k < K; ++k)
                    s += ca.complexElem(i, k) * cb.complexElem(k, j);
                r.complexDataMut()[j * M + i] = s;
            }
        return narrowComplex(r, p);
    }

    if (a.isScalar() || b.isScalar()) {
        auto r = dispatchIntegerBinaryOp(a, b, [](auto x, auto y) { return saturateMul(x, y); }, p);
        if (!r.isUnset()) return r;
        return elementwiseDouble(a, b, std::multiplies<double>{}, p);
    }
    if ((a.type() == ValueType::SINGLE || b.type() == ValueType::SINGLE) &&
        (a.type() == ValueType::SINGLE || a.type() == ValueType::DOUBLE) &&
        (b.type() == ValueType::SINGLE || b.type() == ValueType::DOUBLE)) {
        size_t M = a.dims().rows(), K = a.dims().cols(), N = b.dims().cols();
        if (K != b.dims().rows())
            throw Error("Inner matrix dimensions must agree", 0, 0, "mtimes", "",
                         "numkit:innerdim");
        auto r = Value::matrix(M, N, ValueType::SINGLE, p);
        float *rd = static_cast<float *>(r.rawDataMut());
        std::fill(rd, rd + M * N, 0.0f);
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < K; ++k) {
                float bkj = static_cast<float>(b.elemAsDouble(j * K + k));
                for (size_t i = 0; i < M; ++i) {
                    rd[j * M + i] += static_cast<float>(a.elemAsDouble(k * M + i)) * bkj;
                }
            }
        }
        return r;
    }
    if (a.type() == ValueType::DOUBLE && b.type() == ValueType::DOUBLE) {
        size_t M = a.dims().rows(), K = a.dims().cols(), N = b.dims().cols();
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

Value rdivide(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    std::pmr::memory_resource *p = mr;
    if (a.isEmpty() || b.isEmpty())
        return emptyArithResult(a, b, p);
    if (a.type() == ValueType::LOGICAL || b.type() == ValueType::LOGICAL)
        return rdivide(coerceLogicalToDouble(a, p), coerceLogicalToDouble(b, p), p);
    if (a.type() == ValueType::CHAR || b.type() == ValueType::CHAR)
        return rdivide(coerceCharToDouble(a, p), coerceCharToDouble(b, p), p);
    if (a.isComplex() || b.isComplex())
        return elementwiseComplex(a, b, std::divides<Complex>{}, p);
    if (a.type() == ValueType::DOUBLE && b.type() == ValueType::DOUBLE) {
        if (sameShapeDoubleFastPath(a, b)) {
            auto r = createLike(a, ValueType::DOUBLE, p);
            ops::detail::rdivideLoop(a.doubleData(), b.doubleData(), r.doubleDataMut(), a.numel());
            return r;
        }
        return elementwiseDouble(a, b, std::divides<double>{}, p);
    }
    {
        auto r = dispatchIntegerBinaryOp(a, b, [](auto x, auto y) { return saturateDiv(x, y); }, p);
        if (!r.isUnset()) return r;
    }
    throw Error("Unsupported types for ./", 0, 0, "rdivide", "",
                 "numkit:rdivide:unsupportedTypes");
}

Value ldivide(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    return rdivide(b, a, mr);
}

Value mrdivide(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    std::pmr::memory_resource *p = mr;
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
            return elementwiseComplex(a, b, std::divides<Complex>{}, p);
        return elementwiseDouble(a, b, std::divides<double>{}, p);
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

Value mldivide(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    std::pmr::memory_resource *p = mr;
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
            Complex ca = a.isComplex() ? a.complexData()[0] : Complex(a.doubleData()[0], 0.0);
            Complex cb = b.isComplex() ? b.complexData()[0] : Complex(b.doubleData()[0], 0.0);
            Complex res = cb / ca;
            return narrowIfReal(Value::complexScalar(res.real(), res.imag(), p), p);
        }
        return Value::scalar(b.toScalar() / a.toScalar(), p);
    }
    if (a.isScalar() && !b.isScalar()) {
        if (a.isComplex() || b.isComplex())
            return elementwiseComplex(b, a, std::divides<Complex>{}, p);
        return elementwiseDouble(b, a, std::divides<double>{}, p);
    }
    return matrixSolve(a, b, "mldivide", p);
}

static bool powNeedsComplex(const Value &a, const Value &b)
{
    if (a.type() != ValueType::DOUBLE || b.type() != ValueType::DOUBLE)
        return false;
    auto negBase = [](double x) { return x < 0.0; };
    auto fracExp = [](double y) { return y != std::floor(y); };
    if (a.isScalar() && b.isScalar())
        return negBase(a.toScalar()) && fracExp(b.toScalar());
    if (b.isScalar()) {
        if (!fracExp(b.toScalar())) return false;
        const double *da = a.doubleData();
        for (std::size_t i = 0; i < a.numel(); ++i)
            if (negBase(da[i])) return true;
        return false;
    }
    if (a.isScalar()) {
        if (!negBase(a.toScalar())) return false;
        const double *db = b.doubleData();
        for (std::size_t i = 0; i < b.numel(); ++i)
            if (fracExp(db[i])) return true;
        return false;
    }
    if (a.dims() == b.dims()) {
        const double *da = a.doubleData(), *db = b.doubleData();
        for (std::size_t i = 0; i < a.numel(); ++i)
            if (negBase(da[i]) && fracExp(db[i])) return true;
    }
    return false;
}

Value power(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    std::pmr::memory_resource *p = mr;
    if (a.isEmpty() || b.isEmpty())
        return emptyArithResult(a, b, p);
    if (a.type() == ValueType::LOGICAL || b.type() == ValueType::LOGICAL)
        return power(coerceLogicalToDouble(a, p), coerceLogicalToDouble(b, p), p);
    if (a.type() == ValueType::CHAR || b.type() == ValueType::CHAR)
        return power(coerceCharToDouble(a, p), coerceCharToDouble(b, p), p);
    if (a.isScalar() && b.isScalar()) {
        if (a.isComplex() || b.isComplex()) {
            auto [ca, cb] = promoteToComplex(a, b, p);
            return Value::complexScalar(std::pow(ca.toComplex(), cb.toComplex()), p);
        }
        const double base = a.toScalar(), e = b.toScalar();
        if (base < 0.0 && e != std::floor(e))
            return Value::complexScalar(std::pow(Complex(base, 0.0), e), p);
        return Value::scalar(std::pow(base, e), p);
    }
    if (b.isScalar() && !a.isScalar()
        && a.dims().rows() == a.dims().cols()
        && a.dims().ndim() == 2) {
        const std::size_t R = a.dims().rows();
        const bool bCplx = b.isComplex();
        const double bs = bCplx ? b.toComplex().real() : b.toScalar();
        const bool bRealInt = (!bCplx || b.toComplex().imag() == 0.0)
                              && std::isfinite(bs) && std::floor(bs) == bs
                              && std::abs(bs) <= 1e9;
        if (bRealInt) {
            const long n = static_cast<long>(bs);
            auto makeEye = [&]() {
                auto I = Value::matrix(R, R, ValueType::DOUBLE, p);
                double *id = I.doubleDataMut();
                std::memset(id, 0, sizeof(double) * R * R);
                for (std::size_t i = 0; i < R; ++i) id[i * R + i] = 1.0;
                return I;
            };
            if (n == 0) return makeEye();
            long expAbs = n > 0 ? n : -n;
            Value acc = a;
            for (long k = 1; k < expAbs; ++k) acc = mtimes(acc, a, p);
            if (n < 0) return mldivide(acc, makeEye(), p);
            return acc;
        }
        // Fractional or complex scalar exponent on square matrix A (R x R)
        const Complex expC = bCplx ? b.toComplex() : Complex(bs, 0.0);
        const bool aCplx = a.isComplex();
        // Fast path for diagonal matrices (e.g. D^(-0.5))
        bool isDiag = true;
        for (std::size_t j = 0; j < R && isDiag; ++j) {
            for (std::size_t i = 0; i < R; ++i) {
                if (i == j) continue;
                Complex v = aCplx ? a.complexElem(i + j * R) : Complex(a.elemAsDouble(i + j * R), 0.0);
                if (v.real() != 0.0 || v.imag() != 0.0) {
                    isDiag = false;
                    break;
                }
            }
        }
        if (isDiag) {
            Value outC = Value::matrix(R, R, ValueType::COMPLEX, p);
            Complex *od = outC.complexDataMut();
            std::fill(od, od + R * R, Complex(0.0, 0.0));
            bool anyImag = false;
            for (std::size_t i = 0; i < R; ++i) {
                Complex d = aCplx ? a.complexElem(i + i * R) : Complex(a.elemAsDouble(i + i * R), 0.0);
                Complex dp = (!aCplx && !bCplx && d.real() >= 0.0)
                                 ? Complex(std::pow(d.real(), bs), 0.0)
                                 : std::pow(d, expC);
                od[i + i * R] = dp;
                if (dp.imag() != 0.0) anyImag = true;
            }
            if (!anyImag && !aCplx && !bCplx) {
                Value outR = Value::matrix(R, R, ValueType::DOUBLE, p);
                double *rd = outR.doubleDataMut();
                std::memset(rd, 0, sizeof(double) * R * R);
                for (std::size_t i = 0; i < R; ++i) rd[i + i * R] = od[i + i * R].real();
                return outR;
            }
            return outC;
        }
        // General square matrix: eigendecomposition A = V * D * V^-1 => A^b = (V * D^b) / V
        auto [V, D] = ::numkit::linalg::eig_general_VD(a, p);
        Value Vc = Value::matrix(R, R, ValueType::COMPLEX, p);
        Complex *vcd = Vc.complexDataMut();
        const bool vIsCplx = V.isComplex();
        const bool dIsCplx = D.isComplex();
        for (std::size_t j = 0; j < R; ++j) {
            Complex dj = dIsCplx ? D.complexElem(j + j * R) : Complex(D.elemAsDouble(j + j * R), 0.0);
            Complex dp = (!dIsCplx && !bCplx && dj.real() >= 0.0)
                             ? Complex(std::pow(dj.real(), bs), 0.0)
                             : std::pow(dj, expC);
            for (std::size_t i = 0; i < R; ++i) {
                Complex vij = vIsCplx ? V.complexElem(i + j * R) : Complex(V.elemAsDouble(i + j * R), 0.0);
                vcd[i + j * R] = vij * dp;
            }
        }
        Value res = mrdivide(Vc, V, p);
        if (!aCplx && !bCplx && res.isComplex()) {
            const Complex *rc = res.complexData();
            double maxRe = 0.0, maxIm = 0.0;
            for (std::size_t k = 0; k < R * R; ++k) {
                maxRe = std::max(maxRe, std::abs(rc[k].real()));
                maxIm = std::max(maxIm, std::abs(rc[k].imag()));
            }
            if (maxIm <= 1e-12 * std::max(1.0, maxRe)) {
                Value outR = Value::matrix(R, R, ValueType::DOUBLE, p);
                double *rd = outR.doubleDataMut();
                for (std::size_t k = 0; k < R * R; ++k) rd[k] = rc[k].real();
                return outR;
            }
        }
        return res;
    }
    throw Error("Matrix power not implemented", 0, 0, "power", "",
                 "numkit:power:notImplemented");
}

Value elementPower(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    std::pmr::memory_resource *p = mr;
    if (a.isEmpty() || b.isEmpty())
        return emptyArithResult(a, b, p);
    if (a.type() == ValueType::LOGICAL || b.type() == ValueType::LOGICAL)
        return elementPower(coerceLogicalToDouble(a, p), coerceLogicalToDouble(b, p), p);
    if (a.type() == ValueType::CHAR || b.type() == ValueType::CHAR)
        return elementPower(coerceCharToDouble(a, p), coerceCharToDouble(b, p), p);
    if (a.isComplex() || b.isComplex()) {
        return elementwiseComplex(
            a, b, [](const Complex &x, const Complex &y) { return std::pow(x, y); }, p);
    }
    if (a.type() == ValueType::DOUBLE && b.type() == ValueType::DOUBLE) {
        if (b.isScalar() && b.toScalar() == 2.0)
            return times(a, a, p);
        if (powNeedsComplex(a, b)) {
            Value ca = a, cb = b;
            ca.promoteToComplex(p);
            cb.promoteToComplex(p);
            return elementwiseComplex(
                ca, cb, [](const Complex &x, const Complex &y) { return std::pow(x, y); }, p);
        }
        return elementwiseDouble(a, b, [](double x, double y) { return std::pow(x, y); }, p);
    }
    {
        auto r = dispatchIntegerBinaryOp(a, b, [](auto x, auto y) -> decltype(x) {
            double r = std::pow(static_cast<double>(x), static_cast<double>(y));
            if constexpr (std::is_integral_v<decltype(x)>) {
                r = std::round(r);
                if (r > static_cast<double>(std::numeric_limits<decltype(x)>::max()))
                    return std::numeric_limits<decltype(x)>::max();
                if (r < static_cast<double>(std::numeric_limits<decltype(x)>::min()))
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

Value mpower(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    return power(a, b, mr);
}

// ── Comparisons ──────────────────────────────────────────────────────────

Value eq(const Value &a, const Value &b, std::pmr::memory_resource *) { return compareImpl(Cmp::EQ, a, b); }
Value ne(const Value &a, const Value &b, std::pmr::memory_resource *) { return compareImpl(Cmp::NE, a, b); }
Value lt(const Value &a, const Value &b, std::pmr::memory_resource *) { return compareImpl(Cmp::LT, a, b); }
Value gt(const Value &a, const Value &b, std::pmr::memory_resource *) { return compareImpl(Cmp::GT, a, b); }
Value le(const Value &a, const Value &b, std::pmr::memory_resource *) { return compareImpl(Cmp::LE, a, b); }
Value ge(const Value &a, const Value &b, std::pmr::memory_resource *) { return compareImpl(Cmp::GE, a, b); }

// ── Logical ──────────────────────────────────────────────────────────────

Value logical_and(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    return logicalBinary("&", [](bool x, bool y) { return x && y; }, mr, a, b);
}

Value logical_or(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    return logicalBinary("|", [](bool x, bool y) { return x || y; }, mr, a, b);
}

Value logical_xor(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    return logicalBinary("xor", [](bool x, bool y) { return (x || y) && !(x && y); }, mr, a, b);
}

} // namespace numkit::builtin
