// toolboxes/linalg/src/matrix_functions.cpp
//
// Matrix functions — expm, logm, sqrtm, sylvester, expmv, funm
// Value adapters over keel::linalg kernels.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/linalg.h>
#include <keel/math.h>
#include <numkit/linalg/matrix_functions.hpp>
#include <numkit/linalg/eig.hpp>
#include "linalg_detail.hpp"

// Compute-only TU: Value substrate + Error, no engine. The expm/logm/sqrtm/
// expmv builtins (CallContext wrappers) live in matrix_functions_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>


namespace numkit::linalg {

namespace {

void copyValueToComplex(const Value &V, keel::usize count, keel::c64 *out)
{
    if (V.isComplex()) {
        keel::copy(V.complexData(), V.complexData() + count, out);
    } else {
        const keel::f64 *vd = V.doubleData();
        for (keel::usize i = 0; i < count; ++i) out[i] = keel::c64(vd[i], 0.0);
    }
}

// Apply scalar function f to symmetric A's eigenvalues and reconstruct:
//   result = V * diag(f(eig)) * V'
Value applyScalarFnSym(const Value &A, keel::f64 (*f)(keel::f64),
                       const char *fnName, const char *errId,
                       keel::Allocator *mr)
{
    const bool wantSingle = A.type() == ValueType::SINGLE;
    Value Aw = detail::widenSingleToDouble(A, mr);
    auto [V, D] = eig_symmetric(Aw, mr);
    const keel::usize n = keel::as<keel::usize>(D.dims().dim(0));
    if (n == 0) return Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    const keel::f64 *Vdata = V.doubleData();
    const keel::f64 *Ddata = D.doubleData();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> VfD(n * n, scratch.allocator());
    for (keel::usize k = 0; k < n; ++k) {
        const keel::f64 e = Ddata[k + k * n];
        const keel::f64 fe = f(e);
        if (!keel::is_finite(fe))
            throw Error(keel::format("{}: eigenvalue out of domain (got {})", fnName, e),
                        0, 0, fnName, "", errId);
        for (keel::usize i = 0; i < n; ++i) VfD[i + k * n] = Vdata[i + k * n] * fe;
    }

    auto out = Value::matrix(n, n, ValueType::DOUBLE, mr);
    keel::linalg::gemm(keel::linalg::MatrixTranspose::NoTrans, keel::linalg::MatrixTranspose::Trans,
               n, n, n, 1.0, VfD.data(), n, Vdata, n, 0.0, out.doubleDataMut(), n);
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

} // anonymous namespace

Value expm(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "expm");
    if (A.dims().ndim() != 2)
        throw Error("expm: input must be a 2D matrix",
                    0, 0, "expm", "", "numkit:expm:notMatrix");
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(1));
    if (m != n)
        throw Error("expm: matrix must be square",
                    0, 0, "expm", "", "numkit:expm:notSquare");
    if (n == 0) return Value::matrix(0, 0, A.type(), mr);

    const bool wantSingle = A.type() == ValueType::SINGLE;
    Value Aw = detail::widenSingleToDouble(A, mr);

    if (Aw.isComplex()) {
        auto out = Value::complexMatrix(n, n, mr);
        if (!keel::linalg::expm(Aw.complexData(), n, out.complexDataMut(), mr))
            throw Error("expm: Padé denominator is singular",
                        0, 0, "expm", "", "numkit:expm:singular");
        return detail::narrow_if_real(out, mr);
    }

    auto out = Value::matrix(n, n, ValueType::DOUBLE, mr);
    if (!keel::linalg::expm(Aw.doubleData(), n, out.doubleDataMut(), mr))
        throw Error("expm: Padé denominator is singular",
                    0, 0, "expm", "", "numkit:expm:singular");
    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

Value logm_sym(const Value &A, keel::Allocator *mr)
{
    return applyScalarFnSym(A, [](keel::f64 x) { return keel::log(x); },
                            "logm", "numkit:logm:negativeEigenvalue", mr);
}

Value sqrtm_sym(const Value &A, keel::Allocator *mr)
{
    return applyScalarFnSym(A, [](keel::f64 x) { return keel::sqrt(x); },
                            "sqrtm", "numkit:sqrtm:negativeEigenvalue", mr);
}

Value sqrtm(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "sqrtm");
    if (A.dims().ndim() != 2)
        throw Error("sqrtm: input must be a 2D matrix", 0, 0, "sqrtm", "", "numkit:sqrtm:notMatrix");
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(0));
    if (n != keel::as<keel::usize>(A.dims().dim(1)))
        throw Error("sqrtm: matrix must be square", 0, 0, "sqrtm", "", "numkit:sqrtm:notSquare");
    const bool wantSingle = A.type() == ValueType::SINGLE;
    if (n == 0) return Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    // Closed-form fast paths for 1x1 and 2x2 real matrices with positive real principal root.
    if (!A.isComplex()) {
        if (n == 1) {
            const keel::f64 a = A.elemAsDouble(0);
            if (keel::is_finite(a) && a >= 0.0) {
                Value out = Value::scalar(keel::sqrt(a), mr);
                return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
            }
        } else if (n == 2) {
            const keel::f64 a = A.elemAsDouble(0);
            const keel::f64 c = A.elemAsDouble(1);
            const keel::f64 b = A.elemAsDouble(2);
            const keel::f64 d = A.elemAsDouble(3);
            if (keel::is_finite(a) && keel::is_finite(b) && keel::is_finite(c) && keel::is_finite(d)) {
                const keel::f64 det = a * d - b * c;
                if (det >= 0.0) {
                    const keel::f64 s = keel::sqrt(det);
                    const keel::f64 tau = a + d + 2.0 * s;
                    if (tau > 0.0) {
                        const keel::f64 invT = 1.0 / keel::sqrt(tau);
                        Value out = Value::matrix(2, 2, ValueType::DOUBLE, mr);
                        keel::f64 *o = out.doubleDataMut();
                        o[0] = (a + s) * invT;
                        o[1] = c * invT;
                        o[2] = b * invT;
                        o[3] = (d + s) * invT;
                        return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
                    }
                }
            }
        }
    }

    Value Aw = detail::widenSingleToDouble(A, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> Ac(n * n, scratch.allocator());
    copyValueToComplex(Aw, n * n, Ac.data());

    auto out = Value::complexMatrix(n, n, mr);
    if (!keel::linalg::sqrtm(Ac.data(), n, out.complexDataMut(), mr))
        throw Error("sqrtm: Schur iteration failed to converge", 0, 0, "sqrtm", "", "numkit:sqrtm:noConverge");

    return detail::narrowDoubleToSingleIf(detail::narrow_if_real(out, mr), wantSingle, mr);
}

Value sylvester(const Value &A, const Value &B, const Value &C, keel::Allocator *mr)
{
    for (const Value *v : {&A, &B, &C}) {
        const ValueType vt = v->type();
        if (vt != ValueType::DOUBLE && vt != ValueType::COMPLEX
            && vt != ValueType::SINGLE)
            throw Error(keel::format("sylvester: inputs must be single or double (got a {} array).",
                                     mtypeName(vt)),
                        0, 0, "sylvester", "", "numkit:sylvester:inputType");
    }
    const bool wantSingle = A.type() == ValueType::SINGLE
        || B.type() == ValueType::SINGLE || C.type() == ValueType::SINGLE;
    for (const Value *v : {&A, &B, &C}) {
        const bool cx = v->isComplex();
        const keel::usize n = v->numel();
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 re = cx ? v->complexElem(i).real()
                                 : v->elemAsDouble(i);
            const keel::f64 im = cx ? v->complexElem(i).imag() : 0.0;
            if (!keel::is_finite(re) || !keel::is_finite(im))
                throw Error("sylvester: inputs must be finite (NaN or Inf "
                            "found).",
                            0, 0, "sylvester", "",
                            "numkit:sylvester:inputWithNaNInf");
        }
    }
    if (A.dims().ndim() != 2 || B.dims().ndim() != 2 || C.dims().ndim() != 2)
        throw Error("sylvester: inputs must be 2D matrices", 0, 0, "sylvester", "", "numkit:sylvester:notMatrix");
    const keel::usize m = keel::as<keel::usize>(A.dims().dim(0));
    if (m != keel::as<keel::usize>(A.dims().dim(1)))
        throw Error(keel::format("sylvester: first and second inputs must be square (A is {}x{}).",
                                 A.dims().dim(0), A.dims().dim(1)),
                    0, 0, "sylvester", "", "numkit:sylvester:inputMustBeSquare");
    const keel::usize n = keel::as<keel::usize>(B.dims().dim(0));
    if (n != keel::as<keel::usize>(B.dims().dim(1)))
        throw Error(keel::format("sylvester: first and second inputs must be square (B is {}x{}).",
                                 B.dims().dim(0), B.dims().dim(1)),
                    0, 0, "sylvester", "", "numkit:sylvester:inputMustBeSquare");
    if (keel::as<keel::usize>(C.dims().dim(0)) != m || keel::as<keel::usize>(C.dims().dim(1)) != n)
        throw Error(keel::format("sylvester: inputs must have compatible size (C must be {}x{} to match A and B).",
                                 m, n),
                    0, 0, "sylvester", "", "numkit:sylvester:inputMustBeCompatibleSize");
    if (m == 0 || n == 0)
        throw Error("sylvester: inputs must have compatible size (empty "
                    "operand).",
                    0, 0, "sylvester", "",
                    "numkit:sylvester:inputMustBeCompatibleSize");

    const Value Aw = detail::widenSingleToDouble(A, mr);
    const Value Bw = detail::widenSingleToDouble(B, mr);
    const Value Cw = detail::widenSingleToDouble(C, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> Ac(m * m, scratch.allocator());
    keel::Array<keel::c64> Bc(n * n, scratch.allocator());
    keel::Array<keel::c64> Cc(m * n, scratch.allocator());
    copyValueToComplex(Aw, m * m, Ac.data());
    copyValueToComplex(Bw, n * n, Bc.data());
    copyValueToComplex(Cw, m * n, Cc.data());

    auto out = Value::complexMatrix(m, n, mr);
    if (!keel::linalg::sylvester(Ac.data(), m, Bc.data(), n, Cc.data(), out.complexDataMut(), mr)) {
        const keel::f64 qnan = keel::Limits<keel::f64>::quiet_nan();
        if (!A.isComplex() && !B.isComplex() && !C.isComplex()) {
            Value r = Value::matrix(m, n, ValueType::DOUBLE, mr);
            keel::f64 *rd = r.doubleDataMut();
            for (keel::usize i = 0; i < m * n; ++i) rd[i] = qnan;
            return wantSingle ? detail::narrowDoubleToSingle(r, mr) : r;
        }
        keel::c64 *cd = out.complexDataMut();
        for (keel::usize i = 0; i < m * n; ++i) cd[i] = keel::c64(qnan, qnan);
        return out;
    }

    Value res = detail::narrow_if_real(keel::move(out), mr);
    if (!A.isComplex() && !B.isComplex() && !C.isComplex()
        && res.isComplex()) {
        Value r = Value::matrix(m, n, ValueType::DOUBLE, mr);
        const keel::c64 *cd = res.complexData();
        keel::f64 *rd = r.doubleDataMut();
        for (keel::usize i = 0; i < m * n; ++i)
            rd[i] = cd[i].real();
        res = keel::move(r);
    }
    if (wantSingle && !res.isComplex()) {
        return detail::narrowDoubleToSingle(res, mr);
    }
    return res;
}

Value logm(const Value &A, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "logm");
    if (A.dims().ndim() != 2)
        throw Error("logm: input must be a 2D matrix", 0, 0, "logm", "", "numkit:logm:notMatrix");
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(0));
    if (n != keel::as<keel::usize>(A.dims().dim(1)))
        throw Error("logm: matrix must be square", 0, 0, "logm", "", "numkit:logm:notSquare");
    const bool wantSingle = A.type() == ValueType::SINGLE;
    if (n == 0) return Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    Value Aw = detail::widenSingleToDouble(A, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> Ac(n * n, scratch.allocator());
    copyValueToComplex(Aw, n * n, Ac.data());

    auto out = Value::complexMatrix(n, n, mr);
    if (!keel::linalg::logm(Ac.data(), n, out.complexDataMut(), mr))
        throw Error("logm: Schur iteration failed to converge", 0, 0, "logm", "", "numkit:logm:noConverge");

    return detail::narrowDoubleToSingleIf(detail::narrow_if_real(out, mr), wantSingle, mr);
}

Value expmv(keel::f64 t, const Value &A, const Value &v, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "expmv", "A", 1);
    {
        const ValueType vt = v.type();
        if (vt != ValueType::DOUBLE && vt != ValueType::COMPLEX
            && vt != ValueType::SINGLE && vt != ValueType::LOGICAL
            && vt != ValueType::CHAR && !isIntegerType(vt))
            throw Error(keel::format("expmv: input 3 (v) must be a numeric, logical or"
                                     " char vector (got a {} {} array).",
                                     detail::shapeOf(v), mtypeName(vt)),
                        0, 0, "expmv", "", "numkit:expmv:inputType");
    }
    if (A.dims().ndim() != 2)
        throw Error("expmv: A must be a 2D matrix",
                    0, 0, "expmv", "", "numkit:expmv:notMatrix");
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(0));
    if (keel::as<keel::usize>(A.dims().dim(1)) != n)
        throw Error("expmv: A must be square",
                    0, 0, "expmv", "", "numkit:expmv:notSquare");
    if (v.numel() != n)
        throw Error("expmv: length(v) must equal size(A, 1)",
                    0, 0, "expmv", "", "numkit:expmv:badV");
    const bool wantSingle = A.type() == ValueType::SINGLE || v.type() == ValueType::SINGLE;
    if (n == 0)
        return Value::matrix(0, 1, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    Value Aw = detail::widenSingleToDouble(A, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> vb(n, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) vb[i] = v.elemAsDouble(i);

    auto out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    if (!keel::linalg::expmv(t, Aw.doubleData(), n, vb.data(), out.doubleDataMut(), mr))
        throw Error("expmv: Padé denominator is singular", 0, 0, "expmv", "", "numkit:expmv:singular");

    return detail::narrowDoubleToSingleIf(out, wantSingle, mr);
}

static Value funm_eval(const Value &A, keel::FunctionRef<keel::c64(keel::c64)> f, keel::Allocator *mr)
{
    detail::requireDoubleInput(A, "funm");
    if (A.dims().ndim() != 2)
        throw Error("funm: input must be a 2D matrix", 0, 0, "funm", "", "numkit:funm:notMatrix");
    const keel::usize n = keel::as<keel::usize>(A.dims().dim(0));
    if (n != keel::as<keel::usize>(A.dims().dim(1)))
        throw Error("funm: matrix must be square", 0, 0, "funm", "", "numkit:funm:notSquare");
    const bool wantSingle = A.type() == ValueType::SINGLE;
    if (n == 0) return Value::matrix(0, 0, wantSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);

    Value Aw = detail::widenSingleToDouble(A, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> Ac(n * n, scratch.allocator());
    copyValueToComplex(Aw, n * n, Ac.data());

    auto out = Value::complexMatrix(n, n, mr);
    if (!keel::linalg::funm(Ac.data(), n, f, out.complexDataMut(), mr))
        throw Error("funm: Schur iteration failed to converge", 0, 0, "funm", "", "numkit:funm:noConverge");

    return detail::narrowDoubleToSingleIf(detail::narrow_if_real(out, mr), wantSingle, mr);
}

Value funm(const Value &A, keel::FunctionRef<keel::c64(keel::c64)> f, keel::Allocator *mr)
{
    return funm_eval(A, f, mr);
}

Value funm(const Value &A, keel::StringRef fnName, keel::Allocator *mr)
{
    if (fnName == "exp") return expm(A, mr);
    if (fnName == "log") return logm(A, mr);
    if (fnName == "sqrt") return sqrtm(A, mr);
    if (fnName == "sin") return funm_eval(A, [](keel::c64 x) { return keel::sin(x); }, mr);
    if (fnName == "cos") return funm_eval(A, [](keel::c64 x) { return keel::cos(x); }, mr);
    if (fnName == "sinh") return funm_eval(A, [](keel::c64 x) { return keel::sinh(x); }, mr);
    if (fnName == "cosh") return funm_eval(A, [](keel::c64 x) { return keel::cosh(x); }, mr);

    throw Error(keel::format("funm: unsupported function '{}'", fnName), 0, 0, "funm", "", "numkit:funm:unknownFn");
}

} // namespace numkit::linalg
