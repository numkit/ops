// toolboxes/linalg/src/vector_ops.cpp
//
// cross / dot / kron â€” implementations and engine adapters.
// Migrated from toolboxes/builtin/src/language/arrays/matrix.cpp.

#include <numkit/linalg/vector_ops.hpp>

// Compute-only TU: Value substrate + Error, no engine. The cross/dot/kron
// builtins (CallContext wrappers) live in vector_ops_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/span.hpp>
#include <numkit/value/value_type.hpp>

#include <cmath>
#include <complex>
#include <limits>

namespace numkit::linalg {

// Narrow a DOUBLE workspace to an integer class with MATLAB's saturating
// round-half-away-from-zero cast (mirrors core's castConcatToInteger). Used
// by kron to preserve the integer class of integer operands.
static Value narrowKronToInteger(const Value &d, ValueType vt,
                                 std::pmr::memory_resource *mr)
{
    const auto &dd = d.dims();
    Value r = Value::matrix(dd.rows(), dd.cols(), vt, mr);
    const size_t n = d.numel();
    const double *src = d.doubleData();
    auto fill = [&](auto *dst) {
        using T = std::remove_pointer_t<std::decay_t<decltype(dst)>>;
        const double lo = static_cast<double>(std::numeric_limits<T>::min());
        const double hi = static_cast<double>(std::numeric_limits<T>::max());
        for (size_t i = 0; i < n; ++i) {
            double v = std::round(src[i]);
            if (v < lo) v = lo; else if (v > hi) v = hi;
            dst[i] = static_cast<T>(v);
        }
    };
    switch (vt) {
    case ValueType::INT8:   fill(r.int8DataMut());   break;
    case ValueType::INT16:  fill(r.int16DataMut());  break;
    case ValueType::INT32:  fill(r.int32DataMut());  break;
    case ValueType::INT64:  fill(r.int64DataMut());  break;
    case ValueType::UINT8:  fill(r.uint8DataMut());  break;
    case ValueType::UINT16: fill(r.uint16DataMut()); break;
    case ValueType::UINT32: fill(r.uint32DataMut()); break;
    case ValueType::UINT64: fill(r.uint64DataMut()); break;
    default: break;
    }
    return r;
}

// kron's integer-class rule (MATLAB R2025b): the result keeps an integer
// class iff both operands share the same integer class, or one operand is
// integer and the other a real scalar double (the scalar is cast). Mixed
// integer classes, integer + non-scalar double, and integer + logical all
// ERROR in MATLAB; nk stays lenient there and returns double.
static ValueType kronIntegerClass(const Value &a, const Value &b)
{
    const bool aInt = isIntegerType(a.type());
    const bool bInt = isIntegerType(b.type());
    auto realScalarDouble = [](const Value &v) {
        return v.numel() == 1 && v.type() == ValueType::DOUBLE;
    };
    if (aInt && bInt && a.type() == b.type()) return a.type();
    if (aInt && !bInt && realScalarDouble(b)) return a.type();
    if (bInt && !aInt && realScalarDouble(a)) return b.type();
    return ValueType::DOUBLE;
}

// Integer range [lo, hi] as doubles, for saturating casts.
static void intTypeRange(ValueType vt, double &lo, double &hi)
{
    switch (vt) {
    case ValueType::INT8:   lo = std::numeric_limits<int8_t>::min();   hi = std::numeric_limits<int8_t>::max();   break;
    case ValueType::INT16:  lo = std::numeric_limits<int16_t>::min();  hi = std::numeric_limits<int16_t>::max();  break;
    case ValueType::INT32:  lo = std::numeric_limits<int32_t>::min();  hi = std::numeric_limits<int32_t>::max();  break;
    case ValueType::INT64:  lo = std::numeric_limits<int64_t>::min();  hi = std::numeric_limits<int64_t>::max();  break;
    case ValueType::UINT8:  lo = 0; hi = std::numeric_limits<uint8_t>::max();  break;
    case ValueType::UINT16: lo = 0; hi = std::numeric_limits<uint16_t>::max(); break;
    case ValueType::UINT32: lo = 0; hi = std::numeric_limits<uint32_t>::max(); break;
    case ValueType::UINT64: lo = 0; hi = std::numeric_limits<uint64_t>::max(); break;
    default:                lo = 0; hi = 0; break;
    }
}

// cross's integer-class rule (MATLAB R2025b): the result keeps an integer
// class iff both operands share the same integer class, or one operand is
// integer and the other a real double (ANY shape â€” unlike kron, because each
// element product is int-scalar Ã— double-scalar, an allowed mix). Different
// integer classes and integer + logical ERROR in MATLAB; nk stays lenient
// there and computes in double.
static ValueType crossIntegerClass(const Value &a, const Value &b)
{
    const bool aInt = isIntegerType(a.type());
    const bool bInt = isIntegerType(b.type());
    if (aInt && bInt && a.type() == b.type()) return a.type();
    if (aInt && b.type() == ValueType::DOUBLE) return a.type();
    if (bInt && a.type() == ValueType::DOUBLE) return b.type();
    return ValueType::DOUBLE;
}

// Element-wise copy of any numeric Value to a fresh DOUBLE Value (via
// elemAsDouble, so integer/logical operands are handled). DOUBLE is returned
// as-is. Used for the lenient double-output cross path.
static Value crossToDouble(const Value &v, std::pmr::memory_resource *mr)
{
    if (v.type() == ValueType::DOUBLE) return v;
    const auto &d = v.dims();
    Value r = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    const size_t n = v.numel();
    double *dst = r.doubleDataMut();
    for (size_t i = 0; i < n; ++i) dst[i] = v.elemAsDouble(i);
    return r;
}

// Integer-class cross product with MATLAB's per-operation saturating integer
// arithmetic: each element product saturates to the int range BEFORE the
// subtraction, and the subtraction saturates too. e.g. cross(int8([100 100 0]),
// int8([0 100 100])) = [127 -127 127], not [127 -128 127]. Operands are read
// class-agnostically (elemAsDouble); the result is narrowed to outType.
static Value crossIntegerSaturating(const Value &a, const Value &b, int crossDim,
                                    ValueType outType, size_t nr, size_t nc,
                                    std::pmr::memory_resource *mr)
{
    double lo, hi;
    intTypeRange(outType, lo, hi);
    auto sat = [lo, hi](double v) {
        v = std::round(v);
        return v < lo ? lo : (v > hi ? hi : v);
    };
    auto comp = [&](double p, double q, double r, double s) {
        return sat(sat(p * q) - sat(r * s));
    };
    Value dbl = Value::matrix(nr, nc, ValueType::DOUBLE, mr);
    double *od = dbl.doubleDataMut();
    auto ga = [&](size_t i) { return a.elemAsDouble(i); };
    auto gb = [&](size_t i) { return b.elemAsDouble(i); };
    if (crossDim == 0) {
        for (size_t c = 0; c < nc; ++c) {
            const size_t base = c * 3;
            const double a0 = ga(base), a1 = ga(base + 1), a2 = ga(base + 2);
            const double b0 = gb(base), b1 = gb(base + 1), b2 = gb(base + 2);
            od[base]     = comp(a1, b2, a2, b1);
            od[base + 1] = comp(a2, b0, a0, b2);
            od[base + 2] = comp(a0, b1, a1, b0);
        }
    } else {
        for (size_t r = 0; r < nr; ++r) {
            const double a0 = ga(r), a1 = ga(r + nr), a2 = ga(r + 2 * nr);
            const double b0 = gb(r), b1 = gb(r + nr), b2 = gb(r + 2 * nr);
            od[r]          = comp(a1, b2, a2, b1);
            od[r + nr]     = comp(a2, b0, a0, b2);
            od[r + 2 * nr] = comp(a0, b1, a1, b0);
        }
    }
    return narrowKronToInteger(dbl, outType, mr);
}

// Core cross-product along crossDim (0 = each column is a 3-vec / MATLAB
// dim 1; 1 = each row is a 3-vec / MATLAB dim 2). Shape already validated.
static Value crossCore(const Value &a, const Value &b, int crossDim,
                       std::pmr::memory_resource *mr)
{
    const size_t nr = a.dims().rows();
    const size_t nc = a.dims().cols();

    // Complex cross-product â€” ordinary complex arithmetic, NO conjugation
    // (unlike dot). Either operand complex takes this path; reals -> z+0i.
    if (a.type() == ValueType::COMPLEX || b.type() == ValueType::COMPLEX) {
        const bool aCx = (a.type() == ValueType::COMPLEX);
        const bool bCx = (b.type() == ValueType::COMPLEX);
        auto ga = [&](size_t i) {
            return aCx ? a.complexData()[i] : Complex(a.elemAsDouble(i), 0.0);
        };
        auto gb = [&](size_t i) {
            return bCx ? b.complexData()[i] : Complex(b.elemAsDouble(i), 0.0);
        };
        auto outc = Value::matrix(nr, nc, ValueType::COMPLEX, mr);
        Complex *oc = outc.complexDataMut();
        if (crossDim == 0) {
            for (size_t c = 0; c < nc; ++c) {
                const size_t base = c * 3;
                const Complex a0 = ga(base), a1 = ga(base + 1), a2 = ga(base + 2);
                const Complex b0 = gb(base), b1 = gb(base + 1), b2 = gb(base + 2);
                oc[base    ] = a1 * b2 - a2 * b1;
                oc[base + 1] = a2 * b0 - a0 * b2;
                oc[base + 2] = a0 * b1 - a1 * b0;
            }
        } else {
            for (size_t r = 0; r < nr; ++r) {
                const Complex a0 = ga(r), a1 = ga(r + nr), a2 = ga(r + 2 * nr);
                const Complex b0 = gb(r), b1 = gb(r + nr), b2 = gb(r + 2 * nr);
                oc[r           ] = a1 * b2 - a2 * b1;
                oc[r +     nr  ] = a2 * b0 - a0 * b2;
                oc[r + 2 * nr  ] = a0 * b1 - a1 * b0;
            }
        }
        return outc;
    }

    // Integer/logical operands: same int class (or int + double) preserves the
    // integer class with per-op saturation; other mixes (different int classes,
    // int + logical) are lenient -> computed in double.
    const ValueType intClass = crossIntegerClass(a, b);
    if (intClass != ValueType::DOUBLE)
        return crossIntegerSaturating(a, b, crossDim, intClass, nr, nc, mr);
    Value aHold = crossToDouble(a, mr), bHold = crossToDouble(b, mr);

    auto out = Value::matrix(nr, nc, ValueType::DOUBLE, mr);
    const double *ad = aHold.doubleData();
    const double *bd = bHold.doubleData();
    double *od = out.doubleDataMut();

    if (crossDim == 0) {
        // 3xN: column-major storage, so col c starts at c*3.
        const size_t batches = nc;
        for (size_t c = 0; c < batches; ++c) {
            const size_t base = c * 3;
            const double a0 = ad[base], a1 = ad[base + 1], a2 = ad[base + 2];
            const double b0 = bd[base], b1 = bd[base + 1], b2 = bd[base + 2];
            od[base    ] = a1 * b2 - a2 * b1;
            od[base + 1] = a2 * b0 - a0 * b2;
            od[base + 2] = a0 * b1 - a1 * b0;
        }
    } else {
        // Nx3: column-major, so element (r, k) at r + k*nr.
        const size_t batches = nr;
        for (size_t r = 0; r < batches; ++r) {
            const double a0 = ad[r], a1 = ad[r + nr], a2 = ad[r + 2 * nr];
            const double b0 = bd[r], b1 = bd[r + nr], b2 = bd[r + 2 * nr];
            od[r           ] = a1 * b2 - a2 * b1;
            od[r +     nr  ] = a2 * b0 - a0 * b2;
            od[r + 2 * nr  ] = a0 * b1 - a1 * b0;
        }
    }
    return out;
}

Value cross(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    // MATLAB: cross(A, B) operates along the FIRST dimension of size 3.
    // Common shapes: 1x3, 3x1, 3xN, Nx3. Result has the same shape as inputs.
    const auto &da = a.dims();
    const auto &db = b.dims();
    if (da.rows() != db.rows() || da.cols() != db.cols())
        throw Error("cross: A and B must have the same shape",
                     0, 0, "cross", "", "numkit:cross:shapeMismatch");
    int crossDim;
    if (da.rows() == 3)      crossDim = 0; // each column is a 3-vec (dim 1)
    else if (da.cols() == 3) crossDim = 1; // each row is a 3-vec (dim 2)
    else
        throw Error("cross: A and B must have at least one dimension of length 3",
                     0, 0, "cross", "", "numkit:cross:badSize");
    return crossCore(a, b, crossDim, mr);
}

Value cross(const Value &a, const Value &b, int dim, std::pmr::memory_resource *mr)
{
    if (dim == 0) return cross(a, b, mr);   // default: first length-3 dimension
    const auto &da = a.dims();
    const auto &db = b.dims();
    if (da.rows() != db.rows() || da.cols() != db.cols())
        throw Error("cross: A and B must have the same shape",
                     0, 0, "cross", "", "numkit:cross:shapeMismatch");
    if (da.is3D() || db.is3D())
        throw Error("cross: the dim argument supports 2-D inputs only",
                     0, 0, "cross", "", "numkit:cross:rank");
    if (dim != 1 && dim != 2)
        throw Error("cross: dim must be 1 or 2",
                     0, 0, "cross", "", "numkit:cross:badDim");
    // MATLAB: A and B must have length 3 along the operating dimension.
    const size_t lenAlongDim = (dim == 1) ? da.rows() : da.cols();
    if (lenAlongDim != 3)
        throw Error("cross: A and B must have length 3 in the operating dimension",
                     0, 0, "cross", "", "numkit:cross:badSize");
    return crossCore(a, b, /*crossDim=*/dim - 1, mr);
}

Value dot(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    if (a.numel() != b.numel())
        throw Error("dot: A and B must be the same size",
                     0, 0, "dot", "", "numkit:dot:lengthMismatch");
    const auto &da = a.dims();
    const size_t H = da.rows(), W = da.cols();
    // Vectors (row or column) -> a single scalar.
    const bool isVector = (!da.is3D()) && (H == 1 || W == 1);

    // MATLAB: dot conjugates the FIRST argument â€” dot(a,b) = sum(conj(a).*b).
    // Complex inputs (either operand) take the complex path; the result is
    // complex (per-column for matrices). The real path below drops to scalar
    // doubles via elemAsDouble, which would discard the imaginary part.
    if (a.type() == ValueType::COMPLEX || b.type() == ValueType::COMPLEX) {
        const bool aCx = (a.type() == ValueType::COMPLEX);
        const bool bCx = (b.type() == ValueType::COMPLEX);
        auto getA = [&](size_t i) {
            return aCx ? a.complexData()[i] : Complex(a.elemAsDouble(i), 0.0);
        };
        auto getB = [&](size_t i) {
            return bCx ? b.complexData()[i] : Complex(b.elemAsDouble(i), 0.0);
        };
        if (isVector) {
            Complex s(0.0, 0.0);
            for (size_t i = 0; i < a.numel(); ++i)
                s += std::conj(getA(i)) * getB(i);
            return Value::complexScalar(s, mr);
        }
        Value out = Value::matrix(1, W, ValueType::COMPLEX, mr);
        Complex *od = out.complexDataMut();
        for (size_t j = 0; j < W; ++j) {
            Complex s(0.0, 0.0);
            for (size_t i = 0; i < H; ++i)
                s += std::conj(getA(j * H + i)) * getB(j * H + i);
            od[j] = s;
        }
        return out;
    }

    if (isVector) {
        double s = 0.0;
        for (size_t i = 0; i < a.numel(); ++i)
            s += a.elemAsDouble(i) * b.elemAsDouble(i);
        return Value::scalar(s, mr);
    }
    // Matrices: per-column dot (sum along dim 1) -> 1 x W row vector,
    // matching MATLAB (the previous code flattened everything to a scalar).
    Value out = Value::matrix(1, W, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (size_t j = 0; j < W; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < H; ++i)
            s += a.elemAsDouble(j * H + i) * b.elemAsDouble(j * H + i);
        od[j] = s;
    }
    return out;
}

Value dot(const Value &a, const Value &b, int dim, std::pmr::memory_resource *mr)
{
    if (dim == 0)
        return dot(a, b, mr);   // default (vector â†’ scalar, matrix â†’ per-column)
    if (dim != 1 && dim != 2)
        throw Error("dot: dim must be 1 or 2",
                     0, 0, "dot", "", "numkit:dot:badDim");
    if (a.numel() != b.numel() || a.dims().rows() != b.dims().rows()
        || a.dims().cols() != b.dims().cols())
        throw Error("dot: A and B must be the same size",
                     0, 0, "dot", "", "numkit:dot:lengthMismatch");
    if (a.dims().is3D() || b.dims().is3D())
        throw Error("dot: the dim argument supports 2-D inputs only",
                     0, 0, "dot", "", "numkit:dot:rank");

    const size_t H = a.dims().rows(), W = a.dims().cols();
    // sum(conj(A).*B, dim): dim 1 â†’ 1xW (down columns), dim 2 â†’ Hx1 (across rows).
    const size_t outR = (dim == 1) ? 1 : H;
    const size_t outC = (dim == 1) ? W : 1;
    const bool cplx = (a.type() == ValueType::COMPLEX || b.type() == ValueType::COMPLEX);

    if (cplx) {
        const bool aCx = (a.type() == ValueType::COMPLEX);
        const bool bCx = (b.type() == ValueType::COMPLEX);
        auto getA = [&](size_t i) { return aCx ? a.complexData()[i] : Complex(a.elemAsDouble(i), 0.0); };
        auto getB = [&](size_t i) { return bCx ? b.complexData()[i] : Complex(b.elemAsDouble(i), 0.0); };
        Value out = Value::matrix(outR, outC, ValueType::COMPLEX, mr);
        Complex *od = out.complexDataMut();
        if (dim == 1)
            for (size_t j = 0; j < W; ++j) {
                Complex s(0.0, 0.0);
                for (size_t i = 0; i < H; ++i) s += std::conj(getA(j * H + i)) * getB(j * H + i);
                od[j] = s;
            }
        else
            for (size_t i = 0; i < H; ++i) {
                Complex s(0.0, 0.0);
                for (size_t j = 0; j < W; ++j) s += std::conj(getA(j * H + i)) * getB(j * H + i);
                od[i] = s;
            }
        return out;
    }

    Value out = Value::matrix(outR, outC, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    if (dim == 1)
        for (size_t j = 0; j < W; ++j) {
            double s = 0.0;
            for (size_t i = 0; i < H; ++i) s += a.elemAsDouble(j * H + i) * b.elemAsDouble(j * H + i);
            od[j] = s;
        }
    else
        for (size_t i = 0; i < H; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < W; ++j) s += a.elemAsDouble(j * H + i) * b.elemAsDouble(j * H + i);
            od[i] = s;
        }
    return out;
}

Value kron(const Value &a, const Value &b, std::pmr::memory_resource *mr)
{
    // Class contract probed R2025b (probe66/67): cell/struct reject
    // (numeric, char or logical only); 3-D rejects; integers combine
    // only with the SAME integer class or a real double SCALAR (any
    // single with an integer â€” even a scalar â€” and any non-scalar
    // double is mixedClasses); EMPTY operands reject with the
    // reference's internal badsubscript diagnostic; char/logical widen
    // to double codes; otherwise single contagion.
    for (const Value *v : {&a, &b}) {
        const ValueType t = v->type();
        if (t == ValueType::CELL || t == ValueType::STRUCT
            || t == ValueType::STRING || t == ValueType::FUNC_HANDLE
            || t == ValueType::OBJECT)
            throw Error(std::string("kron: inputs must be numeric, char, "
                        "or logical (got a ")
                        + mtypeName(t) + " array).",
                        0, 0, "kron", "", "numkit:kron:mustBeNumericCharOrLogical");
    }
    if (a.dims().ndim() > 2 || b.dims().ndim() > 2)
        throw Error("kron: inputs must be 2-D.",
                     0, 0, "kron", "", "numkit:kron:TwoDInput");
    {
        const bool aInt = isIntegerType(a.type());
        const bool bInt = isIntegerType(b.type());
        auto dblScalar = [](const Value &v) {
            return v.numel() == 1 && v.type() == ValueType::DOUBLE;
        };
        const bool okMix =
            (!aInt && !bInt)
            || (aInt && bInt && a.type() == b.type())
            || (aInt && !bInt && dblScalar(b))
            || (bInt && !aInt && dblScalar(a));
        if (!okMix)
            throw Error("kron: integers can only be combined with "
                        "integers of the same class, or scalar doubles.",
                        0, 0, "kron", "", "numkit:kron:mixedClasses");
    }
    if (a.numel() == 0 || b.numel() == 0)
        throw Error("kron: empty inputs are rejected.",
                    0, 0, "kron", "", "numkit:kron:badsubscript");

    const size_t rA = a.dims().rows(), cA = a.dims().cols();
    const size_t rB = b.dims().rows(), cB = b.dims().cols();
    const size_t rOut = rA * rB, cOut = cA * cB;

    // Complex Kronecker product â€” ordinary complex element products (no
    // conjugation). Either operand complex takes this path; reals -> z+0i.
    if (a.type() == ValueType::COMPLEX || b.type() == ValueType::COMPLEX) {
        const bool aCx = (a.type() == ValueType::COMPLEX);
        const bool bCx = (b.type() == ValueType::COMPLEX);
        auto outc = Value::matrix(rOut, cOut, ValueType::COMPLEX, mr);
        if (rOut == 0 || cOut == 0) return outc;
        Complex *dst = outc.complexDataMut();
        for (size_t ja = 0; ja < cA; ++ja)
            for (size_t ia = 0; ia < rA; ++ia) {
                const Complex av = aCx ? a.complexData()[ia + ja * rA]
                                       : Complex(a.elemAsDouble(ia + ja * rA), 0.0);
                for (size_t jb = 0; jb < cB; ++jb) {
                    const size_t jOut = ja * cB + jb;
                    for (size_t ib = 0; ib < rB; ++ib) {
                        const size_t iOut = ia * rB + ib;
                        const Complex bv = bCx ? b.complexData()[ib + jb * rB]
                                               : Complex(b.elemAsDouble(ib + jb * rB), 0.0);
                        dst[jOut * rOut + iOut] = av * bv;
                    }
                }
            }
        return outc;
    }

    // Integer operands keep their class (saturating) per MATLAB; otherwise
    // the result is double. The element products are computed in a double
    // workspace via elemAsDouble (correct for every integer operand class)
    // and narrowed at the end.
    const ValueType intClass = kronIntegerClass(a, b);

    auto out = Value::matrix(rOut, cOut, ValueType::DOUBLE, mr);
    if (rOut == 0 || cOut == 0)
        return intClass == ValueType::DOUBLE ? out
                                             : narrowKronToInteger(out, intClass, mr);

    double *dst = out.doubleDataMut();
    for (size_t ja = 0; ja < cA; ++ja)
        for (size_t ia = 0; ia < rA; ++ia) {
            const double av = a.elemAsDouble(ia + ja * rA);
            for (size_t jb = 0; jb < cB; ++jb) {
                const size_t jOut = ja * cB + jb;
                for (size_t ib = 0; ib < rB; ++ib) {
                    const size_t iOut = ia * rB + ib;
                    const double bv = b.elemAsDouble(ib + jb * rB);
                    dst[jOut * rOut + iOut] = av * bv;
                }
            }
        }
    if (intClass != ValueType::DOUBLE)
        return narrowKronToInteger(out, intClass, mr);
    if (a.type() == ValueType::SINGLE || b.type() == ValueType::SINGLE) {
        Value s = Value::matrix(out.dims().rows(), out.dims().cols(),
                                ValueType::SINGLE, mr);
        const double *od = out.doubleData();
        float *sd = s.singleDataMut();
        for (size_t i = 0; i < out.numel(); ++i)
            sd[i] = static_cast<float>(od[i]);
        return s;
    }
    return out;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Engine adapters â€” registered in LinalgLibrary::install
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

} // namespace numkit::linalg
