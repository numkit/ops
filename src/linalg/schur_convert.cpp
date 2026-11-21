// toolboxes/linalg/src/schur_convert.cpp
//
// cdf2rdf / rsf2csf — Value adapters over keel::linalg::cdf2rdf / keel::linalg::rsf2csf.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/linalg.h>
#include <numkit/linalg/schur_convert.hpp>

// Compute-only TU: Value substrate + Error, no engine. The cdf2rdf/rsf2csf
// builtins (CallContext wrappers) live in schur_convert_reg.cpp.
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>

namespace numkit::linalg {

std::tuple<Value, Value>
cdf2rdf(const Value &V, const Value &D, keel::Allocator *mr)
{
    if (V.dims().ndim() != 2 || D.dims().ndim() != 2)
        throw Error("cdf2rdf: V and D must be 2-D matrices",
                    0, 0, "cdf2rdf", "", "numkit:cdf2rdf:notMatrix");
    const keel::usize n = keel::as<keel::usize>(D.dims().dim(0));
    if (n != keel::as<keel::usize>(D.dims().dim(1)))
        throw Error("cdf2rdf: D must be square",
                    0, 0, "cdf2rdf", "", "numkit:cdf2rdf:notSquare");
    if (keel::as<keel::usize>(V.dims().dim(0)) != n
        || keel::as<keel::usize>(V.dims().dim(1)) != n)
        throw Error("cdf2rdf: V and D shape mismatch",
                    0, 0, "cdf2rdf", "", "numkit:cdf2rdf:shapeMismatch");
    if (n == 0)
        return std::make_tuple(
            Value::matrix(0, 0, ValueType::DOUBLE, mr),
            Value::matrix(0, 0, ValueType::DOUBLE, mr));

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::c64> Vc(n * n, scratch.allocator());
    keel::Array<keel::c64> Dc(n * n, scratch.allocator());
    for (keel::usize i = 0; i < n * n; ++i) {
        Vc[i] = V.isComplex() ? V.complexData()[i] : keel::c64(V.elemAsDouble(i), 0.0);
        Dc[i] = D.isComplex() ? D.complexData()[i] : keel::c64(D.elemAsDouble(i), 0.0);
    }

    auto VRout = Value::matrix(n, n, ValueType::DOUBLE, mr);
    auto DRout = Value::matrix(n, n, ValueType::DOUBLE, mr);
    if (!keel::linalg::cdf2rdf(Vc.data(), Dc.data(), n, VRout.doubleDataMut(), DRout.doubleDataMut())) {
        throw Error("cdf2rdf: complex eigenvalues must come in conjugate pairs",
                    0, 0, "cdf2rdf", "", "numkit:cdf2rdf:badPair");
    }
    return std::make_tuple(std::move(VRout), std::move(DRout));
}

std::tuple<Value, Value>
rsf2csf(const Value &UR, const Value &TR, keel::Allocator *mr)
{
    if (UR.dims().ndim() != 2 || TR.dims().ndim() != 2)
        throw Error("rsf2csf: UR and TR must be 2-D matrices",
                    0, 0, "rsf2csf", "", "numkit:rsf2csf:notMatrix");
    const keel::usize n = keel::as<keel::usize>(TR.dims().dim(0));
    if (n != keel::as<keel::usize>(TR.dims().dim(1)))
        throw Error("rsf2csf: TR must be square",
                    0, 0, "rsf2csf", "", "numkit:rsf2csf:notSquare");
    if (keel::as<keel::usize>(UR.dims().dim(0)) != n
        || keel::as<keel::usize>(UR.dims().dim(1)) != n)
        throw Error("rsf2csf: UR and TR shape mismatch",
                    0, 0, "rsf2csf", "", "numkit:rsf2csf:shapeMismatch");
    if (n == 0)
        return std::make_tuple(
            Value::matrix(0, 0, ValueType::COMPLEX, mr),
            Value::matrix(0, 0, ValueType::COMPLEX, mr));

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> Ur(n * n, scratch.allocator());
    keel::Array<keel::f64> Tr(n * n, scratch.allocator());
    for (keel::usize i = 0; i < n * n; ++i) {
        Ur[i] = UR.elemAsDouble(i);
        Tr[i] = TR.elemAsDouble(i);
    }

    auto Uout = Value::matrix(n, n, ValueType::COMPLEX, mr);
    auto Tout = Value::matrix(n, n, ValueType::COMPLEX, mr);
    keel::linalg::rsf2csf(Ur.data(), Tr.data(), n, Uout.complexDataMut(), Tout.complexDataMut());
    return std::make_tuple(std::move(Uout), std::move(Tout));
}

} // namespace numkit::linalg
