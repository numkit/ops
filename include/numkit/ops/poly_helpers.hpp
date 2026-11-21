// ops/include/numkit/ops/poly_helpers.hpp
//
// Polynomial helpers shared between the public roots() builtin and the
// DSP-side filter design code (tf2sos / tf2zp / butter). Thin inline
// forwarding wrappers over keel::poly.

#pragma once

#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/poly.h>
#include <keel/ref.h>

namespace numkit::ops {

inline void polyEvalHorner(const keel::c64 *c, keel::usize n,
                           keel::c64 z, keel::c64 &outP, keel::c64 &outDP)
{
    keel::poly::horner_deriv(keel::ArrayRef<keel::c64>(c, n), z, outP, outDP);
}

inline keel::Array<keel::c64>
polyRootsDurandKerner(const keel::c64 *coeffs, keel::usize coeffsN,
                      keel::Allocator *mr = nullptr)
{
    return keel::poly::roots(keel::ArrayRef<keel::c64>(coeffs, coeffsN), mr);
}

inline keel::Array<keel::c64>
polyRootsDurandKerner(const keel::f64 *coeffs, keel::usize coeffsN,
                      keel::Allocator *mr = nullptr)
{
    return keel::poly::roots(keel::ArrayRef<keel::f64>(coeffs, coeffsN), mr);
}

inline keel::Array<keel::f64>
polyExpandFromRoots(const keel::c64 *roots, keel::usize rootsN,
                    keel::Allocator *mr = nullptr)
{
    return keel::poly::from_roots(keel::ArrayRef<keel::c64>(roots, rootsN), mr);
}

} // namespace numkit::ops
