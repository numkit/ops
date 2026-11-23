/// @file qz.hpp
/// @ingroup group_matfun
// toolboxes/linalg/include/numkit/linalg/qz.hpp
//
// Generalized Schur decomposition: [AA, BB, Q, Z] = qz(A, B)

#pragma once

#include <keel/memory.h>
#include <keel/tuple.h>
#include <numkit/value/value.hpp>

namespace numkit::linalg {

/// @addtogroup group_matfun
/// @{


/// @brief Generalized Schur decomposition of matrix pair (A, B).
/// Returns keel::Tuple<Value, Value, Value, Value>{AA, BB, Q, Z}
/// such that Q * A * Z = AA and Q * B * Z = BB, where AA is upper quasi-triangular,
/// BB is upper triangular, and Q, Z are unitary.
keel::Tuple<Value, Value, Value, Value> qz(const Value &A, const Value &B,
                                           keel::Allocator *mr = nullptr);


/// @}
} // namespace numkit::linalg
