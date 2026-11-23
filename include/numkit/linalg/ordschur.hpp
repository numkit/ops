/// @file ordschur.hpp
/// @ingroup group_matfun
// toolboxes/linalg/include/numkit/linalg/ordschur.hpp
//
// Reorder Schur decomposition: [US, TS] = ordschur(U, T, select)

#pragma once

#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/tuple.h>
#include <numkit/value/value.hpp>

namespace numkit::linalg {

/// @addtogroup group_matfun
/// @{


/// @brief Reorder Schur decomposition A = U * T * U^H using a boolean select vector.
/// Returns keel::Tuple<Value, Value>{U_out, T_out}.
keel::Tuple<Value, Value> ordschur(const Value &U, const Value &T,
                                  const Value &select,
                                  keel::Allocator *mr = nullptr);

/// @brief Reorder Schur decomposition A = U * T * U^H using domain keyword
/// ('lhp', 'rhp', 'uip', 'uop').
keel::Tuple<Value, Value> ordschur(const Value &U, const Value &T,
                                  keel::StringRef domain,
                                  keel::Allocator *mr = nullptr);


/// @}
} // namespace numkit::linalg
