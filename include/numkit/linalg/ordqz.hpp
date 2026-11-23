/// @file ordqz.hpp
/// @ingroup group_matfun
// toolboxes/linalg/include/numkit/linalg/ordqz.hpp
//
// Reorder Generalized Schur decomposition: [AAS, BBS, QS, ZS] = ordqz(AA, BB, Q, Z, select/domain)

#pragma once

#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/tuple.h>
#include <numkit/value/value.hpp>

namespace numkit::linalg {

/// @addtogroup group_matfun
/// @{


/// @brief Reorder Generalized Schur decomposition using a boolean select vector.
keel::Tuple<Value, Value, Value, Value> ordqz(const Value &AA, const Value &BB,
                                              const Value &Q, const Value &Z,
                                              const Value &select,
                                              keel::Allocator *mr = nullptr);

/// @brief Reorder Generalized Schur decomposition using domain keyword ('lhp', 'rhp', 'uip', 'uop').
keel::Tuple<Value, Value, Value, Value> ordqz(const Value &AA, const Value &BB,
                                              const Value &Q, const Value &Z,
                                              keel::StringRef domain,
                                              keel::Allocator *mr = nullptr);


/// @}
} // namespace numkit::linalg
