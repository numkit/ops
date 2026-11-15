// src/builtin/src/elfun/_unary_hint.hpp
//
// Internal SIMD reuse-buffer overloads for the unary functions sin/cos/exp/log/abs.
#pragma once

#include <keel/memory.h>
#include <numkit/value/value.hpp>

namespace numkit::builtin {

Value sin(const Value &x, Value *hint, keel::Allocator *mr);
Value cos(const Value &x, Value *hint, keel::Allocator *mr);
Value exp(const Value &x, Value *hint, keel::Allocator *mr);
Value log(const Value &x, Value *hint, keel::Allocator *mr);
Value abs(const Value &x, Value *hint, keel::Allocator *mr);

} // namespace numkit::builtin
