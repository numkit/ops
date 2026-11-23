// ops/callback_eval.hpp
//
// Thin Value<->double wrappers around a numkit::FnHandle MATLAB-style callback,
// so the ops iterative solver kernels (root_solve: brent/findBracket/brentMin/
// nelderMead) can run in native double precision while the public objective
// stays a generic `void(Span<const Value>, Span<Value>)` handle. Core-free
// (value layer only); FnHandle is value/fn_handle.hpp (L0).

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/value/error.hpp>
#include <numkit/value/fn_handle.hpp>
#include <numkit/value/value.hpp>

namespace numkit::ops {

/// Invoke `fn(x)` with a scalar argument; return the scalar result.
/// Throws if the callback returns a non-scalar value.
inline keel::f64 evalScalar(FnHandle fn, keel::f64 x, keel::Allocator *mr = nullptr)
{
    Value             arg = Value::scalar(x, mr);
    Value             out;
    keel::ArrayRef<Value> args(&arg, 1);
    keel::MutArrayRef<Value>       outs(&out, 1);
    fn(args, outs, mr);
    if (!out.isScalar() && out.numel() != 1)
        throw Error("callback: handle must return a scalar value", 0, 0, "callback", "",
                    "numkit:callback:nonScalar");
    return out.elemAsDouble(0);
}

/// Invoke `fn(x)` with a vector argument (x[0..n) packed into a 1xn DOUBLE row);
/// return the scalar result.
inline keel::f64 evalVecToScalar(FnHandle fn, const keel::f64 *x, keel::usize n,
                              keel::Allocator *mr = nullptr)
{
    Value v = Value::matrix(1, n, ValueType::DOUBLE, mr);
    if (n) {
        keel::f64 *d = v.doubleDataMut();
        for (keel::usize i = 0; i < n; ++i) d[i] = x[i];
    }
    Value out;
    Value args[1] = {keel::move(v)};
    fn(keel::ArrayRef<Value>(args, 1), keel::MutArrayRef<Value>(&out, 1), mr);
    return out.toScalar();
}

} // namespace numkit::ops
