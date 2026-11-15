// src/builtin/src/matfun/matfun.cpp
//
// Matrix functions implementations for numkit::builtin.

#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/builtin/matfun.hpp>
#include <numkit/value/value.hpp>
#include <numkit/builtin/ops.hpp>
#include <numkit/builtin/datatypes.hpp>
#include <numkit/builtin/elfun.hpp>

#include <stdexcept>

namespace numkit::builtin {

Value idivide(const Value &a, const Value &b, keel::StringRef mode, keel::Allocator *mr)
{
    const ValueType t0 = a.type();
    const ValueType t1 = b.type();
    const bool int0 = isIntegerType(t0);
    const bool int1 = isIntegerType(t1);
    const bool dbl0 = (t0 == ValueType::DOUBLE);
    const bool dbl1 = (t1 == ValueType::DOUBLE);

    if (!int0 && !int1)
        throw std::runtime_error("At least one argument must belong to an integer class.");

    ValueType resultType;
    if (int0 && int1) {
        if (t0 != t1)
            throw std::runtime_error("Integers can only be combined with integers of the same class, or scalar doubles.");
        resultType = t0;
    } else if (int0) {
        if (!dbl1 || !b.isScalar())
            throw std::runtime_error("Integers can only be combined with integers of the same class, or scalar doubles.");
        resultType = t0;
    } else {
        if (!dbl0 || !a.isScalar())
            throw std::runtime_error("Integers can only be combined with integers of the same class, or scalar doubles.");
        resultType = t1;
    }

    const Value da = numkit::builtin::toDouble(a, mr);
    const Value db = numkit::builtin::toDouble(b, mr);
    Value q = numkit::builtin::rdivide(da, db, mr);

    if (mode.empty() || mode.equals_ignore_ascii_case("fix"))
        q = numkit::builtin::fix(q, mr);
    else if (mode.equals_ignore_ascii_case("floor"))
        q = numkit::builtin::floor(q, mr);
    else if (mode.equals_ignore_ascii_case("ceil"))
        q = numkit::builtin::ceil(q, mr);
    else if (mode.equals_ignore_ascii_case("round"))
        q = numkit::builtin::round(q, mr);
    else
        throw std::runtime_error("idivide: opt must be 'fix', 'floor', 'ceil', or 'round'");

    return numkit::builtin::cast(q, mtypeName(resultType), mr);
}

} // namespace numkit::builtin
