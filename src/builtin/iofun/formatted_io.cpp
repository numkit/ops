// src/builtin/src/iofun/formatted_io.cpp
//
// Formatted input/output and stream printing implementations for numkit::builtin.

#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/builtin/iofun.hpp>
#include <numkit/value/value.hpp>
#include <numkit/builtin/strfun.hpp>
#include <numkit/builtin/strfun.hpp>
#include <numkit/builtin/strfun.hpp>

namespace numkit::builtin {

std::string sprintf(const std::string &fmt, keel::ArrayRef<Value> args, keel::Allocator *mr) {
    return numkit::builtin::formatCyclic(fmt, args, 0, mr);
}

void disp(const Value &v, std::ostream &os) {
    os << numkit::builtin::dispFormat(v);
}

int fprintf(std::ostream &os, const std::string &fmt, keel::ArrayRef<Value> args, keel::Allocator *mr) {
    std::string s = numkit::builtin::formatCyclic(fmt, args, 0, mr);
    os << s;
    return static_cast<int>(s.size());
}

Value sscanf(const std::string &str, const std::string &fmt, keel::Allocator *mr) {
    Value args[2] = { Value::fromString(str, mr), Value::fromString(fmt, mr) };
    Value out;
    numkit::builtin::sscanf(keel::ArrayRef<Value>(args, 2), 1, keel::MutArrayRef<Value>(&out, 1), mr);
    return out;
}

Value textscan(const std::string &str, const std::string &fmt, keel::Allocator *mr) {
    Value args[2] = { Value::fromString(str, mr), Value::fromString(fmt, mr) };
    Value out;
    numkit::builtin::sscanf(keel::ArrayRef<Value>(args, 2), 1, keel::MutArrayRef<Value>(&out, 1), mr);
    return out;
}

} // namespace numkit::builtin
