// src/builtin/src/iofun/formatted_io.cpp
//
// Formatted input/output and stream printing implementations for numkit::builtin.

#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/builtin/iofun.hpp>
#include <numkit/value/value.hpp>
#include <numkit/builtin/strfun.hpp>

#include <cstdio>

namespace numkit::builtin {

keel::String sprintf(keel::StringRef fmt, keel::ArrayRef<Value> args, keel::Allocator *mr) {
    return numkit::builtin::formatCyclic(fmt, args, 0, mr);
}

void disp(const Value &v, std::FILE *stream) {
    keel::String s = numkit::builtin::dispFormat(v);
    if (stream && !s.empty()) {
        std::fwrite(s.data(), 1, s.size(), stream);
    }
}

keel::i32 fprintf(std::FILE *stream, keel::StringRef fmt, keel::ArrayRef<Value> args, keel::Allocator *mr) {
    keel::String s = numkit::builtin::formatCyclic(fmt, args, 0, mr);
    if (stream && !s.empty()) {
        std::fwrite(s.data(), 1, s.size(), stream);
    }
    return keel::as<keel::i32>(s.size());
}

Value sscanf(keel::StringRef str, keel::StringRef fmt, keel::Allocator *mr) {
    Value args[2] = { Value::fromString(str, mr), Value::fromString(fmt, mr) };
    Value out;
    numkit::builtin::sscanf(keel::ArrayRef<Value>(args, 2), 1, keel::MutArrayRef<Value>(&out, 1), mr);
    return out;
}

Value textscan(keel::StringRef str, keel::StringRef fmt, keel::Allocator *mr) {
    Value args[2] = { Value::fromString(str, mr), Value::fromString(fmt, mr) };
    Value out;
    numkit::builtin::sscanf(keel::ArrayRef<Value>(args, 2), 1, keel::MutArrayRef<Value>(&out, 1), mr);
    return out;
}

} // namespace numkit::builtin
