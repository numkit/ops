// src/builtin/src/timefun/clock.cpp
//
// Clock, current date, CPU time, and execution pausing for numkit::builtin.

#include <keel/format.h>
#include <keel/memory.h>
#include <keel/process.h>
#include <keel/string.h>
#include <keel/time.h>
#include <numkit/builtin/timefun.hpp>
#include <numkit/value/value.hpp>

#include <cstdio>

namespace numkit::builtin {

keel::f64 cputime() {
    return keel::as<keel::f64>(keel::process_cpu_time_ns()) * 1e-9;
}

void pause(keel::f64 seconds) {
    if (seconds > 0.0) {
        const keel::u64 ns = keel::as<keel::u64>(seconds * 1e9);
        keel::sleep_ns(ns);
    }
}

keel::f64 now() {
    return 719529.0 + keel::as<keel::f64>(keel::wall_clock_us()) / 86400000000.0;
}

keel::String date() {
    const keel::CalendarTime t = keel::local_time();
    char buf[64];
    static const char *kMonths[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    const keel::i32 n = std::snprintf(buf, sizeof(buf), "%02d-%s-%04d",
                                keel::as<keel::i32>(t.day),
                                kMonths[(t.month - 1) % 12],
                                keel::as<keel::i32>(t.year));
    return keel::String(keel::StringRef(buf, n > 0 ? keel::as<keel::usize>(n) : 0));
}

Value clock(keel::Allocator *mr) {
    const keel::CalendarTime t = keel::local_time();
    auto out = Value::matrix(1, 6, ValueType::DOUBLE, mr);
    keel::f64 *d = out.doubleDataMut();
    d[0] = keel::as<keel::f64>(t.year);
    d[1] = keel::as<keel::f64>(t.month);
    d[2] = keel::as<keel::f64>(t.day);
    d[3] = keel::as<keel::f64>(t.hour);
    d[4] = keel::as<keel::f64>(t.minute);
    d[5] = keel::as<keel::f64>(t.second) + keel::as<keel::f64>(t.millisecond) / 1000.0;
    return out;
}

} // namespace numkit::builtin
