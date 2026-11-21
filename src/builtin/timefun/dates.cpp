// src/builtin/src/timefun/dates.cpp
//
// Date conversions, calendar operations, and formatting for numkit::builtin.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/format.h>
#include <keel/string.h>
#include <keel/time.h>
#include <keel/utf8.h>
#include <numkit/builtin/timefun.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace numkit::builtin {

static inline keel::f64 civilToSerial(keel::f64 yd, keel::f64 md, keel::f64 dd, keel::f64 hd, keel::f64 mind, keel::f64 sd) {
    keel::f64 dInt;
    const keel::f64 dFrac = std::modf(dd, &dInt);
    keel::i64 y = keel::as<keel::i64>(std::floor(yd));
    keel::i64 m = keel::as<keel::i64>(std::floor(md));
    keel::i64 d = keel::as<keel::i64>(dInt);
    if (m <= 2) y -= 1;
    const keel::i64 era = (y < 0 ? y - 399 : y) / 400;
    const keel::i64 yoe = y - era * 400;
    const keel::i64 doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const keel::i64 doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    const keel::i64 days = era * 146097 + doe - 719468;
    const keel::f64 frac = (hd * 3600.0 + mind * 60.0 + sd) / 86400.0 + dFrac;
    return keel::as<keel::f64>(days) + 719529.0 + frac;
}

static inline void civilFromDays(keel::i64 z, keel::i64 &Y, int &M, int &D) {
    z += 719468;
    const keel::i64 era = (z >= 0 ? z : z - 146096) / 146097;
    const keel::i64 doe = z - era * 146097;
    const keel::i64 yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const keel::i64 y = yoe + era * 400;
    const keel::i64 doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const keel::i64 mp = (5 * doy + 2) / 153;
    D = keel::as<int>(doy - (153 * mp + 2) / 5 + 1);
    M = keel::as<int>(mp < 10 ? mp + 3 : mp - 9);
    Y = y + (M <= 2 ? 1 : 0);
}

static inline void extractTime(keel::f64 frac, int &H, int &MI, keel::f64 &S) {
    const keel::f64 total_ms = std::round(frac * 86400.0 * 1.0e3);
    keel::i64 ms = keel::as<keel::i64>(total_ms);
    H  = keel::as<int>(ms / 3600000LL); ms %= 3600000LL;
    MI = keel::as<int>(ms / 60000LL);   ms %= 60000LL;
    S  = keel::as<keel::f64>(ms) / 1.0e3;
}

static inline void serialToCivil(keel::f64 serial, keel::f64 &yd, keel::f64 &md, keel::f64 &dd, keel::f64 &hd, keel::f64 &mind, keel::f64 &sd) {
    if (serial == 0.0) {
        yd = 0; md = 0; dd = 0; hd = 0; mind = 0; sd = 0;
        return;
    }
    const keel::f64 floored = std::floor(serial);
    const keel::i64 days = keel::as<keel::i64>(floored);
    const keel::f64 frac = serial - floored;
    const keel::i64 z = days - 719529;
    keel::i64 Y; int M, D, H, MI; keel::f64 S;
    civilFromDays(z, Y, M, D);
    extractTime(frac, H, MI, S);
    if (S >= 60.0) { S -= 60.0; ++MI; }
    if (MI >= 60)  { MI -= 60;  ++H;  }
    if (H  >= 24)  { H  -= 24; civilFromDays(z + 1, Y, M, D); }
    yd = keel::as<keel::f64>(Y);
    md = keel::as<keel::f64>(M);
    dd = keel::as<keel::f64>(D);
    hd = keel::as<keel::f64>(H);
    mind = keel::as<keel::f64>(MI);
    sd = S;
}

static constexpr keel::f64 kJDFromSerial  = 1721058.5;
static constexpr keel::f64 kMJDFromSerial = -678942.0;

Value etime(const Value &t2, const Value &t1, keel::Allocator *mr) {
    const keel::usize r2 = t2.dims().rows(), c2 = t2.dims().cols();
    const keel::usize r1 = t1.dims().rows(), c1 = t1.dims().cols();
    if (c2 != 6 || c1 != 6)
        throw std::runtime_error("etime: date vectors must have 6 columns [Y M D H MI S]");
    if (r1 != r2 && r1 != 1 && r2 != 1)
        throw std::runtime_error("etime: t2 and t1 must have the same number of rows (or one a single row)");
    const keel::usize N = std::max(r1, r2);

    auto comp = [&](const Value &v, keel::usize nr, keel::usize row, keel::usize col) {
        return v.elemAsDouble(row + col * nr);
    };

    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize k = 0; k < N; ++k) {
        const keel::usize k2 = (r2 == 1 ? 0 : k);
        const keel::usize k1 = (r1 == 1 ? 0 : k);
        const keel::f64 dDay =
            civilToSerial(comp(t2, r2, k2, 0), comp(t2, r2, k2, 1), comp(t2, r2, k2, 2), 0, 0, 0)
          - civilToSerial(comp(t1, r1, k1, 0), comp(t1, r1, k1, 1), comp(t1, r1, k1, 2), 0, 0, 0);
        const keel::f64 dH  = comp(t2, r2, k2, 3) - comp(t1, r1, k1, 3);
        const keel::f64 dMI = comp(t2, r2, k2, 4) - comp(t1, r1, k1, 4);
        const keel::f64 dS  = comp(t2, r2, k2, 5) - comp(t1, r1, k1, 5);
        o[k] = 86400.0 * dDay + 3600.0 * dH + 60.0 * dMI + dS;
    }
    return out;
}

Value weeknum(keel::ArrayRef<Value> args, keel::Allocator *mr) {
    if (args.empty())
        throw std::runtime_error("weeknum: requires a date argument");

    int weekStart = 1;
    if (args.size() >= 2 && !args[1].isEmpty())
        weekStart = keel::as<int>(args[1].toScalar());
    if (weekStart < 1 || weekStart > 7)
        throw std::runtime_error("weeknum: WeekStart must be an integer in 1..7 (1=Sunday)");
    bool european = false;
    if (args.size() >= 3 && !args[2].isEmpty())
        european = args[2].toScalar() != 0.0;

    auto civilToSerialSimple = [](keel::i64 y, keel::i64 m, keel::i64 d) {
        if (m <= 2) y -= 1;
        const keel::i64 era = (y < 0 ? y - 399 : y) / 400;
        const keel::i64 yoe = y - era * 400;
        const keel::i64 doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
        const keel::i64 doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        return era * 146097 + doe - 719468 + 719529;
    };
    auto yearOf = [](keel::f64 serial) {
        keel::i64 z = keel::as<keel::i64>(std::floor(serial)) - 719529 + 719468;
        const keel::i64 era = (z >= 0 ? z : z - 146096) / 146097;
        const keel::i64 doe = z - era * 146097;
        const keel::i64 yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        keel::i64 y = yoe + era * 400;
        const keel::i64 dy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        const keel::i64 mp = (5 * dy + 2) / 153;
        const keel::i64 m = mp < 10 ? mp + 3 : mp - 9;
        if (m <= 2) y += 1;
        return keel::as<int>(y);
    };
    auto isLeap = [](int y) {
        return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    };
    auto wdJan1 = [](int y) {
        static const int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
        const int yy = y - 1;
        const int dow = ((yy + yy / 4 - yy / 100 + yy / 400 + t[0] + 1) % 7 + 7) % 7;
        return dow + 1;
    };
    auto weekOf = [&](int y, int doy) {
        const int offset = (wdJan1(y) - weekStart + 7) % 7;
        if (!european)
            return (doy - 1 + offset) / 7 + 1;
        const int partial = (offset == 0) ? 0 : (7 - offset);
        const int firstWS = (offset == 0) ? 1 : (8 - offset);
        if (partial >= 4)
            return (doy - 1 + offset) / 7 + 1;
        if (doy >= firstWS)
            return (doy - firstWS) / 7 + 1;
        const int yp = y - 1;
        const int doyp = isLeap(yp) ? 366 : 365;
        const int offp = (wdJan1(yp) - weekStart + 7) % 7;
        const int partp = (offp == 0) ? 0 : (7 - offp);
        const int fwp = (offp == 0) ? 1 : (8 - offp);
        if (partp >= 4)
            return (doyp - 1 + offp) / 7 + 1;
        return (doyp - fwp) / 7 + 1;
    };

    const Value &D = args[0];
    const keel::usize nr = D.dims().rows();
    const keel::usize nc = D.dims().cols();
    const keel::usize n = D.numel();
    auto out = Value::matrix(nr, nc, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 serial = D.elemAsDouble(i);
        const int y = yearOf(serial);
        const int doy = keel::as<int>(std::floor(serial) - civilToSerialSimple(y, 1, 1)) + 1;
        o[i] = keel::as<keel::f64>(weekOf(y, doy));
    }
    return out;
}

Value addtodate(const Value &d, keel::f64 quantity, keel::StringRef unit, keel::Allocator *mr) {
    if (d.numel() != 1)
        throw std::runtime_error("addtodate: date number must be a real numeric scalar");
    const keel::f64 serial = d.toScalar();
    const keel::f64 q = quantity;

    keel::f64 result;
    if (unit.equals_ignore_ascii_case("day"))
        result = serial + q;
    else if (unit.equals_ignore_ascii_case("hour"))
        result = serial + q / 24.0;
    else if (unit.equals_ignore_ascii_case("minute"))
        result = serial + q / 1440.0;
    else if (unit.equals_ignore_ascii_case("second"))
        result = serial + q / 86400.0;
    else if (unit.equals_ignore_ascii_case("millisecond"))
        result = serial + q / 86400000.0;
    else if (unit.equals_ignore_ascii_case("month") || unit.equals_ignore_ascii_case("year")) {
        const keel::f64 dayF = std::floor(serial);
        const keel::f64 frac = serial - dayF;
        const keel::i64 days = keel::as<keel::i64>(dayF) - 719529;
        keel::i64 z = days + 719468;
        const keel::i64 era = (z >= 0 ? z : z - 146096) / 146097;
        const keel::i64 doe = z - era * 146097;
        const keel::i64 yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        keel::i64 Y = yoe + era * 400;
        const keel::i64 doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        const keel::i64 mp = (5 * doy + 2) / 153;
        const keel::i64 D = doy - (153 * mp + 2) / 5 + 1;
        keel::i64 M = mp < 10 ? mp + 3 : mp - 9;
        Y += (M <= 2);

        keel::i64 nY = Y, nM = M;
        if (unit.equals_ignore_ascii_case("month")) {
            keel::i64 tm = (M - 1) + keel::as<keel::i64>(std::llround(q));
            keel::i64 qd = tm / 12, rd = tm % 12;
            if (rd < 0) { qd -= 1; rd += 12; }
            nY = Y + qd;
            nM = rd + 1;
        } else {
            nY = Y + keel::as<keel::i64>(std::llround(q));
        }
        auto leap = [](keel::i64 y) {
            return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
        };
        static const int dim[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        keel::i64 maxD = dim[nM - 1];
        if (nM == 2 && leap(nY)) maxD = 29;
        keel::i64 nD = D < maxD ? D : maxD;
        keel::i64 y2 = nY;
        if (nM <= 2) y2 -= 1;
        const keel::i64 era2 = (y2 < 0 ? y2 - 399 : y2) / 400;
        const keel::i64 yoe2 = y2 - era2 * 400;
        const keel::i64 doy2 = (153 * (nM + (nM > 2 ? -3 : 9)) + 2) / 5 + nD - 1;
        const keel::i64 doe2 = yoe2 * 365 + yoe2 / 4 - yoe2 / 100 + doy2;
        const keel::i64 newDays = era2 * 146097 + doe2 - 719468 + 719529;
        result = keel::as<keel::f64>(newDays) + frac;
    } else {
        throw std::runtime_error("addtodate: units must be one of 'year','month','day','hour','minute','second','millisecond'");
    }
    return Value::scalar(result, mr);
}

Value datenum(keel::ArrayRef<Value> args, keel::Allocator *mr) {
    if (args.empty())
        throw std::runtime_error("datenum requires at least one argument");

    if (args[0].isCell() || (args[0].isString() && args[0].numel() > 1)) {
        const keel::usize n = args[0].numel();
        auto out = Value::matrix(n, 1, ValueType::DOUBLE, mr);
        keel::f64 *o = out.doubleDataMut();
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<Value> subArgs(args.size(), scratch.allocator());
        for (keel::usize k = 0; k < args.size(); ++k) subArgs[k] = args[k];
        for (keel::usize i = 0; i < n; ++i) {
            subArgs[0] = args[0].isCell() ? args[0].cellAt(i)
                                          : Value::fromString(args[0].stringElem(i), mr);
            o[i] = datenum(keel::ArrayRef<Value>(subArgs.data(), subArgs.size()), mr).toScalar();
        }
        return out;
    }

    if (args[0].isChar() || args[0].isString()) {
        const keel::String sOwned = args[0].toString();
        const keel::StringRef s = sOwned.as_ref();
        static const char *MON3[] = {
            "jan","feb","mar","apr","may","jun",
            "jul","aug","sep","oct","nov","dec"};
        auto tryFmt = [&](keel::StringRef fmt, keel::f64 &Y, keel::f64 &Mo, keel::f64 &D, keel::f64 &H, keel::f64 &MI, keel::f64 &S) -> bool {
            Y = 0; Mo = 1; D = 1; H = 0; MI = 0; S = 0;
            keel::usize si = 0, fi = 0;
            auto readNum = [&](int maxD) -> long {
                long v = 0; int n = 0;
                while (si < s.size() && n < maxD && keel::is_ascii_digit(s[si])) {
                    v = v * 10 + (s[si] - '0'); ++si; ++n;
                }
                return n > 0 ? v : -1;
            };
            auto fAt = [&](keel::StringRef tok) {
                return fmt.substr(fi).starts_with(tok);
            };
            while (fi < fmt.size()) {
                if (fAt("yyyy")) {
                    long v = readNum(4); if (v < 0) return false; Y = v; fi += 4;
                } else if (fAt("yy")) {
                    long v = readNum(2); if (v < 0) return false;
                    const keel::CalendarTime nowCal = keel::local_time();
                    long curCent = (keel::as<long>(nowCal.year) / 100) * 100;
                    Y = curCent + v; fi += 2;
                } else if (fAt("mmmm") || fAt("mmm")) {
                    keel::usize mlen = fAt("mmmm") ? 4 : 3;
                    if (si + 3 > s.size()) return false;
                    const keel::StringRef m3 = s.substr(si, 3);
                    int mIdx = -1;
                    for (int i = 0; i < 12; ++i) { if (m3.equals_ignore_ascii_case(MON3[i])) { mIdx = i + 1; break; } }
                    if (mIdx < 0) return false;
                    Mo = mIdx; si += 3;
                    if (mlen == 4) { while (si < s.size() && keel::is_ascii_alpha(s[si])) ++si; }
                    fi += mlen;
                } else if (fAt("mm")) {
                    long v = readNum(2); if (v < 0) return false; Mo = v; fi += 2;
                } else if (fAt("dd")) {
                    long v = readNum(2); if (v < 0) return false; D = v; fi += 2;
                } else if (fAt("HH")) {
                    long v = readNum(2); if (v < 0) return false; H = v; fi += 2;
                } else if (fAt("MM")) {
                    long v = readNum(2); if (v < 0) return false; MI = v; fi += 2;
                } else if (fAt("SS")) {
                    long v = readNum(2); if (v < 0) return false; S = v; fi += 2;
                    if (si < s.size() && (s[si] == '.' || s[si] == ',')) {
                        ++si; keel::f64 scale = 0.1;
                        while (si < s.size() && keel::is_ascii_digit(s[si])) {
                            S += (s[si] - '0') * scale; scale *= 0.1; ++si;
                        }
                    }
                } else {
                    if (si >= s.size() || s[si] != fmt[fi]) return false;
                    ++si; ++fi;
                }
            }
            while (si < s.size() && keel::is_ascii_space(s[si])) ++si;
            return si == s.size();
        };

        keel::f64 Y, Mo, D, H, MI, S;
        if (args.size() >= 2 && (args[1].isChar() || args[1].isString())) {
            const keel::String fmtOwned = args[1].toString();
            const keel::StringRef fmt = fmtOwned.as_ref();
            if (tryFmt(fmt, Y, Mo, D, H, MI, S))
                return Value::scalar(civilToSerial(Y, Mo, D, H, MI, S), mr);
            throw std::runtime_error(keel::format("datenum: date string does not match format '{}'", fmt).c_str());
        }
        static const char *kAutoFmts[] = {
            "yyyy-mm-dd HH:MM:SS", "yyyy-mm-ddTHH:MM:SS", "yyyy-mm-dd",
            "dd-mmm-yyyy HH:MM:SS", "dd-mmm-yyyy", "yyyy/mm/dd HH:MM:SS", "yyyy/mm/dd",
            "mm/dd/yyyy HH:MM:SS", "mm/dd/yyyy", "HH:MM:SS", "dd-mmmm-yyyy", nullptr
        };
        for (int i = 0; kAutoFmts[i]; ++i) {
            if (tryFmt(kAutoFmts[i], Y, Mo, D, H, MI, S))
                return Value::scalar(civilToSerial(Y, Mo, D, H, MI, S), mr);
        }
        throw std::runtime_error(keel::format("datenum: could not auto-detect format of date string '{}'", s).c_str());
    }

    if (args.size() == 1) {
        const Value &A = args[0];
        const keel::usize rows = A.dims().rows(), cols = A.dims().cols();
        if (cols == 6 || cols == 3) {
            auto out = Value::matrix(rows, 1, ValueType::DOUBLE, mr);
            keel::f64 *o = out.doubleDataMut();
            for (keel::usize r = 0; r < rows; ++r) {
                keel::f64 Y  = A.elemAsDouble(r);
                keel::f64 Mo = A.elemAsDouble(r + rows);
                keel::f64 D  = A.elemAsDouble(r + 2 * rows);
                keel::f64 H = 0.0, MI = 0.0, S = 0.0;
                if (cols == 6) {
                    H  = A.elemAsDouble(r + 3 * rows);
                    MI = A.elemAsDouble(r + 4 * rows);
                    S  = A.elemAsDouble(r + 5 * rows);
                }
                o[r] = civilToSerial(Y, Mo, D, H, MI, S);
            }
            return out;
        }
        return A;
    }

    if (args.size() != 3 && args.size() != 6)
        throw std::runtime_error("datenum: expected 1, 3, or 6 arguments");

    keel::usize N = 1;
    for (const auto &a : args) {
        if (a.numel() > 1) {
            if (N > 1 && a.numel() != N)
                throw std::runtime_error("datenum: input vector lengths must match");
            N = a.numel();
        }
    }
    auto pick = [](const Value &a, keel::usize i) {
        return a.numel() == 1 ? a.toScalar() : a.elemAsDouble(i);
    };
    if (N == 1) {
        const keel::f64 h = args.size() == 6 ? args[3].toScalar() : 0.0;
        const keel::f64 mi = args.size() == 6 ? args[4].toScalar() : 0.0;
        const keel::f64 s = args.size() == 6 ? args[5].toScalar() : 0.0;
        return Value::scalar(civilToSerial(args[0].toScalar(), args[1].toScalar(), args[2].toScalar(), h, mi, s), mr);
    }
    auto out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 y = pick(args[0], i);
        const keel::f64 m = pick(args[1], i);
        const keel::f64 d = pick(args[2], i);
        keel::f64 h = 0.0, mi = 0.0, s = 0.0;
        if (args.size() == 6) {
            h  = pick(args[3], i);
            mi = pick(args[4], i);
            s  = pick(args[5], i);
        }
        o[i] = civilToSerial(y, m, d, h, mi, s);
    }
    return out;
}

Value weekday(const Value &d, keel::Allocator *mr) {
    auto dayIndex = [](keel::f64 val) -> int {
        keel::i64 f = keel::as<keel::i64>(std::floor(val)) - 2;
        keel::i64 r = f % 7;
        if (r < 0) r += 7;
        return keel::as<int>(r) + 1;
    };
    const keel::usize N = d.numel();
    if (N == 1) {
        return Value::scalar(keel::as<keel::f64>(dayIndex(d.toScalar())), mr);
    }
    auto out = Value::matrix(d.dims().rows(), d.dims().cols(), ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i)
        o[i] = keel::as<keel::f64>(dayIndex(d.elemAsDouble(i)));
    return out;
}

Value juliandate(keel::ArrayRef<Value> args, keel::Allocator *mr) {
    if (args.empty())
        throw std::runtime_error("juliandate requires at least one argument");
    if (args.size() == 1) {
        const Value &A = args[0];
        if (A.dims().cols() == 6 || A.dims().cols() == 3) {
            Value s = datenum(args, mr);
            keel::f64 *p = s.doubleDataMut();
            for (keel::usize i = 0; i < s.numel(); ++i) p[i] += kJDFromSerial;
            return s;
        }
        const keel::usize R = A.dims().rows(), C = A.dims().cols();
        auto out = Value::matrix(R, C, ValueType::DOUBLE, mr);
        keel::f64 *o = out.doubleDataMut();
        for (keel::usize i = 0; i < R * C; ++i) o[i] = A.elemAsDouble(i) + kJDFromSerial;
        return R == 1 && C == 1 ? Value::scalar(o[0], mr) : out;
    }
    Value s = datenum(args, mr);
    keel::f64 *p = s.doubleDataMut();
    for (keel::usize i = 0; i < s.numel(); ++i) p[i] += kJDFromSerial;
    return s;
}

Value mjuliandate(keel::ArrayRef<Value> args, keel::Allocator *mr) {
    if (args.empty())
        throw std::runtime_error("mjuliandate requires at least one argument");
    if (args.size() == 1) {
        const Value &A = args[0];
        if (A.dims().cols() == 6 || A.dims().cols() == 3) {
            Value s = datenum(args, mr);
            keel::f64 *p = s.doubleDataMut();
            for (keel::usize i = 0; i < s.numel(); ++i) p[i] += kMJDFromSerial;
            return s;
        }
        const keel::usize R = A.dims().rows(), C = A.dims().cols();
        auto out = Value::matrix(R, C, ValueType::DOUBLE, mr);
        keel::f64 *o = out.doubleDataMut();
        for (keel::usize i = 0; i < R * C; ++i) o[i] = A.elemAsDouble(i) + kMJDFromSerial;
        return R == 1 && C == 1 ? Value::scalar(o[0], mr) : out;
    }
    Value s = datenum(args, mr);
    keel::f64 *p = s.doubleDataMut();
    for (keel::usize i = 0; i < s.numel(); ++i) p[i] += kMJDFromSerial;
    return s;
}

Value eomday(const Value &y, const Value &m, keel::Allocator *mr) {
    auto isLeap = [](keel::i64 yr) {
        return (yr % 4 == 0 && yr % 100 != 0) || yr % 400 == 0;
    };
    static const int dim[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    auto eomOne = [&](keel::f64 yr, keel::f64 mo) -> keel::f64 {
        keel::i64 yi = keel::as<keel::i64>(yr);
        keel::i64 mi = keel::as<keel::i64>(mo);
        if (mi < 1 || mi > 12)
            throw std::runtime_error("eomday: month must be in 1..12");
        return (mi == 2 && isLeap(yi)) ? 29.0 : keel::as<keel::f64>(dim[mi]);
    };
    const keel::usize ny = y.numel(), nm = m.numel();
    if (ny == 1 && nm == 1)
        return Value::scalar(eomOne(y.toScalar(), m.toScalar()), mr);
    if (ny != nm && ny != 1 && nm != 1)
        throw std::runtime_error("eomday: input dimensions must agree");
    const keel::usize N = std::max(ny, nm);
    const keel::usize rows = ny > 1 ? y.dims().rows() : m.dims().rows();
    const keel::usize cols = ny > 1 ? y.dims().cols() : m.dims().cols();
    auto out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        keel::f64 yi = ny == 1 ? y.toScalar() : y.elemAsDouble(i);
        keel::f64 mi = nm == 1 ? m.toScalar() : m.elemAsDouble(i);
        o[i] = eomOne(yi, mi);
    }
    return out;
}

Value calendar(keel::ArrayRef<Value> args, keel::Allocator *mr) {
    keel::f64 yr = 0, mo = 0;
    if (args.empty()) {
        const keel::CalendarTime nowCal = keel::local_time();
        yr = keel::as<keel::f64>(nowCal.year);
        mo = keel::as<keel::f64>(nowCal.month);
    } else if (args.size() == 1) {
        keel::f64 d, h, mi, s;
        serialToCivil(args[0].toScalar(), yr, mo, d, h, mi, s);
    } else {
        yr = args[0].toScalar();
        mo = args[1].toScalar();
    }
    keel::i64 yi = keel::as<keel::i64>(yr);
    keel::i64 mi = keel::as<keel::i64>(mo);
    if (mi < 1 || mi > 12)
        throw std::runtime_error("calendar: month must be in 1..12");
    auto isLeap = [](keel::i64 y) {
        return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    };
    static const int dim[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    int daysInMonth = (mi == 2 && isLeap(yi)) ? 29 : dim[mi];
    keel::f64 s1 = civilToSerial(keel::as<keel::f64>(yi), keel::as<keel::f64>(mi), 1.0, 0, 0, 0);
    keel::i64 f = keel::as<keel::i64>(std::floor(s1)) - 2;
    keel::i64 r = f % 7; if (r < 0) r += 7;
    int startCol = keel::as<int>(r);

    auto out = Value::matrix(6, 7, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    std::fill(o, o + 42, 0.0);
    int day = 1;
    for (int col = startCol; col < 7 && day <= daysInMonth; ++col) {
        o[0 + col * 6] = day++;
    }
    for (int row = 1; row < 6 && day <= daysInMonth; ++row) {
        for (int col = 0; col < 7 && day <= daysInMonth; ++col) {
            o[row + col * 6] = day++;
        }
    }
    return out;
}

Value datestr(keel::ArrayRef<Value> args, keel::Allocator *mr) {
    if (args.empty()) {
        const keel::f64 nowSerial = now();
        Value s = Value::scalar(nowSerial, mr);
        return datestr(keel::ArrayRef<Value>(&s, 1), mr);
    }
    const Value &din = args[0];

    struct Comp { int y, mo, d, h, mi, s; };
    auto serialToComp = [&](keel::f64 dval) -> Comp {
        const keel::f64 floored = std::floor(dval);
        const keel::i64 z = keel::as<keel::i64>(floored) - 719529;
        const keel::f64 frac = dval - floored;
        keel::i64 Y; int M, D;
        civilFromDays(z, Y, M, D);
        keel::i64 ms = keel::as<keel::i64>(std::round(frac * 86400.0 * 1.0e3));
        int H  = keel::as<int>(ms / 3600000LL); ms %= 3600000LL;
        int MI = keel::as<int>(ms / 60000LL);   ms %= 60000LL;
        keel::f64 S = keel::as<keel::f64>(ms) / 1.0e3;
        if (S >= 60.0) { S -= 60.0; ++MI; }
        if (MI >= 60)  { MI -= 60;  ++H;  }
        if (H  >= 24)  { H  -= 24; civilFromDays(z + 1, Y, M, D); }
        return Comp{ keel::as<int>(Y), M, D, H, MI, keel::as<int>(std::round(S)) };
    };

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Allocator *salloc = scratch.allocator();
    keel::Array<Comp> dates(salloc);
    if (din.isChar() || din.isString()) {
        static const char *MON3p[] = {
            "jan","feb","mar","apr","may","jun",
            "jul","aug","sep","oct","nov","dec"};
        const keel::String sOwned = din.toString();
        const keel::StringRef s = sOwned.as_ref();
        auto tryFmt = [&](const char *fmt, Comp &c) -> bool {
            int Y = 0, Mo = 1, D = 1, H = 0, MI = 0, S = 0;
            const keel::StringRef F(fmt);
            keel::usize si = 0, fi = 0;
            auto readNum = [&](int maxD) -> long {
                long v = 0; int n = 0;
                while (si < s.size() && n < maxD && keel::is_ascii_digit(s[si])) {
                    v = v * 10 + (s[si] - '0'); ++si; ++n;
                }
                return n > 0 ? v : -1;
            };
            auto fAt = [&](keel::StringRef tok) {
                return F.substr(fi).starts_with(tok);
            };
            while (fi < F.size()) {
                if (fAt("yyyy")) { long v=readNum(4); if(v<0)return false; Y=(int)v; fi+=4; }
                else if (fAt("mmm")) {
                    if (si+3>s.size()) return false;
                    const keel::StringRef m=s.substr(si,3);
                    int mi=-1; for(int k=0;k<12;++k) if(m.equals_ignore_ascii_case(MON3p[k])){mi=k+1;break;}
                    if (mi<0) return false; Mo=mi; si+=3; fi+=3;
                }
                else if (fAt("mm")) { long v=readNum(2); if(v<0)return false; Mo=(int)v; fi+=2; }
                else if (fAt("dd")) { long v=readNum(2); if(v<0)return false; D=(int)v; fi+=2; }
                else if (fAt("HH")) { long v=readNum(2); if(v<0)return false; H=(int)v; fi+=2; }
                else if (fAt("MM")) { long v=readNum(2); if(v<0)return false; MI=(int)v; fi+=2; }
                else if (fAt("SS")) { long v=readNum(2); if(v<0)return false; S=(int)v; fi+=2; }
                else { if (si<s.size() && s[si]==F[fi]) { ++si; ++fi; } else return false; }
            }
            if (si != s.size()) return false;
            c = Comp{Y, Mo, D, H, MI, S};
            return true;
        };
        static const char *cands[] = {
            "yyyy-mm-dd HH:MM:SS", "yyyy-mm-dd",
            "dd-mmm-yyyy HH:MM:SS", "dd-mmm-yyyy"};
        Comp c{}; bool ok = false;
        for (const char *f : cands) if (tryFmt(f, c)) { ok = true; break; }
        if (!ok)
            throw std::runtime_error(
                "datestr: could not parse date string (supported: ISO yyyy-mm-dd and dd-mmm-yyyy forms)");
        dates.push_back(c);
    } else {
        const keel::usize Rr = din.dims().rows();
        const keel::usize Cc = din.dims().cols();
        if (Cc == 6 && !din.dims().is3D()) {
            dates.reserve(Rr);
            for (keel::usize r = 0; r < Rr; ++r)
                dates.push_back(Comp{
                    keel::as<int>(din.elemAsDouble(0 * Rr + r)),
                    keel::as<int>(din.elemAsDouble(1 * Rr + r)),
                    keel::as<int>(din.elemAsDouble(2 * Rr + r)),
                    keel::as<int>(din.elemAsDouble(3 * Rr + r)),
                    keel::as<int>(din.elemAsDouble(4 * Rr + r)),
                    keel::as<int>(std::round(din.elemAsDouble(5 * Rr + r)))});
        } else {
            const keel::usize n = din.numel();
            dates.reserve(n);
            for (keel::usize k = 0; k < n; ++k)
                dates.push_back(serialToComp(din.elemAsDouble(k)));
        }
    }
    if (dates.empty())
        throw std::runtime_error("datestr: empty date input");
    bool anyTime = false;
    for (const auto &c : dates)
        if (c.h != 0 || c.mi != 0 || c.s != 0) { anyTime = true; break; }

    keel::String fmtOwned(salloc);
    keel::StringRef fmt;
    if (args.size() >= 2) {
        const Value &f = args[1];
        if (f.isChar() || f.isString()) {
            fmtOwned = f.toString();
            fmt = fmtOwned.as_ref();
        } else {
            static const char *DATEFORM[] = {
                "dd-mmm-yyyy HH:MM:SS", // 0
                "dd-mmm-yyyy",          // 1
                "mm/dd/yy",             // 2
                "mmm",                  // 3
                "m",                    // 4
                "mm",                   // 5
                "mm/dd",                // 6
                "dd",                   // 7
                "ddd",                  // 8
                "d",                    // 9
                "yyyy",                 // 10
                "yy",                   // 11
                "mmmyy",                // 12
                "HH:MM:SS",             // 13
                "HH:MM:SS PM",          // 14
                "HH:MM",                // 15
                "HH:MM PM",             // 16
                "QQ-yy",                // 17
                "QQ",                   // 18
                "dd/mm",                // 19
                "dd/mm/yy",             // 20
                "mmm.dd,yyyy HH:MM:SS", // 21
                "mmm.dd,yyyy",          // 22
                "mm/dd/yyyy",           // 23
                "dd/mm/yyyy",           // 24
                "yy/mm/dd",             // 25
                "yyyy/mm/dd",           // 26
                "QQ-yyyy",              // 27
                "mmmyyyy",              // 28
                "yyyy-mm-dd",           // 29
                "yyyymmddTHHMMSS",      // 30
                "yyyy-mm-dd HH:MM:SS",  // 31
            };
            const int code = keel::as<int>(std::round(f.elemAsDouble(0)));
            if (code < 0 || code > 31)
                throw std::runtime_error("datestr: unsupported numeric format code (expected 0-31)");
            fmt = DATEFORM[code];
        }
    } else {
        fmt = anyTime ? "dd-mmm-yyyy HH:MM:SS" : "dd-mmm-yyyy";
    }

    static const char *MON3[] = {
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"};
    static const char *MONF[] = {
        "January","February","March","April","May",
        "June","July","August","September","October",
        "November","December"};
    static const char *DOW3[] = {
        "Sun","Mon","Tue","Wed","Thu","Fri","Sat"};
    static const char *DOWF[] = {
        "Sunday","Monday","Tuesday","Wednesday",
        "Thursday","Friday","Saturday"};
    static const int dt[] = {0,3,2,5,0,3,5,1,4,6,2,4};
    bool hour12 = false;
    for (keel::usize k = 0; k + 1 < fmt.size(); ++k) {
        char a = keel::to_ascii_lower(fmt[k]);
        char b = keel::to_ascii_lower(fmt[k+1]);
        if ((a == 'a' || a == 'p') && b == 'm') { hour12 = true; break; }
    }

    auto renderOne = [&](const Comp &cc) -> keel::String {
        const int yi = cc.y, moi = cc.mo, di = cc.d,
                  hi = cc.h, mii = cc.mi, si = cc.s;
        int yw = yi - (moi < 3 ? 1 : 0);
        int dow = ((yw + yw/4 - yw/100 + yw/400 + dt[(moi - 1 + 12) % 12] + di) % 7 + 7) % 7;
        const int h12 = (hi % 12 == 0) ? 12 : (hi % 12);
        keel::String out(salloc);
        char buf[16];
        keel::usize i = 0;
        auto at = [&](const char *t, keel::usize L) {
            return fmt.substr(i).starts_with(keel::StringRef(t, L));
        };
        while (i < fmt.size()) {
            if (at("yyyy", 4)) { std::snprintf(buf,sizeof buf,"%04d",yi); out.append(buf); i+=4; }
            else if (at("yy", 2)) { std::snprintf(buf,sizeof buf,"%02d",((yi%100)+100)%100); out.append(buf); i+=2; }
            else if (at("mmmm", 4)) { out.append(MONF[(moi-1+12)%12]); i+=4; }
            else if (at("mmm", 3)) { out.append(MON3[(moi-1+12)%12]); i+=3; }
            else if (at("mm", 2)) { std::snprintf(buf,sizeof buf,"%02d",moi); out.append(buf); i+=2; }
            else if (at("m", 1)) { out.push_back(MON3[(moi-1+12)%12][0]); i+=1; }
            else if (at("QQ", 2)) { out.push_back('Q'); out.push_back(keel::as<char>('0' + ((moi - 1) / 3 + 1))); i+=2; }
            else if (at("dddd", 4)) { out.append(DOWF[dow]); i+=4; }
            else if (at("ddd", 3)) { out.append(DOW3[dow]); i+=3; }
            else if (at("dd", 2)) { std::snprintf(buf,sizeof buf,"%02d",di); out.append(buf); i+=2; }
            else if (at("d", 1)) { out.push_back(DOW3[dow][0]); i+=1; }
            else if (at("HH", 2)) {
                if (hour12) std::snprintf(buf,sizeof buf,"%2d",h12);
                else        std::snprintf(buf,sizeof buf,"%02d",hi);
                out.append(buf); i+=2;
            }
            else if (at("MM", 2)) { std::snprintf(buf,sizeof buf,"%02d",mii); out.append(buf); i+=2; }
            else if (at("SS", 2)) { std::snprintf(buf,sizeof buf,"%02d",si); out.append(buf); i+=2; }
            else if (hour12 && i + 1 < fmt.size()
                     && ((keel::to_ascii_lower(fmt[i])=='a'
                          || keel::to_ascii_lower(fmt[i])=='p')
                         && keel::to_ascii_lower(fmt[i+1])=='m')) {
                out.append((hi < 12) ? "AM" : "PM"); i += 2;
            }
            else { out.push_back(fmt[i]); ++i; }
        }
        return out;
    };

    if (dates.size() == 1) {
        return Value::fromString(renderOne(dates[0]), mr);
    }
    keel::Array<keel::String> rowstr(salloc);
    rowstr.reserve(dates.size());
    keel::usize maxW = 0;
    for (const auto &c : dates) {
        rowstr.push_back(renderOne(c));
        if (rowstr.back().size() > maxW)
            maxW = rowstr.back().size();
    }
    const keel::usize N = rowstr.size();
    Value Mc = Value::matrix(N, maxW, ValueType::CHAR, mr);
    char *dst = static_cast<char *>(Mc.rawDataMut());
    for (keel::usize r = 0; r < N; ++r)
        for (keel::usize c = 0; c < maxW; ++c)
            dst[c * N + r] = (c < rowstr[r].size()) ? rowstr[r][c] : ' ';
    return Mc;
}

Value datevec(keel::ArrayRef<Value> args, keel::Allocator *mr) {
    if (args.empty())
        throw std::runtime_error("datevec requires at least one argument");
    if (args[0].isChar() || args[0].isString()) {
        const keel::String sOwned = args[0].toString();
        const keel::StringRef s = sOwned.as_ref();
        static const char *MON3s[] = {
            "jan","feb","mar","apr","may","jun",
            "jul","aug","sep","oct","nov","dec"};
        auto tryFmt = [&](keel::StringRef fmt, keel::f64 &Y, keel::f64 &Mo, keel::f64 &D, keel::f64 &H, keel::f64 &MI, keel::f64 &S) -> bool {
            Y = 0; Mo = 1; D = 1; H = 0; MI = 0; S = 0;
            keel::usize si = 0, fi = 0;
            auto readNum = [&](int maxD) -> long {
                long v = 0; int n = 0;
                while (si < s.size() && n < maxD && keel::is_ascii_digit(s[si])) {
                    v = v * 10 + (s[si] - '0'); ++si; ++n;
                }
                return n > 0 ? v : -1;
            };
            auto fAt = [&](keel::StringRef tok) {
                return fmt.substr(fi).starts_with(tok);
            };
            while (fi < fmt.size()) {
                if (fAt("yyyy")) {
                    long v = readNum(4); if (v < 0) return false; Y = keel::as<keel::f64>(v); fi += 4;
                } else if (fAt("mmmm") || fAt("mmm")) {
                    bool full = fAt("mmmm");
                    if (si + 3 > s.size()) return false;
                    const keel::StringRef mon = s.substr(si, 3);
                    int mi = -1;
                    for (int k = 0; k < 12; ++k) if (mon.equals_ignore_ascii_case(MON3s[k])) { mi = k + 1; break; }
                    if (mi < 0) return false;
                    Mo = keel::as<keel::f64>(mi); si += 3;
                    if (full) {
                        while (si < s.size() && keel::is_ascii_alpha(s[si])) ++si;
                        fi += 4;
                    } else { fi += 3; }
                } else if (fAt("mm")) {
                    long v = readNum(2); if (v < 0) return false; Mo = keel::as<keel::f64>(v); fi += 2;
                } else if (fAt("dd")) {
                    long v = readNum(2); if (v < 0) return false; D = keel::as<keel::f64>(v); fi += 2;
                } else if (fAt("HH")) {
                    long v = readNum(2); if (v < 0) return false; H = keel::as<keel::f64>(v); fi += 2;
                } else if (fAt("MM")) {
                    long v = readNum(2); if (v < 0) return false; MI = keel::as<keel::f64>(v); fi += 2;
                } else if (fAt("SS")) {
                    long v = readNum(2); if (v < 0) return false; S = keel::as<keel::f64>(v); fi += 2;
                } else {
                    if (si < s.size() && s[si] == fmt[fi]) { ++si; ++fi; }
                    else return false;
                }
            }
            return si == s.size();
        };
        keel::f64 Y, Mo, D, H, MI, S;
        bool ok = false;
        if (args.size() >= 2 && (args[1].isChar() || args[1].isString())) {
            ok = tryFmt(args[1].toString(), Y, Mo, D, H, MI, S);
        } else {
            static const char *cands[] = {
                "yyyy-mm-dd HH:MM:SS", "yyyy-mm-dd",
                "dd-mmm-yyyy HH:MM:SS", "dd-mmm-yyyy"};
            for (const char *c : cands)
                if (tryFmt(c, Y, Mo, D, H, MI, S)) { ok = true; break; }
        }
        if (!ok)
            throw std::runtime_error(
                "datevec: could not parse date string (supported: explicit format string, or ISO yyyy-mm-dd and dd-mmm-yyyy forms)");
        const keel::f64 vals[6] = {Y, Mo, D, H, MI, S};
        auto out = Value::matrix(1, 6, ValueType::DOUBLE, mr);
        keel::f64 *o = out.doubleDataMut();
        for (int k = 0; k < 6; ++k) o[k] = vals[k];
        return out;
    }

    const Value &Din = args[0];
    const keel::usize N = Din.numel();
    auto out = Value::matrix(N, 6, ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    keel::f64 tmp[6];
    for (keel::usize i = 0; i < N; ++i) {
        serialToCivil(Din.elemAsDouble(i), tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5]);
        for (int c = 0; c < 6; ++c)
            o[i + c * N] = tmp[c];
    }
    return out;
}

Value yyyymmdd(keel::ArrayRef<Value> args, keel::Allocator *mr) {
    if (args.empty())
        throw std::runtime_error("yyyymmdd requires one argument");
    if (args[0].isChar() || args[0].isString())
        throw std::runtime_error("yyyymmdd: string parsing not supported");

    auto packOne = [&](keel::f64 dval) {
        if (dval == 0.0) return 0.0;
        const keel::i64 days = keel::as<keel::i64>(std::floor(dval));
        const keel::i64 z = days - 719529;
        keel::i64 Y; int M, D;
        civilFromDays(z, Y, M, D);
        return keel::as<keel::f64>(Y * 10000 + M * 100 + D);
    };

    const Value &Din = args[0];
    const keel::usize N = Din.numel();
    if (N == 1) {
        return Value::scalar(packOne(Din.toScalar()), mr);
    }
    auto out = Value::matrix(Din.dims().rows(), Din.dims().cols(), ValueType::DOUBLE, mr);
    keel::f64 *o = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i)
        o[i] = packOne(Din.elemAsDouble(i));
    return out;
}

} // namespace numkit::builtin
