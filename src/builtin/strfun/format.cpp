// toolboxes/builtin/src/datatypes/strings/format.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/builtin/strfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace numkit::builtin {

namespace {

inline void appendRepeat(keel::String &out, keel::usize count, char ch)
{
    for (keel::usize i = 0; i < count; ++i)
        out.push_back(ch);
}

// Apply a %s conversion spec (flags / width / precision) to a string value,
// matching MATLAB/C printf semantics: precision caps the number of characters
// emitted, width right-justifies (or left-justifies with the '-' flag) by
// padding with spaces. Length modifiers (l/h) are ignored. Done manually
// rather than via snprintf to avoid passing user-controlled format strings.
void applyStringSpec(keel::StringRef spec, keel::StringRef sv, keel::String &out)
{
    bool leftAlign = false;
    keel::usize k = 1;                       // skip '%'
    for (; k < spec.size(); ++k) {
        const char c = spec[k];
        if (c == '-') leftAlign = true;
        else if (c == '+' || c == ' ' || c == '0' || c == '#') { /* flag */ }
        else break;
    }
    keel::usize width = 0;
    for (; k < spec.size() && keel::is_ascii_digit(spec[k]); ++k)
        width = width * 10 + keel::as<keel::usize>(spec[k] - '0');
    long precision = -1;
    if (k < spec.size() && spec[k] == '.') {
        ++k;
        precision = 0;
        for (; k < spec.size() && keel::is_ascii_digit(spec[k]); ++k)
            precision = precision * 10 + (spec[k] - '0');
    }

    if (precision >= 0 && keel::as<keel::usize>(precision) < sv.size())
        sv = sv.substr(0, keel::as<keel::usize>(precision));
    if (width > sv.size()) {
        const keel::usize pad = width - sv.size();
        if (!leftAlign) appendRepeat(out, pad, ' ');
        out.append(sv);
        if (leftAlign) appendRepeat(out, pad, ' ');
    } else {
        out.append(sv);
    }
}

// Strip the trailing conversion char (and any l/h length modifiers) from a
// printf spec, leaving "%[flags][width][.precision]".
keel::StringRef specBody(keel::StringRef spec)
{
    keel::usize len = spec.size() > 0 ? spec.size() - 1 : 0; // drop type char
    while (len > 0 && (spec[len - 1] == 'l' || spec[len - 1] == 'h'))
        --len;
    return spec.substr(0, len);
}

// Format a non-finite value the way MATLAB does: Inf / -Inf / NaN (capital),
// honouring the field width and the '+'/' ' sign flags (only on +Inf; NaN
// never takes a sign). Precision is ignored. Used for both float (%f/%e/%g)
// and integer (%d/...) conversions.
void formatNonFinite(keel::StringRef spec, keel::f64 v, keel::String &out)
{
    keel::StringRef word = std::isnan(v) ? "NaN" : (v < 0 ? "-Inf" : "Inf");

    bool leftAlign = false, plus = false, space = false;
    keel::usize k = 1; // skip '%'
    for (; k < spec.size(); ++k) {
        const char c = spec[k];
        if (c == '-') leftAlign = true;
        else if (c == '+') plus = true;
        else if (c == ' ') space = true;
        else if (c == '0' || c == '#') { /* flag */ }
        else break;
    }
    keel::usize width = 0;
    for (; k < spec.size() && keel::is_ascii_digit(spec[k]); ++k)
        width = width * 10 + keel::as<keel::usize>(spec[k] - '0');

    if (word == "Inf") {
        if (plus) word = "+Inf";
        else if (space) word = " Inf";
    }
    if (width > word.size()) {
        const keel::usize pad = width - word.size();
        if (!leftAlign) appendRepeat(out, pad, ' ');
        out.append(word);
        if (leftAlign) appendRepeat(out, pad, ' ');
    } else {
        out.append(word);
    }
}

// Format an integer conversion (%d/%i/%u/%o/%x/%X). MATLAB semantics:
//   - a finite whole number prints as an integer;
//   - a non-integer value falls back to %e, keeping flags/width/precision
//     (e.g. sprintf('%d',3.7) -> '3.700000e+00', sprintf('%.2d',3.7) ->
//     '3.70e+00');
//   - Inf / -Inf / NaN print as 'Inf' / '-Inf' / 'NaN' (width honoured).
void formatIntegerConv(keel::StringRef spec, char type, keel::f64 v, keel::String &out)
{
    char buf[160];
    if (!std::isfinite(v)) {
        formatNonFinite(spec, v, out);
        return;
    }

    const keel::StringRef body = specBody(spec);
    keel::SmallString<64> fmtStr;
    fmtStr.append(body);

    const bool whole =
        (v == std::trunc(v)) && std::fabs(v) < 9.2e18; // fits in int64
    if (whole) {
        if (type == 'u') {
            fmtStr.append("llu");
            std::snprintf(buf, sizeof(buf), fmtStr.c_str(),
                          keel::as<unsigned long long>(keel::as<long long>(v)));
        } else if (type == 'x' || type == 'X') {
            fmtStr.append("ll");
            fmtStr.push_back(type);
            std::snprintf(buf, sizeof(buf), fmtStr.c_str(),
                          keel::as<unsigned long long>(keel::as<long long>(v)));
        } else if (type == 'o') {
            fmtStr.append("llo");
            std::snprintf(buf, sizeof(buf), fmtStr.c_str(),
                          keel::as<unsigned long long>(keel::as<long long>(v)));
        } else { // d, i
            fmtStr.append("lld");
            std::snprintf(buf, sizeof(buf), fmtStr.c_str(),
                          keel::as<long long>(v));
        }
        out.append(buf);
        return;
    }

    // Non-integer → %e fallback (MATLAB overrides the integer conversion).
    fmtStr.push_back('e');
    std::snprintf(buf, sizeof(buf), fmtStr.c_str(), v);
    out.append(buf);
}

} // namespace

// ════════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════════

keel::String formatOnce(keel::StringRef fmt, keel::ArrayRef<Value> args, keel::usize argStart,
                        bool literalWhenShort)
{
    keel::String out;
    out.reserve(fmt.size());
    keel::usize ai = argStart;

    for (keel::usize i = 0; i < fmt.size(); ++i) {
        if (fmt[i] == '\\' && i + 1 < fmt.size()) {
            char next = fmt[i + 1];
            if (next == 'n')  { out.push_back('\n'); i++; continue; }
            if (next == 't')  { out.push_back('\t'); i++; continue; }
            if (next == '\\') { out.push_back('\\'); i++; continue; }
            if (next == '\'') { out.push_back('\''); i++; continue; }
            out.push_back(fmt[i]);
            continue;
        }

        if (fmt[i] == '%') {
            if (i + 1 < fmt.size() && fmt[i + 1] == '%') {
                out.push_back('%');
                i++;
                continue;
            }

            keel::usize start = i;
            i++;

            while (i < fmt.size()
                   && (fmt[i] == '-' || fmt[i] == '+' || fmt[i] == '0' || fmt[i] == ' '
                       || fmt[i] == '#'))
                i++;
            if (i < fmt.size() && fmt[i] == '*') {
                i++;
            } else {
                while (i < fmt.size() && fmt[i] >= '0' && fmt[i] <= '9')
                    i++;
            }
            if (i < fmt.size() && fmt[i] == '.') {
                i++;
                if (i < fmt.size() && fmt[i] == '*') {
                    i++;
                } else {
                    while (i < fmt.size() && fmt[i] >= '0' && fmt[i] <= '9')
                        i++;
                }
            }
            while (i < fmt.size() && (fmt[i] == 'l' || fmt[i] == 'h'))
                i++;

            if (i >= fmt.size())
                break;

            char type = fmt[i];
            // The verbatim spec text (before any '*' splicing), emitted as a
            // literal when this conversion runs out of arguments and the
            // caller asked for that behaviour (MATLAB compose short chunks).
            const keel::StringRef literalSpec = fmt.substr(start, i - start + 1);

            // Resolve C-style '*' field width / precision taken from the
            // argument list (MATLAB supports this: sprintf('%*.*f',8,2,pi)).
            // Each '*' consumes one numeric arg, in order (width then
            // precision), BEFORE the conversion consumes its value arg. We
            // splice the concrete integer into the spec so the downstream
            // string / integer / float formatters need no '*' awareness.
            keel::SmallString<64> spec;
            for (keel::usize k = 0; k < literalSpec.size(); ++k) {
                if (literalSpec[k] == '*') {
                    long w = (ai < args.size())
                                 ? keel::as<long>(args[ai].toScalar())
                                 : 0;
                    if (ai < args.size()) ++ai;
                    char wbuf[32];
                    std::snprintf(wbuf, sizeof(wbuf), "%ld", w);
                    spec.append(wbuf);
                } else {
                    spec.push_back(literalSpec[k]);
                }
            }

            if (type == 's') {
                // MATLAB %s accepts both char arrays and string scalars, and
                // honours width / precision in the spec (e.g. %5s, %-5s, %.1s).
                if (ai < args.size()
                    && (args[ai].isChar() || args[ai].isString()))
                    applyStringSpec(spec.as_ref(), args[ai].toString(), out);
                else if (literalWhenShort && ai >= args.size())
                    out.append(literalSpec);
                ai++;
            } else if (type == 'c') {
                if (ai < args.size()) {
                    if (args[ai].isChar()) {
                        auto s = args[ai].toString();
                        out.push_back(s.empty() ? ' ' : s[0]);
                    } else {
                        out.push_back(keel::as<char>(keel::as<int>(args[ai].toScalar())));
                    }
                } else if (literalWhenShort) {
                    out.append(literalSpec);
                }
                ai++;
            } else if (type == 'd' || type == 'i' || type == 'u'
                       || type == 'x' || type == 'X' || type == 'o') {
                if (ai < args.size())
                    formatIntegerConv(spec.as_ref(), type, args[ai].toScalar(), out);
                else if (literalWhenShort)
                    out.append(literalSpec);
                ai++;
            } else if (type == 'f' || type == 'e' || type == 'E' || type == 'g' || type == 'G') {
                if (ai < args.size()) {
                    const keel::f64 v = args[ai].toScalar();
                    if (!std::isfinite(v)) {
                        // MATLAB prints Inf / -Inf / NaN (capitalised) rather
                        // than the C library's lowercase inf/nan.
                        formatNonFinite(spec.as_ref(), v, out);
                    } else {
                        char buf[128];
                        std::snprintf(buf, sizeof(buf), spec.c_str(), v);
                        out.append(buf);
                    }
                } else if (literalWhenShort) {
                    out.append(literalSpec);
                }
                ai++;
            } else {
                out.append(spec.as_ref());
            }
            continue;
        }

        out.push_back(fmt[i]);
    }
    return out;
}

keel::usize countFormatSpecs(keel::StringRef fmt)
{
    keel::usize n = 0;
    for (keel::usize i = 0; i < fmt.size(); ++i) {
        if (fmt[i] != '%') continue;
        if (i + 1 < fmt.size() && fmt[i + 1] == '%') { ++i; continue; }
        ++i;
        // flags
        while (i < fmt.size()
               && (fmt[i] == '-' || fmt[i] == '+' || fmt[i] == '0' || fmt[i] == ' '
                   || fmt[i] == '#'))
            ++i;
        // width: '*' consumes one arg from the list, else literal digits
        if (i < fmt.size() && fmt[i] == '*') { ++n; ++i; }
        else while (i < fmt.size() && keel::is_ascii_digit(fmt[i])) ++i;
        // precision: '.' then '*' (one arg) or literal digits
        if (i < fmt.size() && fmt[i] == '.') {
            ++i;
            if (i < fmt.size() && fmt[i] == '*') { ++n; ++i; }
            else while (i < fmt.size() && keel::is_ascii_digit(fmt[i])) ++i;
        }
        while (i < fmt.size() && (fmt[i] == 'l' || fmt[i] == 'h'))
            ++i;
        if (i < fmt.size()) ++n;   // the conversion itself consumes the value arg
    }
    return n;
}

keel::String formatCyclic(keel::StringRef fmt, keel::ArrayRef<Value> args, keel::usize argStart, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<Value> stream(scratch.allocator());
    stream.reserve(args.size() > argStart ? args.size() - argStart : 0);
    for (keel::usize i = argStart; i < args.size(); ++i) {
        const Value &a = args[i];
        // A string array supplies one argument per element (MATLAB cycles
        // the format over them, like a numeric array). A char array is a
        // single atomic %s argument, so it is NOT expanded here.
        if (a.isString()) {
            keel::usize n = a.numel();
            for (keel::usize j = 0; j < n; ++j)
                stream.push_back(Value::stringScalar(a.stringElem(j), p));
            continue;
        }
        if (a.isChar() || a.isScalar()) {
            // MATLAB uses the REAL part of a complex argument for every numeric
            // conversion (sprintf('%g',1+2i) -> "1"); the imaginary part is
            // discarded. Push the real part so formatOnce's toScalar succeeds.
            if (a.type() == ValueType::COMPLEX)
                stream.push_back(Value::scalar(a.complexData()[0].real(), p));
            else
                stream.push_back(a);
            continue;
        }
        keel::usize n = a.numel();
        for (keel::usize j = 0; j < n; ++j) {
            keel::f64 v;
            if (a.type() == ValueType::DOUBLE) v = a.doubleData()[j];
            else if (a.isLogical())        v = a.logicalData()[j] ? 1.0 : 0.0;
            else if (a.type() == ValueType::COMPLEX) v = a.complexData()[j].real();
            else                           v = a(j);
            stream.push_back(Value::scalar(v, p));
        }
    }

    keel::usize nSpecs = countFormatSpecs(fmt);
    if (nSpecs == 0 || stream.size() <= nSpecs)
        return formatOnce(fmt, keel::ArrayRef<Value>{stream.data(), stream.size()}, 0);

    keel::String out;
    keel::usize pos = 0;
    while (pos < stream.size()) {
        keel::usize end = std::min(pos + nSpecs, stream.size());
        out.append(formatOnce(fmt, keel::ArrayRef<Value>{stream.data() + pos, end - pos}, 0));
        pos = end;
    }
    return out;
}

Value sprintf(const Value &fmt, keel::ArrayRef<Value> args, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (!fmt.isChar())
        return Value::fromString("", p);
    keel::String result = formatCyclic(fmt.toString(), args, 0, mr);
    return Value::fromString(result, p);
}

} // namespace numkit::builtin
