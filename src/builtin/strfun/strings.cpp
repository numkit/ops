// toolboxes/builtin/src/datatypes/strings/strings.cpp
//
// Non-regex string builtins: num2str / str2num / str2double / string /
// char / strcmp / strcmpi / upper / lower / strtrim / strsplit / strcat /
// strlength / strrep / contains / startsWith / endsWith. The regex
// builtins (regexp/regexpi/regexprep) live in regex.cpp.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/format.h>
#include <numkit/builtin/strfun.hpp>
#include <numkit/builtin/iofun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "strings_detail.hpp"

namespace numkit::builtin {

// ════════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════════

// ── Conversion ──────────────────────────────────────────────────────────

// MATLAB num2str signatures (R2025b):
//   num2str(X)         → ~5 significant digits (default)
//   num2str(X, N)      → N significant digits, where N is integer
//   num2str(X, FMT)    → printf-style format
// See BUGS.md #26.


// A newline-joined layout string -> a char MATRIX with one row per line
// (right-padded to max width), matching MATLAB's num2str return shape.
static Value charMatrixFromLines(const std::string &s, keel::Allocator *mr)
{
    std::vector<std::string> lines;
    keel::usize pos = 0;
    for (;;) {
        keel::usize nl = s.find('\n', pos);
        lines.push_back(s.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos));
        if (nl == std::string::npos) break;
        pos = nl + 1;
    }
    if (lines.size() <= 1)
        return Value::fromString(s, mr);
    keel::usize maxW = 0;
    for (auto &L : lines) maxW = std::max(maxW, L.size());
    auto m = Value::matrix(lines.size(), maxW, ValueType::CHAR, mr);
    char *dst = static_cast<char *>(m.rawDataMut());
    for (keel::usize r = 0; r < lines.size(); ++r)
        for (keel::usize c = 0; c < maxW; ++c)
            dst[c * lines.size() + r] = (c < lines[r].size()) ? lines[r][c] : ' ';
    return m;
}

Value num2str(const Value &x, keel::Allocator *mr)
{
    if (x.type() == ValueType::COMPLEX) {
        if (!x.isScalar())
            return charMatrixFromLines(num2strComplexArray(x, -1), mr);
        return Value::fromString(num2strComplexScalar(x.toComplex(), -1), mr);
    }
    if (x.isEmpty()) return Value::fromString("", mr);
    // Real, non-scalar: synthesise MATLAB's default column format and route
    // through the FMT overload (which handles per-row layout, common-blank-
    // column stripping, and NaN/Inf spelling).
    if (!x.isScalar())
        return num2str(x, num2strArrayFormat(x, -1), mr);
    const double v = x.toScalar();
    // MATLAB num2str default precision is MAGNITUDE-AWARE, not a fixed 5 sig
    // figs: it keeps ~4 digits after the integer part, so prec = digits-left-
    // of-decimal + 4 = max(floor(log10|v|),0) + 5. Hence num2str(1000000) =
    // "1000000" and num2str(1000000.5) = "1000000.5" (a fixed "%.5g" wrongly
    // gave "1e+06"). Non-finite values use MATLAB's capitalised spelling.
    if (std::isnan(v)) return Value::fromString("NaN", mr);
    if (std::isinf(v)) return Value::fromString(v < 0 ? "-Inf" : "Inf", mr);
    int prec = 5;
    if (v != 0.0) {
        const int e = static_cast<int>(std::floor(std::log10(std::fabs(v))));
        prec = (e > 0 ? e : 0) + 5;
    }
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
    return Value::fromString(std::string(buf), mr);
}

Value num2str(const Value &x, int N, keel::Allocator *mr)
{
    if (x.type() == ValueType::COMPLEX) {
        if (!x.isScalar()) {
            int n = N; if (n < 1) n = 1;
            return charMatrixFromLines(num2strComplexArray(x, n), mr);
        }
        int n = N; if (n < 1) n = 1;
        return Value::fromString(num2strComplexScalar(x.toComplex(), n), mr);
    }
    if (x.isEmpty()) return Value::fromString("", mr);
    // Real, non-scalar with explicit precision N: MATLAB uses a "%<N+7>.<N>g"
    // column field for every element (no integer-detection in the N form).
    if (!x.isScalar())
        return num2str(x, num2strArrayFormat(x, N), mr);
    const double v = x.toScalar();
    int n = N;
    if (n < 1)  n = 1;
    if (n > 99) n = 99;
    char fmt[16];
    std::snprintf(fmt, sizeof(fmt), "%%.%dg", n);
    char buf[256];
    std::snprintf(buf, sizeof(buf), fmt, v);
    return Value::fromString(std::string(buf), mr);
}

Value num2str(const Value &x, keel::StringRef fmt,
              keel::Allocator *mr)
{
    // Route the value through the sprintf engine rather than a raw
    // snprintf(fmt, double): the engine handles integer conversions
    // (%d/%i/%u/%o/%x read an int from the va_list, so passing a double
    // straight to snprintf printed garbage — e.g. num2str(5,'%05d') gave
    // "00000" instead of "00005"), the non-integer->%e fallback, and the
    // Inf/NaN spelling. MATLAB then strips the leading AND trailing blank
    // COLUMNS common to all rows (keeping leading zeros and internal
    // spacing) and returns an N-row char matrix:
    //   num2str(pi,'%8.4f')                 -> "3.1416"
    //   num2str(5,'%05d')                   -> "00005"
    //   num2str([1.5 2.25 3.125],'%8.3f')   -> "1.500   2.250   3.125"  (1 row)
    //   num2str([1.5 2.25;3.1 4],'%8.3f')   -> 2x13 char matrix
    // The format is applied cyclically across each ROW (MATLAB sprintf
    // semantics), matching nk's sprintf which cycles a format over a
    // vector. NOTE: the DEFAULT-precision and integer-N forms for vector/
    // matrix inputs (num2str([1 2 3]) / num2str(v,4)) use MATLAB's
    // magnitude-dependent column-width algorithm and remain a deferred gap
    // (num2str_reg still routes those to the scalar overloads, which throw
    // on non-scalars) — only the deterministic FMT form is handled here.

    // Scalar complex: apply the format to the real and imaginary parts
    // independently and join them re±|im|i (MATLAB num2str(3.14159-2.71828i,
    // '%.3f') -> "3.142-2.718i"). Complex ARRAYS remain a deferred gap.
    if (x.type() == ValueType::COMPLEX) {
        if (!x.isScalar()) {
            // FMT form on a complex array: format both parts per element
            // and reuse the same column-aligned layout via the scalar N-form
            // approximation (apply fmt to both parts element-wise).
            const keel::usize rows = x.dims().rows();
            const keel::usize cols = x.dims().cols();
            if (rows * cols == 0) return Value::fromString("", mr);
            const keel::c64 *d = x.complexData();
            std::vector<std::vector<std::string>> cells(rows, std::vector<std::string>(cols));
            std::vector<keel::usize> w(cols, 0);
            for (keel::usize r = 0; r < rows; ++r)
                for (keel::usize c = 0; c < cols; ++c) {
                    const keel::c64 z = d[r + c * rows];
                    const std::string sr(num2str(Value::scalar(z.real(), mr), fmt, mr).toString().as_ref());
                    std::string s = sr;
                    if (z.imag() != 0.0) {
                        s += (z.imag() < 0.0 ? '-' : '+');
                        s += std::string_view(num2str(Value::scalar(std::fabs(z.imag()), mr), fmt, mr).toString());
                        s += 'i';
                    }
                    cells[r][c] = s;
                    w[c] = std::max(w[c], s.size());
                }
            std::string out;
            for (keel::usize r = 0; r < rows; ++r) {
                if (r) out += '\n';
                for (keel::usize c = 0; c < cols; ++c) {
                    if (c) out += ' ';
                    out.append(w[c] - cells[r][c].size(), ' ');
                    out += cells[r][c];
                }
            }
            return Value::fromString(out, mr);
        }
        const keel::c64 z = x.toComplex();
        const std::string sr(num2str(Value::scalar(z.real(), mr), fmt, mr).toString().as_ref());
        if (z.imag() == 0.0) return Value::fromString(sr, mr);
        std::string s = sr;
        s += (z.imag() < 0.0 ? '-' : '+');
        s += std::string_view(num2str(Value::scalar(std::fabs(z.imag()), mr), fmt, mr).toString());
        s += 'i';
        return Value::fromString(s, mr);
    }

    const keel::usize nrows = x.dims().rows();
    const keel::usize ncols = x.dims().cols();
    if (nrows * ncols == 0)
        return Value::fromString("", mr);

    Value fmtVal = Value::fromString(fmt, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto rows = keel::Array<std::string>(scratch.allocator());
    rows.reserve(nrows);

    // Build one formatted string per row (format cycled across the row's
    // columns, in column order).
    for (keel::usize r = 0; r < nrows; ++r) {
        Value rowv = Value::matrix(1, ncols, ValueType::DOUBLE, mr);
        double *rd = rowv.doubleDataMut();
        for (keel::usize c = 0; c < ncols; ++c)
            rd[c] = x.elemAsDouble(c * nrows + r);
        keel::ArrayRef<Value> args(&rowv, 1);
        rows.emplace_back(sprintf(fmtVal, args, mr).toString().as_ref());
    }

    auto leadingWS = [](const std::string &s) {
        keel::usize k = 0;
        while (k < s.size() && (s[k] == ' ' || s[k] == '\t')) ++k;
        return k;
    };
    auto trailingWS = [](const std::string &s) {
        keel::usize k = 0;
        while (k < s.size() && (s[s.size() - 1 - k] == ' ' || s[s.size() - 1 - k] == '\t'))
            ++k;
        return k;
    };

    // Strip the leading/trailing blank columns common to ALL rows (= the
    // minimum leading/trailing whitespace run over the rows). For a scalar
    // this reproduces the old find_first_not_of/find_last_not_of trim.
    keel::usize kLead = std::string::npos, kTrail = std::string::npos;
    for (const auto &s : rows) {
        kLead  = std::min(kLead,  leadingWS(s));
        kTrail = std::min(kTrail, trailingWS(s));
    }
    if (kLead == std::string::npos) kLead = 0;
    if (kTrail == std::string::npos) kTrail = 0;

    keel::usize maxW = 0;
    for (auto &s : rows) {
        const keel::usize len = s.size();
        // A row that is entirely whitespace collapses to empty.
        if (kLead + kTrail >= len) s.clear();
        else s = s.substr(kLead, len - kLead - kTrail);
        maxW = std::max(maxW, s.size());
    }

    if (nrows == 1)
        return Value::fromString(rows[0], mr);

    // Right-pad each row to maxW and emit an nrows x maxW char matrix
    // (column-major). Rows are equal length for fixed-width formats; the
    // pad only matters for variable-width formats.
    auto m = Value::matrix(nrows, maxW, ValueType::CHAR, mr);
    char *dst = static_cast<char *>(m.rawDataMut());
    for (keel::usize r = 0; r < nrows; ++r)
        for (keel::usize c = 0; c < maxW; ++c)
            dst[c * nrows + r] = (c < rows[r].size()) ? rows[r][c] : ' ';
    return m;
}

Value int2str(const Value &x, keel::Allocator *mr)
{
    // MATLAB int2str: round half away from zero (std::round), render as a
    // plain integer with no decimals or scientific notation. Inf/-Inf/NaN
    // pass through. For complex input MATLAB operates on the REAL part (the
    // imaginary part is discarded): int2str(3.6+1.2i) = "4".

    // Vector / 2-D matrix: round every element FIRST (printf %.0f rounds
    // half-to-even, but int2str rounds half-away-from-zero), then format with
    // a fixed integer field W = digits(max|rounded|) + 2 and route through the
    // num2str FMT path (per-row layout + common-blank-column strip).
    if (!x.isEmpty() && !x.isScalar() && !x.dims().is3D()) {
        const keel::usize n = x.numel();
        const bool cplx = (x.type() == ValueType::COMPLEX);
        Value rounded = Value::matrix(x.dims().rows(), x.dims().cols(),
                                      ValueType::DOUBLE, mr);
        double *rd = rounded.doubleDataMut();
        double maxabs = 0.0;
        for (keel::usize i = 0; i < n; ++i) {
            const double vi = cplx ? x.complexData()[i].real() : x.elemAsDouble(i);
            double r = std::isfinite(vi) ? std::round(vi) : vi;
            if (r == 0.0) r = 0.0;   // normalise -0 -> 0
            rd[i] = r;
            if (std::isfinite(r) && std::fabs(r) > maxabs) maxabs = std::fabs(r);
        }
        int ndigits = 1;
        if (maxabs >= 1.0)
            ndigits = static_cast<int>(std::floor(std::log10(maxabs))) + 1;
        char fmt[16];
        std::snprintf(fmt, sizeof(fmt), "%%%d.0f", ndigits + 2);
        return num2str(rounded, std::string(fmt), mr);
    }
    if (x.isEmpty()) return Value::fromString("", mr);

    const double v = (x.type() == ValueType::COMPLEX) ? x.toComplex().real()
                                                      : x.toScalar();
    if (std::isnan(v))
        return Value::fromString("NaN", mr);
    if (std::isinf(v))
        return Value::fromString(v < 0 ? "-Inf" : "Inf", mr);
    double r = std::round(v);
    if (r == 0.0) r = 0.0;          // normalise -0 -> 0
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.0f", r);
    return Value::fromString(std::string(buf), mr);
}

Value validatestring(const Value &str, const Value &valid,
                     keel::Allocator *mr)
{
    auto lower = [](std::string s) {
        for (auto &c : s)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    };
    const std::string s(str.toString().as_ref());
    const std::string sl = lower(s);

    // Candidates (original case) from a cell array, string array, or a lone
    // char vector.
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<std::string> cands(scratch.allocator());
    if (valid.isCell() || valid.isString()) {
        const auto &vec = valid.cellDataVec();
        for (const auto &e : vec) cands.emplace_back(e.toString().as_ref());
    } else if (valid.isChar()) {
        cands.emplace_back(valid.toString().as_ref());
    } else {
        throw Error("validatestring: second argument must be a cell array of "
                    "char vectors or a string array",
                     0, 0, "validatestring", "", "numkit:validatestring:badList");
    }

    // 1) Exact (case-insensitive) match wins.
    for (const auto &c : cands)
        if (lower(c) == sl)
            return Value::fromString(c, mr);

    // 2) Case-insensitive leading-substring (prefix) matches.
    keel::Array<const std::string *> pre(scratch.allocator());
    for (const auto &c : cands) {
        const std::string cl = lower(c);
        if (cl.size() >= sl.size() && cl.compare(0, sl.size(), sl) == 0)
            pre.push_back(&c);
    }
    if (pre.empty())
        throw Error("validatestring: '" + s + "' did not match any valid string",
                     0, 0, "validatestring", "", "numkit:validatestring:unrecognized");
    if (pre.size() == 1)
        return Value::fromString(*pre[0], mr);

    // 3) Multiple prefix matches: unambiguous only if the shortest is itself a
    //    leading substring of every other match (then return the shortest).
    const std::string *shortest = pre[0];
    for (auto *p : pre)
        if (p->size() < shortest->size()) shortest = p;
    const std::string shl = lower(*shortest);
    bool prefixOfAll = true;
    for (auto *p : pre) {
        const std::string pl = lower(*p);
        if (!(pl.size() >= shl.size() && pl.compare(0, shl.size(), shl) == 0)) {
            prefixOfAll = false;
            break;
        }
    }
    if (prefixOfAll)
        return Value::fromString(*shortest, mr);
    throw Error("validatestring: '" + s + "' matches multiple valid strings",
                 0, 0, "validatestring", "", "numkit:validatestring:ambiguous");
}

Value str2num(const Value &s, keel::Allocator *mr)
{
    try {
        return Value::scalar(std::stod(std::string(s.toString().as_ref())), mr);
    } catch (...) {
        return Value();
    }
}

// Parse a numeric substring with strtod; the ENTIRE string must be consumed
// (else NaN). Empty -> defVal. ('42abc' / '42 7' -> NaN; 'Inf'/'-Inf'/'NaN'
// parse.) strtod parses leading "inf"/"nan" case-insensitively.
static double strtodFull(const std::string &t, double defVal)
{
    if (t.empty()) return defVal;
    const char *cs = t.c_str();
    char *end = nullptr;
    const double v = std::strtod(cs, &end);
    if (end == cs || *end != '\0') return std::numeric_limits<double>::quiet_NaN();
    return v;
}

// Parse ONE token the way MATLAB str2double does: strip ALL commas (thousands
// separators) and ALL whitespace (MATLAB tolerates spaces even inside complex
// literals, e.g. ' 2 + 3i '), then the ENTIRE remaining token must parse as a
// single real OR complex number — otherwise NaN. Complex: a trailing lowercase
// 'i' or 'j' marks the imaginary part ('2i', '1+2i', '1-2j', 'i', '-i',
// '3.5+1.5i', '1e-3+2i', 'Infi'); the real/imag split is the LAST '+'/'-' that
// is not an exponent sign. Capital 'I'/'J' are NOT imaginary (-> real parse).
struct Str2DoubleResult { double re; double im; bool isComplex; };

static Str2DoubleResult str2doubleParse(keel::StringRef str)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    auto isWs = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    // Strip commas (thousands separators); trim SURROUNDING whitespace only.
    std::string t;
    t.reserve(str.size());
    for (char c : str) if (c != ',') t.push_back(c);
    const char *ws = " \t\n\r\f\v";
    const keel::usize b = t.find_first_not_of(ws);
    if (b == std::string::npos) return {nan, 0.0, false};   // empty / all-whitespace
    const keel::usize e = t.find_last_not_of(ws);
    t = t.substr(b, e - b + 1);

    const char last = t.back();
    if (last != 'i' && last != 'j')               // pure real
        return { strtodFull(t, nan), 0.0, false };   // internal whitespace -> NaN

    // COMPLEX: MATLAB tolerates whitespace around the +/- and the i/j
    // (e.g. ' 2 + 3i '), so strip ALL remaining whitespace before splitting.
    std::string c2;
    c2.reserve(t.size());
    for (char ch : t) if (!isWs(ch)) c2.push_back(ch);
    const std::string body = c2.substr(0, c2.size() - 1);   // drop the i/j
    // Split at the rightmost +/- at index >= 1 whose predecessor is not e/E
    // (so exponent signs like the '-' in '1e-3' are not treated as a split).
    keel::usize split = std::string::npos;
    for (keel::usize k = body.size(); k-- > 1; ) {
        const char c = body[k];
        if ((c == '+' || c == '-') && body[k - 1] != 'e' && body[k - 1] != 'E') {
            split = k; break;
        }
    }
    auto parseImag = [&](const std::string &is) -> double {
        std::string m = is;
        double sign = 1.0;
        if (!m.empty() && (m[0] == '+' || m[0] == '-')) {
            if (m[0] == '-') sign = -1.0;
            m = m.substr(1);
        }
        if (m.empty()) return sign;               // bare i / +i / -i  -> ±1
        return sign * strtodFull(m, nan);
    };
    double re, im;
    if (split == std::string::npos) { re = 0.0; im = parseImag(body); }
    else { re = strtodFull(body.substr(0, split), nan); im = parseImag(body.substr(split)); }
    return { re, im, true };
}

Value str2double(const Value &s, keel::Allocator *mr)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // A cell array of char/string vectors OR a non-scalar string array maps
    // element-wise to a matrix of the SAME shape (MATLAB str2double). NaN where
    // an element fails to parse, is empty, or is not char/string. The output is
    // COMPLEX iff ANY element parses as a complex literal (real elements then
    // carry a zero imaginary part); otherwise it stays DOUBLE (zero regression).
    if (s.isCell() || (s.isString() && s.numel() != 1)) {
        const keel::usize r = static_cast<keel::usize>(s.dims().rows());
        const keel::usize c = static_cast<keel::usize>(s.dims().cols());
        const keel::usize n = s.numel();
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<Str2DoubleResult> res(n, scratch.allocator());
        bool anyComplex = false;
        for (keel::usize i = 0; i < n; ++i) {
            if (s.isCell()) {
                const Value &el = s.cellAt(i);
                res[i] = (el.isChar() || el.isString())
                             ? str2doubleParse(el.toString())
                             : Str2DoubleResult{nan, 0.0, false};
            } else {
                res[i] = str2doubleParse(s.stringElem(i));
            }
            anyComplex = anyComplex || res[i].isComplex;
        }
        if (anyComplex) {
            Value out = Value::matrix(r, c, ValueType::COMPLEX, mr);
            keel::c64 *od = out.complexDataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = keel::c64(res[i].re, res[i].im);
            return out;
        }
        Value out = Value::matrix(r, c, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (keel::usize i = 0; i < n; ++i) od[i] = res[i].re;
        return out;
    }

    // Scalar char array / string scalar → scalar double (or complex scalar).
    const Str2DoubleResult res = str2doubleParse(s.toString());
    if (res.isComplex) {
        Value out = Value::matrix(1, 1, ValueType::COMPLEX, mr);
        out.complexDataMut()[0] = keel::c64(res.re, res.im);
        return out;
    }
    return Value::scalar(res.re, mr);
}

Value toString(const Value &x, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (x.isString())
        return x;
    if (x.isChar())
        return Value::stringScalar(x.toString(), p);
    if (x.isNumeric()) {
        auto fmtNum = [](double d) -> std::string {
            if (std::isnan(d)) return "NaN";
            if (std::isinf(d)) return d < 0 ? "-Inf" : "Inf";
            std::ostringstream os;
            os << d;
            return os.str();
        };
        if (x.isScalar()) {
            return Value::stringScalar(fmtNum(x.toScalar()), p);
        }
        auto result = Value::stringArray(x.dims().rows(), x.dims().cols(), p);
        for (keel::usize i = 0; i < x.numel(); ++i) {
            result.stringElemSet(i, fmtNum(x.elemAsDouble(i)));
        }
        return result;
    }
    if (x.isLogical())
        return Value::stringScalar(x.toBool() ? "true" : "false", p);
    if (x.isCell()) {
        // Cell-of-chars / cell-of-strings → string array of same shape.
        // See BUGS.md #4.
        const keel::usize n = x.numel();
        auto result = Value::stringArray(x.dims().rows(), x.dims().cols(), p);
        for (keel::usize i = 0; i < n; ++i) {
            const Value &el = x.cellAt(i);
            if (el.isChar() || el.isString())
                result.stringElemSet(i, el.toString());
            else if (el.isNumeric() && el.isScalar()) {
                std::ostringstream os;
                os << el.toScalar();
                result.stringElemSet(i, os.str());
            } else {
                throw Error(
                    "string: cell elements must be char, string, or numeric scalar",
                    0, 0, "string", "", "numkit:string:cellElementType");
            }
        }
        return result;
    }
    throw Error("Cannot convert input to string", 0, 0, "string", "",
                 "numkit:string:unsupportedType");
}

Value toChar(const Value &x, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (x.isChar())
        return x;
    if (x.isCell()) {
        const keel::usize n = x.numel();
        if (n == 0)
            return Value::matrix(0, 0, ValueType::CHAR, p);
        std::vector<std::string> rows(n);
        keel::usize maxCols = 0;
        for (keel::usize i = 0; i < n; ++i) {
            const Value &el = x.cellAt(i);
            if (!el.isChar() && !el.isString() && !el.isEmpty())
                throw Error("Cannot convert to char", 0, 0, "char", "", "numkit:char:unsupportedType");
            rows[i] = el.isEmpty() ? std::string{} : std::string(el.toString().as_ref());
            maxCols = std::max(maxCols, rows[i].size());
        }
        if (n == 1)
            return Value::fromString(rows[0], p);
        Value out = Value::matrix(n, maxCols, ValueType::CHAR, p);
        char *dst = out.charDataMut();
        std::memset(dst, ' ', n * maxCols);
        for (keel::usize r = 0; r < n; ++r)
            for (keel::usize c = 0; c < rows[r].size(); ++c)
                dst[c * n + r] = rows[r][c];
        return out;
    }
    if (x.isString()) {
        const keel::usize n = x.numel();
        if (n <= 1)
            return Value::fromString(x.toString(), p);
        keel::usize maxCols = 0;
        for (keel::usize i = 0; i < n; ++i)
            maxCols = std::max(maxCols, x.stringElem(i).size());
        Value out = Value::matrix(n, maxCols, ValueType::CHAR, p);
        char *dst = out.charDataMut();
        std::memset(dst, ' ', n * maxCols);
        for (keel::usize r = 0; r < n; ++r) {
            const auto s = x.stringElem(r);
            for (keel::usize c = 0; c < s.size(); ++c)
                dst[c * n + r] = s[c];
        }
        return out;
    }
    if (x.isNumeric()) {
        std::string s;
        if (x.isScalar()) {
            s += static_cast<char>(static_cast<int>(x.toScalar()));
        } else {
            const double *d = x.doubleData();
            for (keel::usize i = 0; i < x.numel(); ++i)
                s += static_cast<char>(static_cast<int>(d[i]));
        }
        return Value::fromString(s, p);
    }
    throw Error("Cannot convert to char", 0, 0, "char", "", "numkit:char:unsupportedType");
}

// ── Comparisons ─────────────────────────────────────────────────────────


Value strcmp(const Value &a, const Value &b, keel::Allocator *mr)
{
    return strCmpElementwise(a, b,
        [](const std::string &x, const std::string &y) { return x == y; },
        "strcmp", mr);
}

Value strcmpi(const Value &a, const Value &b, keel::Allocator *mr)
{
    return strCmpElementwise(a, b,
        [](const std::string &x, const std::string &y) {
            std::string sa = x, sb = y;
            std::transform(sa.begin(), sa.end(), sa.begin(), ::tolower);
            std::transform(sb.begin(), sb.end(), sb.begin(), ::tolower);
            return sa == sb;
        }, "strcmpi", mr);
}

Value strncmp(const Value &a, const Value &b, keel::usize n, keel::Allocator *mr)
{
    return strCmpElementwise(a, b,
        [n](const std::string &x, const std::string &y) { return strnEq(x, y, n); },
        "strncmp", mr);
}

Value strncmpi(const Value &a, const Value &b, keel::usize n, keel::Allocator *mr)
{
    return strCmpElementwise(a, b,
        [n](const std::string &x, const std::string &y) {
            std::string sa = x, sb = y;
            std::transform(sa.begin(), sa.end(), sa.begin(), ::tolower);
            std::transform(sb.begin(), sb.end(), sb.begin(), ::tolower);
            return strnEq(sa, sb, n);
        }, "strncmpi", mr);
}

// ── Case transforms ─────────────────────────────────────────────────────

// Apply a per-string transform to every element of a CELL array, returning a
// cell of char vectors with the same shape (MATLAB's element-wise behaviour
// for lower/upper/strtrim/deblank/strip). The scalar (non-cell) path is left
// to each caller so the pre-existing scalar return type is preserved exactly.
template <class Op>
static Value mapStringCell(const Value &s, Op op, keel::Allocator *mr)
{
    const keel::usize r = static_cast<keel::usize>(s.dims().rows());
    const keel::usize c = static_cast<keel::usize>(s.dims().cols());
    auto out = Value::cell(r, c, mr);
    const keel::usize n = s.numel();
    for (keel::usize i = 0; i < n; ++i)
        out.cellAt(i) = Value::fromString(op(std::string(s.cellAt(i).toString().as_ref())), mr);
    return out;
}

// Apply a per-string transform preserving the MATLAB container class:
//   cell           -> cell of char vectors (same shape)
//   string array   -> string array (same shape; a scalar string stays a
//                     1×1 string, NOT a char — strtrim/deblank/strip are
//                     class-preserving in MATLAB)
//   char / other   -> char (the whole array is treated as one token, the
//                     pre-existing scalar behaviour)
template <class Op>
static Value mapStringPreserveClass(const Value &s, Op op, keel::Allocator *mr)
{
    if (s.isCell()) return mapStringCell(s, op, mr);
    if (s.isString()) {
        const keel::usize r = static_cast<keel::usize>(s.dims().rows());
        const keel::usize c = static_cast<keel::usize>(s.dims().cols());
        Value out = Value::stringArray(r, c, mr);
        const keel::usize n = s.numel();
        for (keel::usize i = 0; i < n; ++i)
            out.stringElemSet(i, op(std::string(s.stringElem(i).as_ref())));
        return out;
    }
    return Value::fromString(op(std::string(s.toString().as_ref())), mr);
}

Value upper(const Value &s, keel::Allocator *mr)
{
    auto op = [](std::string r) {
        std::transform(r.begin(), r.end(), r.begin(), ::toupper);
        return r;
    };
    return mapStringPreserveClass(s, op, mr);
}

Value lower(const Value &s, keel::Allocator *mr)
{
    auto op = [](std::string r) {
        std::transform(r.begin(), r.end(), r.begin(), ::tolower);
        return r;
    };
    return mapStringPreserveClass(s, op, mr);
}

// ── Trim / split / concat ───────────────────────────────────────────────

Value strtrim(const Value &s, keel::Allocator *mr)
{
    auto op = [](const std::string &r) -> std::string {
        keel::usize start = r.find_first_not_of(" \t\r\n");
        keel::usize end = r.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        return r.substr(start, end - start + 1);
    };
    return mapStringPreserveClass(s, op, mr);
}

Value deblank(const Value &s, keel::Allocator *mr)
{
    auto op = [](const std::string &r) -> std::string {
        keel::usize end = r.find_last_not_of(" \t\r\n\f\v");
        if (end == std::string::npos) return "";
        return r.substr(0, end + 1);
    };
    return mapStringPreserveClass(s, op, mr);
}

Value blanks(keel::usize n, keel::Allocator *mr)
{
    return Value::fromString(std::string(n, ' '), mr);
}


Value strsplit(const Value &s, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<std::string> delims(scratch.allocator());
    appendDefaultWhitespace(delims);
    return strsplitImpl(s.toString(), delims, /*collapse=*/true, mr);
}

Value strsplit(const Value &s, keel::StringRef delim, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<std::string> delims(scratch.allocator());
    delims.emplace_back(delim.data(), delim.size());
    return strsplitImpl(s.toString(), delims, /*collapse=*/true, mr);
}

Value strsplit(const Value &s, const Value &delim, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<std::string> delims(scratch.allocator());
    appendDelims(delim, delims);
    return strsplitImpl(s.toString(), delims, /*collapse=*/true, mr);
}

Value strcat(keel::ArrayRef<Value> parts, keel::Allocator *mr)
{
    bool anyCell = false;
    bool anyNonScalarStr = false;
    keel::usize targetRows = 1, targetCols = 1;
    for (const auto &p : parts) {
        if (p.isCell()) {
            anyCell = true;
            if (p.numel() != 1) {
                targetRows = p.dims().rows();
                targetCols = p.dims().cols();
            }
        } else if (p.isString() && p.numel() != 1) {
            anyNonScalarStr = true;
            targetRows = p.dims().rows();
            targetCols = p.dims().cols();
        }
    }
    if (anyCell) {
        Value out = Value::cell(targetRows, targetCols, mr);
        const keel::usize n = targetRows * targetCols;
        for (keel::usize i = 0; i < n; ++i) {
            std::string s;
            for (const auto &p : parts) {
                if (p.isCell()) {
                    if (p.numel() == 1)
                        s += std::string_view(p.cellAt(0).toString());
                    else if (i < p.numel())
                        s += std::string_view(p.cellAt(i).toString());
                } else if (p.isString() && p.numel() != 1) {
                    if (i < p.numel())
                        s += std::string_view(p.stringElem(i));
                } else {
                    s += std::string_view(p.toString());
                }
            }
            out.cellAt(i) = Value::fromString(s, mr);
        }
        return out;
    }
    if (anyNonScalarStr) {
        Value out = Value::stringArray(targetRows, targetCols, mr);
        const keel::usize n = targetRows * targetCols;
        for (keel::usize i = 0; i < n; ++i) {
            std::string s;
            for (const auto &p : parts) {
                if (p.isString() && p.numel() != 1) {
                    if (i < p.numel())
                        s += std::string_view(p.stringElem(i));
                } else {
                    s += std::string_view(p.toString());
                }
            }
            out.stringElemSet(i, s);
        }
        return out;
    }
    std::string result;
    for (const auto &p : parts)
        result += std::string_view(p.toString());
    return Value::fromString(result, mr);
}

// ── Length ──────────────────────────────────────────────────────────────

Value strlength(const Value &s, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (s.isString()) {
        if (s.isScalar())
            return Value::scalar(static_cast<double>(s.toString().size()), p);
        auto result = createLike(s, ValueType::DOUBLE, p);
        for (keel::usize i = 0; i < s.numel(); ++i)
            result.doubleDataMut()[i] = static_cast<double>(s.stringElem(i).size());
        return result;
    }
    if (s.isChar())
        return Value::scalar(static_cast<double>(s.numel()), p);
    throw Error("Input must be a string or char array", 0, 0, "strlength", "",
                 "numkit:strlength:unsupportedType");
}

// ── Search / replace ────────────────────────────────────────────────────


Value strfind(const Value &s, const Value &pat, keel::Allocator *mr)
{
    const auto pp = pat.toString();
    // A cell / non-scalar string array source -> same-shape CELL of index
    // vectors (MATLAB); a scalar char / string -> a 1×k DOUBLE row vector.
    if (s.isCell() || (s.isString() && !s.isScalar())) {
        const keel::usize r = static_cast<keel::usize>(s.dims().rows());
        const keel::usize c = static_cast<keel::usize>(s.dims().cols());
        Value out = Value::cell(r, c, mr);
        const keel::usize n = s.numel();
        for (keel::usize i = 0; i < n; ++i) {
            const auto el =
                s.isCell() ? s.cellAt(i).toString() : s.stringElem(i);
            out.cellAt(i) = strfindOne(el, pp, mr);
        }
        return out;
    }
    return strfindOne(s.toString(), pp, mr);
}

Value mat2str(const Value &x, int precision, keel::Allocator *mr)
{
    // CHAR: render a quoted char literal (MATLAB R2025b). A char row ->
    // 'abc'; a multi-row char matrix -> ['ab';'cd']; empty char '' -> "''".
    // Internal single quotes are doubled ('a''b'). Checked before the empty
    // guard so '' produces "''" rather than "[]".
    if (x.isChar()) {
        if (x.dims().ndim() > 2)
            throw Error("mat2str: only 2-D inputs are supported",
                         0, 0, "mat2str", "", "numkit:mat2str:rank");
        const keel::usize R = x.dims().rows(), C = x.dims().cols();
        const char *cd = (x.numel() > 0) ? x.charData() : nullptr;
        auto quoteRow = [&](keel::usize r) {
            std::string s;
            s.push_back('\'');
            for (keel::usize c = 0; c < C; ++c) {
                const char ch = cd[c * R + r];   // col-major
                if (ch == '\'') s.push_back('\'');   // double internal quote
                s.push_back(ch);
            }
            s.push_back('\'');
            return s;
        };
        if (R <= 1)
            return Value::fromString(quoteRow(0), mr);
        std::string out;
        out.push_back('[');
        for (keel::usize r = 0; r < R; ++r) {
            if (r > 0) out.push_back(';');
            out += quoteRow(r);
        }
        out.push_back(']');
        return Value::fromString(out, mr);
    }

    if (x.isEmpty())
        return Value::fromString("[]", mr);

    const auto &d = x.dims();
    if (d.ndim() > 2)
        throw Error("mat2str: only 2-D inputs are supported",
                     0, 0, "mat2str", "", "numkit:mat2str:rank");
    const keel::usize R = d.rows(), C = d.cols();

    // Logical: elements print as the words true / false (MATLAB R2025b),
    // e.g. mat2str(true) -> "true", mat2str([true false]) -> "[true false]".
    if (x.isLogical()) {
        auto fmtL = [](double v) -> const char * {
            return v != 0.0 ? "true" : "false";
        };
        if (x.isScalar())
            return Value::fromString(fmtL(x.elemAsDouble(0)), mr);
        std::string out;
        out.push_back('[');
        for (keel::usize r = 0; r < R; ++r) {
            if (r > 0) out.push_back(';');
            for (keel::usize c = 0; c < C; ++c) {
                if (c > 0) out.push_back(' ');
                out += fmtL(x.elemAsDouble(c * R + r));
            }
        }
        out.push_back(']');
        return Value::fromString(out, mr);
    }

    auto fmt = [precision](double v) {
        if (v == 0.0) v = 0.0;   // normalize -0 → 0 (MATLAB never prints "-0")
        std::ostringstream os;
        os.precision(precision);
        os << v;
        return os.str();
    };

    // Complex: format each element INDEPENDENTLY (matches MATLAB mat2str) —
    // an element with exactly-zero imaginary part prints as a bare real, even
    // when other elements of the same array are complex. E.g.
    // mat2str(complex([5 3],[0 4])) -> "[5 3+4i]" (NOT "[5+0i 3+4i]"); an
    // all-zero-imag complex array therefore prints fully real ("[1 2]").
    if (x.type() == ValueType::COMPLEX) {
        const keel::c64 *cd = x.complexData();
        const keel::usize n = x.numel();
        auto fmtC = [&](const keel::c64 &z) -> std::string {
            const double im = z.imag();
            if (im == 0.0) return fmt(z.real());
            std::string s = fmt(z.real());
            s += (im < 0.0 ? '-' : '+');
            s += fmt(im < 0.0 ? -im : im);
            s += 'i';
            return s;
        };
        if (x.isScalar()) return Value::fromString(fmtC(cd[0]), mr);
        std::string out;
        out.reserve(n * (precision + 8) + R + 4);
        out.push_back('[');
        for (keel::usize r = 0; r < R; ++r) {
            if (r > 0) out.push_back(';');
            for (keel::usize c = 0; c < C; ++c) {
                if (c > 0) out.push_back(' ');
                out += fmtC(cd[c * R + r]);
            }
        }
        out.push_back(']');
        return Value::fromString(out, mr);
    }

    // Real numeric (double / single / int8..uint64). Read via elemAsDouble so
    // integer and single arrays format the same way MATLAB mat2str does:
    // bare values with no class wrapper (mat2str(int8([1 2])) -> "[1 2]").
    if (x.isScalar()) {
        return Value::fromString(fmt(x.elemAsDouble(0)), mr);
    }

    std::string out;
    out.reserve(R * C * (precision + 4) + R + 4);
    out.push_back('[');
    for (keel::usize r = 0; r < R; ++r) {
        if (r > 0) out.push_back(';');
        for (keel::usize c = 0; c < C; ++c) {
            if (c > 0) out.push_back(' ');
            out += fmt(x.elemAsDouble(c * R + r));
        }
    }
    out.push_back(']');
    return Value::fromString(out, mr);
}

Value strjoin(const Value &c, const Value &delim, keel::Allocator *mr)
{
    if (!c.isCell())
        throw Error("strjoin: first argument must be a cell array",
                     0, 0, "strjoin", "", "numkit:strjoin:notCell");
    const keel::usize n = c.numel();
    std::string out;
    if (delim.isCell()) {
        // MATLAB R2025b: a cell array of numel(C)-1 delimiters, interleaved
        // between consecutive elements: strjoin({'a','b','c'},{', ',' and '})
        // -> 'a, b and c'.
        const keel::usize nd = delim.numel();
        if (n > 0 && nd != n - 1)
            throw Error("strjoin: delimiter cell array must have one fewer "
                         "element than the first argument",
                         0, 0, "strjoin", "", "numkit:strjoin:badDelimCount");
        for (keel::usize i = 0; i < n; ++i) {
            if (i > 0) out += std::string_view(delim.cellAt(i - 1).toString());
            out += std::string_view(c.cellAt(i).toString());
        }
    } else {
        const std::string sep =
            delim.isEmpty() ? std::string(" ") : std::string(delim.toString().as_ref());
        for (keel::usize i = 0; i < n; ++i) {
            if (i > 0) out += sep;
            out += std::string_view(c.cellAt(i).toString());
        }
    }
    return Value::fromString(out, mr);
}

Value strjoin(const Value &c, keel::StringRef delim, keel::Allocator *mr)
{
    return strjoin(c, Value::fromString(delim, mr), mr);
}

// ── Pack 18 ──────────────────────────────────────────────────────────

Value append(keel::ArrayRef<Value> parts, keel::Allocator *mr)
{
    // Same shape as strcat but does NOT trim trailing whitespace from
    // char-array operands (MATLAB's "append" is the literal-concatenate
    // variant that script users reach for).
    std::string out;
    for (const auto &p : parts)
        out += std::string_view(p.toString());
    return Value::fromString(out, mr);
}


Value count(const Value &s, const Value &pat, keel::Allocator *mr)
{
    // pat may be a single pattern or a cell/string array of patterns; MATLAB
    // sums the non-overlapping occurrence counts across all listed patterns.
    // A cell str input is processed element-wise, returning a DOUBLE array the
    // same shape as the cell (counts per element).
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<std::string> pats(scratch.allocator());
    collectMatchPatterns(pat, pats);
    auto countOne = [&](keel::StringRef ssRef) -> double {
        const std::string_view ss(ssRef);
        keel::usize n = 0;
        for (const auto &pp : pats) {
            if (pp.empty()) continue;
            keel::usize pos = 0;
            while ((pos = ss.find(pp, pos)) != std::string::npos) {
                ++n;
                pos += pp.size();   // non-overlapping (matches MATLAB)
            }
        }
        return static_cast<double>(n);
    };
    if (s.isCell()) {
        const keel::usize r = static_cast<keel::usize>(s.dims().rows());
        const keel::usize c = static_cast<keel::usize>(s.dims().cols());
        auto out = Value::matrix(r, c, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        const keel::usize nn = s.numel();
        for (keel::usize i = 0; i < nn; ++i) od[i] = countOne(s.cellAt(i).toString());
        return out;
    }
    // A string ARRAY is processed element-wise -> DOUBLE array, same shape.
    if (s.isString() && !s.isScalar()) {
        const keel::usize r = static_cast<keel::usize>(s.dims().rows());
        const keel::usize c = static_cast<keel::usize>(s.dims().cols());
        auto out = Value::matrix(r, c, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        const keel::usize nn = s.numel();
        for (keel::usize i = 0; i < nn; ++i) od[i] = countOne(s.stringElem(i));
        return out;
    }
    return Value::scalar(countOne(s.toString()), mr);
}

Value erase(const Value &s, const Value &pat, keel::Allocator *mr)
{
    // pat may be a single pattern or a cell/string array; MATLAB removes every
    // occurrence of each listed pattern, applied in order. A cell str input is
    // processed element-wise, returning a cell of char vectors (same shape).
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<std::string> pats(scratch.allocator());
    collectMatchPatterns(pat, pats);
    auto op = [&](std::string r) -> std::string {
        for (const auto &pp : pats) {
            if (pp.empty()) continue;
            keel::usize pos = 0;
            while ((pos = r.find(pp, pos)) != std::string::npos)
                r.erase(pos, pp.size());
        }
        return r;
    };
    return mapStringPreserveClass(s, op, mr);
}

Value replace(const Value &s, const Value &oldPat, const Value &newPat, keel::Allocator *mr)
{
    // oldPat/newPat may each be a single pattern OR a cell / string array.
    // MATLAB scans the source once, left to right: at each position the FIRST
    // old pattern (in list order) that matches there is replaced with its
    // corresponding new text, then the scan advances past the matched source
    // (no re-scanning, no chain-replacement). A single NEW applies to every
    // OLD; otherwise NEW must pair 1:1 with OLD.
    const std::string ss(s.toString().as_ref());
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<std::string> olds(scratch.allocator()), news(scratch.allocator());
    collectMatchPatterns(oldPat, olds);
    collectMatchPatterns(newPat, news);
    if (news.size() != 1 && news.size() != olds.size())
        throw Error("replace: NEW must be a scalar text or match the number of "
                    "OLD patterns",
                     0, 0, "replace", "", "numkit:replace:sizeMismatch");

    std::string out;
    out.reserve(ss.size());
    keel::usize i = 0;
    while (i < ss.size()) {
        bool matched = false;
        for (keel::usize k = 0; k < olds.size(); ++k) {
            const std::string &o = olds[k];
            if (!o.empty() && i + o.size() <= ss.size()
                && ss.compare(i, o.size(), o) == 0) {
                out += (news.size() == 1) ? news[0] : news[k];
                i += o.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            out += ss[i];
            ++i;
        }
    }
    if (s.isString()) return Value::stringScalar(out, mr);
    return Value::fromString(out, mr);
}

Value reverse(const Value &s, keel::Allocator *mr)
{
    auto op = [](std::string r) {
        std::reverse(r.begin(), r.end());
        return r;
    };
    return mapStringPreserveClass(s, op, mr);
}

Value splitlines(const Value &s, keel::Allocator *mr)
{
    const std::string ss(s.toString().as_ref());
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<std::string> parts(scratch.allocator());
    std::string cur;
    for (keel::usize i = 0; i < ss.size(); ++i) {
        const char c = ss[i];
        if (c == '\r') {
            parts.push_back(std::move(cur));
            cur.clear();
            // Skip a following \n to handle CRLF as one separator.
            if (i + 1 < ss.size() && ss[i + 1] == '\n') ++i;
        } else if (c == '\n') {
            parts.push_back(std::move(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    // Always push the final segment, even if empty (matches MATLAB:
    // splitlines("a\nb") returns 2 entries; splitlines("a\n") returns 2).
    parts.push_back(std::move(cur));
    auto out = Value::cell(parts.size(), 1, mr);
    for (keel::usize i = 0; i < parts.size(); ++i)
        out.cellAt(i) = Value::fromString(parts[i], mr);
    return out;
}


// Pad ONE string to width n on the given side with char ch. MATLAB 'both'
// splits the extra padding floor(pad/2) left, ceil(pad/2) right.
static std::string padOne(std::string r, keel::usize n, const std::string &sd, char ch)
{
    if (r.size() >= n) return r;
    const keel::usize pad = n - r.size();
    if (sd == "left") {
        r.insert(r.begin(), pad, ch);
    } else if (sd == "both") {
        const keel::usize left = pad / 2;
        r.insert(r.begin(), left, ch);
        r.append(pad - left, ch);
    } else {  // "right" (default)
        r.append(pad, ch);
    }
    return r;
}

Value pad(const Value &s, keel::usize n, const Value &side, const Value &padChar, keel::Allocator *mr)
{
    const std::string sd = readSide(side, "right");
    if (sd != "left" && sd != "right" && sd != "both")
        throw Error("pad: side must be 'left', 'right', or 'both'",
                     0, 0, "pad", "", "numkit:pad:badSide");
    char ch = ' ';
    if (!padChar.isEmpty() && (padChar.isChar() || padChar.isString())) {
        const auto p = padChar.toString();
        if (!p.empty()) ch = p[0];
    }
    // string array -> string array same shape; char -> char; cell -> cell
    // of char vectors. Each element is padded to the SAME width n.
    auto op = [&](std::string r) { return padOne(std::move(r), n, sd, ch); };
    return mapStringPreserveClass(s, op, mr);
}

Value strip(const Value &s, const Value &side, const Value &ch, keel::Allocator *mr)
{
    const std::string sd = readSide(side, "both");
    std::string charsToStrip = " \t\r\n\f\v";
    bool noStrip = false;  // explicit empty strip-set => no-op
    if (!ch.isEmpty() && (ch.isChar() || ch.isString())) {
        charsToStrip = std::string(ch.toString().as_ref());
        if (charsToStrip.empty()) noStrip = true;
    }
    auto stripOne = [&](std::string r) -> std::string {
        if (noStrip) return r;
        if (sd == "left" || sd == "both") {
            keel::usize i = 0;
            while (i < r.size() && charsToStrip.find(r[i]) != std::string::npos) ++i;
            if (i > 0) r.erase(0, i);
        }
        if (sd == "right" || sd == "both") {
            while (!r.empty() && charsToStrip.find(r.back()) != std::string::npos)
                r.pop_back();
        }
        return r;
    };
    return mapStringPreserveClass(s, stripOne, mr);
}

Value matches(const Value &s, const Value &pat, keel::Allocator *mr)
{
    // The pattern may be a single string or a cell / string array of
    // alternatives; an element matches if it equals ANY of them.
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<std::string> pats(scratch.allocator());
    collectMatchPatterns(pat, pats);
    auto matchOne = [&](keel::StringRef el) -> bool {
        for (const auto &pp : pats)
            if (el == pp) return true;
        return false;
    };
    // A string ARRAY / cell source -> same-shape LOGICAL array (element-wise).
    if (s.isCell() || (s.isString() && !s.isScalar())) {
        const keel::usize r = static_cast<keel::usize>(s.dims().rows());
        const keel::usize c = static_cast<keel::usize>(s.dims().cols());
        Value out = Value::matrix(r, c, ValueType::LOGICAL, mr);
        keel::u8 *od = out.logicalDataMut();
        const keel::usize nn = s.numel();
        for (keel::usize i = 0; i < nn; ++i) {
            const auto el =
                s.isCell() ? s.cellAt(i).toString() : s.stringElem(i);
            od[i] = matchOne(el) ? 1 : 0;
        }
        return out;
    }
    return Value::logicalScalar(matchOne(s.toString()), mr);
}

// ── Pack 21 ──────────────────────────────────────────────────────────

Value convertCharsToStrings(const Value &x, keel::Allocator *mr)
{
    if (x.isString()) return x;
    if (x.isChar())   return Value::stringScalar(x.toString(), mr);
    if (x.isCell()) {
        const auto &d = x.dims();
        auto c = d.is3D()
                    ? Value::cell3D(d.rows(), d.cols(), d.pages(), mr)
                    : Value::cell(d.rows(), d.cols(), mr);
        for (keel::usize i = 0; i < x.numel(); ++i) {
            const auto &e = x.cellAt(i);
            if (e.isChar())
                c.cellAt(i) = Value::stringScalar(e.toString(), mr);
            else
                c.cellAt(i) = e;
        }
        return c;
    }
    // Numeric / other types: pass through unchanged.
    return x;
}

Value convertStringsToChars(const Value &x, keel::Allocator *mr)
{
    if (x.isChar()) return x;
    if (x.isString()) {
        if (x.numel() <= 1) {
            return Value::fromString(x.toString(), mr);
        }
        // Multi-element string → cell of char rows.
        const auto &d = x.dims();
        auto c = d.is3D()
                    ? Value::cell3D(d.rows(), d.cols(), d.pages(), mr)
                    : Value::cell(d.rows(), d.cols(), mr);
        for (keel::usize i = 0; i < x.numel(); ++i)
            c.cellAt(i) = Value::fromString(x.stringElem(i), mr);
        return c;
    }
    if (x.isCell()) {
        const auto &d = x.dims();
        auto c = d.is3D()
                    ? Value::cell3D(d.rows(), d.cols(), d.pages(), mr)
                    : Value::cell(d.rows(), d.cols(), mr);
        for (keel::usize i = 0; i < x.numel(); ++i) {
            const auto &e = x.cellAt(i);
            if (e.isString())
                c.cellAt(i) = Value::fromString(e.toString(), mr);
            else
                c.cellAt(i) = e;
        }
        return c;
    }
    return x;
}

Value isstringscalar(const Value &x, keel::Allocator *mr)
{
    return Value::logicalScalar(x.isString() && x.numel() == 1, mr);
}


Value isstrprop(const Value &s, const Value &category, keel::Allocator *mr)
{
    if (!category.isChar() && !category.isString())
        throw Error("isstrprop: category must be a string",
                     0, 0, "isstrprop", "", "numkit:isstrprop:cat");
    std::string cat(category.toString().as_ref());
    for (auto &c : cat) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (cat == "alpha")
        return applyCharPred(s, [](unsigned char c) { return std::isalpha(c) != 0; }, mr);
    if (cat == "digit")
        return applyCharPred(s, [](unsigned char c) { return std::isdigit(c) != 0; }, mr);
    if (cat == "alphanum")
        return applyCharPred(s, [](unsigned char c) { return std::isalnum(c) != 0; }, mr);
    if (cat == "lower")
        return applyCharPred(s, [](unsigned char c) { return std::islower(c) != 0; }, mr);
    if (cat == "upper")
        return applyCharPred(s, [](unsigned char c) { return std::isupper(c) != 0; }, mr);
    if (cat == "punct")
        return applyCharPred(s, [](unsigned char c) { return std::ispunct(c) != 0; }, mr);
    if (cat == "space" || cat == "wspace")
        return applyCharPred(s, [](unsigned char c) { return std::isspace(c) != 0; }, mr);
    if (cat == "xdigit")
        return applyCharPred(s, [](unsigned char c) { return std::isxdigit(c) != 0; }, mr);
    if (cat == "cntrl")
        return applyCharPred(s, [](unsigned char c) { return std::iscntrl(c) != 0; }, mr);
    if (cat == "graphic")
        return applyCharPred(s, [](unsigned char c) { return std::isgraph(c) != 0; }, mr);
    if (cat == "print")
        return applyCharPred(s, [](unsigned char c) { return std::isprint(c) != 0; }, mr);
    throw Error("isstrprop: unknown category '" + cat + "'",
                 0, 0, "isstrprop", "", "numkit:isstrprop:badCat");
}

Value isstrprop(const Value &s, keel::StringRef category, keel::Allocator *mr)
{
    return isstrprop(s, Value::fromString(category, mr), mr);
}

Value isletter(const Value &s, keel::Allocator *mr)
{
    return applyCharPred(s, [](unsigned char c) { return std::isalpha(c) != 0; }, mr);
}

Value isspaceFn(const Value &s, keel::Allocator *mr)
{
    return applyCharPred(s, [](unsigned char c) { return std::isspace(c) != 0; }, mr);
}

// ── Pack 22 ──────────────────────────────────────────────────────────
//
// Position-or-pattern lookup: numeric scalar → 0-based index, string
// → find first occurrence and return its [begin, end) range.


Value extractAfter(const Value &s, const Value &p, keel::Allocator *mr)
{
    // A cell str is processed element-wise (cell of char vectors); the
    // position/substring anchor p is scalar and applies to every element.
    auto op = [&](const std::string &ss) -> std::string {
        const auto r = resolvePos(ss, p);
        return r.found ? ss.substr(r.end) : std::string();
    };
    return mapStringPreserveClass(s, op, mr);
}

Value extractBefore(const Value &s, const Value &p, keel::Allocator *mr)
{
    auto op = [&](const std::string &ss) -> std::string {
        const auto r = resolvePos(ss, p);
        return r.found ? ss.substr(0, r.begin) : std::string();
    };
    return mapStringPreserveClass(s, op, mr);
}


Value extractBetween(const Value &s, const Value &start, const Value &end, bool inclusive, keel::Allocator *mr)
{
    const std::string ss(s.toString().as_ref());
    const auto matches = findAllBetweenPairs(ss, start, end);
    // MATLAB returns an M-by-1 cell of inner strings — even for a
    // single match. Empty result is a 0-by-1 cell. See BUGS.md #27.
    auto c = Value::cell(matches.size(), 1, mr);
    for (keel::usize i = 0; i < matches.size(); ++i) {
        const auto &m = matches[i];
        const keel::usize beg = inclusive ? m.openBegin : m.openEnd;
        const keel::usize endIdx = inclusive ? m.closeEnd : m.closeBegin;
        const std::string inner = ss.substr(beg, endIdx - beg);
        c.cellAt(i) = strLikeOf(s, inner, mr);
    }
    return c;
}

Value extractBetween(const Value &s, const Value &start, const Value &end, keel::Allocator *mr)
{
    return extractBetween(s, start, end, false, mr);
}

Value insertAfter(const Value &s, const Value &p, const Value &newText, keel::Allocator *mr)
{
    // A cell str is processed element-wise (cell of char vectors); the anchor
    // p and inserted text are scalar and apply to every element. An element
    // with no match is left unchanged.
    const std::string nt(newText.toString().as_ref());
    auto op = [&](std::string ss) -> std::string {
        const auto r = resolvePos(ss, p);
        if (r.found) ss.insert(r.end, nt);
        return ss;
    };
    return mapStringPreserveClass(s, op, mr);
}

Value insertBefore(const Value &s, const Value &p, const Value &newText, keel::Allocator *mr)
{
    const std::string nt(newText.toString().as_ref());
    auto op = [&](std::string ss) -> std::string {
        const auto r = resolvePos(ss, p);
        if (r.found) ss.insert(r.begin, nt);
        return ss;
    };
    return mapStringPreserveClass(s, op, mr);
}

Value eraseBetween(const Value &s, const Value &start, const Value &end, bool inclusive, keel::Allocator *mr)
{
    const std::string ss(s.toString().as_ref());
    const auto matches = findAllBetweenPairs(ss, start, end);
    if (matches.empty()) return s;
    // Walk left-to-right copying segments outside the matched ranges.
    // See BUGS.md #27.
    std::string out;
    out.reserve(ss.size());
    keel::usize cursor = 0;
    for (const auto &m : matches) {
        const keel::usize keepUntil = inclusive ? m.openBegin : m.openEnd;
        const keel::usize resumeAt  = inclusive ? m.closeEnd  : m.closeBegin;
        out.append(ss, cursor, keepUntil - cursor);
        cursor = resumeAt;
    }
    out.append(ss, cursor, ss.size() - cursor);
    return strLikeOf(s, out, mr);
}

Value eraseBetween(const Value &s, const Value &start, const Value &end, keel::Allocator *mr)
{
    return eraseBetween(s, start, end, false, mr);
}

Value replaceBetween(const Value &s, const Value &start, const Value &end, const Value &newText, bool inclusive, keel::Allocator *mr)
{
    const std::string ss(s.toString().as_ref());
    const std::string nt(newText.toString().as_ref());
    const auto matches = findAllBetweenPairs(ss, start, end);
    if (matches.empty()) return s;
    std::string out;
    out.reserve(ss.size());
    keel::usize cursor = 0;
    for (const auto &m : matches) {
        const keel::usize keepUntil = inclusive ? m.openBegin : m.openEnd;
        const keel::usize resumeAt  = inclusive ? m.closeEnd  : m.closeBegin;
        out.append(ss, cursor, keepUntil - cursor);
        out.append(nt);
        cursor = resumeAt;
    }
    out.append(ss, cursor, ss.size() - cursor);
    return strLikeOf(s, out, mr);
}

Value replaceBetween(const Value &s, const Value &start, const Value &end, const Value &newText, keel::Allocator *mr)
{
    return replaceBetween(s, start, end, newText, false, mr);
}

// ── Pack 23 ──────────────────────────────────────────────────────────


Value dec2bin(const Value &d, int minWidth, keel::Allocator *mr)
{
    return vecToBaseMatrix(d, 2, minWidth, mr);
}

Value dec2hex(const Value &d, int minWidth, keel::Allocator *mr)
{
    return vecToBaseMatrix(d, 16, minWidth, mr);
}

namespace {
Value base2decFixed(const Value &s, int base, keel::Allocator *mr)
{
    const keel::usize rows = static_cast<keel::usize>(s.dims().rows());
    if (s.type() == ValueType::CHAR && rows > 1) {
        const keel::usize cols = s.numel() / rows;
        const char *src = static_cast<const char *>(s.rawData());
        Value out = Value::matrix(rows, 1, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (keel::usize r = 0; r < rows; ++r) {
            std::string row;
            row.reserve(cols);
            for (keel::usize c = 0; c < cols; ++c) row.push_back(src[c * rows + r]);
            od[r] = static_cast<double>(parseBase(row, base));
        }
        return out;
    }
    if (s.isCell() || (s.isString() && !s.isScalar())) {
        const keel::usize n = s.numel();
        Value out = Value::matrix(s.dims().rows(), s.dims().cols(),
                                  ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (keel::usize i = 0; i < n; ++i) {
            const auto str =
                s.isCell() ? s.cellAt(i).toString() : s.stringElem(i);
            od[i] = static_cast<double>(parseBase(str, base));
        }
        return out;
    }
    return Value::scalar(static_cast<double>(parseBase(s.toString(), base)), mr);
}
} // namespace

Value bin2dec(const Value &s, keel::Allocator *mr)
{
    return base2decFixed(s, 2, mr);
}

Value hex2dec(const Value &s, keel::Allocator *mr)
{
    return base2decFixed(s, 16, mr);
}


Value hex2num(const Value &s, keel::Allocator *mr)
{
    // A char MATRIX (>1 row) parses each row as one number -> N×1 column.
    const keel::usize rows = static_cast<keel::usize>(s.dims().rows());
    if (s.type() == ValueType::CHAR && rows > 1) {
        const keel::usize cols = s.numel() / rows;
        const char *src = static_cast<const char *>(s.rawData());
        Value out = Value::matrix(rows, 1, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (keel::usize r = 0; r < rows; ++r) {
            std::string row;
            row.reserve(cols);
            for (keel::usize c = 0; c < cols; ++c) row.push_back(src[c * rows + r]);
            od[r] = hexBitsToDouble(row);
        }
        return out;
    }
    // A cellstr / string array -> same-shape double array.
    if (s.isCell() || (s.isString() && !s.isScalar())) {
        const keel::usize n = s.numel();
        Value out = Value::matrix(s.dims().rows(), s.dims().cols(),
                                  ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (keel::usize i = 0; i < n; ++i) {
            const auto str =
                s.isCell() ? s.cellAt(i).toString() : s.stringElem(i);
            od[i] = hexBitsToDouble(str);
        }
        return out;
    }
    // Scalar char row / string scalar -> scalar double.
    return Value::scalar(hexBitsToDouble(s.toString()), mr);
}

Value num2hex(const Value &x, keel::Allocator *mr)
{
    const ValueType t = x.type();
    const bool single = (t == ValueType::SINGLE);
    const bool dbl    = (t == ValueType::DOUBLE);
    if (!single && !dbl)
        throw Error("num2hex: inputs must be floating point (single or double)",
                     0, 0, "num2hex", "", "numkit:num2hex:notFloat");

    const keel::usize n = x.numel();
    const int W = single ? 8 : 16;
    if (n == 0) return Value::matrix(0, static_cast<keel::usize>(W), ValueType::CHAR, mr);

    // numel × W CHAR matrix, one row per element (column-major order).
    Value m = Value::matrix(n, static_cast<keel::usize>(W), ValueType::CHAR, mr);
    char *dst = static_cast<char *>(m.rawDataMut());
    char buf[24];
    for (keel::usize i = 0; i < n; ++i) {
        const double v = x.elemAsDouble(i);
        if (single) {
            const float f = static_cast<float>(v);
            keel::u32 bits;
            std::memcpy(&bits, &f, sizeof(bits));
            std::snprintf(buf, sizeof(buf), "%08x", static_cast<unsigned int>(bits));
        } else {
            keel::u64 bits;
            std::memcpy(&bits, &v, sizeof(bits));
            std::snprintf(buf, sizeof(buf), "%016llx",
                          static_cast<unsigned long long>(bits));
        }
        for (int c = 0; c < W; ++c)
            dst[static_cast<keel::usize>(c) * n + i] = buf[c];
    }
    return m;
}


Value dec2base(const Value &d, int base, int minWidth, keel::Allocator *mr)
{
    if (base < 2 || base > 36)
        throw Error("dec2base: base must be in 2..36",
                     0, 0, "dec2base", "", "numkit:dec2base:badBase");
    return vecToBaseMatrix(d, base, minWidth, mr);
}

Value base2dec(const Value &s, int base, keel::Allocator *mr)
{
    if (base < 2 || base > 36)
        throw Error("base2dec: base must be in 2..36",
                     0, 0, "base2dec", "", "numkit:base2dec:badBase");
    const keel::usize rows = static_cast<keel::usize>(s.dims().rows());
    // A char MATRIX (multiple rows) parses each row -> column vector.
    if (rows > 1 && s.type() == ValueType::CHAR) {
        const keel::usize cols = s.numel() / rows;
        const char *src = static_cast<const char *>(s.rawData());
        Value out = Value::matrix(rows, 1, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (keel::usize r = 0; r < rows; ++r) {
            std::string row;
            row.reserve(cols);
            for (keel::usize c = 0; c < cols; ++c) row.push_back(src[c * rows + r]);
            od[r] = static_cast<double>(parseBaseN(row, base));
        }
        return out;
    }
    return Value::scalar(static_cast<double>(parseBaseN(s.toString(), base)), mr);
}


// Single-element CF expansion, used by both 1-output (string form) and
// 2-output (numeric N, D) callsites.
Value rat(const Value &x, double tol, keel::Allocator *mr)
{
    const double v = x.toScalar();
    if (!std::isfinite(v))
        return Value::fromString(std::isnan(v) ? "NaN" : (v > 0 ? "Inf" : "-Inf"), mr);
    if (tol <= 0.0) tol = defaultRatTol(v);
    const auto exp = ratExpansion(v, tol);
    return Value::fromString(buildCFString(exp), mr);
}

// rats — fixed-width formatted ratio. MATLAB renders each element as
// `numerator/denominator` and pads to `len` characters per element.
// Default len = 13 per the MATLAB doc, BUT the rendered field is actually
// `len + 1` characters wide — MATLAB reserves one extra column for a
// leading sign on negative values (kept as space when positive). Field
// width = len + 1 here matches `strlength(rats(0.5))` = 14 in MATLAB R2025b.
// For a vector input MATLAB concatenates the per-element fields into one
// big string row.
Value rats(const Value &x, int len, keel::Allocator *mr)
{
    if (len <= 0) len = 13;
    const int field_width = len + 1;  // MATLAB reserves 1 col for leading sign
    const keel::usize n = x.numel();

    // Layout per MATLAB R2025b: numerator is right-justified in the first
    // half of the field (cols 1..field_width/2), '/' sits at the boundary
    // (col field_width/2 + 1), denominator is left-justified in the second
    // half. Empirically: rats(0.5) → '      1/2     ' (numer at col 7,
    // slash at col 8). For integer denominator (q == 1) the whole number
    // is right-justified in the full field.
    const int num_width = field_width / 2;          // 7 for default len=13
    const int den_width = field_width - num_width - 1;  // 6 for default
    auto formatOne = [&](double v) -> std::string {
        if (!std::isfinite(v)) {
            const std::string s = std::isnan(v) ? "NaN" : (v > 0 ? "Inf" : "-Inf");
            const int slack = field_width - static_cast<int>(s.size());
            return std::string(std::max(0, slack), ' ') + s;
        }
        const double tol = defaultRatTol(v);
        const auto exp = ratExpansion(v, tol);
        const std::string ns = std::to_string(exp.p);
        if (exp.q == 1) {
            const int slack = field_width - static_cast<int>(ns.size());
            return std::string(std::max(0, slack), ' ') + ns;
        }
        const std::string ds = std::to_string(exp.q);
        const int n_pad = std::max(0, num_width - static_cast<int>(ns.size()));
        const int d_pad = std::max(0, den_width - static_cast<int>(ds.size()));
        return std::string(n_pad, ' ') + ns + "/" + ds + std::string(d_pad, ' ');
    };

    if (n <= 1) {
        const double v = x.toScalar();
        return Value::fromString(formatOne(v), mr);
    }

    // Vector: concatenate per-element fields end-to-end (one row).
    std::string out;
    out.reserve(n * static_cast<keel::usize>(field_width));
    for (keel::usize i = 0; i < n; ++i)
        out += formatOne(x.elemAsDouble(i));
    return Value::fromString(out, mr);
}

// Single-string literal replacement: replace every non-overlapping occurrence
// of `op` in `r0` with `np`. Empty `op` is a no-op (matches MATLAB).
static std::string strrepOne(keel::StringRef r0, keel::StringRef opRef,
                             keel::StringRef npRef)
{
    std::string r(r0.data(), r0.size());
    const std::string_view op(opRef);
    const std::string_view np(npRef);
    if (!op.empty()) {
        keel::usize pos = 0;
        while ((pos = r.find(op, pos)) != std::string::npos) {
            r.replace(pos, op.length(), np);
            pos += np.length();
        }
    }
    return r;
}

Value strrep(const Value &s, const Value &oldPat, const Value &newPat, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    const bool sc = s.isCell(), oc = oldPat.isCell(), nc = newPat.isCell();

    // Scalar path: no cell arguments — return a char (or string) scalar,
    // preserving the original behaviour exactly.
    if (!sc && !oc && !nc) {
        std::string r = strrepOne(s.toString(), oldPat.toString(), newPat.toString());
        if (s.isString())
            return Value::stringScalar(r, p);
        return Value::fromString(r, p);
    }

    // Cell-aware path (MATLAB: any cell-array argument => cell output). Cell
    // operands must share a common size, or be scalar (1x1) and broadcast;
    // non-cell char/string arguments broadcast to every element. The result
    // is a cell of char vectors shaped like the non-scalar cell operand.
    auto numelOf = [](const Value &v, bool isCellArg) -> keel::usize {
        return isCellArg ? v.numel() : keel::usize{1};
    };
    const keel::usize ns = numelOf(s, sc), no = numelOf(oldPat, oc), nn = numelOf(newPat, nc);
    keel::usize n = 1;
    for (keel::usize v : {ns, no, nn}) {
        if (v == 1) continue;
        if (n == 1) n = v;
        else if (v != n)
            throw Error("strrep: nonscalar arguments must match in size",
                        0, 0, "strrep", "", "numkit:strrep:cellSize");
    }
    keel::usize rr = 1, cc = 1;
    auto setShape = [&](const Value &v) {
        rr = static_cast<keel::usize>(v.dims().rows());
        cc = static_cast<keel::usize>(v.dims().cols());
    };
    // Shape comes from the non-scalar cell operand if any, else the (1x1)
    // cell operand that triggered the cell path.
    if (sc && ns == n && n != 1)       setShape(s);
    else if (oc && no == n && n != 1)  setShape(oldPat);
    else if (nc && nn == n && n != 1)  setShape(newPat);
    else if (sc)                       setShape(s);
    else if (oc)                       setShape(oldPat);
    else                               setShape(newPat);

    const std::string ss0 = sc ? std::string() : std::string(s.toString().as_ref());
    const std::string os0 = oc ? std::string() : std::string(oldPat.toString().as_ref());
    const std::string ns0 = nc ? std::string() : std::string(newPat.toString().as_ref());

    auto out = Value::cell(rr, cc, p);
    for (keel::usize i = 0; i < n; ++i) {
        const std::string si = sc ? std::string(s.cellAt(ns == 1 ? 0 : i).toString().as_ref()) : ss0;
        const std::string oi = oc ? std::string(oldPat.cellAt(no == 1 ? 0 : i).toString().as_ref()) : os0;
        const std::string ni = nc ? std::string(newPat.cellAt(nn == 1 ? 0 : i).toString().as_ref()) : ns0;
        out.cellAt(i) = Value::fromString(strrepOne(si, oi, ni), p);
    }
    return out;
}


Value contains(const Value &s, const Value &pat, bool ignoreCase, keel::Allocator *mr)
{
    return strPredicate(s, pat, ignoreCase, StrPred::Contains, mr);
}

Value startsWith(const Value &s, const Value &prefix, bool ignoreCase, keel::Allocator *mr)
{
    return strPredicate(s, prefix, ignoreCase, StrPred::StartsWith, mr);
}

Value endsWith(const Value &s, const Value &suffix, bool ignoreCase, keel::Allocator *mr)
{
    return strPredicate(s, suffix, ignoreCase, StrPred::EndsWith, mr);
}

// ── Pack 36: compose / strjust / extract / split / join ──────────────


Value compose(const Value &fmt, const Value &x, keel::Allocator *mr)
{
    if (!fmt.isChar() && !fmt.isString())
        throw Error("compose: format must be a char or string",
                     0, 0, "compose", "", "numkit:compose:badFmt");
    const std::string fmtStr(fmt.toString().as_ref());
    // MATLAB: the output class mirrors the FORMAT class — a string format
    // yields a string array, a char format yields a cell of char vectors.
    const bool wantString = fmt.isString();

    // M = number of value-consuming conversion specs in one pass of the
    // format. MATLAB applies the format repeatedly across each ROW of x,
    // consuming M values per output element, so an R×C input yields an
    // R×ceil(C/M) result. A short trailing chunk leaves the unfilled specs
    // as literal text (e.g. compose('%d-%d',[1 2 3]) -> "1-2","3-%d").
    const keel::usize M = std::max<keel::usize>(1, countFormatSpecs(fmtStr));

    const auto &dims = x.dims();
    const keel::usize R = dims.rows();
    const keel::usize C = dims.cols();
    const keel::usize outCols = (C == 0) ? 0 : (C + M - 1) / M;

    auto buildResult = [&](keel::usize rows, keel::usize cols) -> Value {
        return wantString ? Value::stringArray(rows, cols, mr)
                          : Value::cell(rows, cols, mr);
    };
    auto setElem = [&](Value &v, keel::usize idx, const std::string &s) {
        if (wantString) v.stringElemSet(idx, s);
        else            v.cellAt(idx) = Value::fromString(s, mr);
    };

    if (R == 0 || C == 0)
        return buildResult(R, outCols);

    Value result = buildResult(R, outCols);

    std::vector<Value> chunk;
    chunk.reserve(M);
    for (keel::usize r = 0; r < R; ++r) {
        for (keel::usize j = 0; j < outCols; ++j) {
            chunk.clear();
            for (keel::usize k = 0; k < M; ++k) {
                const keel::usize col = j * M + k;
                if (col >= C) break;  // short trailing chunk
                // Column-major linear index of x(r, col).
                chunk.push_back(elemScalar(x, col * R + r, mr));
            }
            const std::string s =
                formatOnce(fmtStr, {chunk.data(), chunk.size()}, 0,
                           /*literalWhenShort=*/true);
            setElem(result, j * R + r, s);  // column-major out(r, j)
        }
    }
    return result;
}

Value strjust(const Value &M, keel::StringRef side, keel::Allocator *mr)
{
    if (!M.isChar())
        throw Error("strjust: input must be a char matrix",
                     0, 0, "strjust", "", "numkit:strjust:badInput");
    const auto &dims = M.dims();
    const keel::usize rows = dims.rows();
    const keel::usize cols = dims.cols();
    const auto src = M.toString();
    if (rows == 0 || cols == 0) return M;

    std::string out(rows * cols, ' ');
    for (keel::usize r = 0; r < rows; ++r) {
        // Column-major: row r, col c -> src[c*rows + r].
        keel::usize firstNonSp = cols, lastNonSp = 0;
        bool hasNonSp = false;
        for (keel::usize c = 0; c < cols; ++c) {
            char ch = src[c * rows + r];
            if (ch != ' ') {
                if (!hasNonSp) { firstNonSp = c; hasNonSp = true; }
                lastNonSp = c;
            }
        }
        if (!hasNonSp) continue;  // whole row blank

        const keel::usize len = lastNonSp - firstNonSp + 1;
        keel::usize target;
        if      (side == "left")   target = 0;
        else if (side == "center") target = (cols - len) / 2;
        else if (side == "right")  target = cols - len;
        else throw Error("strjust: side must be 'left', 'right', or 'center'",
                          0, 0, "strjust", "", "numkit:strjust:badSide");

        for (keel::usize c = 0; c < len; ++c)
            out[(target + c) * rows + r] = src[(firstNonSp + c) * rows + r];
    }

    // Build a char matrix of the same shape as M.
    Value r = Value::matrix(rows, cols, ValueType::CHAR, mr);
    char *p = r.charDataMut();
    std::memcpy(p, out.data(), rows * cols);
    return r;
}

Value extract(const Value &s, const Value &pat, keel::Allocator *mr)
{
    const std::string sStr = strInput(s);
    const std::string pStr = strInput(pat);

    std::vector<std::string> hits;
    findAllLiteral(sStr, pStr, hits);

    if (hits.empty()) return Value::cell(0, 0, mr);
    Value c = Value::cell(hits.size(), 1, mr);
    for (keel::usize i = 0; i < hits.size(); ++i)
        c.cellAt(i) = Value::fromString(hits[i], mr);
    return c;
}

Value split(const Value &s, const Value &delim, keel::Allocator *mr)
{
    const std::string sStr = strInput(s);
    const std::string dStr = strInput(delim);

    std::vector<std::string> parts;
    splitKeepEmpty(sStr, dStr, parts);

    // MATLAB: a STRING input yields a string-array result; a char or cellstr
    // input yields a cell array of character vectors. The values and shape
    // (an N×1 column) are identical either way — only the container differs.
    if (s.isString()) {
        Value r = Value::stringArray(parts.size(), 1, mr);
        for (keel::usize i = 0; i < parts.size(); ++i)
            r.stringElemSet(i, parts[i]);
        return r;
    }

    Value c = Value::cell(parts.size(), 1, mr);
    for (keel::usize i = 0; i < parts.size(); ++i)
        c.cellAt(i) = Value::fromString(parts[i], mr);
    return c;
}

Value join(const Value &arr, const Value &delim, keel::Allocator *mr)
{
    const std::string d = delim.isEmpty() ? std::string(" ") : strInput(delim);

    auto getElem = [&](keel::usize i) -> std::string {
        if (arr.isString()) return std::string(arr.stringElem(i).as_ref());
        if (arr.isCell())   return std::string(arr.cellAt(i).toString().as_ref());
        return std::string(arr.toString().as_ref());  // scalar char/string fallback
    };

    if (arr.isScalar()) {
        return Value::stringScalar(getElem(0), mr);
    }

    const auto &dims = arr.dims();
    const keel::usize rows = dims.rows();
    const keel::usize cols = dims.cols();

    if (rows == 1 || cols == 1) {
        // 1-D: glue all elements with `d`.
        std::string out;
        const keel::usize n = arr.numel();
        for (keel::usize i = 0; i < n; ++i) {
            if (i > 0) out += d;
            out += getElem(i);
        }
        return Value::stringScalar(out, mr);
    }

    // 2-D: join along columns -> N×1 string column.
    Value r = Value::stringArray(rows, 1, mr);
    for (keel::usize row = 0; row < rows; ++row) {
        std::string out;
        for (keel::usize col = 0; col < cols; ++col) {
            if (col > 0) out += d;
            out += getElem(col * rows + row);
        }
        r.stringElemSet(row, out);
    }
    return r;
}

// ── Pack 36: array constructors / character constants ───────────────
Value newlineFn(keel::Allocator *mr)
{
    return Value::fromString("\n", mr);
}

Value stringsND(keel::ArrayRef<keel::usize> dims, keel::Allocator *mr)
{
    const keel::usize ndim = dims.size();
    if (ndim == 0)
        return Value::stringScalar("", mr);

    const keel::usize r = dims[0];
    const keel::usize c = (ndim >= 2) ? dims[1] : r;  // strings(n) → n×n.
    keel::usize p = 0;
    if (ndim >= 3) {
        p = 1;
        for (keel::usize i = 2; i < ndim; ++i)
            p *= dims[i];  // collapse extras into pages (3D cap).
    }

    auto v = (p == 0) ? Value::stringArray(r, c, mr)
                      : Value::stringArray3D(r, c, p, mr);
    const keel::usize n = v.numel();
    for (keel::usize i = 0; i < n; ++i)
        v.stringElemSet(i, "");
    return v;
}

// strtok: first token + remainder. See language/strings/strings.hpp.
std::pair<Value, Value> strtok(const Value &str, keel::StringRef delimRef,
                               keel::Allocator *mr)
{
    const std::string s(str.toString().as_ref());
    const std::string_view delim(delimRef);
    auto isDelim = [&](char c) { return delim.find(c) != std::string_view::npos; };

    // Skip leading delim chars.
    keel::usize start = 0;
    while (start < s.size() && isDelim(s[start])) ++start;
    // Find end of token.
    keel::usize end = start;
    while (end < s.size() && !isDelim(s[end])) ++end;

    Value tok = Value::fromString(s.substr(start, end - start), mr);
    Value rem = Value::fromString(end < s.size() ? s.substr(end) : std::string{},
                                  mr);
    return {std::move(tok), std::move(rem)};
}

Value convertContainedStringsToChars(const Value &v, keel::Allocator *mr)
{
    if (v.isString()) {
        if (v.numel() <= 1)
            return Value::fromString(v.toString(), mr);
        auto c = Value::cell(v.numel(), 1, mr);
        for (keel::usize i = 0; i < v.numel(); ++i)
            c.cellAt(i) = Value::fromString(v.stringElem(i), mr);
        return c;
    }
    if (v.isCell()) {
        const auto &d = v.dims();
        auto c = d.is3D()
                    ? Value::cell3D(d.rows(), d.cols(), d.pages(), mr)
                    : Value::cell(d.rows(), d.cols(), mr);
        for (keel::usize i = 0; i < v.numel(); ++i)
            c.cellAt(i) = convertContainedStringsToChars(v.cellAt(i), mr);
        return c;
    }
    if (v.isStruct() && !v.isStructArray()) {
        auto s = Value::structure(mr);
        for (auto entry : v.structFields())
            s.field(entry.key) = convertContainedStringsToChars(entry.value, mr);
        return s;
    }
    return v;
}

} // namespace numkit::builtin
