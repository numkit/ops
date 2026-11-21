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
#include <keel/string.h>
#include <keel/complex.h>
#include <keel/format.h>
#include <keel/utf8.h>
#include <numkit/builtin/strfun.hpp>
#include <numkit/builtin/iofun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <numkit/ops/helpers.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

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
static Value charMatrixFromLines(keel::StringRef s, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::StringRef> lines(scratch.allocator());
    keel::usize pos = 0;
    for (;;) {
        const keel::StringRef rem = s.substr(pos, s.size() - pos);
        const keel::usize nl = rem.find('\n');
        if (nl == keel::kInvalidIndex) {
            lines.push_back(rem);
            break;
        }
        lines.push_back(rem.substr(0, nl));
        pos += nl + 1;
    }
    if (lines.size() <= 1)
        return Value::fromString(s, mr);
    keel::usize maxW = 0;
    for (const auto &L : lines) maxW = std::max(maxW, L.size());
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
        if (!x.isScalar()) {
            const keel::String layout = num2strComplexArray(x, -1);
            return charMatrixFromLines(layout.as_ref(), mr);
        }
        return Value::fromString(num2strComplexScalar(x.toComplex(), -1), mr);
    }
    if (x.isEmpty()) return Value::fromString("", mr);
    // Real, non-scalar: synthesise MATLAB's default column format and route
    // through the FMT overload (which handles per-row layout, common-blank-
    // column stripping, and NaN/Inf spelling).
    if (!x.isScalar()) {
        const keel::String afmt = num2strArrayFormat(x, -1);
        return num2str(x, afmt.as_ref(), mr);
    }
    const keel::f64 v = x.toScalar();
    // MATLAB num2str default precision is MAGNITUDE-AWARE, not a fixed 5 sig
    // figs: it keeps ~4 digits after the integer part, so prec = digits-left-
    // of-decimal + 4 = max(floor(log10|v|),0) + 5. Hence num2str(1000000) =
    // "1000000" and num2str(1000000.5) = "1000000.5" (a fixed "%.5g" wrongly
    // gave "1e+06"). Non-finite values use MATLAB's capitalised spelling.
    if (std::isnan(v)) return Value::fromString("NaN", mr);
    if (std::isinf(v)) return Value::fromString(v < 0 ? "-Inf" : "Inf", mr);
    int prec = 5;
    if (v != 0.0) {
        const int e = keel::as<int>(std::floor(std::log10(std::fabs(v))));
        prec = (e > 0 ? e : 0) + 5;
    }
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
    return Value::fromString(buf, mr);
}

Value num2str(const Value &x, int N, keel::Allocator *mr)
{
    if (x.type() == ValueType::COMPLEX) {
        if (!x.isScalar()) {
            int n = N; if (n < 1) n = 1;
            const keel::String layout = num2strComplexArray(x, n);
            return charMatrixFromLines(layout.as_ref(), mr);
        }
        int n = N; if (n < 1) n = 1;
        return Value::fromString(num2strComplexScalar(x.toComplex(), n), mr);
    }
    if (x.isEmpty()) return Value::fromString("", mr);
    // Real, non-scalar with explicit precision N: MATLAB uses a "%<N+7>.<N>g"
    // column field for every element (no integer-detection in the N form).
    if (!x.isScalar()) {
        const keel::String afmt = num2strArrayFormat(x, N);
        return num2str(x, afmt.as_ref(), mr);
    }
    const keel::f64 v = x.toScalar();
    int n = N;
    if (n < 1)  n = 1;
    if (n > 99) n = 99;
    char fmt[16];
    std::snprintf(fmt, sizeof(fmt), "%%.%dg", n);
    char buf[256];
    std::snprintf(buf, sizeof(buf), fmt, v);
    return Value::fromString(buf, mr);
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
            keel::ArenaTemp scratch = keel::get_scratch(mr);
            auto *salloc = scratch.allocator();
            keel::Array<keel::String> cells(rows * cols, salloc);
            keel::Array<keel::usize> w(cols, keel::usize{0}, salloc);
            for (keel::usize r = 0; r < rows; ++r) {
                for (keel::usize c = 0; c < cols; ++c) {
                    const keel::c64 z = d[r + c * rows];
                    keel::String s = num2str(Value::scalar(z.real(), mr), fmt, mr).toString();
                    if (z.imag() != 0.0) {
                        s.push_back(z.imag() < 0.0 ? '-' : '+');
                        const keel::String si =
                            num2str(Value::scalar(std::fabs(z.imag()), mr), fmt, mr).toString();
                        s.append(si.as_ref());
                        s.push_back('i');
                    }
                    w[c] = std::max(w[c], s.size());
                    cells[r * cols + c] = std::move(s);
                }
            }
            keel::String out(salloc);
            for (keel::usize r = 0; r < rows; ++r) {
                if (r) out.push_back('\n');
                for (keel::usize c = 0; c < cols; ++c) {
                    if (c) out.push_back(' ');
                    const keel::String &cell = cells[r * cols + c];
                    for (keel::usize p = 0; p < w[c] - cell.size(); ++p)
                        out.push_back(' ');
                    out.append(cell.as_ref());
                }
            }
            return Value::fromString(out, mr);
        }
        const keel::c64 z = x.toComplex();
        keel::String s = num2str(Value::scalar(z.real(), mr), fmt, mr).toString();
        if (z.imag() == 0.0) return Value::fromString(s, mr);
        s.push_back(z.imag() < 0.0 ? '-' : '+');
        const keel::String si =
            num2str(Value::scalar(std::fabs(z.imag()), mr), fmt, mr).toString();
        s.append(si.as_ref());
        s.push_back('i');
        return Value::fromString(s, mr);
    }

    const keel::usize nrows = x.dims().rows();
    const keel::usize ncols = x.dims().cols();
    if (nrows * ncols == 0)
        return Value::fromString("", mr);

    Value fmtVal = Value::fromString(fmt, mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto *salloc = scratch.allocator();
    auto rawRows = keel::Array<keel::String>(salloc);
    rawRows.reserve(nrows);

    // Build one formatted string per row (format cycled across the row's
    // columns, in column order).
    for (keel::usize r = 0; r < nrows; ++r) {
        Value rowv = Value::matrix(1, ncols, ValueType::DOUBLE, mr);
        keel::f64 *rd = rowv.doubleDataMut();
        for (keel::usize c = 0; c < ncols; ++c)
            rd[c] = x.elemAsDouble(c * nrows + r);
        keel::ArrayRef<Value> args(&rowv, 1);
        rawRows.push_back(sprintf(fmtVal, args, mr).toString());
    }

    auto leadingWS = [](keel::StringRef s) {
        keel::usize k = 0;
        while (k < s.size() && (s[k] == ' ' || s[k] == '\t')) ++k;
        return k;
    };
    auto trailingWS = [](keel::StringRef s) {
        keel::usize k = 0;
        while (k < s.size() && (s[s.size() - 1 - k] == ' ' || s[s.size() - 1 - k] == '\t'))
            ++k;
        return k;
    };

    // Strip the leading/trailing blank columns common to ALL rows (= the
    // minimum leading/trailing whitespace run over the rows). For a scalar
    // this reproduces the old find_first_not_of/find_last_not_of trim.
    keel::usize kLead = keel::kInvalidIndex, kTrail = keel::kInvalidIndex;
    for (const auto &s : rawRows) {
        kLead  = std::min(kLead,  leadingWS(s.as_ref()));
        kTrail = std::min(kTrail, trailingWS(s.as_ref()));
    }
    if (kLead == keel::kInvalidIndex) kLead = 0;
    if (kTrail == keel::kInvalidIndex) kTrail = 0;

    keel::Array<keel::StringRef> rows(salloc);
    rows.reserve(nrows);
    keel::usize maxW = 0;
    for (const auto &s : rawRows) {
        const keel::usize len = s.size();
        // A row that is entirely whitespace collapses to empty.
        keel::StringRef trimmed = (kLead + kTrail >= len)
            ? keel::StringRef("", 0)
            : s.as_ref().substr(kLead, len - kLead - kTrail);
        rows.push_back(trimmed);
        maxW = std::max(maxW, trimmed.size());
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
        keel::f64 *rd = rounded.doubleDataMut();
        keel::f64 maxabs = 0.0;
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 vi = cplx ? x.complexData()[i].real() : x.elemAsDouble(i);
            keel::f64 r = std::isfinite(vi) ? std::round(vi) : vi;
            if (r == 0.0) r = 0.0;   // normalise -0 -> 0
            rd[i] = r;
            if (std::isfinite(r) && std::fabs(r) > maxabs) maxabs = std::fabs(r);
        }
        int ndigits = 1;
        if (maxabs >= 1.0)
            ndigits = keel::as<int>(std::floor(std::log10(maxabs))) + 1;
        char fmt[16];
        std::snprintf(fmt, sizeof(fmt), "%%%d.0f", ndigits + 2);
        return num2str(rounded, keel::StringRef(fmt), mr);
    }
    if (x.isEmpty()) return Value::fromString("", mr);

    const keel::f64 v = (x.type() == ValueType::COMPLEX) ? x.toComplex().real()
                                                      : x.toScalar();
    if (std::isnan(v))
        return Value::fromString("NaN", mr);
    if (std::isinf(v))
        return Value::fromString(v < 0 ? "-Inf" : "Inf", mr);
    keel::f64 r = std::round(v);
    if (r == 0.0) r = 0.0;          // normalise -0 -> 0
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.0f", r);
    return Value::fromString(buf, mr);
}

Value validatestring(const Value &str, const Value &valid,
                     keel::Allocator *mr)
{
    const keel::StringRef sRef = str.asStringRef();

    // Candidates (original case) from a cell array, string array, or a lone
    // char vector.
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::StringRef> cands(scratch.allocator());
    if (valid.isCell() || valid.isString()) {
        const auto &vec = valid.cellDataVec();
        for (const auto &e : vec) cands.push_back(e.asStringRef());
    } else if (valid.isChar()) {
        cands.push_back(valid.asStringRef());
    } else {
        throw Error("validatestring: second argument must be a cell array of "
                    "char vectors or a string array",
                     0, 0, "validatestring", "", "numkit:validatestring:badList");
    }

    // 1) Exact (case-insensitive) match wins.
    for (const auto &c : cands)
        if (c.equals_ignore_ascii_case(sRef))
            return Value::fromString(c, mr);

    // 2) Case-insensitive leading-substring (prefix) matches.
    keel::Array<keel::StringRef> pre(scratch.allocator());
    for (const auto &c : cands) {
        if (c.size() >= sRef.size() &&
            c.substr(0, sRef.size()).equals_ignore_ascii_case(sRef))
            pre.push_back(c);
    }
    if (pre.empty())
        throw Error(keel::format("validatestring: '{}' did not match any valid string", sRef),
                     0, 0, "validatestring", "", "numkit:validatestring:unrecognized");
    if (pre.size() == 1)
        return Value::fromString(pre[0], mr);

    // 3) Multiple prefix matches: unambiguous only if the shortest is itself a
    //    leading substring of every other match (then return the shortest).
    keel::StringRef shortest = pre[0];
    for (const auto p : pre)
        if (p.size() < shortest.size()) shortest = p;
    bool prefixOfAll = true;
    for (const auto p : pre) {
        if (!(p.size() >= shortest.size() &&
              p.substr(0, shortest.size()).equals_ignore_ascii_case(shortest))) {
            prefixOfAll = false;
            break;
        }
    }
    if (prefixOfAll)
        return Value::fromString(shortest, mr);
    throw Error(keel::format("validatestring: '{}' matches multiple valid strings", sRef),
                 0, 0, "validatestring", "", "numkit:validatestring:ambiguous");
}

Value str2num(const Value &s, keel::Allocator *mr)
{
    const keel::String str = s.toString();
    const char *cs = str.c_str();
    errno = 0;
    char *end = nullptr;
    const keel::f64 v = std::strtod(cs, &end);
    if (end == cs || errno == ERANGE)
        return Value();
    return Value::scalar(v, mr);
}

// Parse a numeric substring with strtod; the ENTIRE string must be consumed
// (else NaN). Empty -> defVal. ('42abc' / '42 7' -> NaN; 'Inf'/'-Inf'/'NaN'
// parse.) strtod parses leading "inf"/"nan" case-insensitively.
static keel::f64 strtodFull(keel::StringRef t, keel::f64 defVal)
{
    if (t.empty()) return defVal;
    const keel::String tmp = keel::String::from(t);
    const char *cs = tmp.c_str();
    char *end = nullptr;
    const keel::f64 v = std::strtod(cs, &end);
    if (end != cs + tmp.size()) return std::numeric_limits<keel::f64>::quiet_NaN();
    return v;
}

// Parse ONE token the way MATLAB str2double does: strip ALL commas (thousands
// separators) and ALL whitespace (MATLAB tolerates spaces even inside complex
// literals, e.g. ' 2 + 3i '), then the ENTIRE remaining token must parse as a
// single real OR complex number — otherwise NaN. Complex: a trailing lowercase
// 'i' or 'j' marks the imaginary part ('2i', '1+2i', '1-2j', 'i', '-i',
// '3.5+1.5i', '1e-3+2i', 'Infi'); the real/imag split is the LAST '+'/'-' that
// is not an exponent sign. Capital 'I'/'J' are NOT imaginary (-> real parse).
struct Str2DoubleResult { keel::f64 re; keel::f64 im; bool isComplex; };

static Str2DoubleResult str2doubleParse(keel::StringRef str)
{
    const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();
    auto isWs = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    // Strip commas (thousands separators); trim SURROUNDING whitespace only.
    keel::String tBuf;
    tBuf.reserve(str.size());
    for (char c : str) if (c != ',') tBuf.push_back(c);
    keel::usize b = 0;
    while (b < tBuf.size() && isWs(tBuf[b])) ++b;
    if (b == tBuf.size()) return {nan, 0.0, false};   // empty / all-whitespace
    keel::usize e = tBuf.size() - 1;
    while (e > b && isWs(tBuf[e])) --e;
    const keel::StringRef t = tBuf.as_ref().substr(b, e - b + 1);

    const char last = t[t.size() - 1];
    if (last != 'i' && last != 'j')               // pure real
        return { strtodFull(t, nan), 0.0, false };   // internal whitespace -> NaN

    // COMPLEX: MATLAB tolerates whitespace around the +/- and the i/j
    // (e.g. ' 2 + 3i '), so strip ALL remaining whitespace before splitting.
    keel::String c2;
    c2.reserve(t.size());
    for (char ch : t) if (!isWs(ch)) c2.push_back(ch);
    const keel::StringRef body = c2.as_ref().substr(0, c2.size() - 1);   // drop the i/j
    // Split at the rightmost +/- at index >= 1 whose predecessor is not e/E
    // (so exponent signs like the '-' in '1e-3' are not treated as a split).
    keel::usize split = keel::kInvalidIndex;
    for (keel::usize k = body.size(); k-- > 1; ) {
        const char c = body[k];
        if ((c == '+' || c == '-') && body[k - 1] != 'e' && body[k - 1] != 'E') {
            split = k; break;
        }
    }
    auto parseImag = [&](keel::StringRef is) -> keel::f64 {
        keel::StringRef m = is;
        keel::f64 sign = 1.0;
        if (!m.empty() && (m[0] == '+' || m[0] == '-')) {
            if (m[0] == '-') sign = -1.0;
            m = m.substr(1, m.size() - 1);
        }
        if (m.empty()) return sign;               // bare i / +i / -i  -> ±1
        return sign * strtodFull(m, nan);
    };
    keel::f64 re, im;
    if (split == keel::kInvalidIndex) {
        re = 0.0;
        im = parseImag(body);
    } else {
        re = strtodFull(body.substr(0, split), nan);
        im = parseImag(body.substr(split, body.size() - split));
    }
    return { re, im, true };
}

Value str2double(const Value &s, keel::Allocator *mr)
{
    const keel::f64 nan = std::numeric_limits<keel::f64>::quiet_NaN();

    // A cell array of char/string vectors OR a non-scalar string array maps
    // element-wise to a matrix of the SAME shape (MATLAB str2double). NaN where
    // an element fails to parse, is empty, or is not char/string. The output is
    // COMPLEX iff ANY element parses as a complex literal (real elements then
    // carry a zero imaginary part); otherwise it stays DOUBLE (zero regression).
    if (s.isCell() || (s.isString() && s.numel() != 1)) {
        const keel::usize r = keel::as<keel::usize>(s.dims().rows());
        const keel::usize c = keel::as<keel::usize>(s.dims().cols());
        const keel::usize n = s.numel();
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<Str2DoubleResult> res(n, scratch.allocator());
        bool anyComplex = false;
        for (keel::usize i = 0; i < n; ++i) {
            if (s.isCell()) {
                const Value &el = s.cellAt(i);
                res[i] = (el.isChar() || el.isString())
                             ? str2doubleParse(el.asStringRef())
                             : Str2DoubleResult{nan, 0.0, false};
            } else {
                res[i] = str2doubleParse(s.stringElemRef(i));
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
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize i = 0; i < n; ++i) od[i] = res[i].re;
        return out;
    }

    // Scalar char array / string scalar → scalar double (or complex scalar).
    const Str2DoubleResult res = str2doubleParse(s.asStringRef());
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
    auto fmtNum = [](keel::f64 d) -> keel::String {
        if (std::isnan(d)) return keel::String::from("NaN");
        if (std::isinf(d)) return keel::String::from(d < 0 ? "-Inf" : "Inf");
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6g", d);
        return keel::String::from(buf);
    };
    if (x.isString())
        return x;
    if (x.isChar())
        return Value::stringScalar(x.asStringRef(), p);
    if (x.isNumeric()) {
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
                result.stringElemSet(i, el.asStringRef());
            else if (el.isNumeric() && el.isScalar()) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.6g", el.toScalar());
                result.stringElemSet(i, buf);
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
        keel::ArenaTemp scratch = keel::get_scratch(p);
        keel::Array<keel::StringRef> rows(n, scratch.allocator());
        keel::usize maxCols = 0;
        for (keel::usize i = 0; i < n; ++i) {
            const Value &el = x.cellAt(i);
            if (!el.isChar() && !el.isString() && !el.isEmpty())
                throw Error("Cannot convert to char", 0, 0, "char", "", "numkit:char:unsupportedType");
            rows[i] = el.isEmpty() ? keel::StringRef{} : el.asStringRef();
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
            return Value::fromString(x.asStringRef(), p);
        keel::usize maxCols = 0;
        for (keel::usize i = 0; i < n; ++i)
            maxCols = std::max(maxCols, x.stringElemRef(i).size());
        Value out = Value::matrix(n, maxCols, ValueType::CHAR, p);
        char *dst = out.charDataMut();
        std::memset(dst, ' ', n * maxCols);
        for (keel::usize r = 0; r < n; ++r) {
            const keel::StringRef s = x.stringElemRef(r);
            for (keel::usize c = 0; c < s.size(); ++c)
                dst[c * n + r] = s[c];
        }
        return out;
    }
    if (x.isNumeric()) {
        keel::String s;
        if (x.isScalar()) {
            s.push_back(keel::as<char>(keel::as<int>(x.toScalar())));
        } else {
            const keel::f64 *d = x.doubleData();
            s.reserve(x.numel());
            for (keel::usize i = 0; i < x.numel(); ++i)
                s.push_back(keel::as<char>(keel::as<int>(d[i])));
        }
        return Value::fromString(s, p);
    }
    throw Error("Cannot convert to char", 0, 0, "char", "", "numkit:char:unsupportedType");
}

// ── Comparisons ─────────────────────────────────────────────────────────


Value strcmp(const Value &a, const Value &b, keel::Allocator *mr)
{
    return strCmpElementwise(a, b,
        [](keel::StringRef x, keel::StringRef y) { return x == y; },
        "strcmp", mr);
}

Value strcmpi(const Value &a, const Value &b, keel::Allocator *mr)
{
    return strCmpElementwise(a, b,
        [](keel::StringRef x, keel::StringRef y) {
            return x.equals_ignore_ascii_case(y);
        }, "strcmpi", mr);
}

Value strncmp(const Value &a, const Value &b, keel::usize n, keel::Allocator *mr)
{
    return strCmpElementwise(a, b,
        [n](keel::StringRef x, keel::StringRef y) { return strnEq(x, y, n); },
        "strncmp", mr);
}

Value strncmpi(const Value &a, const Value &b, keel::usize n, keel::Allocator *mr)
{
    return strCmpElementwise(a, b,
        [n](keel::StringRef x, keel::StringRef y) {
            if (x.size() >= n && y.size() >= n)
                return x.substr(0, n).equals_ignore_ascii_case(y.substr(0, n));
            return x.equals_ignore_ascii_case(y);
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
    const keel::usize r = keel::as<keel::usize>(s.dims().rows());
    const keel::usize c = keel::as<keel::usize>(s.dims().cols());
    auto out = Value::cell(r, c, mr);
    const keel::usize n = s.numel();
    for (keel::usize i = 0; i < n; ++i) {
        out.cellAt(i) = Value::fromString(op(s.cellAt(i).asStringRef()), mr);
    }
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
        const keel::usize r = keel::as<keel::usize>(s.dims().rows());
        const keel::usize c = keel::as<keel::usize>(s.dims().cols());
        Value out = Value::stringArray(r, c, mr);
        const keel::usize n = s.numel();
        for (keel::usize i = 0; i < n; ++i)
            out.stringElemSet(i, op(s.stringElemRef(i)));
        return out;
    }
    return Value::fromString(op(s.asStringRef()), mr);
}

Value upper(const Value &s, keel::Allocator *mr)
{
    auto op = [](keel::StringRef r) -> keel::String {
        keel::String out = keel::String::from(r);
        for (char &ch : out)
            ch = keel::to_ascii_upper(ch);
        return out;
    };
    return mapStringPreserveClass(s, op, mr);
}

Value lower(const Value &s, keel::Allocator *mr)
{
    auto op = [](keel::StringRef r) -> keel::String {
        keel::String out = keel::String::from(r);
        for (char &ch : out)
            ch = keel::to_ascii_lower(ch);
        return out;
    };
    return mapStringPreserveClass(s, op, mr);
}

// ── Trim / split / concat ───────────────────────────────────────────────

Value strtrim(const Value &s, keel::Allocator *mr)
{
    auto op = [](keel::StringRef r) -> keel::String {
        auto isTrimWs = [](char c) {
            return c == ' ' || c == '\t' || c == '\r' || c == '\n';
        };
        keel::usize start = 0;
        while (start < r.size() && isTrimWs(r[start])) ++start;
        if (start == r.size()) return keel::String();
        keel::usize end = r.size() - 1;
        while (end > start && isTrimWs(r[end])) --end;
        return keel::String::from(r.substr(start, end - start + 1));
    };
    return mapStringPreserveClass(s, op, mr);
}

Value deblank(const Value &s, keel::Allocator *mr)
{
    auto op = [](keel::StringRef r) -> keel::String {
        auto isDeblankWs = [](char c) {
            return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
        };
        keel::usize len = r.size();
        while (len > 0 && isDeblankWs(r[len - 1])) --len;
        return keel::String::from(r.substr(0, len));
    };
    return mapStringPreserveClass(s, op, mr);
}

Value blanks(keel::usize n, keel::Allocator *mr)
{
    keel::String s;
    s.reserve(n);
    for (keel::usize i = 0; i < n; ++i) s.push_back(' ');
    return Value::fromString(s, mr);
}


Value strsplit(const Value &s, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::String> delims(scratch.allocator());
    appendDefaultWhitespace(delims);
    return strsplitImpl(s.asStringRef(), delims, /*collapse=*/true, mr);
}

Value strsplit(const Value &s, keel::StringRef delim, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::String> delims(scratch.allocator());
    delims.push_back(keel::String::from(delim));
    return strsplitImpl(s.asStringRef(), delims, /*collapse=*/true, mr);
}

Value strsplit(const Value &s, const Value &delim, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::String> delims(scratch.allocator());
    appendDelims(delim, delims);
    return strsplitImpl(s.asStringRef(), delims, /*collapse=*/true, mr);
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
            keel::String s;
            for (const auto &p : parts) {
                if (p.isCell()) {
                    if (p.numel() == 1) {
                        s.append(p.cellAt(0).asStringRef());
                    } else if (i < p.numel()) {
                        s.append(p.cellAt(i).asStringRef());
                    }
                } else if (p.isString() && p.numel() != 1) {
                    if (i < p.numel())
                        s.append(p.stringElemRef(i));
                } else {
                    s.append(p.asStringRef());
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
            keel::String s;
            for (const auto &p : parts) {
                if (p.isString() && p.numel() != 1) {
                    if (i < p.numel())
                        s.append(p.stringElemRef(i));
                } else {
                    s.append(p.asStringRef());
                }
            }
            out.stringElemSet(i, s);
        }
        return out;
    }
    keel::String result;
    for (const auto &p : parts) {
        result.append(p.asStringRef());
    }
    return Value::fromString(result, mr);
}

// ── Length ──────────────────────────────────────────────────────────────

Value strlength(const Value &s, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    if (s.isString()) {
        if (s.isScalar())
            return Value::scalar(keel::as<keel::f64>(s.asStringRef().size()), p);
        auto result = createLike(s, ValueType::DOUBLE, p);
        for (keel::usize i = 0; i < s.numel(); ++i)
            result.doubleDataMut()[i] = keel::as<keel::f64>(s.stringElemRef(i).size());
        return result;
    }
    if (s.isChar())
        return Value::scalar(keel::as<keel::f64>(s.numel()), p);
    throw Error("Input must be a string or char array", 0, 0, "strlength", "",
                 "numkit:strlength:unsupportedType");
}

// ── Search / replace ────────────────────────────────────────────────────


Value strfind(const Value &s, const Value &pat, keel::Allocator *mr)
{
    const keel::StringRef pp = pat.asStringRef();
    // A cell / non-scalar string array source -> same-shape CELL of index
    // vectors (MATLAB); a scalar char / string -> a 1×k DOUBLE row vector.
    if (s.isCell() || (s.isString() && !s.isScalar())) {
        const keel::usize r = keel::as<keel::usize>(s.dims().rows());
        const keel::usize c = keel::as<keel::usize>(s.dims().cols());
        Value out = Value::cell(r, c, mr);
        const keel::usize n = s.numel();
        for (keel::usize i = 0; i < n; ++i) {
            if (s.isCell()) {
                out.cellAt(i) = strfindOne(s.cellAt(i).asStringRef(), pp, mr);
            } else {
                out.cellAt(i) = strfindOne(s.stringElemRef(i), pp, mr);
            }
        }
        return out;
    }
    return strfindOne(s.asStringRef(), pp, mr);
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
        auto appendQuotedRow = [&](keel::String &dst, keel::usize r) {
            dst.push_back('\'');
            for (keel::usize c = 0; c < C; ++c) {
                const char ch = cd[c * R + r];   // col-major
                if (ch == '\'') dst.push_back('\'');   // double internal quote
                dst.push_back(ch);
            }
            dst.push_back('\'');
        };
        if (R <= 1) {
            keel::String s;
            appendQuotedRow(s, 0);
            return Value::fromString(s, mr);
        }
        keel::String out;
        out.push_back('[');
        for (keel::usize r = 0; r < R; ++r) {
            if (r > 0) out.push_back(';');
            appendQuotedRow(out, r);
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
        auto fmtL = [](keel::f64 v) -> const char * {
            return v != 0.0 ? "true" : "false";
        };
        if (x.isScalar())
            return Value::fromString(fmtL(x.elemAsDouble(0)), mr);
        keel::String out;
        out.push_back('[');
        for (keel::usize r = 0; r < R; ++r) {
            if (r > 0) out.push_back(';');
            for (keel::usize c = 0; c < C; ++c) {
                if (c > 0) out.push_back(' ');
                out.append(fmtL(x.elemAsDouble(c * R + r)));
            }
        }
        out.push_back(']');
        return Value::fromString(out, mr);
    }

    const int prec = std::clamp(precision, 1, 99);
    auto appendFmt = [prec](keel::String &dst, keel::f64 v) {
        if (v == 0.0) v = 0.0;   // normalize -0 → 0 (MATLAB never prints "-0")
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
        dst.append(buf);
    };

    // Complex: format each element INDEPENDENTLY (matches MATLAB mat2str) —
    // an element with exactly-zero imaginary part prints as a bare real, even
    // when other elements of the same array are complex. E.g.
    // mat2str(complex([5 3],[0 4])) -> "[5 3+4i]" (NOT "[5+0i 3+4i]"); an
    // all-zero-imag complex array therefore prints fully real ("[1 2]").
    if (x.type() == ValueType::COMPLEX) {
        const keel::c64 *cd = x.complexData();
        const keel::usize n = x.numel();
        auto appendFmtC = [&](keel::String &dst, const keel::c64 &z) {
            const keel::f64 im = z.imag();
            appendFmt(dst, z.real());
            if (im != 0.0) {
                dst.push_back(im < 0.0 ? '-' : '+');
                appendFmt(dst, im < 0.0 ? -im : im);
                dst.push_back('i');
            }
        };
        if (x.isScalar()) {
            keel::String s;
            appendFmtC(s, cd[0]);
            return Value::fromString(s, mr);
        }
        keel::String out;
        out.reserve(n * keel::as<keel::usize>(prec + 8) + R + 4);
        out.push_back('[');
        for (keel::usize r = 0; r < R; ++r) {
            if (r > 0) out.push_back(';');
            for (keel::usize c = 0; c < C; ++c) {
                if (c > 0) out.push_back(' ');
                appendFmtC(out, cd[c * R + r]);
            }
        }
        out.push_back(']');
        return Value::fromString(out, mr);
    }

    // Real numeric (double / single / int8..uint64). Read via elemAsDouble so
    // integer and single arrays format the same way MATLAB mat2str does:
    // bare values with no class wrapper (mat2str(int8([1 2])) -> "[1 2]").
    if (x.isScalar()) {
        keel::String s;
        appendFmt(s, x.elemAsDouble(0));
        return Value::fromString(s, mr);
    }

    keel::String out;
    out.reserve(R * C * keel::as<keel::usize>(prec + 4) + R + 4);
    out.push_back('[');
    for (keel::usize r = 0; r < R; ++r) {
        if (r > 0) out.push_back(';');
        for (keel::usize c = 0; c < C; ++c) {
            if (c > 0) out.push_back(' ');
            appendFmt(out, x.elemAsDouble(c * R + r));
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
    keel::String out;
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
            if (i > 0) {
                out.append(delim.cellAt(i - 1).asStringRef());
            }
            out.append(c.cellAt(i).asStringRef());
        }
    } else {
        const keel::StringRef sep =
            delim.isEmpty() ? keel::StringRef(" ") : delim.asStringRef();
        for (keel::usize i = 0; i < n; ++i) {
            if (i > 0) out.append(sep);
            out.append(c.cellAt(i).asStringRef());
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
    keel::String out;
    for (const auto &p : parts) {
        out.append(p.asStringRef());
    }
    return Value::fromString(out, mr);
}


Value count(const Value &s, const Value &pat, keel::Allocator *mr)
{
    // pat may be a single pattern or a cell/string array of patterns; MATLAB
    // sums the non-overlapping occurrence counts across all listed patterns.
    // A cell str input is processed element-wise, returning a DOUBLE array the
    // same shape as the cell (counts per element).
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::String> pats(scratch.allocator());
    collectMatchPatterns(pat, pats);
    auto countOne = [&](keel::StringRef ss) -> keel::f64 {
        keel::usize n = 0;
        for (const auto &pp : pats) {
            if (pp.empty()) continue;
            keel::usize pos = 0;
            while (pos <= ss.size()) {
                const keel::usize rel = ss.substr(pos, ss.size() - pos).find(pp.as_ref());
                if (rel == keel::kInvalidIndex) break;
                ++n;
                pos += rel + pp.size();   // non-overlapping (matches MATLAB)
            }
        }
        return keel::as<keel::f64>(n);
    };
    if (s.isCell()) {
        const keel::usize r = keel::as<keel::usize>(s.dims().rows());
        const keel::usize c = keel::as<keel::usize>(s.dims().cols());
        auto out = Value::matrix(r, c, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        const keel::usize nn = s.numel();
        for (keel::usize i = 0; i < nn; ++i) {
            od[i] = countOne(s.cellAt(i).asStringRef());
        }
        return out;
    }
    // A string ARRAY is processed element-wise -> DOUBLE array, same shape.
    if (s.isString() && !s.isScalar()) {
        const keel::usize r = keel::as<keel::usize>(s.dims().rows());
        const keel::usize c = keel::as<keel::usize>(s.dims().cols());
        auto out = Value::matrix(r, c, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        const keel::usize nn = s.numel();
        for (keel::usize i = 0; i < nn; ++i) od[i] = countOne(s.stringElemRef(i));
        return out;
    }
    return Value::scalar(countOne(s.asStringRef()), mr);
}

Value erase(const Value &s, const Value &pat, keel::Allocator *mr)
{
    // pat may be a single pattern or a cell/string array; MATLAB removes every
    // occurrence of each listed pattern, applied in order. A cell str input is
    // processed element-wise, returning a cell of char vectors (same shape).
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::String> pats(scratch.allocator());
    collectMatchPatterns(pat, pats);
    auto op = [&](keel::StringRef r0) -> keel::String {
        keel::String cur = keel::String::from(r0);
        for (const auto &pp : pats) {
            if (pp.empty()) continue;
            const keel::StringRef curRef = cur.as_ref();
            if (curRef.find(pp.as_ref()) == keel::kInvalidIndex) continue;
            keel::String next;
            next.reserve(cur.size());
            keel::usize pos = 0;
            while (pos <= curRef.size()) {
                const keel::usize rel = curRef.substr(pos, curRef.size() - pos).find(pp.as_ref());
                if (rel == keel::kInvalidIndex) {
                    next.append(curRef.substr(pos, curRef.size() - pos));
                    break;
                }
                next.append(curRef.substr(pos, rel));
                pos += rel + pp.size();
            }
            cur = std::move(next);
        }
        return cur;
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
    const keel::StringRef ss = s.asStringRef();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::String> olds(scratch.allocator()), news(scratch.allocator());
    collectMatchPatterns(oldPat, olds);
    collectMatchPatterns(newPat, news);
    if (news.size() != 1 && news.size() != olds.size())
        throw Error("replace: NEW must be a scalar text or match the number of "
                    "OLD patterns",
                     0, 0, "replace", "", "numkit:replace:sizeMismatch");

    keel::String out;
    out.reserve(ss.size());
    keel::usize i = 0;
    while (i < ss.size()) {
        bool matched = false;
        for (keel::usize k = 0; k < olds.size(); ++k) {
            const keel::StringRef o = olds[k].as_ref();
            if (!o.empty() && i + o.size() <= ss.size()
                && ss.substr(i, o.size()) == o) {
                out.append(((news.size() == 1) ? news[0] : news[k]).as_ref());
                i += o.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            out.push_back(ss[i]);
            ++i;
        }
    }
    if (s.isString()) return Value::stringScalar(out, mr);
    return Value::fromString(out, mr);
}

Value reverse(const Value &s, keel::Allocator *mr)
{
    auto op = [](keel::StringRef r) -> keel::String {
        keel::String out = keel::String::from(r);
        std::reverse(out.begin(), out.end());
        return out;
    };
    return mapStringPreserveClass(s, op, mr);
}

Value splitlines(const Value &s, keel::Allocator *mr)
{
    const keel::StringRef ss = s.asStringRef();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto *salloc = scratch.allocator();
    keel::Array<keel::String> parts(salloc);
    keel::String cur(salloc);
    for (keel::usize i = 0; i < ss.size(); ++i) {
        const char c = ss[i];
        if (c == '\r') {
            parts.push_back(std::move(cur));
            cur = keel::String(salloc);
            // Skip a following \n to handle CRLF as one separator.
            if (i + 1 < ss.size() && ss[i + 1] == '\n') ++i;
        } else if (c == '\n') {
            parts.push_back(std::move(cur));
            cur = keel::String(salloc);
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
static keel::String padOne(keel::StringRef r, keel::usize n, keel::StringRef sd, char ch)
{
    if (r.size() >= n) return keel::String::from(r);
    const keel::usize padLen = n - r.size();
    keel::String out;
    out.reserve(n);
    if (sd == "left") {
        for (keel::usize i = 0; i < padLen; ++i) out.push_back(ch);
        out.append(r);
    } else if (sd == "both") {
        const keel::usize left = padLen / 2;
        for (keel::usize i = 0; i < left; ++i) out.push_back(ch);
        out.append(r);
        for (keel::usize i = 0; i < padLen - left; ++i) out.push_back(ch);
    } else {  // "right" (default)
        out.append(r);
        for (keel::usize i = 0; i < padLen; ++i) out.push_back(ch);
    }
    return out;
}

Value pad(const Value &s, keel::usize n, const Value &side, const Value &padChar, keel::Allocator *mr)
{
    const keel::String sd = readSide(side, "right");
    const keel::StringRef sdRef = sd.as_ref();
    if (sdRef != "left" && sdRef != "right" && sdRef != "both")
        throw Error("pad: side must be 'left', 'right', or 'both'",
                     0, 0, "pad", "", "numkit:pad:badSide");
    char ch = ' ';
    if (!padChar.isEmpty() && (padChar.isChar() || padChar.isString())) {
        const keel::StringRef p = padChar.asStringRef();
        if (!p.empty()) ch = p[0];
    }
    // string array -> string array same shape; char -> char; cell -> cell
    // of char vectors. Each element is padded to the SAME width n.
    auto op = [&](keel::StringRef r) { return padOne(r, n, sdRef, ch); };
    return mapStringPreserveClass(s, op, mr);
}

Value strip(const Value &s, const Value &side, const Value &ch, keel::Allocator *mr)
{
    const keel::String sd = readSide(side, "both");
    const keel::StringRef sdRef = sd.as_ref();
    keel::StringRef csRef(" \t\r\n\f\v");
    bool noStrip = false;  // explicit empty strip-set => no-op
    if (!ch.isEmpty() && (ch.isChar() || ch.isString())) {
        csRef = ch.asStringRef();
        if (csRef.empty()) noStrip = true;
    }
    auto stripOne = [&](keel::StringRef r) -> keel::String {
        if (noStrip) return keel::String::from(r);
        keel::usize beg = 0;
        keel::usize end = r.size();
        if (sdRef == "left" || sdRef == "both") {
            while (beg < end && csRef.find(r[beg]) != keel::kInvalidIndex) ++beg;
        }
        if (sdRef == "right" || sdRef == "both") {
            while (end > beg && csRef.find(r[end - 1]) != keel::kInvalidIndex) --end;
        }
        return keel::String::from(r.substr(beg, end - beg));
    };
    return mapStringPreserveClass(s, stripOne, mr);
}

Value matches(const Value &s, const Value &pat, keel::Allocator *mr)
{
    // The pattern may be a single string or a cell / string array of
    // alternatives; an element matches if it equals ANY of them.
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::String> pats(scratch.allocator());
    collectMatchPatterns(pat, pats);
    auto matchOne = [&](keel::StringRef el) -> bool {
        for (const auto &pp : pats)
            if (el == pp.as_ref()) return true;
        return false;
    };
    // A string ARRAY / cell source -> same-shape LOGICAL array (element-wise).
    if (s.isCell() || (s.isString() && !s.isScalar())) {
        const keel::usize r = keel::as<keel::usize>(s.dims().rows());
        const keel::usize c = keel::as<keel::usize>(s.dims().cols());
        Value out = Value::matrix(r, c, ValueType::LOGICAL, mr);
        keel::u8 *od = out.logicalDataMut();
        const keel::usize nn = s.numel();
        for (keel::usize i = 0; i < nn; ++i) {
            if (s.isCell()) {
                od[i] = matchOne(s.cellAt(i).asStringRef()) ? 1 : 0;
            } else {
                od[i] = matchOne(s.stringElemRef(i)) ? 1 : 0;
            }
        }
        return out;
    }
    return Value::logicalScalar(matchOne(s.asStringRef()), mr);
}

// ── Pack 21 ──────────────────────────────────────────────────────────

Value convertCharsToStrings(const Value &x, keel::Allocator *mr)
{
    if (x.isString()) return x;
    if (x.isChar())   return Value::stringScalar(x.asStringRef(), mr);
    if (x.isCell()) {
        const auto &d = x.dims();
        auto c = d.is3D()
                    ? Value::cell3D(d.rows(), d.cols(), d.pages(), mr)
                    : Value::cell(d.rows(), d.cols(), mr);
        for (keel::usize i = 0; i < x.numel(); ++i) {
            const auto &e = x.cellAt(i);
            if (e.isChar())
                c.cellAt(i) = Value::stringScalar(e.asStringRef(), mr);
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
            return Value::fromString(x.asStringRef(), mr);
        }
        // Multi-element string → cell of char rows.
        const auto &d = x.dims();
        auto c = d.is3D()
                    ? Value::cell3D(d.rows(), d.cols(), d.pages(), mr)
                    : Value::cell(d.rows(), d.cols(), mr);
        for (keel::usize i = 0; i < x.numel(); ++i)
            c.cellAt(i) = Value::fromString(x.stringElemRef(i), mr);
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
                c.cellAt(i) = Value::fromString(e.asStringRef(), mr);
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
    const keel::StringRef cat = category.asStringRef();
    if (cat.equals_ignore_ascii_case("alpha"))
        return applyCharPred(s, keel::is_ascii_alpha, mr);
    if (cat.equals_ignore_ascii_case("digit"))
        return applyCharPred(s, keel::is_ascii_digit, mr);
    if (cat.equals_ignore_ascii_case("alphanum"))
        return applyCharPred(s, keel::is_ascii_alnum, mr);
    if (cat.equals_ignore_ascii_case("lower"))
        return applyCharPred(s, keel::is_ascii_lower, mr);
    if (cat.equals_ignore_ascii_case("upper"))
        return applyCharPred(s, keel::is_ascii_upper, mr);
    if (cat.equals_ignore_ascii_case("punct"))
        return applyCharPred(s, keel::is_ascii_punct, mr);
    if (cat.equals_ignore_ascii_case("space") || cat.equals_ignore_ascii_case("wspace"))
        return applyCharPred(s, keel::is_ascii_space, mr);
    if (cat.equals_ignore_ascii_case("xdigit"))
        return applyCharPred(s, keel::is_ascii_hex_digit, mr);
    if (cat.equals_ignore_ascii_case("cntrl"))
        return applyCharPred(s, keel::is_ascii_cntrl, mr);
    if (cat.equals_ignore_ascii_case("graphic"))
        return applyCharPred(s, keel::is_ascii_graph, mr);
    if (cat.equals_ignore_ascii_case("print"))
        return applyCharPred(s, keel::is_ascii_print, mr);
    keel::String catLow = keel::String::from(cat);
    asciiLowerInPlace(catLow);
    throw Error(keel::format("isstrprop: unknown category '{}'", catLow.as_ref()),
                 0, 0, "isstrprop", "", "numkit:isstrprop:badCat");
}

Value isstrprop(const Value &s, keel::StringRef category, keel::Allocator *mr)
{
    return isstrprop(s, Value::fromString(category, mr), mr);
}

Value isletter(const Value &s, keel::Allocator *mr)
{
    return applyCharPred(s, keel::is_ascii_alpha, mr);
}

Value isspaceFn(const Value &s, keel::Allocator *mr)
{
    return applyCharPred(s, keel::is_ascii_space, mr);
}

// ── Pack 22 ──────────────────────────────────────────────────────────
//
// Position-or-pattern lookup: numeric scalar → 0-based index, string
// → find first occurrence and return its [begin, end) range.


Value extractAfter(const Value &s, const Value &p, keel::Allocator *mr)
{
    // A cell str is processed element-wise (cell of char vectors); the
    // position/substring anchor p is scalar and applies to every element.
    auto op = [&](keel::StringRef ss) -> keel::String {
        const auto r = resolvePos(ss, p);
        return r.found ? keel::String::from(ss.substr(r.end, ss.size() - r.end))
                       : keel::String();
    };
    return mapStringPreserveClass(s, op, mr);
}

Value extractBefore(const Value &s, const Value &p, keel::Allocator *mr)
{
    auto op = [&](keel::StringRef ss) -> keel::String {
        const auto r = resolvePos(ss, p);
        return r.found ? keel::String::from(ss.substr(0, r.begin))
                       : keel::String();
    };
    return mapStringPreserveClass(s, op, mr);
}


Value extractBetween(const Value &s, const Value &start, const Value &end, bool inclusive, keel::Allocator *mr)
{
    const keel::StringRef ss = s.asStringRef();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto matches = findAllBetweenPairs(ss, start, end, scratch.allocator());
    // MATLAB returns an M-by-1 cell of inner strings — even for a
    // single match. Empty result is a 0-by-1 cell. See BUGS.md #27.
    auto c = Value::cell(matches.size(), 1, mr);
    for (keel::usize i = 0; i < matches.size(); ++i) {
        const auto &m = matches[i];
        const keel::usize beg = inclusive ? m.openBegin : m.openEnd;
        const keel::usize endIdx = inclusive ? m.closeEnd : m.closeBegin;
        const keel::StringRef inner = ss.substr(beg, endIdx - beg);
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
    const keel::StringRef nt = newText.asStringRef();
    auto op = [&](keel::StringRef ss) -> keel::String {
        const auto r = resolvePos(ss, p);
        if (!r.found) return keel::String::from(ss);
        keel::String out;
        out.reserve(ss.size() + nt.size());
        out.append(ss.substr(0, r.end));
        out.append(nt);
        out.append(ss.substr(r.end, ss.size() - r.end));
        return out;
    };
    return mapStringPreserveClass(s, op, mr);
}

Value insertBefore(const Value &s, const Value &p, const Value &newText, keel::Allocator *mr)
{
    const keel::StringRef nt = newText.asStringRef();
    auto op = [&](keel::StringRef ss) -> keel::String {
        const auto r = resolvePos(ss, p);
        if (!r.found) return keel::String::from(ss);
        keel::String out;
        out.reserve(ss.size() + nt.size());
        out.append(ss.substr(0, r.begin));
        out.append(nt);
        out.append(ss.substr(r.begin, ss.size() - r.begin));
        return out;
    };
    return mapStringPreserveClass(s, op, mr);
}

Value eraseBetween(const Value &s, const Value &start, const Value &end, bool inclusive, keel::Allocator *mr)
{
    const keel::StringRef ss = s.asStringRef();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto matches = findAllBetweenPairs(ss, start, end, scratch.allocator());
    if (matches.empty()) return s;
    // Walk left-to-right copying segments outside the matched ranges.
    // See BUGS.md #27.
    keel::String out;
    out.reserve(ss.size());
    keel::usize cursor = 0;
    for (const auto &m : matches) {
        const keel::usize keepUntil = inclusive ? m.openBegin : m.openEnd;
        const keel::usize resumeAt  = inclusive ? m.closeEnd  : m.closeBegin;
        out.append(ss.substr(cursor, keepUntil - cursor));
        cursor = resumeAt;
    }
    out.append(ss.substr(cursor, ss.size() - cursor));
    return strLikeOf(s, out.as_ref(), mr);
}

Value eraseBetween(const Value &s, const Value &start, const Value &end, keel::Allocator *mr)
{
    return eraseBetween(s, start, end, false, mr);
}

Value replaceBetween(const Value &s, const Value &start, const Value &end, const Value &newText, bool inclusive, keel::Allocator *mr)
{
    const keel::StringRef ss = s.asStringRef();
    const keel::StringRef nt = newText.asStringRef();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const auto matches = findAllBetweenPairs(ss, start, end, scratch.allocator());
    if (matches.empty()) return s;
    keel::String out;
    out.reserve(ss.size());
    keel::usize cursor = 0;
    for (const auto &m : matches) {
        const keel::usize keepUntil = inclusive ? m.openBegin : m.openEnd;
        const keel::usize resumeAt  = inclusive ? m.closeEnd  : m.closeBegin;
        out.append(ss.substr(cursor, keepUntil - cursor));
        out.append(nt);
        cursor = resumeAt;
    }
    out.append(ss.substr(cursor, ss.size() - cursor));
    return strLikeOf(s, out.as_ref(), mr);
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
    const keel::usize rows = keel::as<keel::usize>(s.dims().rows());
    if (s.type() == ValueType::CHAR && rows > 1) {
        const keel::usize cols = s.numel() / rows;
        const char *src = static_cast<const char *>(s.rawData());
        Value out = Value::matrix(rows, 1, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::String row(scratch.allocator());
        row.reserve(cols);
        for (keel::usize r = 0; r < rows; ++r) {
            row.clear();
            for (keel::usize c = 0; c < cols; ++c) row.push_back(src[c * rows + r]);
            od[r] = keel::as<keel::f64>(parseBase(row.as_ref(), base));
        }
        return out;
    }
    if (s.isCell() || (s.isString() && !s.isScalar())) {
        const keel::usize n = s.numel();
        Value out = Value::matrix(s.dims().rows(), s.dims().cols(),
                                  ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize i = 0; i < n; ++i) {
            if (s.isCell()) {
                od[i] = keel::as<keel::f64>(parseBase(s.cellAt(i).asStringRef(), base));
            } else {
                od[i] = keel::as<keel::f64>(parseBase(s.stringElemRef(i), base));
            }
        }
        return out;
    }
    return Value::scalar(keel::as<keel::f64>(parseBase(s.asStringRef(), base)), mr);
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
    const keel::usize rows = keel::as<keel::usize>(s.dims().rows());
    if (s.type() == ValueType::CHAR && rows > 1) {
        const keel::usize cols = s.numel() / rows;
        const char *src = static_cast<const char *>(s.rawData());
        Value out = Value::matrix(rows, 1, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::String row(scratch.allocator());
        row.reserve(cols);
        for (keel::usize r = 0; r < rows; ++r) {
            row.clear();
            for (keel::usize c = 0; c < cols; ++c) row.push_back(src[c * rows + r]);
            od[r] = hexBitsToDouble(row.as_ref());
        }
        return out;
    }
    // A cellstr / string array -> same-shape double array.
    if (s.isCell() || (s.isString() && !s.isScalar())) {
        const keel::usize n = s.numel();
        Value out = Value::matrix(s.dims().rows(), s.dims().cols(),
                                  ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize i = 0; i < n; ++i) {
            if (s.isCell()) {
                od[i] = hexBitsToDouble(s.cellAt(i).asStringRef());
            } else {
                od[i] = hexBitsToDouble(s.stringElemRef(i));
            }
        }
        return out;
    }
    // Scalar char row / string scalar -> scalar double.
    return Value::scalar(hexBitsToDouble(s.asStringRef()), mr);
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
    if (n == 0) return Value::matrix(0, keel::as<keel::usize>(W), ValueType::CHAR, mr);

    // numel × W CHAR matrix, one row per element (column-major order).
    Value m = Value::matrix(n, keel::as<keel::usize>(W), ValueType::CHAR, mr);
    char *dst = static_cast<char *>(m.rawDataMut());
    char buf[24];
    for (keel::usize i = 0; i < n; ++i) {
        const keel::f64 v = x.elemAsDouble(i);
        if (single) {
            const keel::f32 f = keel::as<keel::f32>(v);
            keel::u32 bits;
            std::memcpy(&bits, &f, sizeof(bits));
            std::snprintf(buf, sizeof(buf), "%08x", keel::as<unsigned int>(bits));
        } else {
            keel::u64 bits;
            std::memcpy(&bits, &v, sizeof(bits));
            std::snprintf(buf, sizeof(buf), "%016llx",
                          keel::as<unsigned long long>(bits));
        }
        for (int c = 0; c < W; ++c)
            dst[keel::as<keel::usize>(c) * n + i] = buf[c];
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
    const keel::usize rows = keel::as<keel::usize>(s.dims().rows());
    // A char MATRIX (multiple rows) parses each row -> column vector.
    if (rows > 1 && s.type() == ValueType::CHAR) {
        const keel::usize cols = s.numel() / rows;
        const char *src = static_cast<const char *>(s.rawData());
        Value out = Value::matrix(rows, 1, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::String row(scratch.allocator());
        row.reserve(cols);
        for (keel::usize r = 0; r < rows; ++r) {
            row.clear();
            for (keel::usize c = 0; c < cols; ++c) row.push_back(src[c * rows + r]);
            od[r] = keel::as<keel::f64>(parseBaseN(row.as_ref(), base));
        }
        return out;
    }
    return Value::scalar(keel::as<keel::f64>(parseBaseN(s.asStringRef(), base)), mr);
}


// Single-element CF expansion, used by both 1-output (string form) and
// 2-output (numeric N, D) callsites.
Value rat(const Value &x, keel::f64 tol, keel::Allocator *mr)
{
    const keel::f64 v = x.toScalar();
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
    auto appendOne = [&](keel::String &dst, keel::f64 v) {
        if (!std::isfinite(v)) {
            const keel::StringRef s = std::isnan(v) ? "NaN" : (v > 0 ? "Inf" : "-Inf");
            const int slack = std::max(0, field_width - keel::as<int>(s.size()));
            for (int k = 0; k < slack; ++k) dst.push_back(' ');
            dst.append(s);
            return;
        }
        const keel::f64 tol = defaultRatTol(v);
        const auto exp = ratExpansion(v, tol);
        char nbuf[32];
        const int nlen = std::snprintf(nbuf, sizeof(nbuf), "%lld", exp.p);
        if (exp.q == 1) {
            const int slack = std::max(0, field_width - nlen);
            for (int k = 0; k < slack; ++k) dst.push_back(' ');
            dst.append(nbuf);
            return;
        }
        char dbuf[32];
        const int dlen = std::snprintf(dbuf, sizeof(dbuf), "%lld", exp.q);
        const int n_pad = std::max(0, num_width - nlen);
        const int d_pad = std::max(0, den_width - dlen);
        for (int k = 0; k < n_pad; ++k) dst.push_back(' ');
        dst.append(nbuf);
        dst.push_back('/');
        dst.append(dbuf);
        for (int k = 0; k < d_pad; ++k) dst.push_back(' ');
    };

    if (n <= 1) {
        const keel::f64 v = x.toScalar();
        keel::String out;
        out.reserve(field_width);
        appendOne(out, v);
        return Value::fromString(out, mr);
    }

    // Vector: concatenate per-element fields end-to-end (one row).
    keel::String out;
    out.reserve(n * keel::as<keel::usize>(field_width));
    for (keel::usize i = 0; i < n; ++i)
        appendOne(out, x.elemAsDouble(i));
    return Value::fromString(out, mr);
}

// Single-string literal replacement: replace every non-overlapping occurrence
// of `op` in `r0` with `np`. Empty `op` is a no-op (matches MATLAB).
static keel::String strrepOne(keel::StringRef r0, keel::StringRef op,
                              keel::StringRef np)
{
    if (op.empty()) return keel::String::from(r0);
    keel::String out;
    out.reserve(r0.size());
    keel::usize pos = 0;
    while (pos <= r0.size()) {
        const keel::usize rel = r0.substr(pos, r0.size() - pos).find(op);
        if (rel == keel::kInvalidIndex) {
            out.append(r0.substr(pos, r0.size() - pos));
            break;
        }
        out.append(r0.substr(pos, rel));
        out.append(np);
        pos += rel + op.size();
    }
    return out;
}

Value strrep(const Value &s, const Value &oldPat, const Value &newPat, keel::Allocator *mr)
{
    keel::Allocator *p = mr;
    const bool sc = s.isCell(), oc = oldPat.isCell(), nc = newPat.isCell();

    // Scalar path: no cell arguments — return a char (or string) scalar,
    // preserving the original behaviour exactly.
    if (!sc && !oc && !nc) {
        const keel::String r = strrepOne(s.asStringRef(), oldPat.asStringRef(), newPat.asStringRef());
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
    const keel::usize sizes[3] = {ns, no, nn};
    for (keel::usize v : sizes) {
        if (v == 1) continue;
        if (n == 1) n = v;
        else if (v != n)
            throw Error("strrep: nonscalar arguments must match in size",
                        0, 0, "strrep", "", "numkit:strrep:cellSize");
    }
    keel::usize rr = 1, cc = 1;
    auto setShape = [&](const Value &v) {
        rr = keel::as<keel::usize>(v.dims().rows());
        cc = keel::as<keel::usize>(v.dims().cols());
    };
    // Shape comes from the non-scalar cell operand if any, else the (1x1)
    // cell operand that triggered the cell path.
    if (sc && ns == n && n != 1)       setShape(s);
    else if (oc && no == n && n != 1)  setShape(oldPat);
    else if (nc && nn == n && n != 1)  setShape(newPat);
    else if (sc)                       setShape(s);
    else if (oc)                       setShape(oldPat);
    else                               setShape(newPat);

    const keel::StringRef ss0 = sc ? keel::StringRef{} : s.asStringRef();
    const keel::StringRef os0 = oc ? keel::StringRef{} : oldPat.asStringRef();
    const keel::StringRef ns0 = nc ? keel::StringRef{} : newPat.asStringRef();

    auto out = Value::cell(rr, cc, p);
    for (keel::usize i = 0; i < n; ++i) {
        const keel::StringRef si = sc ? s.cellAt(ns == 1 ? 0 : i).asStringRef() : ss0;
        const keel::StringRef oi = oc ? oldPat.cellAt(no == 1 ? 0 : i).asStringRef() : os0;
        const keel::StringRef ni = nc ? newPat.cellAt(nn == 1 ? 0 : i).asStringRef() : ns0;
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
    const keel::StringRef fmtRef = fmt.asStringRef();
    // MATLAB: the output class mirrors the FORMAT class — a string format
    // yields a string array, a char format yields a cell of char vectors.
    const bool wantString = fmt.isString();

    // M = number of value-consuming conversion specs in one pass of the
    // format. MATLAB applies the format repeatedly across each ROW of x,
    // consuming M values per output element, so an R×C input yields an
    // R×ceil(C/M) result. A short trailing chunk leaves the unfilled specs
    // as literal text (e.g. compose('%d-%d',[1 2 3]) -> "1-2","3-%d").
    const keel::usize M = std::max<keel::usize>(1, countFormatSpecs(fmtRef));

    const auto &dims = x.dims();
    const keel::usize R = dims.rows();
    const keel::usize C = dims.cols();
    const keel::usize outCols = (C == 0) ? 0 : (C + M - 1) / M;

    auto buildResult = [&](keel::usize rows, keel::usize cols) -> Value {
        return wantString ? Value::stringArray(rows, cols, mr)
                          : Value::cell(rows, cols, mr);
    };
    auto setElem = [&](Value &v, keel::usize idx, keel::StringRef s) {
        if (wantString) v.stringElemSet(idx, s);
        else            v.cellAt(idx) = Value::fromString(s, mr);
    };

    if (R == 0 || C == 0)
        return buildResult(R, outCols);

    Value result = buildResult(R, outCols);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<Value> chunk(scratch.allocator());
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
            const keel::String s =
                formatOnce(fmtRef, {chunk.data(), chunk.size()}, 0,
                           /*literalWhenShort=*/true);
            setElem(result, j * R + r, s.as_ref());  // column-major out(r, j)
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
    const keel::StringRef src = M.asStringRef();
    if (rows == 0 || cols == 0) return M;

    Value res = Value::matrix(rows, cols, ValueType::CHAR, mr);
    char *out = res.charDataMut();
    std::memset(out, ' ', rows * cols);
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

    return res;
}

Value extract(const Value &s, const Value &pat, keel::Allocator *mr)
{
    const keel::StringRef sRef = strInputRef(s);
    const keel::StringRef pRef = strInputRef(pat);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::StringRef> hits(scratch.allocator());
    findAllLiteral(sRef, pRef, hits);

    if (hits.empty()) return Value::cell(0, 0, mr);
    Value c = Value::cell(hits.size(), 1, mr);
    for (keel::usize i = 0; i < hits.size(); ++i)
        c.cellAt(i) = Value::fromString(hits[i], mr);
    return c;
}

Value split(const Value &s, const Value &delim, keel::Allocator *mr)
{
    const keel::StringRef sRef = strInputRef(s);
    const keel::StringRef dRef = strInputRef(delim);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::StringRef> parts(scratch.allocator());
    splitKeepEmpty(sRef, dRef, parts);

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
    const keel::StringRef d = delim.isEmpty() ? keel::StringRef(" ") : strInputRef(delim);

    auto appendElem = [&](keel::String &dst, keel::usize i) {
        if (arr.isString()) {
            dst.append(arr.stringElemRef(i));
        } else if (arr.isCell()) {
            dst.append(arr.cellAt(i).asStringRef());
        } else {
            dst.append(arr.asStringRef());  // scalar char/string fallback
        }
    };

    if (arr.isScalar()) {
        keel::String s;
        appendElem(s, 0);
        return Value::stringScalar(s, mr);
    }

    const auto &dims = arr.dims();
    const keel::usize rows = dims.rows();
    const keel::usize cols = dims.cols();

    if (rows == 1 || cols == 1) {
        // 1-D: glue all elements with `d`.
        keel::String out;
        const keel::usize n = arr.numel();
        for (keel::usize i = 0; i < n; ++i) {
            if (i > 0) out.append(d);
            appendElem(out, i);
        }
        return Value::stringScalar(out, mr);
    }

    // 2-D: join along columns -> N×1 string column.
    Value r = Value::stringArray(rows, 1, mr);
    for (keel::usize row = 0; row < rows; ++row) {
        keel::String out;
        for (keel::usize col = 0; col < cols; ++col) {
            if (col > 0) out.append(d);
            appendElem(out, col * rows + row);
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
std::pair<Value, Value> strtok(const Value &str, keel::StringRef delim,
                               keel::Allocator *mr)
{
    const keel::StringRef s = str.asStringRef();
    auto isDelim = [&](char c) { return delim.find(c) != keel::kInvalidIndex; };

    // Skip leading delim chars.
    keel::usize start = 0;
    while (start < s.size() && isDelim(s[start])) ++start;
    // Find end of token.
    keel::usize end = start;
    while (end < s.size() && !isDelim(s[end])) ++end;

    Value tok = Value::fromString(s.substr(start, end - start), mr);
    Value rem = Value::fromString(
        end < s.size() ? s.substr(end, s.size() - end) : keel::StringRef("", 0),
        mr);
    return {std::move(tok), std::move(rem)};
}

Value convertContainedStringsToChars(const Value &v, keel::Allocator *mr)
{
    if (v.isString()) {
        if (v.numel() <= 1)
            return Value::fromString(v.asStringRef(), mr);
        auto c = Value::cell(v.numel(), 1, mr);
        for (keel::usize i = 0; i < v.numel(); ++i)
            c.cellAt(i) = Value::fromString(v.stringElemRef(i), mr);
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
