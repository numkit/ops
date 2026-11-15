// toolboxes/.../strings_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by strings.cpp + strings_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::builtin {

namespace {
// Format a SCALAR complex value as "re±|im|i" (MATLAB num2str). A common
// precision is used for both parts, derived from max(|re|,|im|) with the same
// magnitude-aware rule as the real default: prec = max(floor(log10 M),0)+5.
// precOverride >= 1 forces that precision (the num2str(z,N) form). An element
// with exactly-zero imaginary part prints as a bare real (num2str(complex(5,0))
// = "5"). The DEFAULT-precision/N forms for complex ARRAYS use MATLAB's
// column-aligned layout and remain a deferred gap (see num2str_reg).
std::string num2strComplexScalar(keel::c64 z, int precOverride)
{
    const double re = z.real(), im = z.imag();
    int prec = precOverride;
    if (prec < 1) {
        const double M = std::max(std::fabs(re), std::fabs(im));
        prec = 5;
        if (M != 0.0) {
            const int e = static_cast<int>(std::floor(std::log10(M)));
            prec = (e > 0 ? e : 0) + 5;
        }
    }
    if (prec > 99) prec = 99;
    auto fmtP = [prec](double v) {
        if (v == 0.0) v = 0.0;   // normalise -0 -> 0
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
        return std::string(buf);
    };
    if (im == 0.0) return fmtP(re);
    std::string s = fmtP(re);
    s += (im < 0.0 ? '-' : '+');
    s += fmtP(std::fabs(im));
    s += 'i';
    return s;
}

// Column-aligned layout for a COMPLEX, non-scalar array: each element
// formatted by num2strComplexScalar (imag==0 prints as plain real — the
// common complex-typed-storage-of-real-data case), columns padded to the
// max element width of the column, rows joined with '\n'.
// (bugs/opened/lang/num2str-complex-array.md)
inline std::string num2strComplexArray(const Value &x, int precOverride)
{
    const keel::usize rows = x.dims().rows();
    const keel::usize cols = x.dims().cols();
    if (rows * cols == 0) return "";
    const keel::c64 *d = x.complexData();

    // Split into two probed regimes (MATLAB R2025b):
    // 1) ALL imaginary parts == 0 (complex-typed storage of real data):
    //    MATLAB formats EXACTLY like the real path — all-integer arrays
    //    use "%<maxchars+2>" left-justified fields; fractional arrays use
    //    "%<P+7>.<P>g" right-aligned fields; common blank columns stripped.
    // 2) Any nonzero imaginary part: EVERY element prints in a+bi form
    //    (imag==0 elements get "+0i"); per-element scalar formatting.
    //    (Exact field-width model for this regime is a documented
    //    divergence — see the bug file.)
    bool anyImag = false, allInt = true;
    double maxAbs = 0.0;
    for (keel::usize i = 0; i < rows * cols; ++i) {
        if (d[i].imag() != 0.0) anyImag = true;
        if (d[i].real() != std::floor(d[i].real())) allInt = false;
        maxAbs = std::max(maxAbs, std::max(keel::abs(d[i].real()), keel::abs(d[i].imag())));
    }
    if (!anyImag) {
        std::vector<std::string> cells;
        keel::usize wmax = 0;
        for (keel::usize i = 0; i < rows * cols; ++i) {
            std::string s = num2strComplexScalar(d[i], precOverride);
            wmax = std::max(wmax, s.size());
            cells.push_back(std::move(s));
        }
        const keel::usize W = allInt ? wmax + 2
                                     : (static_cast<keel::usize>(std::max(0, static_cast<int>(std::floor(std::log10(maxAbs > 0 ? maxAbs : 1.0))))) + 5 + 7);
        std::string out;
        for (keel::usize r = 0; r < rows; ++r) {
            if (r) out += '\n';
            for (keel::usize c = 0; c < cols; ++c) {
                const std::string &s = cells[r + c * rows];
                const keel::usize pad = W > s.size() ? W - s.size() : 0;
                if (allInt) { out += s; out.append(pad, ' '); }
                else { out.append(pad, ' '); out += s; }
            }
        }
        // strip common leading/trailing blank columns (row-preserving)
        std::vector<std::string> lines;
        keel::usize pos = 0;
        while (pos <= out.size()) {
            keel::usize nl = out.find('\n', pos);
            lines.push_back(out.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos));
            if (nl == std::string::npos) break;
            pos = nl + 1;
        }
        keel::usize commonLead = lines.empty() ? 0 : lines[0].size();
        keel::usize commonTrail = lines.empty() ? 0 : lines[0].size();
        for (auto &L : lines) {
            keel::usize a = L.find_first_not_of(' ');
            if (a == std::string::npos) a = L.size();
            commonLead = std::min(commonLead, a);
            keel::usize b = L.find_last_not_of(' ');
            if (b == std::string::npos) b = 0; else b = L.size() - 1 - b;
            commonTrail = std::min(commonTrail, b);
        }
        std::string stripped;
        for (keel::usize i = 0; i < lines.size(); ++i) {
            if (i) stripped += '\n';
            keel::usize e = lines[i].size() > commonTrail ? lines[i].size() - commonTrail : 0;
            stripped += lines[i].substr(commonLead, e > commonLead ? e - commonLead : 0);
        }
        return stripped;
    }
    // regime 2: any nonzero imag — a+bi for every element
    std::string out;
    for (keel::usize r = 0; r < rows; ++r) {
        if (r) out += '\n';
        for (keel::usize c = 0; c < cols; ++c) {
            if (c) out += "   ";
            const keel::c64 z = d[r + c * rows];
            char buf[64], buf2[64];
            snprintf(buf, sizeof buf, "%g", z.real());
            snprintf(buf2, sizeof buf2, "%g", keel::abs(z.imag()));
            out += buf;
            out += (z.imag() < 0.0) ? "-" : "+";
            out += buf2;
            out += "i";
        }
    }
    return out;
}

// MATLAB's default num2str column format for a REAL, non-scalar array.
// N >= 1 forces the precision (the num2str(X,N) form); N <= 0 selects the
// default (auto): an all-integer array uses a fixed "%<W>.0f" field where
// W = (max element character count incl sign) + 2; otherwise a magnitude-
// aware "%<W>.<P>g" where P = max(floor(log10(maxabs)),0)+5 and W = P + 7.
// The returned format is applied cyclically per row; the caller routes it
// through the FMT overload, which performs MATLAB's common-blank-column
// stripping and NaN/Inf spelling. Matches MATLAB R2025b across integer,
// fractional, negative, and large/small-magnitude probes.
std::string num2strArrayFormat(const Value &x, int N)
{
    const keel::usize n = x.numel();
    char fmt[32];
    if (N >= 1) {
        int p = N; if (p > 99) p = 99;
        std::snprintf(fmt, sizeof(fmt), "%%%d.%dg", p + 7, p);
        return std::string(fmt);
    }
    double maxabs = 0.0;
    bool allInt = true;
    int maxChars = 1;
    for (keel::usize i = 0; i < n; ++i) {
        const double v = x.elemAsDouble(i);
        if (!std::isfinite(v)) { allInt = false; continue; }
        const double a = std::fabs(v);
        if (a > maxabs) maxabs = a;
        if (v != std::floor(v)) allInt = false;
        // Column width is driven by the DIGIT count of |v| (sign excluded):
        // MATLAB's integer field is max-abs-digits + 2, and a minus sign is
        // absorbed into the field rather than widening it — num2str([-1 10
        // -100])='-1   10 -100' (width 5), not width 6. Measuring "%.0f" of v
        // (which includes '-') over-counted by one for negatives.
        char b[64];
        const int len = std::snprintf(b, sizeof(b), "%.0f", a);
        if (len > maxChars) maxChars = len;
    }
    if (allInt) {
        std::snprintf(fmt, sizeof(fmt), "%%%d.0f", maxChars + 2);
        return std::string(fmt);
    }
    int p = 5;
    if (maxabs != 0.0) {
        const int e = static_cast<int>(std::floor(std::log10(maxabs)));
        p = (e > 0 ? e : 0) + 5;
    }
    if (p > 99) p = 99;
    std::snprintf(fmt, sizeof(fmt), "%%%d.%dg", p + 7, p);
    return std::string(fmt);
}
} // namespace
namespace {

// Element-wise string comparison with MATLAB cell-array broadcasting:
//   char/string vs char/string -> scalar logical (whole-string compare)
//   cell vs char/string-scalar  -> logical array shaped like the cell
//   cell vs cell                -> element-wise; sizes must match, or one
//                                  is a scalar (1x1) cell that broadcasts
// `cmp(sa, sb)` is the per-pair predicate (captures n for strncmp).
template <class Pred>
Value strCmpElementwise(const Value &a, const Value &b, Pred cmp,
                        const char *fn, keel::Allocator *mr)
{
    // An operand is an "array" (compared element-wise) when it is a cell OR
    // a non-scalar string array; a char / string scalar broadcasts against it.
    auto isArr = [](const Value &v) {
        return v.isCell() || (v.isString() && !v.isScalar());
    };
    auto isText = [](const Value &v) {
        return v.isChar() || v.isString();
    };
    auto elemIsText = [&](const Value &v, keel::usize idx) -> bool {
        return v.isCell() ? isText(v.cellAt(idx)) : v.isString();
    };
    auto elemAt = [](const Value &v, keel::usize idx) -> std::string {
        return v.isCell() ? std::string(v.cellAt(idx).toString().as_ref())
                          : std::string(v.stringElem(idx).as_ref());
    };
    const bool ac = isArr(a), bc = isArr(b);
    if (!ac && !bc) {
        if (!isText(a) || !isText(b))
            return Value::logicalScalar(false, mr);
        return Value::logicalScalar(
            cmp(std::string(a.toString().as_ref()), std::string(b.toString().as_ref())), mr);
    }

    const keel::usize na = ac ? a.numel() : 1;
    const keel::usize nb = bc ? b.numel() : 1;
    keel::usize n = 1, rr = 1, cc = 1;
    auto shapeOf = [](const Value &v, keel::usize &r, keel::usize &c) {
        r = static_cast<keel::usize>(v.dims().rows());
        c = static_cast<keel::usize>(v.dims().cols());
    };
    if (ac && bc) {
        if (na == nb)      { n = na; shapeOf(a, rr, cc); }
        else if (na == 1)  { n = nb; shapeOf(b, rr, cc); }
        else if (nb == 1)  { n = na; shapeOf(a, rr, cc); }
        else
            throw Error(std::string(fn) + ": cell array sizes must match",
                        0, 0, fn, "", std::string("numkit:") + fn + ":cellSize");
    } else if (ac) { n = na; shapeOf(a, rr, cc); }
    else           { n = nb; shapeOf(b, rr, cc); }

    auto out = Value::matrix(rr, cc, ValueType::LOGICAL, mr);
    auto *od = out.logicalDataMut();
    if ((!ac && !isText(a)) || (!bc && !isText(b))) {
        std::memset(od, 0, n);
        return out;
    }
    const std::string scalA = ac ? std::string() : std::string(a.toString().as_ref());
    const std::string scalB = bc ? std::string() : std::string(b.toString().as_ref());
    for (keel::usize i = 0; i < n; ++i) {
        const keel::usize ia = (na == 1 ? 0 : i);
        const keel::usize ib = (nb == 1 ? 0 : i);
        if ((ac && !elemIsText(a, ia)) || (bc && !elemIsText(b, ib))) {
            od[i] = 0;
            continue;
        }
        const std::string sa = ac ? elemAt(a, ia) : scalA;
        const std::string sb = bc ? elemAt(b, ib) : scalB;
        od[i] = cmp(sa, sb) ? 1 : 0;
    }
    return out;
}

// strncmp/strncmpi predicate: if BOTH strings are at least n chars, compare
// the first n; otherwise require full equality (MATLAB: strncmp('ab','ab',5)
// is true, strncmp('ab','abc',5) is false).
inline bool strnEq(const std::string &sa, const std::string &sb, keel::usize n)
{
    if (sa.size() >= n && sb.size() >= n)
        return sa.compare(0, n, sb, 0, n) == 0;
    return sa == sb;
}

} // namespace
namespace {

// Length of the LONGEST delimiter (literal) in `delims` matching `s` at
// position `i`, or 0 if none match. Longest-match so a delimiter ", " wins
// over "," when both are listed.
keel::usize matchDelimAt(std::string_view s, keel::usize i,
                    keel::ArrayRef<std::string> delims)
{
    keel::usize best = 0;
    for (const auto &d : delims)
        if (!d.empty() && d.size() > best && i + d.size() <= s.size() &&
            s.compare(i, d.size(), d) == 0)
            best = d.size();
    return best;
}

Value strsplitImpl(keel::StringRef sRef,
                   keel::ArrayRef<std::string> delims, bool collapse,
                   keel::Allocator *mr,
                   keel::Array<std::string> *matchesOut = nullptr)
{
    const std::string_view s(sRef);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<std::string> parts(scratch.allocator());
    std::string current;
    const keel::usize n = s.size();
    keel::usize i = 0;
    while (i < n) {
        keel::usize mlen = matchDelimAt(s, i, delims);
        if (mlen > 0) {
            parts.push_back(current);
            current.clear();
            const keel::usize matchStart = i;
            i += mlen;
            if (collapse) {
                keel::usize m2;
                while (i < n && (m2 = matchDelimAt(s, i, delims)) > 0)
                    i += m2;
            }
            if (matchesOut)
                matchesOut->emplace_back(s.substr(matchStart, i - matchStart));
        } else {
            current.push_back(s[i++]);
        }
    }
    parts.push_back(current);
    auto c = Value::cell(1, parts.size());
    for (keel::usize k = 0; k < parts.size(); ++k)
        c.cellAt(k) = Value::fromString(parts[k], mr);
    return c;
}

// Append the MATLAB default whitespace delimiter set to `delims`.
void appendDefaultWhitespace(keel::Array<std::string> &delims)
{
    const char ws[] = {' ', '\t', '\n', '\r', '\f', '\v'};
    for (char w : ws)
        delims.push_back(std::string(1, w));
}

// Build the delimiter list from a string OR a cell array of strings.
void appendDelims(const Value &delim, keel::Array<std::string> &delims)
{
    if (delim.isCell())
        for (keel::usize i = 0; i < delim.numel(); ++i)
            delims.emplace_back(delim.cellAt(i).toString().as_ref());
    else
        delims.emplace_back(delim.toString().as_ref());
}

} // namespace
namespace {
// All 1-based indices of `pp` in `ss` (overlapping, like MATLAB) as a 1×k
// DOUBLE row vector; [] (0×0) if no match or `pp` is empty.
Value strfindOne(keel::StringRef ssRef, keel::StringRef ppRef,
                 keel::Allocator *mr)
{
    const std::string_view ss(ssRef);
    const std::string_view pp(ppRef);
    if (pp.empty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);  // strfind(s,'') -> []
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::usize> hits(scratch.allocator());
    keel::usize pos = 0;
    while ((pos = ss.find(pp, pos)) != std::string::npos) {
        hits.push_back(pos + 1);  // 1-based
        pos += 1;                 // overlapping matches like MATLAB
    }
    if (hits.empty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    auto r = Value::matrix(1, hits.size(), ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < hits.size(); ++i)
        r.doubleDataMut()[i] = static_cast<double>(hits[i]);
    return r;
}
} // anonymous
namespace {

// Collect the pattern strings from `pat`: a char/string scalar yields one
// pattern; a cell array of char vectors or a multi-element string array yields
// one per element (both store their elements as Values in cellDataVec()).
// Shared by contains/startsWith/endsWith (match-any) and count/erase
// (apply each listed pattern).
void collectMatchPatterns(const Value &pat, keel::Array<std::string> &out)
{
    if (pat.isCell() || (pat.isString() && pat.numel() != 1)) {
        const auto &vec = pat.cellDataVec();
        for (const auto &e : vec) out.emplace_back(e.toString().as_ref());
    } else {
        out.emplace_back(pat.toString().as_ref());
    }
}

// ASCII lower-case in place — used by the 'IgnoreCase' option of
// contains/startsWith/endsWith.
inline void asciiLowerInPlace(std::string &x)
{
    for (char &c : x)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

} // namespace
namespace {
inline std::string readSide(const Value &side, const char *def)
{
    if (side.isEmpty()) return def;
    if (!side.isChar() && !side.isString())
        throw std::runtime_error("string-side argument must be a string");
    std::string s(side.toString().as_ref());
    for (auto &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
} // anon
namespace {
// Build a logical array shaped like the input char/string by running
// `predFn(c)` over each character.
template <typename PredFn>
Value applyCharPred(const Value &s, PredFn pred, keel::Allocator *mr)
{
    if (s.isChar()) {
        const auto str = s.toString();
        const keel::usize n = str.size();
        // Mirror char's shape (1×N for typical char rows).
        const auto &d = s.dims();
        auto r = (d.is3D()
                     ? Value::matrix3d(d.rows(), d.cols(), d.pages(),
                                       ValueType::LOGICAL, mr)
                     : Value::matrix(d.rows(), d.cols(),
                                     ValueType::LOGICAL, mr));
        for (keel::usize i = 0; i < n && i < r.numel(); ++i)
            r.logicalDataMut()[i] = pred(static_cast<unsigned char>(str[i])) ? 1 : 0;
        return r;
    }
    if (s.isString()) {
        // For string array, MATLAB returns a cell array of logical
        // arrays (one per element). For now support scalar strings.
        if (s.numel() == 1) {
            const auto str = s.toString();
            auto r = Value::matrix(1, str.size(), ValueType::LOGICAL, mr);
            for (keel::usize i = 0; i < str.size(); ++i)
                r.logicalDataMut()[i] = pred(static_cast<unsigned char>(str[i])) ? 1 : 0;
            return r;
        }
        throw Error("char-predicate: string-array form not supported",
                     0, 0, "isstrprop", "", "numkit:isstrprop:stringArray");
    }
    throw Error("char-predicate: input must be char or string",
                 0, 0, "isstrprop", "", "numkit:isstrprop:type");
}
} // anon
namespace {
struct PosRange { keel::usize begin; keel::usize end; bool found; };

// Resolve `p` to a [begin, end) range in `s`. For a numeric position
// p, [p-1, p) is returned (so callers can treat it as a "single
// character" anchor). For a string pattern, [pos, pos+len).
PosRange resolvePos(keel::StringRef sRef, const Value &p)
{
    const std::string_view s(sRef);
    if (p.isChar() || p.isString()) {
        const auto pat = p.toString();
        if (pat.empty()) return {0, 0, false};
        const keel::usize pos = s.find(std::string_view(pat));
        if (pos == std::string_view::npos) return {0, 0, false};
        return {pos, pos + pat.size(), true};
    }
    // Numeric scalar (1-based).
    const auto p1 = static_cast<long long>(p.toScalar());
    if (p1 < 1 || static_cast<keel::usize>(p1) > s.size())
        return {0, 0, false};
    return {static_cast<keel::usize>(p1 - 1), static_cast<keel::usize>(p1), true};
}

inline Value strLikeOf(const Value &s, keel::StringRef out, keel::Allocator *mr)
{
    if (s.isString()) return Value::stringScalar(out, mr);
    return Value::fromString(out, mr);
}
} // anon
namespace {

// Find all non-overlapping (start, end) delimiter pairs in `ss`. Each
// match's `begin` is the index of the opening delimiter, `end` is one
// past the closing delimiter, and `between` brackets just the inner
// text (between the two delimiters). Numeric (position-arg) inputs
// return at most one pair. See BUGS.md #27.
struct BetweenMatch {
    keel::usize openBegin;   // index of first char of opening delimiter
    keel::usize openEnd;     // index just past opening delimiter (start of inner)
    keel::usize closeBegin;  // index of first char of closing delimiter (end of inner)
    keel::usize closeEnd;    // index just past closing delimiter
};

std::vector<BetweenMatch>
findAllBetweenPairs(keel::StringRef ssRef, const Value &start, const Value &end)
{
    const std::string_view ss(ssRef);
    std::vector<BetweenMatch> matches;
    const bool startIsPattern = (start.isChar() || start.isString());
    const bool endIsPattern   = (end.isChar()   || end.isString());

    if (!startIsPattern || !endIsPattern) {
        // Numeric-anchor form: at most one match.
        const auto rs = resolvePos(ssRef, start);
        if (!rs.found) return matches;
        const std::string_view tail = ss.substr(rs.end);
        const auto re = resolvePos(keel::StringRef(tail.data(), tail.size()), end);
        if (!re.found) return matches;
        matches.push_back({rs.begin, rs.end, rs.end + re.begin, rs.end + re.end});
        return matches;
    }

    const std::string sp(start.toString().as_ref());
    const std::string ep(end.toString().as_ref());
    if (sp.empty() || ep.empty()) return matches;

    keel::usize cursor = 0;
    while (cursor < ss.size()) {
        const keel::usize sPos = ss.find(sp, cursor);
        if (sPos == std::string_view::npos) break;
        const keel::usize innerBeg = sPos + sp.size();
        const keel::usize ePos = ss.find(ep, innerBeg);
        if (ePos == std::string_view::npos) break;
        matches.push_back({sPos, innerBeg, ePos, ePos + ep.size()});
        // Advance past the closing delimiter so non-overlapping pairs
        // are matched — `<<a>><<b>>` yields 2 matches, not 1.
        cursor = ePos + ep.size();
    }
    return matches;
}

} // namespace
namespace {
std::string toBaseString(keel::u64 v, int base, int minWidth)
{
    if (v == 0) {
        std::string s(std::max(1, minWidth), '0');
        return s;
    }
    std::string out;
    while (v > 0) {
        const int d = static_cast<int>(v % static_cast<keel::u64>(base));
        out.push_back(d < 10 ? char('0' + d) : char('A' + d - 10));
        v /= static_cast<keel::u64>(base);
    }
    while (static_cast<int>(out.size()) < minWidth) out.push_back('0');
    std::reverse(out.begin(), out.end());
    return out;
}

keel::u64 parseBase(keel::StringRef s, int base)
{
    keel::u64 v = 0;
    for (char c : s) {
        if (std::isspace(static_cast<unsigned char>(c))) continue;
        int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = 10 + (c - 'a');
        else if (c >= 'A' && c <= 'F') d = 10 + (c - 'A');
        else throw Error(std::string("invalid digit '") + c + "' for base "
                          + std::to_string(base),
                          0, 0, "base", "", "numkit:base:badDigit");
        if (d >= base)
            throw Error(std::string("digit '") + c + "' out of range for base "
                          + std::to_string(base),
                          0, 0, "base", "", "numkit:base:badDigit");
        v = v * static_cast<keel::u64>(base) + static_cast<keel::u64>(d);
    }
    return v;
}

// Convert an N-element double vector to a 2-D char matrix where each
// row holds the base-`base` representation, padded to the maximum
// observed width (and at least minWidth).
Value vecToBaseMatrix(const Value &d, int base, int minWidth, keel::Allocator *mr)
{
    const keel::usize n = d.numel();
    if (n == 0) return Value::fromString("", mr);
    if (n == 1) {
        const double v = d.toScalar();
        if (v < 0) throw Error("dec2*: value must be non-negative",
                                0, 0, "dec2", "", "numkit:dec2:negative");
        return Value::fromString(
            toBaseString(static_cast<keel::u64>(v), base, minWidth), mr);
    }
    // Compute max width.
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto rows = keel::Array<std::string>(scratch.allocator());
    rows.reserve(n);
    int maxW = minWidth;
    for (keel::usize i = 0; i < n; ++i) {
        const double v = d.elemAsDouble(i);
        if (v < 0) throw Error("dec2*: value must be non-negative",
                                0, 0, "dec2", "", "numkit:dec2:negative");
        rows.emplace_back(toBaseString(static_cast<keel::u64>(v), base, 0));
        maxW = std::max<int>(maxW, static_cast<int>(rows.back().size()));
    }
    // Pad each row to maxW with leading zeros.
    for (auto &r : rows)
        while (static_cast<int>(r.size()) < maxW)
            r.insert(r.begin(), '0');
    // Build a CHAR matrix n × maxW, column-major.
    auto m = Value::matrix(n, maxW, ValueType::CHAR, mr);
    char *dst = static_cast<char *>(m.rawDataMut());
    for (keel::usize r = 0; r < n; ++r)
        for (int c = 0; c < maxW; ++c)
            dst[c * n + r] = rows[r][c];
    return m;
}
} // anon
namespace {
// Interpret a hex string as the raw IEEE-754 bit pattern of a double.
// Whitespace is skipped (a char matrix pads short rows with spaces); the
// digits are right-padded with '0' to 16 (MATLAB: hex2num('4') == 2).
double hexBitsToDouble(keel::StringRef raw)
{
    std::string h;
    h.reserve(16);
    for (char c : raw) {
        if (std::isspace(static_cast<unsigned char>(c))) continue;
        h.push_back(c);
    }
    if (h.size() > 16) h.resize(16);
    while (h.size() < 16) h.push_back('0');  // right-pad with zeros
    keel::u64 bits = 0;
    for (char c : h) {
        int d;
        if      (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = 10 + (c - 'a');
        else if (c >= 'A' && c <= 'F') d = 10 + (c - 'A');
        else throw Error(std::string("hex2num: invalid hex digit '") + c + "'",
                          0, 0, "hex2num", "", "numkit:hex2num:badDigit");
        bits = (bits << 4) | static_cast<keel::u64>(d);
    }
    double out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}
} // anon
namespace {
// Parse a base-`base` (2..36) digit string: 0-9 then A-Z / a-z. Whitespace
// is skipped (MATLAB pads short rows of a char matrix with spaces).
keel::u64 parseBaseN(keel::StringRef s, int base)
{
    keel::u64 v = 0;
    for (char c : s) {
        if (std::isspace(static_cast<unsigned char>(c))) continue;
        int d;
        if      (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'z') d = 10 + (c - 'a');
        else if (c >= 'A' && c <= 'Z') d = 10 + (c - 'A');
        else throw Error(std::string("base2dec: invalid digit '") + c + "'",
                          0, 0, "base2dec", "", "numkit:base2dec:badDigit");
        if (d >= base)
            throw Error(std::string("base2dec: digit '") + c +
                            "' out of range for base " + std::to_string(base),
                          0, 0, "base2dec", "", "numkit:base2dec:badDigit");
        v = v * static_cast<keel::u64>(base) + static_cast<keel::u64>(d);
    }
    return v;
}
} // anon
namespace {

// MATLAB's rat() uses the **regularized** continued-fraction expansion:
// at each step it picks `a = round(r)` (nearest integer, half-away-from-
// zero), not floor(r). This produces signed coefficients (e.g. 0.5 →
// `[1, -2]`, recovering 1 + 1/-2 = 0.5) and matches MATLAB's display
// `'1 + 1/(-2)'` exactly. The standard CF recurrence h_n = a_n·h_{n-1}
// + h_{n-2}, k_n = a_n·k_{n-1} + k_{n-2} works unchanged for signed a_n.
//
// Returns the sequence of CF coefficients (a_0, a_1, ...) plus the
// final converged p/q (denominator normalised positive). Stops when
// |x - p/q| ≤ tol, with tol defaulting (caller-side) to 1e-6·max(1,|x|).
struct RatExpansion {
    long long p = 0;
    long long q = 0;
    // Up to 64 coefficients — same safety bound as before. Inline
    // storage avoids any allocation for the typical 2-7 coefficient case.
    std::array<long long, 64> coeffs;
    int n_coeffs = 0;
};

RatExpansion ratExpansion(double x, double tol)
{
    RatExpansion r;
    if (!std::isfinite(x)) return r;

    double rem = x;
    // Track the last two convergents.
    long long prev_p = 0, curr_p = 1;
    long long prev_q = 1, curr_q = 0;

    for (int i = 0; i < 64; ++i) {
        const long long a = static_cast<long long>(std::round(rem));
        const long long new_p = a * curr_p + prev_p;
        const long long new_q = a * curr_q + prev_q;
        r.coeffs[r.n_coeffs++] = a;
        prev_p = curr_p; curr_p = new_p;
        prev_q = curr_q; curr_q = new_q;

        if (curr_q != 0) {
            const double approx = static_cast<double>(curr_p)
                                / static_cast<double>(curr_q);
            if (std::fabs(approx - x) <= tol) break;
        }
        const double frac = rem - static_cast<double>(a);
        if (frac == 0.0) break;
        rem = 1.0 / frac;
        if (std::fabs(rem) > 1e15) break;
    }

    r.p = curr_p;
    r.q = curr_q;
    if (r.q < 0) { r.p = -r.p; r.q = -r.q; }
    return r;
}

// Build MATLAB's nested CF-string from the coefficient sequence:
//   1 coeff   →  "a0"
//   2 coeffs  →  "a0 + 1/(a1)"
//   3 coeffs  →  "a0 + 1/(a1 + 1/(a2))"
//   n coeffs  →  "a0 + 1/(a1 + 1/( ... + 1/(an-1)))"
std::string buildCFString(const RatExpansion &r)
{
    if (r.n_coeffs == 0) return "0";
    std::string s = std::to_string(r.coeffs[r.n_coeffs - 1]);
    for (int i = r.n_coeffs - 2; i >= 0; --i)
        s = std::to_string(r.coeffs[i]) + " + 1/(" + s + ")";
    return s;
}

// Default tol per MATLAB docs: `1e-6 * norm(X(:),1)`. For a scalar
// input that's `1e-6 * |x|`, with a small floor at 1e-6 so x=0 still
// terminates (the loop hits frac == 0 at i=0 anyway, but keep safe).
inline double defaultRatTol(double x)
{
    return 1e-6 * std::max(1.0, std::fabs(x));
}

} // anonymous
namespace {
enum class StrPred { Contains, StartsWith, EndsWith };

// Test one source string against the collected patterns (any-match) under the
// chosen predicate. ignoreCase ASCII-lowercases both sides.
bool strMatchAny(keel::StringRef srcRef, const keel::Array<std::string> &pats,
                 bool ignoreCase, StrPred pred)
{
    std::string src(srcRef.data(), srcRef.size());
    if (ignoreCase) asciiLowerInPlace(src);
    for (std::string pp : pats) {
        if (ignoreCase) asciiLowerInPlace(pp);
        bool hit = false;
        switch (pred) {
        case StrPred::Contains:
            hit = src.find(pp) != std::string::npos;
            break;
        case StrPred::StartsWith:
            hit = src.size() >= pp.size() && src.compare(0, pp.size(), pp) == 0;
            break;
        case StrPred::EndsWith:
            hit = src.size() >= pp.size()
                  && src.compare(src.size() - pp.size(), pp.size(), pp) == 0;
            break;
        }
        if (hit) return true;
    }
    return false;
}

// Shared driver for contains/startsWith/endsWith: a cell array or a non-scalar
// string-array SOURCE maps element-wise to a LOGICAL array of the same shape
// (MATLAB); a scalar char/string source returns a logical scalar. The pattern
// argument may itself be a scalar or a cell/string array (any-match).
Value strPredicate(const Value &s, const Value &pat, bool ignoreCase,
                   StrPred pred, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<std::string> pats(scratch.allocator());
    collectMatchPatterns(pat, pats);

    if (s.isCell() || (s.isString() && s.numel() != 1)) {
        const keel::usize r = static_cast<keel::usize>(s.dims().rows());
        const keel::usize c = static_cast<keel::usize>(s.dims().cols());
        Value out = Value::matrix(r, c, ValueType::LOGICAL, mr);
        keel::u8 *od = out.logicalDataMut();
        const keel::usize n = s.numel();
        for (keel::usize i = 0; i < n; ++i) {
            const auto el =
                s.isCell() ? s.cellAt(i).toString() : s.stringElem(i);
            od[i] = strMatchAny(el, pats, ignoreCase, pred) ? 1 : 0;
        }
        return out;
    }
    return Value::logicalScalar(
        strMatchAny(s.toString(), pats, ignoreCase, pred), mr);
}
} // namespace
namespace {

// Read element i of x as a Value scalar to feed sprintf.
Value elemScalar(const Value &x, keel::usize i, keel::Allocator *mr)
{
    if (x.isNumeric()) {
        return Value::scalar(x.elemAsDouble(i), mr);
    }
    if (x.isLogical()) {
        return Value::logicalScalar(x.elemAsDouble(i) != 0, mr);
    }
    if (x.isString())  return Value::stringScalar(x.stringElem(i), mr);
    if (x.isCell())    return x.cellAt(i);  // cell-of-strings allowed
    return Value::scalar(0.0, mr);
}

// Get a char input string from char or string-scalar.
const std::string strInput(const Value &v)
{
    if (v.isChar() || (v.isString() && v.isScalar())) return std::string(v.toString().as_ref());
    if (v.isString()) return std::string(v.stringElem(0).as_ref());  // first elem of array
    return std::string(v.toString().as_ref());  // last-resort coercion
}

// Find every non-overlapping occurrence of `needle` in `hay` and append
// each matched substring to `out`.
void findAllLiteral(keel::StringRef hayRef, keel::StringRef needleRef,
                    std::vector<std::string> &out)
{
    const std::string_view hay(hayRef);
    const std::string_view needle(needleRef);
    if (needle.empty()) return;
    keel::usize pos = 0;
    while (true) {
        keel::usize f = hay.find(needle, pos);
        if (f == std::string_view::npos) break;
        out.emplace_back(needle);
        pos = f + needle.size();
    }
}

// Split `s` by `delim` keeping empty tokens (MATLAB `split` semantics).
void splitKeepEmpty(keel::StringRef sRef, keel::StringRef delimRef,
                    std::vector<std::string> &out)
{
    const std::string_view s(sRef);
    const std::string_view delim(delimRef);
    if (delim.empty()) { out.emplace_back(s); return; }
    keel::usize pos = 0;
    while (true) {
        keel::usize f = s.find(delim, pos);
        if (f == std::string_view::npos) {
            out.emplace_back(s.substr(pos));
            return;
        }
        out.emplace_back(s.substr(pos, f - pos));
        pos = f + delim.size();
    }
}

} // namespace

} // namespace numkit::builtin
