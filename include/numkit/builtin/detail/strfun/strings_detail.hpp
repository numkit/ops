// toolboxes/.../strings_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by strings.cpp + strings_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/string.h>
#include <keel/format.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <keel/utf8.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

#include <algorithm>
#include <keel/static_array.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

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
keel::String num2strComplexScalar(keel::c64 z, int precOverride,
                                  keel::Allocator *alloc = keel::default_allocator())
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
    char bufRe[64];
    const double rv = (re == 0.0) ? 0.0 : re;   // normalise -0 -> 0
    std::snprintf(bufRe, sizeof(bufRe), "%.*g", prec, rv);
    if (im == 0.0) return keel::String::from(bufRe, alloc);

    char bufIm[64];
    double iv = std::fabs(im);
    if (iv == 0.0) iv = 0.0;
    std::snprintf(bufIm, sizeof(bufIm), "%.*g", prec, iv);

    keel::String s(alloc);
    s.append(bufRe);
    s.push_back(im < 0.0 ? '-' : '+');
    s.append(bufIm);
    s.push_back('i');
    return s;
}

// Column-aligned layout for a COMPLEX, non-scalar array: each element
// formatted by num2strComplexScalar (imag==0 prints as plain real — the
// common complex-typed-storage-of-real-data case), columns padded to the
// max element width of the column, rows joined with '\n'.
// (bugs/opened/lang/num2str-complex-array.md)
inline keel::String num2strComplexArray(const Value &x, int precOverride,
                                        keel::Allocator *alloc = keel::default_allocator())
{
    const keel::usize rows = x.dims().rows();
    const keel::usize cols = x.dims().cols();
    if (rows * cols == 0) return keel::String(alloc);
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
        keel::ArenaTemp scratch = keel::get_scratch(alloc);
        auto *salloc = scratch.allocator();
        keel::Array<keel::String> cells(salloc);
        cells.reserve(rows * cols);
        keel::usize wmax = 0;
        for (keel::usize i = 0; i < rows * cols; ++i) {
            keel::String s = num2strComplexScalar(d[i], precOverride, salloc);
            wmax = std::max(wmax, s.size());
            cells.push_back(std::move(s));
        }
        const keel::usize W = allInt ? wmax + 2
                                     : (static_cast<keel::usize>(std::max(0, static_cast<int>(std::floor(std::log10(maxAbs > 0 ? maxAbs : 1.0))))) + 5 + 7);
        keel::String out(salloc);
        for (keel::usize r = 0; r < rows; ++r) {
            if (r) out.push_back('\n');
            for (keel::usize c = 0; c < cols; ++c) {
                const keel::String &s = cells[r + c * rows];
                const keel::usize pad = W > s.size() ? W - s.size() : 0;
                if (allInt) {
                    out.append(s.as_ref());
                    for (keel::usize p = 0; p < pad; ++p) out.push_back(' ');
                } else {
                    for (keel::usize p = 0; p < pad; ++p) out.push_back(' ');
                    out.append(s.as_ref());
                }
            }
        }
        // strip common leading/trailing blank columns (row-preserving)
        const keel::StringRef outRef = out.as_ref();
        keel::Array<keel::StringRef> lines(salloc);
        keel::usize pos = 0;
        while (pos <= outRef.size()) {
            const keel::StringRef rem = outRef.substr(pos, outRef.size() - pos);
            const keel::usize nl = rem.find('\n');
            if (nl == keel::kInvalidIndex) {
                lines.push_back(rem);
                break;
            }
            lines.push_back(rem.substr(0, nl));
            pos += nl + 1;
        }
        keel::usize commonLead = lines.empty() ? 0 : lines[0].size();
        keel::usize commonTrail = lines.empty() ? 0 : lines[0].size();
        for (const auto &L : lines) {
            keel::usize a = 0;
            while (a < L.size() && L[a] == ' ') ++a;
            commonLead = std::min(commonLead, a);
            keel::usize b = 0;
            if (a < L.size()) {
                while (b < L.size() && L[L.size() - 1 - b] == ' ') ++b;
            }
            commonTrail = std::min(commonTrail, b);
        }
        keel::String stripped(alloc);
        for (keel::usize i = 0; i < lines.size(); ++i) {
            if (i) stripped.push_back('\n');
            const keel::usize e = lines[i].size() > commonTrail ? lines[i].size() - commonTrail : 0;
            stripped.append(lines[i].substr(commonLead, e > commonLead ? e - commonLead : 0));
        }
        return stripped;
    }
    // regime 2: any nonzero imag — a+bi for every element
    keel::String out(alloc);
    for (keel::usize r = 0; r < rows; ++r) {
        if (r) out.push_back('\n');
        for (keel::usize c = 0; c < cols; ++c) {
            if (c) out.append("   ");
            const keel::c64 z = d[r + c * rows];
            char buf[64], buf2[64];
            std::snprintf(buf, sizeof buf, "%g", z.real());
            std::snprintf(buf2, sizeof buf2, "%g", keel::abs(z.imag()));
            out.append(buf);
            out.push_back((z.imag() < 0.0) ? '-' : '+');
            out.append(buf2);
            out.push_back('i');
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
[[maybe_unused]] keel::String num2strArrayFormat(const Value &x, int N)
{
    const keel::usize n = x.numel();
    char fmt[32];
    if (N >= 1) {
        int p = N; if (p > 99) p = 99;
        std::snprintf(fmt, sizeof(fmt), "%%%d.%dg", p + 7, p);
        return keel::String::from(fmt);
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
        return keel::String::from(fmt);
    }
    int p = 5;
    if (maxabs != 0.0) {
        const int e = static_cast<int>(std::floor(std::log10(maxabs)));
        p = (e > 0 ? e : 0) + 5;
    }
    if (p > 99) p = 99;
    std::snprintf(fmt, sizeof(fmt), "%%%d.%dg", p + 7, p);
    return keel::String::from(fmt);
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
    const bool ac = isArr(a), bc = isArr(b);
    if (!ac && !bc) {
        if (!isText(a) || !isText(b))
            return Value::logicalScalar(false, mr);
        return Value::logicalScalar(cmp(a.asStringRef(), b.asStringRef()), mr);
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
            throw Error(keel::format("{}: cell array sizes must match", fn),
                        0, 0, fn, "", keel::format("numkit:{}:cellSize", fn));
    } else if (ac) { n = na; shapeOf(a, rr, cc); }
    else           { n = nb; shapeOf(b, rr, cc); }

    auto out = Value::matrix(rr, cc, ValueType::LOGICAL, mr);
    auto *od = out.logicalDataMut();
    if ((!ac && !isText(a)) || (!bc && !isText(b))) {
        std::memset(od, 0, n);
        return out;
    }
    const keel::StringRef scalA = ac ? keel::StringRef{} : a.asStringRef();
    const keel::StringRef scalB = bc ? keel::StringRef{} : b.asStringRef();
    for (keel::usize i = 0; i < n; ++i) {
        const keel::usize ia = (na == 1 ? 0 : i);
        const keel::usize ib = (nb == 1 ? 0 : i);
        if ((ac && !elemIsText(a, ia)) || (bc && !elemIsText(b, ib))) {
            od[i] = 0;
            continue;
        }
        const keel::StringRef sa = !ac ? scalA
            : (a.isCell() ? a.cellAt(ia).asStringRef()
                          : a.stringElemRef(ia));
        const keel::StringRef sb = !bc ? scalB
            : (b.isCell() ? b.cellAt(ib).asStringRef()
                          : b.stringElemRef(ib));
        od[i] = cmp(sa, sb) ? 1 : 0;
    }
    return out;
}

// strncmp/strncmpi predicate: if BOTH strings are at least n chars, compare
// the first n; otherwise require full equality (MATLAB: strncmp('ab','ab',5)
// is true, strncmp('ab','abc',5) is false).
inline bool strnEq(keel::StringRef sa, keel::StringRef sb, keel::usize n)
{
    if (sa.size() >= n && sb.size() >= n)
        return sa.substr(0, n) == sb.substr(0, n);
    return sa == sb;
}

} // namespace
namespace {

// Length of the LONGEST delimiter (literal) in `delims` matching `s` at
// position `i`, or 0 if none match. Longest-match so a delimiter ", " wins
// over "," when both are listed.
keel::usize matchDelimAt(keel::StringRef s, keel::usize i,
                         keel::ArrayRef<keel::String> delims)
{
    keel::usize best = 0;
    for (const auto &d : delims)
        if (!d.empty() && d.size() > best && i + d.size() <= s.size() &&
            s.substr(i, d.size()) == d.as_ref())
            best = d.size();
    return best;
}

Value strsplitImpl(keel::StringRef s,
                   keel::ArrayRef<keel::String> delims, bool collapse,
                   keel::Allocator *mr,
                   keel::Array<keel::String> *matchesOut = nullptr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto *salloc = scratch.allocator();
    keel::Array<keel::String> parts(salloc);
    keel::String current(salloc);
    const keel::usize n = s.size();
    keel::usize i = 0;
    while (i < n) {
        keel::usize mlen = matchDelimAt(s, i, delims);
        if (mlen > 0) {
            parts.push_back(std::move(current));
            current = keel::String(salloc);
            const keel::usize matchStart = i;
            i += mlen;
            if (collapse) {
                keel::usize m2;
                while (i < n && (m2 = matchDelimAt(s, i, delims)) > 0)
                    i += m2;
            }
            if (matchesOut)
                matchesOut->push_back(
                    keel::String::from(s.substr(matchStart, i - matchStart)));
        } else {
            current.push_back(s[i++]);
        }
    }
    parts.push_back(std::move(current));
    auto c = Value::cell(1, parts.size());
    for (keel::usize k = 0; k < parts.size(); ++k)
        c.cellAt(k) = Value::fromString(parts[k], mr);
    return c;
}

// Append the MATLAB default whitespace delimiter set to `delims`.
void appendDefaultWhitespace(keel::Array<keel::String> &delims)
{
    const char ws[] = {' ', '\t', '\n', '\r', '\f', '\v'};
    for (char w : ws)
        delims.push_back(keel::String(keel::StringRef(&w, 1)));
}

// Build the delimiter list from a string OR a cell array of strings.
void appendDelims(const Value &delim, keel::Array<keel::String> &delims)
{
    if (delim.isCell())
        for (keel::usize i = 0; i < delim.numel(); ++i)
            delims.push_back(delim.cellAt(i).toString());
    else
        delims.push_back(delim.toString());
}

} // namespace
namespace {
// All 1-based indices of `pp` in `ss` (overlapping, like MATLAB) as a 1×k
// DOUBLE row vector; [] (0×0) if no match or `pp` is empty.
[[maybe_unused]] Value strfindOne(keel::StringRef ss, keel::StringRef pp,
                 keel::Allocator *mr)
{
    if (pp.empty())
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);  // strfind(s,'') -> []
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::usize> hits(scratch.allocator());
    keel::usize pos = 0;
    while (pos <= ss.size()) {
        const keel::usize rel = ss.substr(pos, ss.size() - pos).find(pp);
        if (rel == keel::kInvalidIndex) break;
        const keel::usize hit = pos + rel;
        hits.push_back(hit + 1);  // 1-based
        pos = hit + 1;            // overlapping matches like MATLAB
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
void collectMatchPatterns(const Value &pat, keel::Array<keel::String> &out)
{
    if (pat.isCell() || (pat.isString() && pat.numel() != 1)) {
        const auto &vec = pat.cellDataVec();
        for (const auto &e : vec) out.push_back(e.toString());
    } else {
        out.push_back(pat.toString());
    }
}

// ASCII lower-case in place — used by the 'IgnoreCase' option of
// contains/startsWith/endsWith.
inline void asciiLowerInPlace(keel::String &x)
{
    for (char &c : x)
        c = keel::to_ascii_lower(c);
}

} // namespace
namespace {
inline keel::String readSide(const Value &side, const char *def)
{
    if (side.isEmpty()) return keel::String::from(def);
    if (!side.isChar() && !side.isString())
        throw std::runtime_error("string-side argument must be a string");
    keel::String s = side.toString();
    for (char &c : s) c = keel::to_ascii_lower(c);
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
        const keel::StringRef str = s.asStringRef();
        const keel::usize n = str.size();
        // Mirror char's shape (1×N for typical char rows).
        const auto &d = s.dims();
        auto r = (d.is3D()
                     ? Value::matrix3d(d.rows(), d.cols(), d.pages(),
                                       ValueType::LOGICAL, mr)
                     : Value::matrix(d.rows(), d.cols(),
                                     ValueType::LOGICAL, mr));
        for (keel::usize i = 0; i < n && i < r.numel(); ++i)
            r.logicalDataMut()[i] = pred(str[i]) ? 1 : 0;
        return r;
    }
    if (s.isString()) {
        // For string array, MATLAB returns a cell array of logical
        // arrays (one per element). For now support scalar strings.
        if (s.numel() == 1) {
            const keel::StringRef str = s.asStringRef();
            auto r = Value::matrix(1, str.size(), ValueType::LOGICAL, mr);
            for (keel::usize i = 0; i < str.size(); ++i)
                r.logicalDataMut()[i] = pred(str[i]) ? 1 : 0;
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
PosRange resolvePos(keel::StringRef s, const Value &p)
{
    if (p.isChar() || p.isString()) {
        const keel::StringRef pat = p.asStringRef();
        if (pat.empty()) return {0, 0, false};
        const keel::usize pos = s.find(pat);
        if (pos == keel::kInvalidIndex) return {0, 0, false};
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

[[maybe_unused]] keel::Array<BetweenMatch>
findAllBetweenPairs(keel::StringRef ss, const Value &start, const Value &end,
                    keel::Allocator *alloc = keel::default_allocator())
{
    keel::Array<BetweenMatch> matches(alloc);
    const bool startIsPattern = (start.isChar() || start.isString());
    const bool endIsPattern   = (end.isChar()   || end.isString());

    if (!startIsPattern || !endIsPattern) {
        // Numeric-anchor form: at most one match.
        const auto rs = resolvePos(ss, start);
        if (!rs.found) return matches;
        const keel::StringRef tail = ss.substr(rs.end, ss.size() - rs.end);
        const auto re = resolvePos(tail, end);
        if (!re.found) return matches;
        matches.push_back({rs.begin, rs.end, rs.end + re.begin, rs.end + re.end});
        return matches;
    }

    const keel::StringRef sp = start.asStringRef();
    const keel::StringRef ep = end.asStringRef();
    if (sp.empty() || ep.empty()) return matches;

    keel::usize cursor = 0;
    while (cursor < ss.size()) {
        const keel::usize relS = ss.substr(cursor, ss.size() - cursor).find(sp);
        if (relS == keel::kInvalidIndex) break;
        const keel::usize sPos = cursor + relS;
        const keel::usize innerBeg = sPos + sp.size();
        const keel::usize relE = ss.substr(innerBeg, ss.size() - innerBeg).find(ep);
        if (relE == keel::kInvalidIndex) break;
        const keel::usize ePos = innerBeg + relE;
        matches.push_back({sPos, innerBeg, ePos, ePos + ep.size()});
        // Advance past the closing delimiter so non-overlapping pairs
        // are matched — `<<a>><<b>>` yields 2 matches, not 1.
        cursor = ePos + ep.size();
    }
    return matches;
}

} // namespace
namespace {
keel::String toBaseString(keel::u64 v, int base, int minWidth,
                          keel::Allocator *alloc = keel::default_allocator())
{
    keel::String out(alloc);
    if (v == 0) {
        const int w = std::max(1, minWidth);
        out.reserve(w);
        for (int i = 0; i < w; ++i) out.push_back('0');
        return out;
    }
    while (v > 0) {
        const int d = static_cast<int>(v % static_cast<keel::u64>(base));
        out.push_back(d < 10 ? char('0' + d) : char('A' + d - 10));
        v /= static_cast<keel::u64>(base);
    }
    while (static_cast<int>(out.size()) < minWidth) out.push_back('0');
    std::reverse(out.begin(), out.end());
    return out;
}

[[maybe_unused]] keel::u64 parseBase(keel::StringRef s, int base)
{
    keel::u64 v = 0;
    for (char c : s) {
        if (keel::is_ascii_space(c)) continue;
        int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = 10 + (c - 'a');
        else if (c >= 'A' && c <= 'F') d = 10 + (c - 'A');
        else throw Error(keel::format("invalid digit '{}' for base {}", c, base),
                          0, 0, "base", "", "numkit:base:badDigit");
        if (d >= base)
            throw Error(keel::format("digit '{}' out of range for base {}", c, base),
                          0, 0, "base", "", "numkit:base:badDigit");
        v = v * static_cast<keel::u64>(base) + static_cast<keel::u64>(d);
    }
    return v;
}

// Convert an N-element double vector to a 2-D char matrix where each
// row holds the base-`base` representation, padded to the maximum
// observed width (and at least minWidth).
[[maybe_unused]] Value vecToBaseMatrix(const Value &d, int base, int minWidth, keel::Allocator *mr)
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
    auto *salloc = scratch.allocator();
    auto rows = keel::Array<keel::String>(salloc);
    rows.reserve(n);
    int maxW = minWidth;
    for (keel::usize i = 0; i < n; ++i) {
        const double v = d.elemAsDouble(i);
        if (v < 0) throw Error("dec2*: value must be non-negative",
                                0, 0, "dec2", "", "numkit:dec2:negative");
        rows.push_back(toBaseString(static_cast<keel::u64>(v), base, 0, salloc));
        maxW = std::max<int>(maxW, static_cast<int>(rows.back().size()));
    }
    // Build a CHAR matrix n × maxW, column-major, left-padded with leading zeros.
    auto m = Value::matrix(n, maxW, ValueType::CHAR, mr);
    char *dst = static_cast<char *>(m.rawDataMut());
    for (keel::usize r = 0; r < n; ++r) {
        const int pad = maxW - static_cast<int>(rows[r].size());
        for (int c = 0; c < pad; ++c)
            dst[c * n + r] = '0';
        for (int c = pad; c < maxW; ++c)
            dst[c * n + r] = rows[r][static_cast<keel::usize>(c - pad)];
    }
    return m;
}
} // anon
namespace {
// Interpret a hex string as the raw IEEE-754 bit pattern of a double.
// Whitespace is skipped (a char matrix pads short rows with spaces); the
// digits are right-padded with '0' to 16 (MATLAB: hex2num('4') == 2).
[[maybe_unused]] double hexBitsToDouble(keel::StringRef raw)
{
    char h[16];
    int hlen = 0;
    for (char c : raw) {
        if (keel::is_ascii_space(c)) continue;
        if (hlen < 16) h[hlen++] = c;
    }
    while (hlen < 16) h[hlen++] = '0';  // right-pad with zeros
    keel::u64 bits = 0;
    for (int i = 0; i < 16; ++i) {
        const char c = h[i];
        int d;
        if      (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = 10 + (c - 'a');
        else if (c >= 'A' && c <= 'F') d = 10 + (c - 'A');
        else throw Error(keel::format("hex2num: invalid hex digit '{}'", c),
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
[[maybe_unused]] keel::u64 parseBaseN(keel::StringRef s, int base)
{
    keel::u64 v = 0;
    for (char c : s) {
        if (keel::is_ascii_space(c)) continue;
        int d;
        if      (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'z') d = 10 + (c - 'a');
        else if (c >= 'A' && c <= 'Z') d = 10 + (c - 'A');
        else throw Error(keel::format("base2dec: invalid digit '{}'", c),
                          0, 0, "base2dec", "", "numkit:base2dec:badDigit");
        if (d >= base)
            throw Error(keel::format("base2dec: digit '{}' out of range for base {}", c, base),
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
    keel::StaticArray<long long, 64> coeffs;
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
[[maybe_unused]] keel::String buildCFString(const RatExpansion &r)
{
    if (r.n_coeffs == 0) return keel::String::from("0");
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%lld", r.coeffs[r.n_coeffs - 1]);
    keel::String s = keel::String::from(buf);
    for (int i = r.n_coeffs - 2; i >= 0; --i) {
        std::snprintf(buf, sizeof(buf), "%lld", r.coeffs[i]);
        keel::String next;
        next.reserve(std::strlen(buf) + 6 + s.size() + 1);
        next.append(buf);
        next.append(" + 1/(");
        next.append(s.as_ref());
        next.push_back(')');
        s = std::move(next);
    }
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
bool strMatchAny(keel::StringRef srcRef, keel::ArrayRef<keel::String> pats,
                 bool ignoreCase, StrPred pred)
{
    keel::String srcLower;
    keel::StringRef src = srcRef;
    if (ignoreCase) {
        srcLower = keel::String::from(srcRef);
        asciiLowerInPlace(srcLower);
        src = srcLower.as_ref();
    }
    for (const auto &patItem : pats) {
        if (patItem.empty()) return true;
        keel::String ppLower;
        keel::StringRef pp = patItem.as_ref();
        if (ignoreCase) {
            ppLower = patItem.clone();
            asciiLowerInPlace(ppLower);
            pp = ppLower.as_ref();
        }
        bool hit = false;
        switch (pred) {
        case StrPred::Contains:
            hit = src.find(pp) != keel::kInvalidIndex;
            break;
        case StrPred::StartsWith:
            hit = src.starts_with(pp);
            break;
        case StrPred::EndsWith:
            hit = src.ends_with(pp);
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
[[maybe_unused]] Value strPredicate(const Value &s, const Value &pat, bool ignoreCase,
                   StrPred pred, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::String> pats(scratch.allocator());
    collectMatchPatterns(pat, pats);

    if (s.isCell() || (s.isString() && s.numel() != 1)) {
        const keel::usize r = static_cast<keel::usize>(s.dims().rows());
        const keel::usize c = static_cast<keel::usize>(s.dims().cols());
        Value out = Value::matrix(r, c, ValueType::LOGICAL, mr);
        keel::u8 *od = out.logicalDataMut();
        const keel::usize n = s.numel();
        for (keel::usize i = 0; i < n; ++i) {
            const keel::StringRef el = s.isCell()
                ? s.cellAt(i).asStringRef()
                : s.stringElemRef(i);
            od[i] = strMatchAny(el, pats, ignoreCase, pred) ? 1 : 0;
        }
        return out;
    }
    return Value::logicalScalar(
        strMatchAny(s.asStringRef(), pats, ignoreCase, pred), mr);
}
} // namespace
namespace {

// Read element i of x as a Value scalar to feed sprintf.
[[maybe_unused]] Value elemScalar(const Value &x, keel::usize i, keel::Allocator *mr)
{
    if (x.isNumeric()) {
        return Value::scalar(x.elemAsDouble(i), mr);
    }
    if (x.isLogical()) {
        return Value::logicalScalar(x.elemAsDouble(i) != 0, mr);
    }
    if (x.isString())  return Value::stringScalar(x.stringElemRef(i), mr);
    if (x.isCell())    return x.cellAt(i);  // cell-of-strings allowed
    return Value::scalar(0.0, mr);
}

// Get a char input string view from char or string-scalar.
[[maybe_unused]] keel::StringRef strInputRef(const Value &v)
{
    if (v.isChar() || (v.isString() && v.isScalar())) return v.asStringRef();
    if (v.isString()) return v.stringElemRef(0);  // first elem of array
    return v.asStringRef();  // last-resort coercion
}

// Get a char input string from char or string-scalar.
[[maybe_unused]] keel::String strInput(const Value &v)
{
    return keel::String::from(strInputRef(v));
}

// Find every non-overlapping occurrence of `needle` in `hay` and append
// each matched substring to `out`.
[[maybe_unused]] void findAllLiteral(keel::StringRef hay, keel::StringRef needle,
                    keel::Array<keel::StringRef> &out)
{
    if (needle.empty()) return;
    keel::usize pos = 0;
    while (pos <= hay.size()) {
        const keel::usize rel = hay.substr(pos, hay.size() - pos).find(needle);
        if (rel == keel::kInvalidIndex) break;
        out.push_back(needle);
        pos += rel + needle.size();
    }
}

// Split `s` by `delim` keeping empty tokens (MATLAB `split` semantics).
[[maybe_unused]] void splitKeepEmpty(keel::StringRef s, keel::StringRef delim,
                    keel::Array<keel::StringRef> &out)
{
    if (delim.empty()) { out.push_back(s); return; }
    keel::usize pos = 0;
    while (true) {
        const keel::usize rel = s.substr(pos, s.size() - pos).find(delim);
        if (rel == keel::kInvalidIndex) {
            out.push_back(s.substr(pos, s.size() - pos));
            return;
        }
        out.push_back(s.substr(pos, rel));
        pos += rel + delim.size();
    }
}

} // namespace

} // namespace numkit::builtin
