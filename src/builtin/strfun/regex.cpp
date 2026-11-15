// toolboxes/builtin/src/datatypes/strings/regex.cpp
//
// regexp / regexpi / regexprep — ECMAScript regex via std::regex.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/string.h>
#include <numkit/builtin/strfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <cctype>
#include <iterator>
#include <regex>
#include <utility>

#include "regex_detail.hpp"

namespace numkit::builtin {

namespace {

inline keel::StringRef subMatchRef(const std::cmatch &m, keel::usize g)
{
    if (g >= m.size() || !m[g].matched) return keel::StringRef("", 0);
    return keel::StringRef(m[g].first, static_cast<keel::usize>(m[g].length()));
}

struct KeelStringInserter {
    using iterator_category = std::output_iterator_tag;
    using value_type        = void;
    using difference_type   = std::ptrdiff_t;
    using pointer           = void;
    using reference         = void;

    keel::String *out;
    KeelStringInserter &operator=(char c) { out->push_back(c); return *this; }
    KeelStringInserter &operator*() { return *this; }
    KeelStringInserter &operator++() { return *this; }
    KeelStringInserter operator++(int) { return *this; }
};

} // namespace

Value regexpFind(const Value &s, const Value &pat, keel::StringRef option, bool ignoreCase, keel::Allocator *mr)
{
    if ((!s.isChar() && !s.isString()) || (!pat.isChar() && !pat.isString()))
        throw Error("regexp: s and pat must be strings",
                     0, 0, "regexp", "", "numkit:regexp:badArg");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Allocator *salloc = scratch.allocator();

    const keel::String textOwned = s.toString();
    const keel::StringRef text = textOwned.as_ref();
    const char *tbeg = text.data();
    const char *tend = tbeg + text.size();

    // Strip MATLAB named-group syntax to a std::regex-compatible pattern,
    // keeping the name → capture-group-index map for the 'names' option.
    const NamedGroups ng = extractNamedGroups(pat.toString(), salloc);
    const std::regex  re = compileRegex(ng.cleaned, ignoreCase);

    if (option.equals_ignore_ascii_case("names")) {
        // Count matches first to choose scalar struct (1) vs struct array.
        keel::usize count = 0;
        for (auto it = std::cregex_iterator(tbeg, tend, re),
                  end = std::cregex_iterator(); it != end; ++it)
            ++count;

        if (count == 0) {
            // MATLAB returns a 0×0 struct that still carries the field names.
            Value out = Value::structArray(0, 0, mr);
            for (const auto &nm : ng.names) out.setFieldAll(nm.first, Value::Empty);
            return out;
        }
        if (count == 1) {
            auto it = std::cregex_iterator(tbeg, tend, re);
            const auto &m = *it;
            Value out = Value::structure(mr);
            for (const auto &nm : ng.names)
                out.setFieldAll(nm.first, Value::fromString(subMatchRef(m, nm.second), mr));
            return out;
        }
        Value out = Value::structArray(1, count, mr);
        keel::usize i = 0;
        for (auto it = std::cregex_iterator(tbeg, tend, re),
                  end = std::cregex_iterator(); it != end; ++it, ++i) {
            const auto &m = *it;
            for (const auto &nm : ng.names)
                out.setField(i, nm.first, Value::fromString(subMatchRef(m, nm.second), mr));
        }
        return out;
    }

    if (option.equals_ignore_ascii_case("split")) {
        keel::Array<keel::StringRef> parts(salloc);
        auto begin = std::cregex_iterator(tbeg, tend, re);
        auto end   = std::cregex_iterator();
        keel::usize prev = 0;
        for (auto it = begin; it != end; ++it) {
            const auto &m = *it;
            const keel::usize mpos = static_cast<keel::usize>(m.position());
            parts.push_back(text.substr(prev, mpos - prev));
            prev = mpos + static_cast<keel::usize>(m.length());
        }
        parts.push_back(text.substr(prev));
        return rowCellOfStrings(parts.data(), parts.size(), mr);
    }

    if (option.equals_ignore_ascii_case("match")) {
        keel::Array<keel::StringRef> matches(salloc);
        for (auto it = std::cregex_iterator(tbeg, tend, re),
                  end = std::cregex_iterator(); it != end; ++it)
            matches.push_back(subMatchRef(*it, 0));
        return rowCellOfStrings(matches.data(), matches.size(), mr);
    }

    if (option.equals_ignore_ascii_case("tokens")) {
        // 1×N cell, each entry is a 1×k cell of capture group strings.
        keel::Array<keel::Array<keel::StringRef>> all(salloc);
        for (auto it = std::cregex_iterator(tbeg, tend, re),
                  end = std::cregex_iterator(); it != end; ++it) {
            keel::Array<keel::StringRef> grp(salloc);
            for (keel::usize g = 1; g < it->size(); ++g)
                grp.push_back(subMatchRef(*it, g));
            all.push_back(std::move(grp));
        }
        auto out = Value::cell(1, all.size());
        for (keel::usize i = 0; i < all.size(); ++i)
            out.cellAt(i) = rowCellOfStrings(all[i].data(), all[i].size(), mr);
        return out;
    }

    if (option.equals_ignore_ascii_case("end")) {
        // 1-based end index of each match (= position + length).
        keel::Array<double> idx(salloc);
        for (auto it = std::cregex_iterator(tbeg, tend, re),
                  e = std::cregex_iterator(); it != e; ++it)
            idx.push_back(static_cast<double>(it->position() + it->length()));
        return rowFromIndices(idx.data(), idx.size(), mr);
    }

    if (option.equals_ignore_ascii_case("tokenextents")) {
        // 1×N cell; each entry is a k×2 matrix of [startCol endCol] (1-based)
        // for the capture groups. With no capture groups, the whole match is
        // the single token.
        keel::usize count = 0;
        for (auto it = std::cregex_iterator(tbeg, tend, re),
                  e = std::cregex_iterator(); it != e; ++it)
            ++count;
        auto out = Value::cell(1, count);
        keel::usize i = 0;
        for (auto it = std::cregex_iterator(tbeg, tend, re),
                  e = std::cregex_iterator(); it != e; ++it, ++i) {
            const auto &m = *it;
            const keel::usize ng2 = (m.size() > 1) ? m.size() - 1 : 1;
            auto te = Value::matrix(ng2, 2, ValueType::DOUBLE, mr);
            double *td = te.doubleDataMut();
            for (keel::usize g = 0; g < ng2; ++g) {
                const auto &sub = (m.size() > 1) ? m[g + 1] : m[0];
                const keel::isize st = sub.first - tbeg;
                td[0 * ng2 + g] = static_cast<double>(st + 1);
                td[1 * ng2 + g] = static_cast<double>(st + sub.length());
            }
            out.cellAt(i) = te;
        }
        return out;
    }

    if (!option.empty() && !option.equals_ignore_ascii_case("start"))
        throw Error(keel::format("regexp: unknown option '{}' (supported: 'start' / 'end' / 'tokenExtents' / "
                                 "'match' / 'tokens' / 'names' / 'split')", option),
                     0, 0, "regexp", "", "numkit:regexp:badOption");

    // Default ('start'): 1-based start indices.
    keel::Array<double> idx(salloc);
    for (auto it = std::cregex_iterator(tbeg, tend, re),
              end = std::cregex_iterator(); it != end; ++it)
        idx.push_back(static_cast<double>(it->position() + 1));
    return rowFromIndices(idx.data(), idx.size(), mr);
}

// regexp(..., 'once'): match only the FIRST occurrence and return the
// "scalarised" form of the requested output (MATLAB R2025b):
//   'start'/'end'      -> scalar index (or [] when no match)
//   'match'            -> char row     (or '' when no match)
//   'tokens'           -> 1×k cell of capture-group chars (or {} no match)
//   'tokenExtents'     -> k×2 matrix   (or [] when no match)
//   'names'            -> scalar struct (or 0×0 struct w/ fields, no match)
//   'split'            -> {prefix, remainder} split at the first match only
// Kept separate from regexpFind so the all-matches paths stay byte-for-byte
// unchanged.
Value regexpFindOnce(const Value &s, const Value &pat, keel::StringRef option,
                     bool ignoreCase, keel::Allocator *mr)
{
    if ((!s.isChar() && !s.isString()) || (!pat.isChar() && !pat.isString()))
        throw Error("regexp: s and pat must be strings",
                     0, 0, "regexp", "", "numkit:regexp:badArg");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Allocator *salloc = scratch.allocator();

    const keel::String textOwned = s.toString();
    const keel::StringRef text = textOwned.as_ref();
    const char *tbeg = text.data();
    const char *tend = tbeg + text.size();

    const NamedGroups ng = extractNamedGroups(pat.toString(), salloc);
    const std::regex  re = compileRegex(ng.cleaned, ignoreCase);

    auto it  = std::cregex_iterator(tbeg, tend, re);
    auto end = std::cregex_iterator();
    const bool has = (it != end);

    if (option.equals_ignore_ascii_case("match"))
        return Value::fromString(has ? subMatchRef(*it, 0) : keel::StringRef("", 0), mr);

    if (option.equals_ignore_ascii_case("tokens")) {
        if (!has) return Value::cell(0, 0, mr);
        const auto &m = *it;
        keel::Array<keel::StringRef> grp(salloc);
        for (keel::usize g = 1; g < m.size(); ++g) grp.push_back(subMatchRef(m, g));
        return rowCellOfStrings(grp.data(), grp.size(), mr);
    }

    if (option.equals_ignore_ascii_case("split")) {
        keel::Array<keel::StringRef> parts(salloc);
        if (!has) {
            parts.push_back(text);
        } else {
            const auto &m = *it;
            const keel::usize mpos = static_cast<keel::usize>(m.position());
            parts.push_back(text.substr(0, mpos));
            parts.push_back(text.substr(mpos + static_cast<keel::usize>(m.length())));
        }
        return rowCellOfStrings(parts.data(), parts.size(), mr);
    }

    if (option.equals_ignore_ascii_case("names")) {
        if (!has) {
            Value out = Value::structArray(0, 0, mr);
            for (const auto &nm : ng.names) out.setFieldAll(nm.first, Value::Empty);
            return out;
        }
        const auto &m = *it;
        Value out = Value::structure(mr);
        for (const auto &nm : ng.names)
            out.setFieldAll(nm.first, Value::fromString(subMatchRef(m, nm.second), mr));
        return out;
    }

    if (option.equals_ignore_ascii_case("tokenextents")) {
        if (!has) return Value::matrix(0, 0, ValueType::DOUBLE, mr);
        const auto &m = *it;
        const keel::usize ng2 = (m.size() > 1) ? m.size() - 1 : 1;
        auto te = Value::matrix(ng2, 2, ValueType::DOUBLE, mr);
        double *td = te.doubleDataMut();
        for (keel::usize g = 0; g < ng2; ++g) {
            const auto &sub = (m.size() > 1) ? m[g + 1] : m[0];
            const keel::isize st = sub.first - tbeg;
            td[0 * ng2 + g] = static_cast<double>(st + 1);
            td[1 * ng2 + g] = static_cast<double>(st + sub.length());
        }
        return te;
    }

    if (option.equals_ignore_ascii_case("end")) {
        if (!has) return Value::matrix(0, 0, ValueType::DOUBLE, mr);
        return Value::scalar(static_cast<double>(it->position() + it->length()), mr);
    }

    if (!option.empty() && !option.equals_ignore_ascii_case("start"))
        throw Error(keel::format("regexp: unknown option '{}' (supported: 'start' / 'end' / 'tokenExtents' / "
                                 "'match' / 'tokens' / 'names' / 'split')", option),
                     0, 0, "regexp", "", "numkit:regexp:badOption");

    // 'start' (default): scalar 1-based index of the first match.
    if (!has) return Value::matrix(0, 0, ValueType::DOUBLE, mr);
    return Value::scalar(static_cast<double>(it->position() + 1), mr);
}

// Apply ONE pattern/replacement to a single string. With `once`, only the
// first match is replaced (MATLAB's 'once' option).
static keel::String regexrepOne(keel::StringRef text, keel::StringRef pat,
                                const keel::String &rep, bool ignoreCase, bool once,
                                keel::Allocator *alloc)
{
    const NamedGroups ng = extractNamedGroups(pat, alloc);
    const std::regex re = compileRegex(ng.cleaned, ignoreCase);
    const auto flags = once ? std::regex_constants::format_first_only
                            : std::regex_constants::format_default;
    keel::String out(alloc);
    out.reserve(text.size());
    std::regex_replace(KeelStringInserter{&out},
                       text.data(), text.data() + text.size(),
                       re, rep.c_str(), flags);
    return out;
}

// Apply a LIST of patterns to one string, in sequence (MATLAB regexprep with
// a cell pattern array applies each pattern/replacement in turn, feeding the
// result of one into the next). A single replacement is recycled for every
// pattern; otherwise the replacement count must equal the pattern count.
static keel::String regexrepSeq(keel::StringRef text0,
                                const keel::Array<keel::String> &pats,
                                const keel::Array<keel::String> &reps,
                                bool ignoreCase, bool once,
                                keel::Allocator *alloc)
{
    keel::String text(text0, alloc);
    for (keel::usize i = 0; i < pats.size(); ++i)
        text = regexrepOne(text.as_ref(), pats[i].as_ref(),
                           reps.size() == 1 ? reps[0] : reps[i],
                           ignoreCase, once, alloc);
    return text;
}

Value regexprep(const Value &s, const Value &pat, const Value &rep, bool ignoreCase, bool once, keel::Allocator *mr)
{
    auto isStrLike = [](const Value &v) { return v.isChar() || v.isString() || v.isCell(); };
    if (!isStrLike(s) || !isStrLike(pat) || !isStrLike(rep))
        throw Error("regexprep: s, pat, rep must be strings or cell arrays of strings",
                     0, 0, "regexprep", "", "numkit:regexprep:badArg");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Allocator *salloc = scratch.allocator();
    keel::Array<keel::String> pats(salloc), reps(salloc);
    auto collect = [salloc](const Value &v, keel::Array<keel::String> &out) {
        if (v.isCell()) {
            const keel::usize n = v.numel();
            for (keel::usize i = 0; i < n; ++i) {
                auto sv = v.cellAt(i).toString();
                out.emplace_back(sv.as_ref(), salloc);
            }
        } else {
            auto sv = v.toString();
            out.emplace_back(sv.as_ref(), salloc);
        }
    };
    collect(pat, pats);
    collect(rep, reps);
    if (reps.size() != 1 && reps.size() != pats.size())
        throw Error("regexprep: Multiple replace strings and patterns given "
                    "must have the same quantity.",
                     0, 0, "regexprep", "", "numkit:regexprep:repCount");

    // A cell string operand processes element-wise -> cell of char vectors,
    // same shape; a char/string scalar processes to a single char row (the
    // pre-existing scalar return type is preserved exactly).
    if (s.isCell()) {
        const keel::usize r = static_cast<keel::usize>(s.dims().rows());
        const keel::usize c = static_cast<keel::usize>(s.dims().cols());
        auto out = Value::cell(r, c, mr);
        const keel::usize n = s.numel();
        for (keel::usize i = 0; i < n; ++i)
            out.cellAt(i) = Value::fromString(
                regexrepSeq(s.cellAt(i).toString(), pats, reps, ignoreCase, once, salloc), mr);
        return out;
    }
    return Value::fromString(regexrepSeq(s.toString(), pats, reps, ignoreCase, once, salloc), mr);
}

// ── Pack 36: regexptranslate ─────────────────────────────────────────
Value regexptranslate(keel::StringRef op, keel::StringRef s, keel::Allocator *mr)
{
    // Characters that carry special meaning in ECMAScript regex syntax —
    // matches the set MATLAB's regexptranslate('escape', …) prepends `\`
    // to.
    auto isMeta = [](char c) {
        switch (c) {
        case '.': case '*': case '+': case '?': case '|':
        case '(': case ')': case '[': case ']': case '{': case '}':
        case '^': case '$': case '\\':
            return true;
        default:
            return false;
        }
    };

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::String out(scratch.allocator());
    out.reserve(s.size() * 2);

    if (op == "escape") {
        for (char c : s) {
            if (isMeta(c)) out.push_back('\\');
            out.push_back(c);
        }
        return Value::fromString(out, mr);
    }
    if (op == "wildcard") {
        // MATLAB rules: `*` → `.*`, `?` → `.`, every other regex meta is
        // escaped. Non-meta non-wildcard chars pass through.
        for (char c : s) {
            if (c == '*')      out.append(".*");
            else if (c == '?') out.push_back('.');
            else {
                if (isMeta(c)) out.push_back('\\');
                out.push_back(c);
            }
        }
        return Value::fromString(out, mr);
    }
    throw Error(keel::format("regexptranslate: unsupported op '{}' (supported: 'escape', 'wildcard')", op),
                 0, 0, "regexptranslate", "", "numkit:regexptranslate:badOp");
}

Value regexptranslate(keel::StringRef op, const Value &str, keel::Allocator *mr)
{
    return regexptranslate(op, str.toString(), mr);
}

} // namespace numkit::builtin
