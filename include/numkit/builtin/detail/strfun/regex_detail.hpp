// toolboxes/.../regex_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by regex.cpp + regex_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/array.h>
#include <keel/format.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/value/value.hpp>

#include <regex>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <utility>


namespace numkit::builtin {

namespace {

[[maybe_unused]] std::regex compileRegex(keel::StringRef pat, bool ignoreCase)
{
    auto flags = std::regex::ECMAScript;
    if (ignoreCase) flags |= std::regex::icase;
    try {
        return std::regex(pat.data(), pat.size(), flags);
    } catch (const std::regex_error &e) {
        throw Error(keel::format("regex: invalid pattern — {}", e.what()),
                     0, 0, "regexp", "", "numkit:regexp:badPattern");
    }
}

// Named-token support. std::regex (ECMAScript) does not accept MATLAB's
// `(?<name>...)` named groups, so rewrite each one to a plain capture
// group `(...)` and record name → 1-based capture-group index. Lookbehind
// `(?<=` / `(?<!`, lookahead `(?=` / `(?!`, and non-capturing `(?:` are
// left untouched (and don't consume a capture index). Escapes and
// character classes are honoured so `\(` and `[(]` are not miscounted.
struct NamedGroups {
    keel::String cleaned;
    keel::Array<std::pair<keel::String, keel::usize>> names;  // name → group #
    explicit NamedGroups(keel::Allocator *alloc = keel::default_allocator())
        : cleaned(alloc), names(alloc) {}
};

[[maybe_unused]] NamedGroups extractNamedGroups(keel::StringRef pat,
                               keel::Allocator *alloc = keel::default_allocator())
{
    NamedGroups ng(alloc);
    ng.cleaned.reserve(pat.size());
    keel::usize group = 0;
    bool inClass = false;
    for (keel::usize i = 0; i < pat.size();) {
        const char c = pat[i];
        if (c == '\\' && i + 1 < pat.size()) {           // escaped pair
            ng.cleaned.push_back(c);
            ng.cleaned.push_back(pat[i + 1]);
            i += 2;
            continue;
        }
        if (inClass) {
            if (c == ']') inClass = false;
            ng.cleaned.push_back(c);
            ++i;
            continue;
        }
        if (c == '[') { inClass = true; ng.cleaned.push_back(c); ++i; continue; }
        if (c == '(') {
            const bool named = i + 3 < pat.size() && pat[i + 1] == '?'
                               && pat[i + 2] == '<'
                               && (keel::is_ascii_alpha(pat[i + 3])
                                   || pat[i + 3] == '_');
            if (named) {
                keel::usize j = i + 3;
                keel::String name(alloc);
                while (j < pat.size() && pat[j] != '>') name.push_back(pat[j++]);
                ++group;
                ng.names.emplace_back(std::move(name), group);
                ng.cleaned.push_back('(');          // plain capture group
                i = (j < pat.size()) ? j + 1 : j;   // skip past '>'
                continue;
            }
            if (i + 1 < pat.size() && pat[i + 1] == '?') {   // (?: (?= (?! (?<= (?<!
                ng.cleaned.push_back(c);             // non-capturing / lookaround
                ++i;
                continue;
            }
            ++group;                         // plain capture group
            ng.cleaned.push_back(c);
            ++i;
            continue;
        }
        ng.cleaned.push_back(c);
        ++i;
    }
    return ng;
}

[[maybe_unused]] Value rowFromIndices(const keel::f64 *v, keel::usize n, keel::Allocator *mr)
{
    auto out = Value::matrix(1, n, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < n; ++i)
        out.doubleDataMut()[i] = v[i];
    return out;
}

[[maybe_unused]] Value rowCellOfStrings(const keel::StringRef *v, keel::usize n, keel::Allocator *mr)
{
    auto out = Value::cell(1, n);
    for (keel::usize i = 0; i < n; ++i)
        out.cellAt(i) = Value::fromString(v[i], mr);
    return out;
}

} // namespace

// regexp single-match worker (def in regex.cpp, external).
Value regexpFindOnce(const Value &s, const Value &pat, keel::StringRef option,
                     bool ignoreCase, keel::Allocator *mr);

} // namespace numkit::builtin
