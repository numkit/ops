// toolboxes/.../discrete_detail.hpp — private compute/register substrate (anon-in-
// header, internal linkage per TU) shared by discrete.cpp + discrete_reg.cpp.
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/reductions.hpp>  // engine-free numkit::builtin::detail dim-infra (ops re-export)

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/format.h>
#include <keel/hash_map.h>
#include <keel/hash_set.h>
#include <keel/math.h>
#include <keel/ref.h>

#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>

#include <numkit/builtin/datafun.hpp>


namespace numkit::builtin {

namespace {

// Phase P3 — hash-based set ops. The previous sort-based path was
// O(N log N) on every call; for the bench input (1M doubles drawn from
// a small integer range, K ≈ 8000 unique) it spent ~95% of its time
// sorting duplicates. Hash dedupe drops that to O(N) + O(K log K).

// Hash that normalises -0 → +0 so the two share a bucket.
struct DoubleHashEq0 {
    keel::u64 operator()(keel::f64 v) const noexcept {
        if (v == 0.0) return 0;          // covers both +0 and -0
        keel::u64 bits = keel::bit_cast<keel::u64>(v);
        bits ^= bits >> 33;
        bits *= 0xff51afd7ed558ccdULL;
        bits ^= bits >> 33;
        bits *= 0xc4ceb9fe1a85ec53ULL;
        bits ^= bits >> 33;
        return bits;
    }
};

struct IndexedVal {
    keel::f64 v;
    keel::usize origIdx;
};

inline Value emptyRow(keel::Allocator *alloc)
{
    return Value::matrix(1, 0, ValueType::DOUBLE, alloc);
}

inline Value rowFromVec(const keel::f64 *data, keel::usize n, keel::Allocator *alloc)
{
    auto r = Value::matrix(1, n, ValueType::DOUBLE, alloc);
    if (n > 0)
        std::copy(data, data + n, r.doubleDataMut());
    return r;
}

// ── complex unique ─────────────────────────────────────────────────
inline bool cxUniqLess(keel::c64 a, keel::c64 b)
{
    const keel::f64 am = keel::abs(a), bm = keel::abs(b);
    const bool an = keel::is_nan(am), bn = keel::is_nan(bm);
    if (an || bn) { if (an && bn) return false; return bn; }   // non-NaN < NaN
    if (am != bm) return am < bm;
    return keel::arg(a) < keel::arg(b);
}
inline bool cxUniqEqual(keel::c64 a, keel::c64 b)
{
    return a.real() == b.real() && a.imag() == b.imag();        // NaN never equal
}

[[maybe_unused]] std::tuple<Value, Value, Value>
uniqueComplexFull(const Value &x, bool stable, bool last = false,
                  keel::Allocator *alloc = nullptr)
{
    const keel::usize n = x.numel();
    const keel::c64 *p = x.complexData();
    keel::ArenaTemp scratch = keel::get_scratch(alloc);

    struct UC { keel::c64 v; keel::usize firstIdx; };
    keel::Array<UC> uniq(scratch.allocator());
    keel::Array<keel::usize> ic0(n, scratch.allocator());   // original -> 0-based pos in uniq
    for (keel::usize i = 0; i < n; ++i) {
        const keel::c64 z = p[i];
        const bool nanComp = keel::is_nan(z.real()) || keel::is_nan(z.imag());
        keel::usize found = keel::as<keel::usize>(-1);
        if (!nanComp)
            for (keel::usize k = 0; k < uniq.size(); ++k)
                if (cxUniqEqual(uniq[k].v, z)) { found = k; break; }
        if (found == keel::as<keel::usize>(-1)) { ic0[i] = uniq.size(); uniq.push_back({z, i}); }
        else { ic0[i] = found; if (last) uniq[found].firstIdx = i; }  // 'last': keep last occurrence
    }

    const keel::usize u = uniq.size();
    keel::Array<keel::usize> perm(u, scratch.allocator());
    for (keel::usize k = 0; k < u; ++k) perm[k] = k;
    if (!stable)
        std::stable_sort(perm.begin(), perm.end(),
                         [&](keel::usize a, keel::usize b) { return cxUniqLess(uniq[a].v, uniq[b].v); });
    keel::Array<keel::usize> newRank(u, scratch.allocator());
    for (keel::usize k = 0; k < u; ++k) newRank[perm[k]] = k;

    auto cOut  = Value::matrix(1, u, ValueType::COMPLEX, alloc);
    auto iaRow = Value::matrix(1, u, ValueType::DOUBLE, alloc);
    auto icRow = Value::matrix(1, n, ValueType::DOUBLE, alloc);
    keel::c64 *cd = cOut.complexDataMut();
    keel::f64 *ia = iaRow.doubleDataMut();
    keel::f64 *ici = icRow.doubleDataMut();
    for (keel::usize k = 0; k < u; ++k) {
        cd[k] = uniq[perm[k]].v;
        ia[k] = keel::as<keel::f64>(uniq[perm[k]].firstIdx + 1);
    }
    for (keel::usize i = 0; i < n; ++i)
        ici[i] = keel::as<keel::f64>(newRank[ic0[i]] + 1);
    return std::make_tuple(std::move(cOut), std::move(iaRow), std::move(icRow));
}

// ── 'rows' helpers ─────────────────────────────────────────────

// Hash-key type for unique('rows') / setops('rows'). Arena-backed so the
// keys themselves bump into the per-call scratch arena along with the
// keel::HashMap's flat tables.
using RowKey = keel::Array<keel::f64>;

struct RowKeyHash {
    keel::u64 operator()(const RowKey &k) const noexcept {
        keel::u64 h = 0xcbf29ce484222325ULL;  // FNV-ish seed
        for (keel::f64 v : k) {
            keel::u64 bits = (v == 0.0) ? 0ULL : keel::bit_cast<keel::u64>(v); // collapse +0 / -0
            bits ^= bits >> 33;
            bits *= 0xff51afd7ed558ccdULL;
            bits ^= bits >> 33;
            h ^= bits + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct RowKeyEq {
    bool operator()(const RowKey &a, const RowKey &b) const noexcept {
        if (a.size() != b.size()) return false;
        for (keel::usize i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }
};

inline bool rowHasNan(const keel::f64 *p, keel::usize cols, keel::usize rows, keel::usize r)
{
    for (keel::usize c = 0; c < cols; ++c)
        if (keel::is_nan(p[c * rows + r])) return true;
    return false;
}

inline RowKey extractRow(const keel::f64 *p, keel::usize cols, keel::usize rows, keel::usize r, keel::Allocator *alloc)
{
    RowKey k(cols, alloc);
    for (keel::usize c = 0; c < cols; ++c) {
        const keel::f64 v = p[c * rows + r];
        k[c] = (v == 0.0) ? 0.0 : v;
    }
    return k;
}

inline keel::i32 rowLexCmp(const keel::f64 *p, keel::usize cols, keel::usize rows, keel::usize a, keel::usize b)
{
    for (keel::usize c = 0; c < cols; ++c) {
        const keel::f64 av = p[c * rows + a];
        const keel::f64 bv = p[c * rows + b];
        if (av < bv) return -1;
        if (av > bv) return  1;
    }
    return 0;
}

inline Value emptyRowsResult(keel::usize cols, keel::Allocator *alloc)
{
    return Value::matrix(0, cols, ValueType::DOUBLE, alloc);
}

[[maybe_unused]] void validateUniqueRowsInput(const Value &x, const char *fn)
{
    if (x.type() != ValueType::DOUBLE)
        throw Error(keel::format("{}: 'rows' flag requires a DOUBLE matrix", fn),
                     0, 0, fn, "", keel::format("numkit:{}:rowsType", fn));
    if (x.dims().ndim() > 2)
        throw Error(keel::format("{}: 'rows' flag requires a 2D matrix", fn),
                     0, 0, fn, "", keel::format("numkit:{}:rowsND", fn));
}

[[maybe_unused]] keel::HashSet<keel::f64, DoubleHashEq0>
hashSetNoNaN(const Value &x, keel::Allocator *alloc)
{
    keel::HashSet<keel::f64, DoubleHashEq0> s(alloc);
    s.reserve(x.numel() / 2 + 1);
    const keel::f64 *p = x.doubleData();
    const keel::usize n = x.numel();
    for (keel::usize i = 0; i < n; ++i)
        if (!keel::is_nan(p[i])) s.insert(p[i]);
    return s;
}

[[maybe_unused]] void validateEdges(const Value &edges, const char *fn)
{
    if (edges.numel() < 2)
        throw Error(keel::format("{}: edges must have length >= 2", fn),
                     0, 0, fn, "", keel::format("numkit:{}:shortEdges", fn));
    const keel::f64 *e = edges.doubleData();
    for (keel::usize i = 1; i < edges.numel(); ++i)
        if (!(e[i] >= e[i - 1]))
            throw Error(keel::format("{}: edges must be ascending", fn),
                         0, 0, fn, "", keel::format("numkit:{}:badEdges", fn));
}

[[maybe_unused]] bool edgesAreUniform(const keel::f64 *e, keel::usize nEdges, keel::f64 &outStep)
{
    if (nEdges < 3) return false;
    const keel::f64 step = e[1] - e[0];
    if (!(step > 0)) return false;
    const keel::f64 tol = keel::abs(step) * 1e-12;
    for (keel::usize i = 2; i < nEdges; ++i) {
        const keel::f64 g = e[i] - e[i - 1];
        if (keel::abs(g - step) > tol) return false;
    }
    outStep = step;
    return true;
}

} // namespace (set ops helpers)
namespace {
inline keel::usize histcBin(keel::f64 v, const keel::f64 *e, keel::usize nE)
{
    if (keel::is_nan(v) || v < e[0] || v > e[nE - 1]) return keel::Limits<keel::usize>::kMax;
    if (v == e[nE - 1]) return nE - 1;
    keel::usize lo = 0, hi = nE;
    while (lo < hi) {
        const keel::usize mid = lo + (hi - lo) / 2;
        if (e[mid] <= v) lo = mid + 1; else hi = mid;
    }
    return lo - 1;            // e[lo-1] <= v < e[lo]
}
} // namespace
namespace {

bool isExactNonnegInt(keel::f64 v, keel::u64 &outU)
{
    if (!keel::is_finite(v) || v < 0) return false;
    if (v != keel::floor(v))         return false;
    if (v > keel::as<keel::f64>(keel::Limits<keel::u64>::kMax))
        return false;
    outU = keel::as<keel::u64>(v);
    return true;
}

bool isPrimeU64(keel::u64 n)
{
    if (n < 2)                return false;
    if (n == 2 || n == 3)     return true;
    if (n % 2 == 0)           return false;
    if (n % 3 == 0)           return false;
    // 6k ± 1 trial division.
    for (keel::u64 i = 5; i * i <= n; i += 6) {
        if (n % i == 0)        return false;
        if (n % (i + 2) == 0)  return false;
    }
    return true;
}

[[maybe_unused]] bool isPrimeDouble(keel::f64 v)
{
    keel::u64 u;
    if (!isExactNonnegInt(v, u)) return false;
    return isPrimeU64(u);
}

} // namespace (number-theory helpers)
namespace {

// 11! = 39 916 800 rows of doubles ≈ 3.5 GB if we let n=12 through.
// Capping at n=11 is exactly MATLAB's documented hard limit on perms.
constexpr keel::i32 kPermMaxN = 11;

[[maybe_unused]] keel::u64 permFactorial(keel::i32 n)
{
    keel::u64 f = 1;
    for (keel::i32 i = 2; i <= n; ++i) f *= keel::as<keel::u64>(i);
    return f;
}

[[maybe_unused]] keel::f64 factorialDouble(keel::f64 v, const char *fn)
{
    if (!keel::is_finite(v) || v < 0 || v != keel::floor(v))
        throw Error(keel::format("{}: entries must be non-negative integers", fn),
                     0, 0, fn, "", keel::format("numkit:{}:badArg", fn));
    if (v > 170.0)
        return keel::Limits<keel::f64>::infinity();
    keel::f64 r = 1.0;
    const keel::i32 n = keel::as<keel::i32>(v);
    for (keel::i32 i = 2; i <= n; ++i) r *= keel::as<keel::f64>(i);
    return r;
}

} // namespace (combinatorics helpers)
namespace {
inline bool nearlyEqualTol(keel::f64 x, keel::f64 y, keel::f64 tol)
{
    if (x == y) return true;
    if (keel::is_nan(x) || keel::is_nan(y)) return false;
    const keel::f64 s = std::max(keel::abs(x), keel::abs(y));
    return keel::abs(x - y) <= tol * std::max(1.0, s);
}
} // anon

// file-scope workers used by the reg (defs in discrete.cpp, external).
std::pair<Value, Value> ismemberComplex(const Value &a, const Value &b, bool wantLoc, keel::Allocator *mr);
Value nchoosekCombinations(const Value &v, keel::f64 kd, keel::Allocator *mr);

} // namespace numkit::builtin
