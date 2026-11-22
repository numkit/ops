// src/builtin/src/datafun/group.cpp
//
// Pure C++ group-based data operations:
//   findgroups     — assign 1-based group IDs to each element
//   groupcounts    — count elements per group
//   groupsummary   — per-group, per-column reduction
//   grouptransform — per-group data transformation

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/format.h>
#include <numkit/builtin/datafun.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <algorithm>

namespace numkit::builtin {

namespace {

// Build a sorted-unique value list with a parallel "group id per
// element" vector (1-based). Operates on numeric input via
// elemAsDouble (CHAR / LOGICAL fall through naturally).
//
// out_groups[i] is the group id of g[i]; out_unique is the sorted
// unique value list (length = max(out_groups)). NaN entries get
// out_groups[i] = 0 (sentinel for "missing"). The caller decides
// what to do with them (findgroups -> NaN; groupcounts -> trailing
// NaN bucket).
void groupOf(const Value &g, keel::Array<keel::usize> &out_groups,
             keel::Array<keel::f64> &out_unique, keel::Allocator *alloc)
{
    const keel::usize n = g.numel();
    out_groups.assign(n, 0);
    if (n == 0) { out_unique.clear(); return; }

    keel::Array<keel::f64> vals(n, alloc);
    for (keel::usize i = 0; i < n; ++i) vals[i] = g.elemAsDouble(i);

    // Build sorted-unique list of finite values (skip NaN — handled
    // separately by callers).
    keel::Array<keel::f64> sorted(alloc);
    sorted.reserve(n);
    for (keel::f64 v : vals)
        if (!keel::is_nan(v)) sorted.push_back(v);
    std::sort(sorted.begin(), sorted.end());
    auto newEnd = std::unique(sorted.begin(), sorted.end(),
                              [](keel::f64 a, keel::f64 b) {
                                  return a == b;
                              });
    sorted.resize(newEnd - sorted.begin());
    for (keel::usize i = 0; i < n; ++i) {
        if (keel::is_nan(vals[i])) { out_groups[i] = 0; continue; }
        const auto it = std::lower_bound(sorted.begin(), sorted.end(), vals[i]);
        out_groups[i] = (keel::usize)(it - sorted.begin()) + 1;   // 1-based
    }
    out_unique = std::move(sorted);
}

} // namespace

// ── Public C++ API ──────────────────────────────────────────────────────────

FindgroupsResult findgroups(const Value &g, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::usize> groups(scratch.allocator());
    keel::Array<keel::f64> uniqueVals(scratch.allocator());
    groupOf(g, groups, uniqueVals, scratch.allocator());
    auto G = Value::matrix(g.dims().rows(), g.dims().cols(),
                           ValueType::DOUBLE, mr);
    keel::f64 *gd = G.doubleDataMut();
    const keel::f64 nan = keel::Limits<keel::f64>::quiet_nan();
    for (keel::usize i = 0; i < groups.size(); ++i)
        gd[i] = (groups[i] == 0) ? nan : keel::f64(groups[i]);
    auto ID = Value::matrix(uniqueVals.size(), 1, ValueType::DOUBLE, mr);
    if (!uniqueVals.empty())
        keel::copy_n(ID.doubleDataMut(), uniqueVals.data(), uniqueVals.size());
    return {std::move(G), std::move(ID)};
}

GroupcountsResult groupcounts(const Value &g, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::usize> groups(scratch.allocator());
    keel::Array<keel::f64> uniqueVals(scratch.allocator());
    groupOf(g, groups, uniqueVals, scratch.allocator());
    keel::usize nan_count = 0;
    for (auto gg : groups) if (gg == 0) ++nan_count;
    const bool have_nan = nan_count > 0;
    const keel::usize nGroups = uniqueVals.size() + (have_nan ? 1 : 0);
    GroupcountsResult R;
    if (nGroups == 0) {
        R.C  = Value::matrix(0, 1, ValueType::DOUBLE, mr);
        R.GR = Value::matrix(0, 1, ValueType::DOUBLE, mr);
        R.P  = Value::matrix(0, 1, ValueType::DOUBLE, mr);
        return R;
    }
    keel::Array<keel::usize> counts(nGroups, keel::as<keel::usize>(0), scratch.allocator());
    for (auto gg : groups) {
        if (gg == 0) counts[uniqueVals.size()]++;  // trailing NaN bucket
        else         counts[gg - 1]++;
    }
    R.C = Value::matrix(nGroups, 1, ValueType::DOUBLE, mr);
    {
        keel::f64 *dst = R.C.doubleDataMut();
        for (keel::usize i = 0; i < nGroups; ++i) dst[i] = keel::f64(counts[i]);
    }
    R.GR = Value::matrix(nGroups, 1, ValueType::DOUBLE, mr);
    {
        keel::f64 *gd = R.GR.doubleDataMut();
        for (keel::usize i = 0; i < uniqueVals.size(); ++i) gd[i] = uniqueVals[i];
        if (have_nan)
            gd[uniqueVals.size()] = keel::Limits<keel::f64>::quiet_nan();
    }
    const keel::f64 total = keel::f64(groups.size());
    R.P = Value::matrix(nGroups, 1, ValueType::DOUBLE, mr);
    {
        keel::f64 *pd = R.P.doubleDataMut();
        for (keel::usize i = 0; i < nGroups; ++i)
            pd[i] = (total > 0.0) ? 100.0 * keel::f64(counts[i]) / total : 0.0;
    }
    return R;
}

GroupsummaryResult groupsummary(const Value &A, const Value &G,
                                keel::StringRef method,
                                keel::Allocator *mr)
{
    const keel::usize nRows = A.dims().rows();
    const keel::usize nCols = (A.dims().ndim() >= 2) ? A.dims().cols() : 1;
    if (G.numel() != nRows)
        throw Error("groupsummary: groupvars must have length size(A, 1)",
                    0, 0, "groupsummary", "", "numkit:groupsummary:shape");

    keel::ArenaTemp scratch = keel::get_scratch(mr);

    // Group the rows.
    keel::Array<keel::usize> groups(scratch.allocator());
    keel::Array<keel::f64> uniqueVals(scratch.allocator());
    groupOf(G, groups, uniqueVals, scratch.allocator());
    keel::usize nan_count = 0;
    for (auto g : groups) if (g == 0) ++nan_count;
    const bool have_nan = nan_count > 0;
    const keel::usize nGroups = uniqueVals.size() + (have_nan ? 1 : 0);

    // Bucket row indices by group ID (0 = NaN bucket -> trailing).
    keel::Array<keel::Array<keel::usize>> buckets(scratch.allocator());
    buckets.reserve(nGroups);
    for (keel::usize i = 0; i < nGroups; ++i)
        buckets.push_back(keel::Array<keel::usize>(scratch.allocator()));
    for (keel::usize i = 0; i < groups.size(); ++i) {
        const keel::usize gi = (groups[i] == 0) ? uniqueVals.size()
                                                : (groups[i] - 1);
        buckets[gi].push_back(i);
    }

    auto B = (nCols == 1) ? Value::matrix(nGroups, 1, ValueType::DOUBLE, mr)
                          : Value::matrix(nGroups, nCols, ValueType::DOUBLE, mr);
    keel::f64 *bd = B.doubleDataMut();
    auto Aget = [&](keel::usize r, keel::usize c) {
        return A.elemAsDouble(r + c * nRows);
    };

    for (keel::usize c = 0; c < nCols; ++c) {
        for (keel::usize g = 0; g < nGroups; ++g) {
            const auto &rows = buckets[g];
            const keel::usize kn = rows.size();
            keel::f64 out = 0.0;
            if (method == "sum") {
                for (auto r : rows) out += Aget(r, c);
            } else if (method == "mean") {
                if (kn == 0) out = keel::Limits<keel::f64>::quiet_nan();
                else {
                    keel::f64 s = 0.0;
                    for (auto r : rows) s += Aget(r, c);
                    out = s / keel::f64(kn);
                }
            } else if (method == "median") {
                keel::Array<keel::f64> v(scratch.allocator()); v.reserve(kn);
                for (auto r : rows) v.push_back(Aget(r, c));
                std::sort(v.begin(), v.end());
                out = (kn == 0)            ? keel::Limits<keel::f64>::quiet_nan()
                    : (kn % 2 == 1)        ? v[kn / 2]
                                            : 0.5 * (v[kn / 2 - 1] + v[kn / 2]);
            } else if (method == "max") {
                out = -keel::Limits<keel::f64>::infinity();
                for (auto r : rows) {
                    const keel::f64 v = Aget(r, c);
                    if (v > out) out = v;
                }
                if (kn == 0) out = keel::Limits<keel::f64>::quiet_nan();
            } else if (method == "min") {
                out = keel::Limits<keel::f64>::infinity();
                for (auto r : rows) {
                    const keel::f64 v = Aget(r, c);
                    if (v < out) out = v;
                }
                if (kn == 0) out = keel::Limits<keel::f64>::quiet_nan();
            } else if (method == "std" || method == "var") {
                if (kn < 2) out = keel::Limits<keel::f64>::quiet_nan();
                else {
                    keel::f64 s = 0.0, ss = 0.0;
                    for (auto r : rows) { keel::f64 v = Aget(r, c); s += v; ss += v*v; }
                    const keel::f64 m = s / keel::f64(kn);
                    const keel::f64 v = (ss - keel::f64(kn) * m * m) / keel::f64(kn - 1);
                    out = (method == "var") ? std::max(0.0, v)
                                             : keel::sqrt(std::max(0.0, v));
                }
            } else if (method == "numunique") {
                keel::Array<keel::f64> v(scratch.allocator()); v.reserve(kn);
                for (auto r : rows) v.push_back(Aget(r, c));
                std::sort(v.begin(), v.end());
                auto newEnd = std::unique(v.begin(), v.end());
                out = keel::f64(newEnd - v.begin());
            } else if (method == "nnz") {
                keel::usize k = 0;
                for (auto r : rows) if (Aget(r, c) != 0.0) ++k;
                out = keel::f64(k);
            } else if (method == "mode") {
                keel::Array<keel::f64> v(scratch.allocator()); v.reserve(kn);
                for (auto r : rows) v.push_back(Aget(r, c));
                std::sort(v.begin(), v.end());
                keel::usize best_run = 0, cur_run = 0;
                keel::f64 best_val = keel::Limits<keel::f64>::quiet_nan();
                for (keel::usize i = 0; i < v.size(); ++i) {
                    if (i == 0 || v[i] != v[i - 1]) cur_run = 1;
                    else                            ++cur_run;
                    if (cur_run > best_run) { best_run = cur_run; best_val = v[i]; }
                }
                out = best_val;
            } else if (method == "all") {
                bool ok = true;
                for (auto r : rows) if (Aget(r, c) == 0.0) { ok = false; break; }
                out = ok ? 1.0 : 0.0;
            } else if (method == "any") {
                bool ok = false;
                for (auto r : rows) if (Aget(r, c) != 0.0) { ok = true; break; }
                out = ok ? 1.0 : 0.0;
            } else {
                throw Error(keel::format("groupsummary: method '{}' not supported in this revision", method),
                            0, 0, "groupsummary", "",
                            "numkit:groupsummary:badMethod");
            }
            bd[g + c * nGroups] = out;
        }
    }

    GroupsummaryResult R;
    R.B = std::move(B);
    R.BG = Value::matrix(nGroups, 1, ValueType::DOUBLE, mr);
    {
        keel::f64 *gd = R.BG.doubleDataMut();
        for (keel::usize i = 0; i < uniqueVals.size(); ++i) gd[i] = uniqueVals[i];
        if (have_nan)
            gd[uniqueVals.size()] = keel::Limits<keel::f64>::quiet_nan();
    }
    R.BC = Value::matrix(nGroups, 1, ValueType::DOUBLE, mr);
    {
        keel::f64 *cd = R.BC.doubleDataMut();
        for (keel::usize g = 0; g < nGroups; ++g) cd[g] = keel::f64(buckets[g].size());
    }
    return R;
}

Value grouptransform(const Value &A, const Value &G,
                     keel::StringRef method,
                     keel::Allocator *mr)
{
    const keel::usize nRows = A.dims().rows();
    const keel::usize nCols = (A.dims().ndim() >= 2) ? A.dims().cols() : 1;
    if (G.numel() != nRows)
        throw Error("grouptransform: groupvars must have length size(A,1)",
                    0, 0, "grouptransform", "",
                    "numkit:grouptransform:shape");

    if (method != "zscore" && method != "norm" &&
        method != "meancenter" && method != "rescale" &&
        method != "meanfill" && method != "linearfill")
    {
        throw Error("grouptransform: method must be 'zscore', 'norm', "
                    "'meancenter', 'rescale', 'meanfill', or 'linearfill'",
                    0, 0, "grouptransform", "",
                    "numkit:grouptransform:badMethod");
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::usize> groups(scratch.allocator());
    keel::Array<keel::f64> uniqueVals(scratch.allocator());
    groupOf(G, groups, uniqueVals, scratch.allocator());
    keel::usize nan_count = 0;
    for (auto g : groups) if (g == 0) ++nan_count;
    const bool have_nan = nan_count > 0;
    const keel::usize nGroups = uniqueVals.size() + (have_nan ? 1 : 0);
    keel::Array<keel::Array<keel::usize>> buckets(scratch.allocator());
    buckets.reserve(nGroups);
    for (keel::usize i = 0; i < nGroups; ++i)
        buckets.push_back(keel::Array<keel::usize>(scratch.allocator()));
    for (keel::usize i = 0; i < groups.size(); ++i) {
        const keel::usize gi = (groups[i] == 0) ? uniqueVals.size()
                                                : (groups[i] - 1);
        buckets[gi].push_back(i);
    }

    auto B = (nCols == 1) ? Value::matrix(nRows, 1, ValueType::DOUBLE, mr)
                          : Value::matrix(nRows, nCols, ValueType::DOUBLE, mr);
    keel::f64 *bd = B.doubleDataMut();
    for (keel::usize i = 0; i < nRows * nCols; ++i)
        bd[i] = keel::Limits<keel::f64>::quiet_nan();

    auto Aget = [&](keel::usize r, keel::usize c) {
        return A.elemAsDouble(r + c * nRows);
    };

    for (keel::usize c = 0; c < nCols; ++c) {
        for (const auto &rows : buckets) {
            const keel::usize kn = rows.size();
            if (kn == 0) continue;

            if (method == "meancenter" || method == "zscore" ||
                method == "norm") {
                keel::f64 sum = 0.0, sq = 0.0;
                keel::usize k = 0;
                for (auto r : rows) {
                    const keel::f64 v = Aget(r, c);
                    if (!keel::is_nan(v)) { sum += v; sq += v*v; ++k; }
                }
                const keel::f64 mean = (k > 0) ? sum / keel::f64(k) : 0.0;
                if (method == "meancenter") {
                    for (auto r : rows)
                        bd[r + c * nRows] = Aget(r, c) - mean;
                } else if (method == "zscore") {
                    const keel::f64 var = (k > 1)
                        ? (sq - keel::f64(k) * mean * mean) / keel::f64(k - 1)
                        : 0.0;
                    const keel::f64 sd  = keel::sqrt(std::max(0.0, var));
                    for (auto r : rows)
                        bd[r + c * nRows] = (sd > 0.0)
                            ? (Aget(r, c) - mean) / sd
                            : 0.0;
                } else {  // norm
                    const keel::f64 nm = keel::sqrt(sq);
                    for (auto r : rows)
                        bd[r + c * nRows] = (nm > 0.0)
                            ? Aget(r, c) / nm : 0.0;
                }
            } else if (method == "rescale") {
                keel::f64 lo = keel::Limits<keel::f64>::infinity();
                keel::f64 hi = -keel::Limits<keel::f64>::infinity();
                for (auto r : rows) {
                    const keel::f64 v = Aget(r, c);
                    if (keel::is_nan(v)) continue;
                    if (v < lo) lo = v;
                    if (v > hi) hi = v;
                }
                const keel::f64 range = hi - lo;
                for (auto r : rows) {
                    const keel::f64 v = Aget(r, c);
                    if (keel::is_nan(v))            bd[r + c * nRows] = v;
                    else if (range == 0.0)         bd[r + c * nRows] = 0.0;
                    else                            bd[r + c * nRows] = (v - lo) / range;
                }
            } else if (method == "meanfill") {
                keel::f64 sum = 0.0;
                keel::usize k = 0;
                for (auto r : rows) {
                    const keel::f64 v = Aget(r, c);
                    if (!keel::is_nan(v)) { sum += v; ++k; }
                }
                const keel::f64 mean = (k > 0) ? sum / keel::f64(k)
                                            : keel::Limits<keel::f64>::quiet_nan();
                for (auto r : rows) {
                    const keel::f64 v = Aget(r, c);
                    bd[r + c * nRows] = keel::is_nan(v) ? mean : v;
                }
            } else if (method == "linearfill") {
                keel::Array<keel::f64> vals(kn, scratch.allocator());
                for (keel::usize j = 0; j < kn; ++j) vals[j] = Aget(rows[j], c);
                keel::Array<keel::usize> good(scratch.allocator());
                good.reserve(kn);
                for (keel::usize j = 0; j < kn; ++j)
                    if (!keel::is_nan(vals[j])) good.push_back(j);
                keel::Array<keel::f64> out(keel::ArrayRef<keel::f64>(vals), scratch.allocator());
                if (good.size() >= 2) {
                    for (keel::usize k = 0; k + 1 < good.size(); ++k) {
                        const keel::usize a = good[k];
                        const keel::usize b = good[k + 1];
                        if (b == a + 1) continue;
                        const keel::f64 slope = (vals[b] - vals[a]) / keel::f64(b - a);
                        for (keel::usize j = a + 1; j < b; ++j)
                            out[j] = vals[a] + slope * keel::f64(j - a);
                    }
                    if (good[0] > 0) {
                        const keel::usize a = good[0];
                        const keel::usize b = good[1];
                        const keel::f64 slope = (vals[b] - vals[a]) / keel::f64(b - a);
                        for (keel::usize j = 0; j < a; ++j)
                            out[j] = vals[a] - slope * keel::f64(a - j);
                    }
                    if (good[good.size() - 1] + 1 < kn) {
                        const keel::usize a = good[good.size() - 2];
                        const keel::usize b = good[good.size() - 1];
                        const keel::f64 slope = (vals[b] - vals[a]) / keel::f64(b - a);
                        for (keel::usize j = b + 1; j < kn; ++j)
                            out[j] = vals[b] + slope * keel::f64(j - b);
                    }
                }
                for (keel::usize j = 0; j < kn; ++j)
                    bd[rows[j] + c * nRows] = out[j];
            }
        }
    }
    return B;
}

} // namespace numkit::builtin
