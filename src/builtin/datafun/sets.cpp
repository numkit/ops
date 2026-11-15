// src/builtin/src/datafun/sets.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/builtin/datafun.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/helpers.hpp>
#include <numkit/ops/rows_helpers.hpp>

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/hash_map.h>
#include <keel/hash_set.h>
#include <keel/math.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "discrete_detail.hpp"

namespace numkit::builtin {


// ── unique ─────────────────────────────────────────────────────────

Value unique(const Value &x, bool stable, keel::Allocator *alloc)
{
    const keel::usize n = x.numel();
    if (n == 0) return emptyRow(alloc);
    if (x.type() == ValueType::COMPLEX)
        return std::get<0>(uniqueComplexFull(x, stable, false, alloc));

    keel::ArenaTemp scratch = keel::get_scratch(alloc);
    const keel::f64 *p = x.doubleData();

    if (stable) {
        // First-occurrence order, no sort. Each NaN is distinct (kept).
        keel::HashSet<keel::f64, DoubleHashEq0> seen(scratch.allocator());
        seen.reserve(n / 2 + 1);
        keel::Array<keel::f64> out(scratch.allocator());
        out.reserve(n);
        for (keel::usize i = 0; i < n; ++i) {
            if (keel::is_nan(p[i])) out.push_back(keel::Limits<keel::f64>::quiet_nan());
            else if (seen.insert(p[i])) out.push_back(p[i]);
        }
        return rowFromVec(out.data(), out.size(), alloc);
    }

    keel::HashSet<keel::f64, DoubleHashEq0> seen(scratch.allocator());
    seen.reserve(n / 2 + 1);
    keel::usize nanCount = 0;
    for (keel::usize i = 0; i < n; ++i) {
        if (keel::is_nan(p[i])) ++nanCount;
        else seen.insert(p[i]);
    }

    keel::Array<keel::f64> out(scratch.allocator());
    out.reserve(seen.size() + nanCount);
    for (keel::f64 v : seen)
        out.push_back(v);
    std::sort(out.begin(), out.end());
    for (keel::usize i = 0; i < nanCount; ++i)
        out.push_back(keel::Limits<keel::f64>::quiet_nan());
    return rowFromVec(out.data(), out.size(), alloc);
}

std::tuple<Value, Value, Value>
uniqueWithIndices(const Value &x, bool stable, bool last,
                  keel::Allocator *alloc)
{
    const keel::usize n = x.numel();
    if (n == 0) {
        return std::make_tuple(emptyRow(alloc), emptyRow(alloc),
                               emptyRow(alloc));
    }
    if (x.type() == ValueType::COMPLEX)
        return uniqueComplexFull(x, stable, last, alloc);

    if (stable) {
        keel::ArenaTemp scratch = keel::get_scratch(alloc);
        keel::HashMap<keel::f64, keel::usize, DoubleHashEq0> posByVal(scratch.allocator());
        posByVal.reserve(n / 2 + 1);
        keel::Array<keel::f64> uVals(scratch.allocator());
        keel::Array<keel::f64> iaVec(scratch.allocator());
        keel::Array<keel::f64> ic(n, scratch.allocator());
        const keel::f64 *p = x.doubleData();
        if (last) {
            keel::HashMap<keel::f64, keel::usize, DoubleHashEq0> lastIdx(scratch.allocator());
            lastIdx.reserve(n / 2 + 1);
            for (keel::usize i = 0; i < n; ++i)
                if (!keel::is_nan(p[i]))
                    lastIdx.put(p[i], i);
            keel::HashSet<keel::f64, DoubleHashEq0> seen(scratch.allocator());
            seen.reserve(lastIdx.size() + n / 2);
            for (keel::usize i = 0; i < n; ++i) {
                if (keel::is_nan(p[i])) {
                    uVals.push_back(keel::Limits<keel::f64>::quiet_nan());
                    iaVec.push_back(static_cast<keel::f64>(i + 1));
                    ic[i] = static_cast<keel::f64>(uVals.size());
                    continue;
                }
                if (const keel::usize *li = lastIdx.get(p[i]); li && *li == i) {
                    if (seen.insert(p[i])) {
                        posByVal.put(p[i], uVals.size());
                        uVals.push_back(p[i]);
                        iaVec.push_back(static_cast<keel::f64>(i + 1));
                    }
                }
            }
            for (keel::usize i = 0; i < n; ++i) {
                if (keel::is_nan(p[i])) continue;
                if (const keel::usize *pv = posByVal.get(p[i]))
                    ic[i] = static_cast<keel::f64>(*pv + 1);
            }
        } else {
            // 'stable' (default): first-occurrence order.
            for (keel::usize i = 0; i < n; ++i) {
                if (keel::is_nan(p[i])) {
                    uVals.push_back(keel::Limits<keel::f64>::quiet_nan());
                    iaVec.push_back(static_cast<keel::f64>(i + 1));
                    ic[i] = static_cast<keel::f64>(uVals.size());
                } else {
                    auto res = posByVal.try_insert(p[i], uVals.size());
                    if (res.inserted) {
                        uVals.push_back(p[i]);
                        iaVec.push_back(static_cast<keel::f64>(i + 1));
                        ic[i] = static_cast<keel::f64>(uVals.size());
                    } else {
                        ic[i] = static_cast<keel::f64>(*res.value + 1);
                    }
                }
            }
        }
        auto cOut  = Value::matrix(1, uVals.size(), ValueType::DOUBLE, alloc);
        auto iaRow = Value::matrix(1, iaVec.size(), ValueType::DOUBLE, alloc);
        std::copy(uVals.begin(), uVals.end(), cOut.doubleDataMut());
        std::copy(iaVec.begin(), iaVec.end(), iaRow.doubleDataMut());
        auto icRow = Value::matrix(1, n, ValueType::DOUBLE, alloc);
        std::copy(ic.begin(), ic.end(), icRow.doubleDataMut());
        return std::make_tuple(std::move(cOut), std::move(iaRow), std::move(icRow));
    }

    keel::ArenaTemp scratch = keel::get_scratch(alloc);
    keel::HashMap<keel::f64, keel::usize, DoubleHashEq0> firstIdx(scratch.allocator());
    firstIdx.reserve(n / 2 + 1);
    keel::Array<keel::usize> nanIdxOrder(scratch.allocator());
    const keel::f64 *p = x.doubleData();
    for (keel::usize i = 0; i < n; ++i) {
        if (keel::is_nan(p[i])) {
            nanIdxOrder.push_back(i);
        } else if (last) {
            firstIdx.put(p[i], i);         // 'last': keep the last occurrence
        } else {
            firstIdx.try_insert(p[i], i);  // default: keep the first occurrence
        }
    }

    keel::Array<IndexedVal> sorted(scratch.allocator());
    sorted.reserve(firstIdx.size() + nanIdxOrder.size());
    for (const auto &entry : firstIdx)
        sorted.push_back({entry.key, entry.value});
    std::sort(sorted.begin(), sorted.end(),
              [](const IndexedVal &a, const IndexedVal &b) {
                  return a.v < b.v;
              });
    for (keel::usize idx : nanIdxOrder)
        sorted.push_back({keel::Limits<keel::f64>::quiet_nan(), idx});

    keel::HashMap<keel::f64, keel::usize, DoubleHashEq0> rankByValue(scratch.allocator());
    rankByValue.reserve(firstIdx.size());
    const keel::usize nanRankBase = sorted.size() - nanIdxOrder.size();
    for (keel::usize r = 0; r < nanRankBase; ++r)
        rankByValue.put(sorted[r].v, r);

    keel::Array<keel::f64> ic(n, scratch.allocator());
    keel::usize nanSeen = 0;
    for (keel::usize i = 0; i < n; ++i) {
        if (keel::is_nan(p[i])) {
            ic[i] = static_cast<keel::f64>(nanRankBase + nanSeen + 1);
            ++nanSeen;
        } else {
            ic[i] = static_cast<keel::f64>(*rankByValue.get(p[i]) + 1);
        }
    }

    auto cOut = Value::matrix(1, sorted.size(), ValueType::DOUBLE, alloc);
    auto iaRow = Value::matrix(1, sorted.size(), ValueType::DOUBLE, alloc);
    for (keel::usize i = 0; i < sorted.size(); ++i) {
        cOut.doubleDataMut()[i]  = sorted[i].v;
        iaRow.doubleDataMut()[i] = static_cast<keel::f64>(sorted[i].origIdx + 1);
    }
    auto icRow = Value::matrix(1, n, ValueType::DOUBLE, alloc);
    std::copy(ic.begin(), ic.end(), icRow.doubleDataMut());

    return std::make_tuple(std::move(cOut), std::move(iaRow), std::move(icRow));
}

// ── unique with 'rows' flag ────────────────────────────────────────

Value uniqueRows(const Value &x, bool stable, keel::Allocator *alloc)
{
    validateUniqueRowsInput(x, "unique");
    const keel::usize rows = x.dims().rows();
    const keel::usize cols = x.dims().cols();
    if (rows == 0) return emptyRowsResult(cols, alloc);

    const keel::f64 *src = x.doubleData();
    keel::ArenaTemp scratch = keel::get_scratch(alloc);

    if (stable) {
        keel::HashMap<RowKey, keel::usize, RowKeyHash, RowKeyEq> seen(scratch.allocator());
        seen.reserve(rows);
        keel::Array<keel::usize> uniqRows(scratch.allocator());
        for (keel::usize r = 0; r < rows; ++r) {
            if (rowHasNan(src, cols, rows, r)) {
                uniqRows.push_back(r);
            } else if (seen.try_insert(extractRow(src, cols, rows, r, scratch.allocator()), r).inserted) {
                uniqRows.push_back(r);
            }
        }
        return numkit::ops::collectRowsByIndex(x, uniqRows.data(), uniqRows.size(), alloc);
    }

    keel::HashMap<RowKey, keel::usize, RowKeyHash, RowKeyEq> firstIdx(scratch.allocator());
    firstIdx.reserve(rows);
    keel::Array<keel::usize> nanRows(scratch.allocator());
    for (keel::usize r = 0; r < rows; ++r) {
        if (rowHasNan(src, cols, rows, r)) {
            nanRows.push_back(r);
        } else {
            firstIdx.try_insert(extractRow(src, cols, rows, r, scratch.allocator()), r);
        }
    }

    keel::Array<keel::usize> uniqRows(scratch.allocator());
    uniqRows.reserve(firstIdx.size() + nanRows.size());
    for (const auto &entry : firstIdx) uniqRows.push_back(entry.value);
    std::sort(uniqRows.begin(), uniqRows.end(),
              [src, cols, rows](keel::usize a, keel::usize b) {
                  return rowLexCmp(src, cols, rows, a, b) < 0;
              });
    uniqRows.append(nanRows);

    return numkit::ops::collectRowsByIndex(x, uniqRows.data(), uniqRows.size(), alloc);
}

std::tuple<Value, Value, Value>
uniqueRowsWithIndices(const Value &x, bool stable, bool last,
                      keel::Allocator *alloc)
{
    validateUniqueRowsInput(x, "unique");
    const keel::usize rows = x.dims().rows();
    const keel::usize cols = x.dims().cols();
    if (rows == 0) {
        return std::make_tuple(emptyRowsResult(cols, alloc),
                               emptyRow(alloc), emptyRow(alloc));
    }

    const keel::f64 *src = x.doubleData();
    keel::ArenaTemp scratch = keel::get_scratch(alloc);

    if (stable) {
        keel::HashMap<RowKey, keel::usize, RowKeyHash, RowKeyEq> posByKey(scratch.allocator());
        keel::HashMap<keel::usize, keel::usize> posByNanRow(scratch.allocator());
        posByKey.reserve(rows);
        keel::Array<keel::usize> uniqRows(scratch.allocator());
        for (keel::usize r = 0; r < rows; ++r) {
            if (rowHasNan(src, cols, rows, r)) {
                posByNanRow.put(r, uniqRows.size());
                uniqRows.push_back(r);
            } else if (posByKey.try_insert(extractRow(src, cols, rows, r, scratch.allocator()),
                                           uniqRows.size()).inserted) {
                uniqRows.push_back(r);
            }
        }

        auto icRow = Value::matrix(rows, 1, ValueType::DOUBLE, alloc);
        keel::f64 *ic = icRow.doubleDataMut();
        for (keel::usize r = 0; r < rows; ++r) {
            const keel::usize pos = rowHasNan(src, cols, rows, r)
                ? *posByNanRow.get(r)
                : *posByKey.get(extractRow(src, cols, rows, r, scratch.allocator()));
            ic[r] = static_cast<keel::f64>(pos + 1);
        }

        auto iaCol = Value::matrix(uniqRows.size(), 1, ValueType::DOUBLE, alloc);
        keel::f64 *ia = iaCol.doubleDataMut();
        for (keel::usize i = 0; i < uniqRows.size(); ++i)
            ia[i] = static_cast<keel::f64>(uniqRows[i] + 1);

        return std::make_tuple(
            numkit::ops::collectRowsByIndex(x, uniqRows.data(), uniqRows.size(), alloc),
            std::move(iaCol), std::move(icRow));
    }

    keel::HashMap<RowKey, keel::usize, RowKeyHash, RowKeyEq> firstIdx(scratch.allocator());
    firstIdx.reserve(rows);
    keel::Array<keel::usize> nanRowOrder(scratch.allocator());
    for (keel::usize r = 0; r < rows; ++r) {
        if (rowHasNan(src, cols, rows, r)) {
            nanRowOrder.push_back(r);
        } else if (last) {
            firstIdx.put(extractRow(src, cols, rows, r, scratch.allocator()), r);
        } else {
            firstIdx.try_insert(extractRow(src, cols, rows, r, scratch.allocator()), r);
        }
    }

    keel::Array<keel::usize> uniqRows(scratch.allocator());
    uniqRows.reserve(firstIdx.size() + nanRowOrder.size());
    for (const auto &entry : firstIdx) uniqRows.push_back(entry.value);
    std::sort(uniqRows.begin(), uniqRows.end(),
              [src, cols, rows](keel::usize a, keel::usize b) {
                  return rowLexCmp(src, cols, rows, a, b) < 0;
              });
    const keel::usize nanRankBase = uniqRows.size();
    uniqRows.append(nanRowOrder);

    keel::HashMap<RowKey, keel::usize, RowKeyHash, RowKeyEq> rankByKey(scratch.allocator());
    rankByKey.reserve(nanRankBase);
    for (keel::usize r = 0; r < nanRankBase; ++r)
        rankByKey.put(extractRow(src, cols, rows, uniqRows[r], scratch.allocator()), r);

    auto icRow = Value::matrix(rows, 1, ValueType::DOUBLE, alloc);
    keel::f64 *ic = icRow.doubleDataMut();
    keel::usize nanSeen = 0;
    for (keel::usize r = 0; r < rows; ++r) {
        if (rowHasNan(src, cols, rows, r)) {
            ic[r] = static_cast<keel::f64>(nanRankBase + nanSeen + 1);
            ++nanSeen;
        } else {
            ic[r] = static_cast<keel::f64>(*rankByKey.get(extractRow(src, cols, rows, r, scratch.allocator())) + 1);
        }
    }

    auto iaCol = Value::matrix(uniqRows.size(), 1, ValueType::DOUBLE, alloc);
    keel::f64 *ia = iaCol.doubleDataMut();
    for (keel::usize i = 0; i < uniqRows.size(); ++i)
        ia[i] = static_cast<keel::f64>(uniqRows[i] + 1);

    return std::make_tuple(numkit::ops::collectRowsByIndex(x, uniqRows.data(), uniqRows.size(), alloc),
                           std::move(iaCol), std::move(icRow));
}

// ── ismember ───────────────────────────────────────────────────────

inline keel::c64 elemAsComplex(const Value &v, keel::usize k)
{
    return (v.type() == ValueType::COMPLEX) ? v.complexData()[k]
                                            : keel::c64(v.elemAsDouble(k), 0.0);
}

std::pair<Value, Value>
ismemberComplex(const Value &a, const Value &b, bool wantLoc, keel::Allocator *alloc)
{
    const keel::usize na = a.numel(), nb = b.numel();
    Value tf = createLike(a, ValueType::LOGICAL, alloc);
    Value loc;
    keel::f64 *lo = nullptr;
    if (wantLoc) { loc = createLike(a, ValueType::DOUBLE, alloc); lo = loc.doubleDataMut(); }
    if (na == 0) return {std::move(tf), std::move(loc)};
    keel::u8 *out = tf.logicalDataMut();

    keel::ArenaTemp scratch = keel::get_scratch(alloc);
    keel::HashMap<keel::c64, keel::f64> idxB(scratch.allocator());
    idxB.reserve(nb);
    for (keel::usize i = 0; i < nb; ++i) {
        const keel::c64 z = elemAsComplex(b, i);
        if (keel::is_nan(z.real()) || keel::is_nan(z.imag())) continue;
        idxB.try_insert(z, static_cast<keel::f64>(i + 1)); // lowest index wins
    }
    for (keel::usize i = 0; i < na; ++i) {
        const keel::c64 z = elemAsComplex(a, i);
        if (keel::is_nan(z.real()) || keel::is_nan(z.imag())) { out[i] = 0; if (lo) lo[i] = 0.0; continue; }
        const keel::f64 *hit = idxB.get(z);
        out[i] = hit ? 1 : 0;
        if (lo) lo[i] = hit ? *hit : 0.0;
    }
    return {std::move(tf), std::move(loc)};
}

enum class CxSetOp { Intersect, Union, Diff };
Value complexSetOp(const Value &a, const Value &b, CxSetOp op, bool stable,
                   keel::Allocator *alloc)
{
    keel::ArenaTemp scratch = keel::get_scratch(alloc);
    keel::HashSet<keel::c64> setB(scratch.allocator());
    setB.reserve(b.numel());
    for (keel::usize i = 0; i < b.numel(); ++i) {
        const keel::c64 z = elemAsComplex(b, i);
        if (keel::is_nan(z.real()) || keel::is_nan(z.imag())) continue;
        setB.insert(z);
    }
    keel::HashSet<keel::c64> seen(scratch.allocator());
    seen.reserve(a.numel() + b.numel());
    keel::Array<keel::c64> out(scratch.allocator());

    auto consider = [&](keel::c64 z) {
        if (keel::is_nan(z.real()) || keel::is_nan(z.imag())) return;
        const bool inB = setB.contains(z);
        bool keep = (op == CxSetOp::Union) ? true
                  : (op == CxSetOp::Intersect) ? inB
                                               : !inB;
        if (keep && seen.insert(z)) out.push_back(z);
    };
    for (keel::usize i = 0; i < a.numel(); ++i) consider(elemAsComplex(a, i));
    if (op == CxSetOp::Union)
        for (keel::usize i = 0; i < b.numel(); ++i) consider(elemAsComplex(b, i));

    if (!stable)
        std::stable_sort(out.begin(), out.end(),
                         [](keel::c64 x, keel::c64 y) { return cxUniqLess(x, y); });

    auto r = Value::matrix(1, out.size(), ValueType::COMPLEX, alloc);
    if (!out.empty())
        std::copy(out.begin(), out.end(), r.complexDataMut());
    return r;
}

Value ismember(const Value &a, const Value &b, keel::Allocator *alloc)
{
    if (a.type() == ValueType::COMPLEX || b.type() == ValueType::COMPLEX)
        return ismemberComplex(a, b, /*wantLoc=*/false, alloc).first;
    const keel::usize na = a.numel();
    const keel::usize nb = b.numel();

    auto r = createLike(a, ValueType::LOGICAL, alloc);
    if (na == 0) return r;
    if (nb == 0) {
        std::fill(r.logicalDataMut(), r.logicalDataMut() + na, keel::u8{0});
        return r;
    }

    keel::ArenaTemp scratch = keel::get_scratch(alloc);
    keel::HashSet<keel::f64, DoubleHashEq0> setB(scratch.allocator());
    setB.reserve(nb);
    const keel::f64 *pb = b.doubleData();
    for (keel::usize i = 0; i < nb; ++i)
        if (!keel::is_nan(pb[i])) setB.insert(pb[i]);

    const keel::f64 *pa = a.doubleData();
    keel::u8 *out = r.logicalDataMut();
    for (keel::usize i = 0; i < na; ++i) {
        const keel::f64 v = pa[i];
        out[i] = (!keel::is_nan(v) && setB.contains(v)) ? 1 : 0;
    }
    return r;
}

// ── union / intersect / setdiff ────────────────────────────────────

Value setUnion(const Value &a, const Value &b, bool stable, keel::Allocator *alloc)
{
    if (a.type() == ValueType::COMPLEX || b.type() == ValueType::COMPLEX)
        return complexSetOp(a, b, CxSetOp::Union, stable, alloc);
    keel::ArenaTemp scratch = keel::get_scratch(alloc);
    if (stable) {
        keel::HashSet<keel::f64, DoubleHashEq0> seen(scratch.allocator());
        keel::Array<keel::f64> out(scratch.allocator());
        out.reserve(a.numel() + b.numel());
        const keel::f64 *pa = a.doubleData();
        for (keel::usize i = 0; i < a.numel(); ++i)
            if (!keel::is_nan(pa[i]) && seen.insert(pa[i])) out.push_back(pa[i]);
        const keel::f64 *pbS = b.doubleData();
        for (keel::usize i = 0; i < b.numel(); ++i)
            if (!keel::is_nan(pbS[i]) && seen.insert(pbS[i])) out.push_back(pbS[i]);
        return rowFromVec(out.data(), out.size(), alloc);
    }
    auto s = hashSetNoNaN(a, scratch.allocator());
    const keel::f64 *pb = b.doubleData();
    for (keel::usize i = 0; i < b.numel(); ++i)
        if (!keel::is_nan(pb[i])) s.insert(pb[i]);
    keel::Array<keel::f64> out(scratch.allocator());
    out.reserve(s.size());
    for (keel::f64 v : s) out.push_back(v);
    std::sort(out.begin(), out.end());
    return rowFromVec(out.data(), out.size(), alloc);
}

Value setIntersect(const Value &a, const Value &b, bool stable, keel::Allocator *alloc)
{
    if (a.type() == ValueType::COMPLEX || b.type() == ValueType::COMPLEX)
        return complexSetOp(a, b, CxSetOp::Intersect, stable, alloc);
    if (stable) {
        keel::ArenaTemp scr = keel::get_scratch(alloc);
        auto setB = hashSetNoNaN(b, scr.allocator());
        keel::HashSet<keel::f64, DoubleHashEq0> seen(scr.allocator());
        keel::Array<keel::f64> out(scr.allocator());
        const keel::f64 *pa = a.doubleData();
        for (keel::usize i = 0; i < a.numel(); ++i) {
            const keel::f64 v = pa[i];
            if (!keel::is_nan(v) && setB.contains(v) && seen.insert(v))
                out.push_back(v);
        }
        return rowFromVec(out.data(), out.size(), alloc);
    }
    const bool aSmaller = a.numel() <= b.numel();
    const Value &small = aSmaller ? a : b;
    const Value &large = aSmaller ? b : a;

    keel::ArenaTemp scratch = keel::get_scratch(alloc);
    auto smallSet = hashSetNoNaN(small, scratch.allocator());
    keel::HashSet<keel::f64, DoubleHashEq0> seenInLarge(scratch.allocator());
    seenInLarge.reserve(smallSet.size());
    keel::Array<keel::f64> out(scratch.allocator());
    out.reserve(smallSet.size());

    const keel::f64 *pl = large.doubleData();
    for (keel::usize i = 0; i < large.numel(); ++i) {
        const keel::f64 v = pl[i];
        if (keel::is_nan(v)) continue;
        if (smallSet.contains(v) && seenInLarge.insert(v))
            out.push_back(v);
    }
    std::sort(out.begin(), out.end());
    return rowFromVec(out.data(), out.size(), alloc);
}

Value setDiff(const Value &a, const Value &b, bool stable, keel::Allocator *alloc)
{
    if (a.type() == ValueType::COMPLEX || b.type() == ValueType::COMPLEX)
        return complexSetOp(a, b, CxSetOp::Diff, stable, alloc);
    keel::ArenaTemp scratch = keel::get_scratch(alloc);
    auto setB = hashSetNoNaN(b, scratch.allocator());
    keel::HashSet<keel::f64, DoubleHashEq0> seen(scratch.allocator());
    seen.reserve(a.numel() / 2 + 1);
    keel::Array<keel::f64> out(scratch.allocator());
    out.reserve(a.numel());
    const keel::f64 *pa = a.doubleData();
    for (keel::usize i = 0; i < a.numel(); ++i) {
        const keel::f64 v = pa[i];
        if (keel::is_nan(v)) continue;
        if (!setB.contains(v) && seen.insert(v))
            out.push_back(v);
    }
    if (!stable) std::sort(out.begin(), out.end());
    return rowFromVec(out.data(), out.size(), alloc);
}

// ── histcounts / discretize ────────────────────────────────────────

Value histcounts(const Value &x, const Value &edges, keel::Allocator *mr)
{
    validateEdges(edges, "histcounts");
    const keel::usize nBins = edges.numel() - 1;
    auto r = Value::matrix(1, nBins, ValueType::DOUBLE, mr);
    double *dst = r.doubleDataMut();
    std::fill(dst, dst + nBins, 0.0);

    const double *e = edges.doubleData();
    const double *p = x.doubleData();
    const keel::usize n = x.numel();

    double step;
    if (edgesAreUniform(e, edges.numel(), step)) {
        const double e0 = e[0];
        const double eN = e[nBins];
        const double invStep = 1.0 / step;
        for (keel::usize i = 0; i < n; ++i) {
            const double v = p[i];
            if (v >= e0 && v <= eN) {
                keel::usize bin;
                if (v == eN) {
                    bin = nBins - 1;          // last bin is right-closed
                } else {
                    bin = static_cast<keel::usize>((v - e0) * invStep);
                    if (bin >= nBins) bin = nBins - 1;  // FP rounding guard
                }
                dst[bin] += 1.0;
            }
        }
        return r;
    }

    // Irregular-edges path. Inline bounds check + upper_bound.
    const double e0  = e[0];
    const double eN  = e[nBins];
    const keel::usize nE  = edges.numel();
    if (nBins <= 8) {
        for (keel::usize i = 0; i < n; ++i) {
            const double v = p[i];
            if (!(v >= e0 && v <= eN)) continue;
            if (v == eN) { dst[nBins - 1] += 1.0; continue; }
            keel::usize k = 0;
            while (k + 1 < nBins && e[k + 1] <= v) ++k;
            dst[k] += 1.0;
        }
    } else {
        for (keel::usize i = 0; i < n; ++i) {
            const double v = p[i];
            if (!(v >= e0 && v <= eN)) continue;
            if (v == eN) { dst[nBins - 1] += 1.0; continue; }
            keel::usize lo = 0, hi = nE;
            while (lo < hi) {
                const keel::usize mid = lo + (hi - lo) / 2;
                if (e[mid] <= v) lo = mid + 1;
                else hi = mid;
            }
            dst[lo - 1] += 1.0;
        }
    }
    return r;
}

Value histcounts(const Value &x, const Value &edges, HistNorm norm, keel::Allocator *mr)
{
    Value counts = histcounts(x, edges, mr);   // raw count row vector
    if (norm == HistNorm::Count) return counts;

    const keel::usize nBins = counts.numel();
    double *c = counts.doubleDataMut();
    const double N = static_cast<double>(x.numel());
    const double *e = edges.doubleData();

    switch (norm) {
    case HistNorm::Count:
        break;                              // handled above
    case HistNorm::Probability:
        if (N > 0) for (keel::usize i = 0; i < nBins; ++i) c[i] /= N;
        break;
    case HistNorm::CountDensity:
        for (keel::usize i = 0; i < nBins; ++i) c[i] /= (e[i + 1] - e[i]);
        break;
    case HistNorm::Pdf:
        if (N > 0)
            for (keel::usize i = 0; i < nBins; ++i) c[i] /= (N * (e[i + 1] - e[i]));
        break;
    case HistNorm::CumCount: {
        double acc = 0.0;
        for (keel::usize i = 0; i < nBins; ++i) { acc += c[i]; c[i] = acc; }
        break;
    }
    case HistNorm::Cdf: {
        double acc = 0.0;
        for (keel::usize i = 0; i < nBins; ++i) {
            acc += c[i];
            c[i] = (N > 0) ? acc / N : 0.0;
        }
        break;
    }
    }
    return counts;
}

// ── automatic bin selection (MATLAB binpicker rules) ───────────────
// Faithful port of MATLAB R2025b's histcounts auto-binning:
//   datafun/histcounts.m (dispatch + scott/fd/sturges/sqrt/auto rules),
//   datafun/+matlab/+internal/+math/{binpicker,binpickerbl}.m,
//   datafun/private/integerrule.m.
namespace {

// MATLAB eps(x): distance from |x| to the next larger double.
inline double matlabEps(double x)
{
    x = std::fabs(x);
    return std::nextafter(x, std::numeric_limits<double>::infinity()) - x;
}

// Append the colon range lo:step:hi (MATLAB colon point count, with the usual
// floating tolerance so an exact-multiple endpoint is included). step > 0.
void appendColon(std::vector<double> &e, double lo, double step, double hi)
{
    if (step <= 0.0 || !std::isfinite(step)) { e.push_back(lo); return; }
    const double span = (hi - lo) / step;
    const double tol  = 4.0 * std::numeric_limits<double>::epsilon()
                            * std::max(std::fabs(span), 1.0);
    const long n = static_cast<long>(std::floor(span + tol));
    for (long i = 0; i <= n; ++i) e.push_back(lo + static_cast<double>(i) * step);
}

// linspace(a, b, nbins+1) — nbins uniform bins, exact endpoints.
std::vector<double> linspaceEdges(double a, double b, long nbins)
{
    if (nbins < 1) nbins = 1;
    std::vector<double> e(static_cast<keel::usize>(nbins) + 1);
    for (long i = 0; i <= nbins; ++i)
        e[i] = a + (b - a) * (static_cast<double>(i) / static_cast<double>(nbins));
    e[0] = a; e[nbins] = b;
    return e;
}

double sampleStd(const std::vector<double> &x)
{
    const keel::usize n = x.size();
    if (n < 2) return 0.0;
    double mean = 0.0;
    for (double v : x) mean += v;
    mean /= static_cast<double>(n);
    double s = 0.0;
    for (double v : x) { const double d = v - mean; s += d * d; }
    return std::sqrt(s / static_cast<double>(n - 1));
}

// MATLAB iqr = prctile(x,75) - prctile(x,25); prctile linearly interpolates the
// sorted samples placed at percentiles 100*((1:n)-0.5)/n.
double iqrMatlab(std::vector<double> x)
{
    const keel::usize n = x.size();
    if (n == 0) return 0.0;
    std::sort(x.begin(), x.end());
    auto prc = [&](double p) -> double {
        if (n == 1) return x[0];
        const double pos = p / 100.0 * static_cast<double>(n) - 0.5;  // 0-based
        if (pos <= 0.0) return x.front();
        if (pos >= static_cast<double>(n - 1)) return x.back();
        const keel::usize lo = static_cast<keel::usize>(std::floor(pos));
        const double frac = pos - static_cast<double>(lo);
        return x[lo] + frac * (x[lo + 1] - x[lo]);
    };
    return prc(75.0) - prc(25.0);
}

// Assemble [leftEdge, leftEdge+1..n-1*binWidth, rightEdge] (binpicker tail).
std::vector<double> niceEdges(double leftEdge, double binWidth,
                              double rightEdge, long nbins)
{
    if (nbins < 1) nbins = 1;
    if (!std::isfinite(binWidth)) return linspaceEdges(leftEdge, rightEdge, nbins);
    std::vector<double> e(static_cast<keel::usize>(nbins) + 1);
    e[0] = leftEdge;
    for (long i = 1; i < nbins; ++i) e[i] = leftEdge + static_cast<double>(i) * binWidth;
    e[nbins] = rightEdge;
    return e;
}

std::vector<double> binpicker(double xmin, double xmax,
                              bool haveNbins, double nbins, double rawBinWidth)
{
    static const double kSqrtEps = std::sqrt(std::numeric_limits<double>::epsilon());
    const double dmax    = std::numeric_limits<double>::max();
    const double xscale  = std::max(std::fabs(xmin), std::fabs(xmax));
    const double xrange  = xmax - xmin;
    rawBinWidth = std::max(rawBinWidth, matlabEps(xscale));

    if (xrange > std::max(kSqrtEps * xscale, std::numeric_limits<double>::min())) {
        const double powOfTen = std::pow(10.0, std::floor(std::log10(rawBinWidth)));
        const double relSize  = rawBinWidth / powOfTen;          // in [1, 10)
        double binWidth, leftEdge, rightEdge, nbinsActual;
        if (!haveNbins) {                                        // automatic rule
            if      (relSize < 1.5) binWidth = 1.0  * powOfTen;
            else if (relSize < 2.5) binWidth = 2.0  * powOfTen;
            else if (relSize < 4.0) binWidth = 3.0  * powOfTen;
            else if (relSize < 7.5) binWidth = 5.0  * powOfTen;
            else                    binWidth = 10.0 * powOfTen;
            leftEdge    = std::max(std::min(binWidth * std::floor(xmin / binWidth), xmin), -dmax);
            nbinsActual = std::max(1.0, std::ceil((xmax - leftEdge) / binWidth));
            rightEdge   = std::min(std::max(leftEdge + nbinsActual * binWidth, xmax), dmax);
        } else {                                                 // bin count fixed
            binWidth = powOfTen * std::floor(relSize);
            leftEdge = std::max(std::min(binWidth * std::floor(xmin / binWidth), xmin), -dmax);
            if (nbins > 1.0) {
                const double ll  = (xmax - leftEdge) / nbins;
                const double ul  = (xmax - leftEdge) / (nbins - 1.0);
                const double p10 = std::pow(10.0, std::floor(std::log10(ul - ll)));
                binWidth = p10 * std::ceil(ll / p10);
            }
            nbinsActual = nbins;
            rightEdge   = std::min(std::max(leftEdge + nbinsActual * binWidth, xmax), dmax);
        }
        return niceEdges(leftEdge, binWidth, rightEdge, static_cast<long>(nbinsActual));
    }

    // nearly-constant data
    const double nb       = haveNbins ? nbins : 1.0;
    const double binRange = std::max(1.0, std::ceil(nb * matlabEps(xscale)));
    const double leftEdge  = std::floor(2.0 * (xmin - binRange / 4.0)) / 2.0;
    const double rightEdge = std::ceil(2.0 * (xmax + binRange / 4.0)) / 2.0;
    const long   nbi       = static_cast<long>(nb);
    return niceEdges(leftEdge, (rightEdge - leftEdge) / nb, rightEdge, nbi);
}

std::vector<double> binpickerbl(double dmin, double dmax,
                                double minlimit, double maxlimit, double binwidth)
{
    static const double kSqrtEps = std::sqrt(std::numeric_limits<double>::epsilon());
    const double xscale = std::max(std::fabs(dmin), std::fabs(dmax));
    const double xrange = dmax - dmin;
    binwidth = std::max(binwidth, matlabEps(xscale));
    if (xrange > std::max(kSqrtEps * xscale, std::numeric_limits<double>::min())) {
        const long nbins = std::max(1L,
            static_cast<long>(std::ceil((maxlimit - minlimit) / binwidth)));
        return linspaceEdges(minlimit, maxlimit, nbins);
    }
    return {minlimit, maxlimit};
}

// integerrule (non-empty data). dataMin/dataMax/dataScale over the finite data.
std::vector<double> integerrule(double minx, double maxx,
                                double dataMin, double dataMax, double dataScale,
                                bool hardlimits, double maximumbins)
{
    const double dataRange = dataMax - dataMin;
    double bw;
    if (dataRange > maximumbins)
        bw = std::pow(10.0, std::ceil(std::log10(dataRange / maximumbins)));
    else if (matlabEps(dataScale) > 1.0)
        bw = std::pow(10.0, std::ceil(std::log10(matlabEps(dataScale))));
    else
        bw = 1.0;

    std::vector<double> e;
    if (!hardlimits) {
        const double mn = bw * std::round(minx / bw);
        const double mx = bw * std::round(maxx / bw);
        appendColon(e, std::floor(mn) - 0.5 * bw, bw, std::ceil(mx) + 0.5 * bw);
    } else {
        e.push_back(minx);
        appendColon(e, bw * std::ceil(minx / bw) + 0.5, bw, bw * std::floor(maxx / bw) - 0.5);
        e.push_back(maxx);
    }
    return e;
}

// Apply a BinMethod to data `xc` (finite) with range [minx,maxx]. dmin/dmax are
// the finite-data extremes (== minx/maxx unless hardlimits restrict the data).
std::vector<double> applyMethod(HistBinMethod m, const std::vector<double> &xc,
                                double minx, double maxx, bool hard,
                                double dmin, double dmax)
{
    const double dataScale = std::max(std::fabs(dmin), std::fabs(dmax));
    const double maxbins   = 65536.0;
    const double n         = static_cast<double>(xc.size());

    HistBinMethod eff = m;
    if (m == HistBinMethod::Auto) {
        bool integerLike = !xc.empty() && (maxx - minx) <= 50.0
                        && maxx <= 4503599627370496.0 && minx >= -4503599627370496.0;
        if (integerLike)
            for (double v : xc) if (std::round(v) != v) { integerLike = false; break; }
        eff = integerLike ? HistBinMethod::Integers : HistBinMethod::Scott;
    }

    switch (eff) {
    case HistBinMethod::Integers:
        return integerrule(minx, maxx, dmin, dmax, dataScale, hard, maxbins);
    case HistBinMethod::Scott: {
        const double bw = 3.5 * sampleStd(xc) / std::pow(n, 1.0 / 3.0);
        return hard ? binpickerbl(dmin, dmax, minx, maxx, bw)
                    : binpicker(minx, maxx, false, 0.0, bw);
    }
    case HistBinMethod::Fd: {
        double bw;
        if (xc.size() > 1) {
            const double iq = std::max(iqrMatlab(xc), (dmax - dmin) / 10.0);
            bw = 2.0 * iq * std::pow(n, -1.0 / 3.0);
        } else {
            bw = 1.0;
        }
        return hard ? binpickerbl(dmin, dmax, minx, maxx, bw)
                    : binpicker(minx, maxx, false, 0.0, bw);
    }
    case HistBinMethod::Sturges: {
        const long nb = std::max(1L, static_cast<long>(std::ceil(std::log2(n) + 1.0)));
        if (hard) return linspaceEdges(minx, maxx, nb);
        const double bw = (maxx - minx) / static_cast<double>(nb);
        return std::isfinite(bw) ? binpicker(minx, maxx, false, 0.0, bw)
                                 : binpicker(minx, maxx, true, static_cast<double>(nb), bw);
    }
    case HistBinMethod::Sqrt: {
        const long nb = std::max(1L, static_cast<long>(std::ceil(std::sqrt(n))));
        if (hard) return linspaceEdges(minx, maxx, nb);
        const double bw = (maxx - minx) / static_cast<double>(nb);
        return std::isfinite(bw) ? binpicker(minx, maxx, false, 0.0, bw)
                                 : binpicker(minx, maxx, true, static_cast<double>(nb), bw);
    }
    default:
        return binpicker(minx, maxx, false, 0.0, 0.0);
    }
}

} // namespace

Value histcountsAutoEdges(const Value &x, const HistBinSpec &spec,
                          keel::Allocator *mr)
{
    // Finite-data extremes (MATLAB excludes Inf/NaN from the range).
    const keel::usize nx = x.numel();
    std::vector<double> xc;
    xc.reserve(nx);
    bool any = false;
    double dmin = 0.0, dmax = 0.0;
    for (keel::usize k = 0; k < nx; ++k) {
        const double v = x.elemAsDouble(k);
        if (!std::isfinite(v)) continue;
        xc.push_back(v);
        if (!any) { dmin = dmax = v; any = true; }
        else { if (v < dmin) dmin = v; if (v > dmax) dmax = v; }
    }

    std::vector<double> edges;

    if (spec.hasBinLimits) {
        const double lo = spec.limLo, hi = spec.limHi;
        if (spec.hasNumBins) {
            const long n = std::max(1L, static_cast<long>(spec.numBins));
            edges.resize(static_cast<keel::usize>(n) + 1);
            const double w = (hi - lo) / static_cast<double>(n);
            for (long i = 0; i < n; ++i) edges[i] = lo + static_cast<double>(i) * w;
            edges[n] = hi;
        } else if (spec.hasBinWidth) {
            const double bw = std::max(spec.binWidth, (hi - lo) / 65536.0);
            appendColon(edges, lo, bw, hi);
            if (edges.empty() || edges.back() < hi || edges.size() == 1) edges.push_back(hi);
        } else {
            std::vector<double> xr;
            for (double v : xc) if (v >= lo && v <= hi) xr.push_back(v);
            double rmin = lo, rmax = hi;
            if (!xr.empty()) {
                rmin = *std::min_element(xr.begin(), xr.end());
                rmax = *std::max_element(xr.begin(), xr.end());
            }
            edges = applyMethod(spec.method, xr, lo, hi, true, rmin, rmax);
        }
    } else if (spec.hasNumBins) {
        const long n = std::max(1L, static_cast<long>(spec.numBins));
        if (!any) { edges.resize(static_cast<keel::usize>(n) + 1);
                    for (long i = 0; i <= n; ++i) edges[i] = static_cast<double>(i); }
        else edges = binpicker(dmin, dmax, true, static_cast<double>(n),
                               (dmax - dmin) / static_cast<double>(n));
    } else if (spec.hasBinWidth) {
        const double bw = spec.binWidth;
        if (!any) { edges = {0.0, bw}; }
        else {
            const double xrange = dmax - dmin;
            double binWidth = bw;
            double leftEdge = binWidth * std::floor(dmin / binWidth);
            long nbins = std::max(1L, static_cast<long>(std::ceil((dmax - leftEdge) / binWidth)));
            const double maxbins = 65536.0;
            if (static_cast<double>(nbins) > maxbins) {
                nbins = static_cast<long>(maxbins);
                binWidth = xrange / (maxbins - 1.0);
                leftEdge = binWidth * std::floor(dmin / binWidth);
                if (dmax <= leftEdge + (nbins - 1) * binWidth) {
                    binWidth = xrange / maxbins;
                    leftEdge = dmin;
                }
            }
            edges.resize(static_cast<keel::usize>(nbins) + 1);
            for (long i = 0; i <= nbins; ++i) edges[i] = leftEdge + static_cast<double>(i) * binWidth;
            if (edges.back() < dmax) edges.back() = dmax;
        }
    } else {  // BinMethod (default Auto)
        if (!any)
            edges = (spec.method == HistBinMethod::Integers)
                        ? std::vector<double>{-0.5, 0.5}
                        : std::vector<double>{0.0, 1.0};
        else
            edges = applyMethod(spec.method, xc, dmin, dmax, false, dmin, dmax);
    }

    Value e = Value::matrix(1, edges.size(), ValueType::DOUBLE, mr);
    std::copy(edges.begin(), edges.end(), e.doubleDataMut());
    return e;
}

Value discretize(const Value &x, const Value &edges, keel::Allocator *mr)
{
    validateEdges(edges, "discretize");
    auto r = createLike(x, ValueType::DOUBLE, mr);
    const double *e = edges.doubleData();
    const double *p = x.doubleData();
    double *dst = r.doubleDataMut();
    const keel::usize n = x.numel();
    const keel::usize nBins = edges.numel() - 1;

    double step;
    if (edgesAreUniform(e, edges.numel(), step)) {
        const double e0 = e[0];
        const double eN = e[nBins];
        const double invStep = 1.0 / step;
        for (keel::usize i = 0; i < n; ++i) {
            const double v = p[i];
            if (v >= e0 && v <= eN) {
                keel::usize bin;
                if (v == eN) {
                    bin = nBins - 1;
                } else {
                    bin = static_cast<keel::usize>((v - e0) * invStep);
                    if (bin >= nBins) bin = nBins - 1;
                }
                dst[i] = static_cast<double>(bin + 1); // 1-based
            } else {
                dst[i] = std::nan("");
            }
        }
        return r;
    }

    const double e0 = e[0];
    const double eN = e[nBins];
    const keel::usize nE = edges.numel();
    if (nBins <= 8) {
        for (keel::usize i = 0; i < n; ++i) {
            const double v = p[i];
            if (!(v >= e0 && v <= eN)) { dst[i] = std::nan(""); continue; }
            if (v == eN) { dst[i] = static_cast<double>(nBins); continue; }
            keel::usize k = 0;
            while (k + 1 < nBins && e[k + 1] <= v) ++k;
            dst[i] = static_cast<double>(k + 1); // 1-based
        }
    } else {
        for (keel::usize i = 0; i < n; ++i) {
            const double v = p[i];
            if (!(v >= e0 && v <= eN)) { dst[i] = std::nan(""); continue; }
            if (v == eN) { dst[i] = static_cast<double>(nBins); continue; }
            keel::usize lo = 0, hi = nE;
            while (lo < hi) {
                const keel::usize mid = lo + (hi - lo) / 2;
                if (e[mid] <= v) lo = mid + 1;
                else hi = mid;
            }
            dst[i] = static_cast<double>(lo); // (lo-1)+1
        }
    }
    return r;
}

// ── histc (legacy) ──────────────────────────────────────────────────
//
// 0-based bin for v given ascending edges e[0..nE-1]: k in [0, nE-2] when
// e[k] <= v < e[k+1]; nE-1 when v == e[nE-1] (last bin = exact equal to
// last edge). Returns SIZE_MAX when v is NaN or outside [e[0], e[nE-1]].

Value histc(const Value &x, const Value &edges, keel::Allocator *mr)
{
    validateEdges(edges, "histc");
    const keel::usize nE = edges.numel();
    const double *e = edges.doubleData();

    const keel::usize rows = static_cast<keel::usize>(x.dims().rows());
    const keel::usize cols = x.numel() == 0 ? 0 : x.numel() / std::max<keel::usize>(rows, 1);
    const double *p = x.doubleData();

    // Row vector -> count along the row, return a 1 × nE row. Otherwise
    // (column vector / matrix) count each column -> nE × ncols.
    if (rows == 1) {
        Value r = Value::matrix(1, nE, ValueType::DOUBLE, mr);
        double *dst = r.doubleDataMut();
        std::fill(dst, dst + nE, 0.0);
        const keel::usize n = x.numel();
        for (keel::usize i = 0; i < n; ++i) {
            const keel::usize b = histcBin(p[i], e, nE);
            if (b != SIZE_MAX) dst[b] += 1.0;
        }
        return r;
    }

    const keel::usize H = rows, W = cols;
    Value r = Value::matrix(nE, W, ValueType::DOUBLE, mr);
    double *dst = r.doubleDataMut();
    std::fill(dst, dst + nE * W, 0.0);
    for (keel::usize c = 0; c < W; ++c)
        for (keel::usize i = 0; i < H; ++i) {
            const keel::usize b = histcBin(p[c * H + i], e, nE);
            if (b != SIZE_MAX) dst[c * nE + b] += 1.0;
        }
    return r;
}


// ════════════════════════════════════════════════════════════════════════
// Pack 16: setxor / allunique / numunique / ismembertol / uniquetol
// ════════════════════════════════════════════════════════════════════════
//
// Builds on the same hash-set / sort+merge primitives used by `unique` /
// `union` / `intersect`. The tolerant variants do an O(N²) sweep — fine
// for the typical cell sizes (≤ 1e4) where these get called.

Value setxor(const Value &a, const Value &b, keel::Allocator *alloc)
{
    // Symmetric difference: { x : x in A xor x in B }.
    if (a.numel() == 0 && b.numel() == 0) return emptyRow(alloc);

    keel::ArenaTemp scratch = keel::get_scratch(alloc);
    keel::HashSet<keel::f64, DoubleHashEq0> sa(scratch.allocator()), sb(scratch.allocator());
    sa.reserve(a.numel());
    sb.reserve(b.numel());
    for (keel::usize i = 0; i < a.numel(); ++i) sa.insert(a.doubleData()[i]);
    for (keel::usize i = 0; i < b.numel(); ++i) sb.insert(b.doubleData()[i]);

    keel::Array<keel::f64> out(scratch.allocator());
    out.reserve(sa.size() + sb.size());
    for (keel::f64 v : sa) if (!sb.contains(v)) out.push_back(v);
    for (keel::f64 v : sb) if (!sa.contains(v)) out.push_back(v);
    std::sort(out.begin(), out.end());
    return rowFromVec(out.data(), out.size(), alloc);
}

Value allunique(const Value &x, keel::Allocator *alloc)
{
    const keel::usize n = x.numel();
    if (n <= 1) return Value::logicalScalar(true, alloc);
    keel::ArenaTemp scratch = keel::get_scratch(alloc);
    keel::HashSet<keel::f64, DoubleHashEq0> seen(scratch.allocator());
    seen.reserve(n);
    const keel::f64 *p = x.doubleData();
    for (keel::usize i = 0; i < n; ++i) {
        if (keel::is_nan(p[i])) continue;
        if (!seen.insert(p[i]))
            return Value::logicalScalar(false, alloc);
    }
    return Value::logicalScalar(true, alloc);
}

Value numunique(const Value &x, keel::Allocator *alloc)
{
    const keel::usize n = x.numel();
    if (n == 0) return Value::scalar(0.0, alloc);
    keel::ArenaTemp scratch = keel::get_scratch(alloc);
    keel::HashSet<keel::f64, DoubleHashEq0> seen(scratch.allocator());
    seen.reserve(n);
    keel::usize nanCount = 0;
    const keel::f64 *p = x.doubleData();
    for (keel::usize i = 0; i < n; ++i) {
        if (keel::is_nan(p[i])) ++nanCount;
        else seen.insert(p[i]);
    }
    return Value::scalar(static_cast<keel::f64>(seen.size() + nanCount), alloc);
}


Value ismembertol(const Value &a, const Value &s, double tol, keel::Allocator *mr)
{
    // Returns logical of size(a). For each a[i], true if there exists
    // s[j] with |a[i] - s[j]| ≤ tol * max(1, |a|, |s|). Naive O(|a||s|).
    auto r = createLike(a, ValueType::LOGICAL, mr);
    const keel::usize na = a.numel(), ns = s.numel();
    const double *pa = a.doubleData();
    const double *ps = s.doubleData();
    for (keel::usize i = 0; i < na; ++i) {
        bool hit = false;
        for (keel::usize j = 0; j < ns; ++j) {
            if (nearlyEqualTol(pa[i], ps[j], tol)) { hit = true; break; }
        }
        r.logicalDataMut()[i] = hit ? 1 : 0;
    }
    return r;
}

Value uniquetol(const Value &x, double tol, keel::Allocator *mr)
{
    // MATLAB convention (doc uniquetol): two values u and v are within
    // tolerance iff
    //     |u - v| <= tol * max(|A(:)|)
    // i.e. an ABSOLUTE global tolerance scaled by the input's largest
    // magnitude. Cluster representatives are kept in input/sort order
    // and a candidate joins an existing cluster iff it is within
    // tol*DS of the cluster's representative (i.e. the previously
    // emitted unique value, since input is sorted ascending).
    //
    // Per-pair relative scaling (the previous nk behaviour) gave
    // wildly different cluster counts on dense inputs because the
    // scale shifted with each comparison.
    const keel::usize n = x.numel();
    if (n == 0) return emptyRow(mr);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto vals = keel::Array<double>(scratch.allocator());
    vals.reserve(n);
    const double *p = x.doubleData();
    double dataScale = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        vals.push_back(p[i]);
        if (!std::isnan(p[i])) {
            const double m = keel::abs(p[i]);
            if (m > dataScale) dataScale = m;
        }
    }
    std::sort(vals.begin(), vals.end(),
              [](double a, double b) {
                  if (std::isnan(b)) return !std::isnan(a);
                  if (std::isnan(a)) return false;
                  return a < b;
              });
    const double tolAbs = tol * dataScale;
    auto out = keel::Array<double>(scratch.allocator());
    out.reserve(n);
    for (double v : vals) {
        if (std::isnan(v)) {
            // NaNs are unique to themselves -- always emit.
            out.push_back(v);
            continue;
        }
        if (out.empty() || std::isnan(out.back())
            || keel::abs(v - out.back()) > tolAbs) {
            out.push_back(v);
        }
    }
    return rowFromVec(out.data(), out.size(), mr);
}

} // namespace numkit::builtin
