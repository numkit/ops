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
#include <numeric>
#include <utility>

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
                    iaVec.push_back(keel::as<keel::f64>(i + 1));
                    ic[i] = keel::as<keel::f64>(uVals.size());
                    continue;
                }
                if (const keel::usize *li = lastIdx.get(p[i]); li && *li == i) {
                    if (seen.insert(p[i])) {
                        posByVal.put(p[i], uVals.size());
                        uVals.push_back(p[i]);
                        iaVec.push_back(keel::as<keel::f64>(i + 1));
                    }
                }
            }
            for (keel::usize i = 0; i < n; ++i) {
                if (keel::is_nan(p[i])) continue;
                if (const keel::usize *pv = posByVal.get(p[i]))
                    ic[i] = keel::as<keel::f64>(*pv + 1);
            }
        } else {
            // 'stable' (default): first-occurrence order.
            for (keel::usize i = 0; i < n; ++i) {
                if (keel::is_nan(p[i])) {
                    uVals.push_back(keel::Limits<keel::f64>::quiet_nan());
                    iaVec.push_back(keel::as<keel::f64>(i + 1));
                    ic[i] = keel::as<keel::f64>(uVals.size());
                } else {
                    auto res = posByVal.try_insert(p[i], uVals.size());
                    if (res.inserted) {
                        uVals.push_back(p[i]);
                        iaVec.push_back(keel::as<keel::f64>(i + 1));
                        ic[i] = keel::as<keel::f64>(uVals.size());
                    } else {
                        ic[i] = keel::as<keel::f64>(*res.value + 1);
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
            ic[i] = keel::as<keel::f64>(nanRankBase + nanSeen + 1);
            ++nanSeen;
        } else {
            ic[i] = keel::as<keel::f64>(*rankByValue.get(p[i]) + 1);
        }
    }

    auto cOut = Value::matrix(1, sorted.size(), ValueType::DOUBLE, alloc);
    auto iaRow = Value::matrix(1, sorted.size(), ValueType::DOUBLE, alloc);
    for (keel::usize i = 0; i < sorted.size(); ++i) {
        cOut.doubleDataMut()[i]  = sorted[i].v;
        iaRow.doubleDataMut()[i] = keel::as<keel::f64>(sorted[i].origIdx + 1);
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
            ic[r] = keel::as<keel::f64>(pos + 1);
        }

        auto iaCol = Value::matrix(uniqRows.size(), 1, ValueType::DOUBLE, alloc);
        keel::f64 *ia = iaCol.doubleDataMut();
        for (keel::usize i = 0; i < uniqRows.size(); ++i)
            ia[i] = keel::as<keel::f64>(uniqRows[i] + 1);

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
            ic[r] = keel::as<keel::f64>(nanRankBase + nanSeen + 1);
            ++nanSeen;
        } else {
            ic[r] = keel::as<keel::f64>(*rankByKey.get(extractRow(src, cols, rows, r, scratch.allocator())) + 1);
        }
    }

    auto iaCol = Value::matrix(uniqRows.size(), 1, ValueType::DOUBLE, alloc);
    keel::f64 *ia = iaCol.doubleDataMut();
    for (keel::usize i = 0; i < uniqRows.size(); ++i)
        ia[i] = keel::as<keel::f64>(uniqRows[i] + 1);

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
        idxB.try_insert(z, keel::as<keel::f64>(i + 1)); // lowest index wins
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
    keel::f64 *dst = r.doubleDataMut();
    std::fill(dst, dst + nBins, 0.0);

    const keel::f64 *e = edges.doubleData();
    const keel::f64 *p = x.doubleData();
    const keel::usize n = x.numel();

    keel::f64 step;
    if (edgesAreUniform(e, edges.numel(), step)) {
        const keel::f64 e0 = e[0];
        const keel::f64 eN = e[nBins];
        const keel::f64 invStep = 1.0 / step;
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 v = p[i];
            if (v >= e0 && v <= eN) {
                keel::usize bin;
                if (v == eN) {
                    bin = nBins - 1;          // last bin is right-closed
                } else {
                    bin = keel::as<keel::usize>((v - e0) * invStep);
                    if (bin >= nBins) bin = nBins - 1;  // FP rounding guard
                }
                dst[bin] += 1.0;
            }
        }
        return r;
    }

    // Irregular-edges path. Inline bounds check + upper_bound.
    const keel::f64 e0  = e[0];
    const keel::f64 eN  = e[nBins];
    const keel::usize nE  = edges.numel();
    if (nBins <= 8) {
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 v = p[i];
            if (!(v >= e0 && v <= eN)) continue;
            if (v == eN) { dst[nBins - 1] += 1.0; continue; }
            keel::usize k = 0;
            while (k + 1 < nBins && e[k + 1] <= v) ++k;
            dst[k] += 1.0;
        }
    } else {
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 v = p[i];
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
    keel::f64 *c = counts.doubleDataMut();
    const keel::f64 N = keel::as<keel::f64>(x.numel());
    const keel::f64 *e = edges.doubleData();

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
        keel::f64 acc = 0.0;
        for (keel::usize i = 0; i < nBins; ++i) { acc += c[i]; c[i] = acc; }
        break;
    }
    case HistNorm::Cdf: {
        keel::f64 acc = 0.0;
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
inline keel::f64 matlabEps(keel::f64 x)
{
    x = keel::abs(x);
    return keel::nextafter(x, keel::Limits<keel::f64>::infinity()) - x;
}

// Append the colon range lo:step:hi (MATLAB colon point count, with the usual
// floating tolerance so an exact-multiple endpoint is included). step > 0.
void appendColon(keel::Array<keel::f64> &e, keel::f64 lo, keel::f64 step, keel::f64 hi)
{
    if (step <= 0.0 || !keel::is_finite(step)) { e.push_back(lo); return; }
    const keel::f64 span = (hi - lo) / step;
    const keel::f64 tol  = 4.0 * keel::Limits<keel::f64>::kEpsilon
                            * std::max(keel::abs(span), 1.0);
    const keel::i64 n = keel::as<keel::i64>(keel::floor(span + tol));
    for (keel::i64 i = 0; i <= n; ++i) e.push_back(lo + keel::as<keel::f64>(i) * step);
}

// linspace(a, b, nbins+1) — nbins uniform bins, exact endpoints.
keel::Array<keel::f64> linspaceEdges(keel::f64 a, keel::f64 b, keel::i64 nbins, keel::Allocator *alloc)
{
    if (nbins < 1) nbins = 1;
    keel::Array<keel::f64> e(keel::as<keel::usize>(nbins) + 1, alloc);
    for (keel::i64 i = 0; i <= nbins; ++i)
        e[i] = a + (b - a) * (keel::as<keel::f64>(i) / keel::as<keel::f64>(nbins));
    e[0] = a; e[nbins] = b;
    return e;
}

keel::f64 sampleStd(keel::ArrayRef<keel::f64> x)
{
    const keel::usize n = x.size();
    if (n < 2) return 0.0;
    keel::f64 mean = 0.0;
    for (keel::f64 v : x) mean += v;
    mean /= keel::as<keel::f64>(n);
    keel::f64 s = 0.0;
    for (keel::f64 v : x) { const keel::f64 d = v - mean; s += d * d; }
    return keel::sqrt(s / keel::as<keel::f64>(n - 1));
}

// MATLAB iqr = prctile(x,75) - prctile(x,25); prctile linearly interpolates the
// sorted samples placed at percentiles 100*((1:n)-0.5)/n.
keel::f64 iqrMatlab(keel::ArrayRef<keel::f64> x_in, keel::Allocator *alloc)
{
    const keel::usize n = x_in.size();
    if (n == 0) return 0.0;
    keel::Array<keel::f64> x(x_in, alloc);
    std::sort(x.begin(), x.end());
    auto prc = [&](keel::f64 p) -> keel::f64 {
        if (n == 1) return x[0];
        const keel::f64 pos = p / 100.0 * keel::as<keel::f64>(n) - 0.5;  // 0-based
        if (pos <= 0.0) return x[0];
        if (pos >= keel::as<keel::f64>(n - 1)) return x[n - 1];
        const keel::usize lo = keel::as<keel::usize>(keel::floor(pos));
        const keel::f64 frac = pos - keel::as<keel::f64>(lo);
        return x[lo] + frac * (x[lo + 1] - x[lo]);
    };
    return prc(75.0) - prc(25.0);
}

// Assemble [leftEdge, leftEdge+1..n-1*binWidth, rightEdge] (binpicker tail).
keel::Array<keel::f64> niceEdges(keel::f64 leftEdge, keel::f64 binWidth,
                              keel::f64 rightEdge, keel::i64 nbins, keel::Allocator *alloc)
{
    if (nbins < 1) nbins = 1;
    if (!keel::is_finite(binWidth)) return linspaceEdges(leftEdge, rightEdge, nbins, alloc);
    keel::Array<keel::f64> e(keel::as<keel::usize>(nbins) + 1, alloc);
    e[0] = leftEdge;
    for (keel::i64 i = 1; i < nbins; ++i) e[i] = leftEdge + keel::as<keel::f64>(i) * binWidth;
    e[nbins] = rightEdge;
    return e;
}

keel::Array<keel::f64> binpicker(keel::f64 xmin, keel::f64 xmax,
                              bool haveNbins, keel::f64 nbins, keel::f64 rawBinWidth,
                              keel::Allocator *alloc)
{
    static const keel::f64 kSqrtEps = keel::sqrt(keel::Limits<keel::f64>::kEpsilon);
    const keel::f64 dmax    = keel::Limits<keel::f64>::kMax;
    const keel::f64 xscale  = std::max(keel::abs(xmin), keel::abs(xmax));
    const keel::f64 xrange  = xmax - xmin;
    rawBinWidth = std::max(rawBinWidth, matlabEps(xscale));

    if (xrange > std::max(kSqrtEps * xscale, keel::Limits<keel::f64>::kMin)) {
        const keel::f64 powOfTen = keel::pow(10.0, keel::floor(keel::log10(rawBinWidth)));
        const keel::f64 relSize  = rawBinWidth / powOfTen;          // in [1, 10)
        keel::f64 binWidth, leftEdge, rightEdge, nbinsActual;
        if (!haveNbins) {                                        // automatic rule
            if      (relSize < 1.5) binWidth = 1.0  * powOfTen;
            else if (relSize < 2.5) binWidth = 2.0  * powOfTen;
            else if (relSize < 4.0) binWidth = 3.0  * powOfTen;
            else if (relSize < 7.5) binWidth = 5.0  * powOfTen;
            else                    binWidth = 10.0 * powOfTen;
            leftEdge    = std::max(std::min(binWidth * keel::floor(xmin / binWidth), xmin), -dmax);
            nbinsActual = std::max(1.0, keel::ceil((xmax - leftEdge) / binWidth));
            rightEdge   = std::min(std::max(leftEdge + nbinsActual * binWidth, xmax), dmax);
        } else {                                                 // bin count fixed
            binWidth = powOfTen * keel::floor(relSize);
            leftEdge = std::max(std::min(binWidth * keel::floor(xmin / binWidth), xmin), -dmax);
            if (nbins > 1.0) {
                const keel::f64 ll  = (xmax - leftEdge) / nbins;
                const keel::f64 ul  = (xmax - leftEdge) / (nbins - 1.0);
                const keel::f64 p10 = keel::pow(10.0, keel::floor(keel::log10(ul - ll)));
                binWidth = p10 * keel::ceil(ll / p10);
            }
            nbinsActual = nbins;
            rightEdge   = std::min(std::max(leftEdge + nbinsActual * binWidth, xmax), dmax);
        }
        return niceEdges(leftEdge, binWidth, rightEdge, keel::as<keel::i64>(nbinsActual), alloc);
    }

    // nearly-constant data
    const keel::f64 nb       = haveNbins ? nbins : 1.0;
    const keel::f64 binRange = std::max(1.0, keel::ceil(nb * matlabEps(xscale)));
    const keel::f64 leftEdge  = keel::floor(2.0 * (xmin - binRange / 4.0)) / 2.0;
    const keel::f64 rightEdge = keel::ceil(2.0 * (xmax + binRange / 4.0)) / 2.0;
    const keel::i64   nbi       = keel::as<keel::i64>(nb);
    return niceEdges(leftEdge, (rightEdge - leftEdge) / nb, rightEdge, nbi, alloc);
}

keel::Array<keel::f64> binpickerbl(keel::f64 dmin, keel::f64 dmax,
                                keel::f64 minlimit, keel::f64 maxlimit, keel::f64 binwidth,
                                keel::Allocator *alloc)
{
    static const keel::f64 kSqrtEps = keel::sqrt(keel::Limits<keel::f64>::kEpsilon);
    const keel::f64 xscale = std::max(keel::abs(dmin), keel::abs(dmax));
    const keel::f64 xrange = dmax - dmin;
    binwidth = std::max(binwidth, matlabEps(xscale));
    if (xrange > std::max(kSqrtEps * xscale, keel::Limits<keel::f64>::kMin)) {
        const keel::i64 nbins = std::max(1L,
            keel::as<keel::i64>(keel::ceil((maxlimit - minlimit) / binwidth)));
        return linspaceEdges(minlimit, maxlimit, nbins, alloc);
    }
    return keel::Array<keel::f64>({minlimit, maxlimit}, alloc);
}

// integerrule (non-empty data). dataMin/dataMax/dataScale over the finite data.
keel::Array<keel::f64> integerrule(keel::f64 minx, keel::f64 maxx,
                                keel::f64 dataMin, keel::f64 dataMax, keel::f64 dataScale,
                                bool hardlimits, keel::f64 maximumbins,
                                keel::Allocator *alloc)
{
    const keel::f64 dataRange = dataMax - dataMin;
    keel::f64 bw;
    if (dataRange > maximumbins)
        bw = keel::pow(10.0, keel::ceil(keel::log10(dataRange / maximumbins)));
    else if (matlabEps(dataScale) > 1.0)
        bw = keel::pow(10.0, keel::ceil(keel::log10(matlabEps(dataScale))));
    else
        bw = 1.0;

    keel::Array<keel::f64> e(alloc);
    if (!hardlimits) {
        const keel::f64 mn = bw * keel::round(minx / bw);
        const keel::f64 mx = bw * keel::round(maxx / bw);
        appendColon(e, keel::floor(mn) - 0.5 * bw, bw, keel::ceil(mx) + 0.5 * bw);
    } else {
        e.push_back(minx);
        appendColon(e, bw * keel::ceil(minx / bw) + 0.5, bw, bw * keel::floor(maxx / bw) - 0.5);
        e.push_back(maxx);
    }
    return e;
}

// Apply a BinMethod to data `xc` (finite) with range [minx,maxx]. dmin/dmax are
// the finite-data extremes (== minx/maxx unless hardlimits restrict the data).
keel::Array<keel::f64> applyMethod(HistBinMethod m, keel::ArrayRef<keel::f64> xc,
                                keel::f64 minx, keel::f64 maxx, bool hard,
                                keel::f64 dmin, keel::f64 dmax,
                                keel::Allocator *alloc)
{
    const keel::f64 dataScale = std::max(keel::abs(dmin), keel::abs(dmax));
    const keel::f64 maxbins   = 65536.0;
    const keel::f64 n         = keel::as<keel::f64>(xc.size());

    HistBinMethod eff = m;
    if (m == HistBinMethod::Auto) {
        bool integerLike = !xc.empty() && (maxx - minx) <= 50.0
                        && maxx <= 4503599627370496.0 && minx >= -4503599627370496.0;
        if (integerLike)
            for (keel::f64 v : xc) if (keel::round(v) != v) { integerLike = false; break; }
        eff = integerLike ? HistBinMethod::Integers : HistBinMethod::Scott;
    }

    switch (eff) {
    case HistBinMethod::Integers:
        return integerrule(minx, maxx, dmin, dmax, dataScale, hard, maxbins, alloc);
    case HistBinMethod::Scott: {
        const keel::f64 bw = 3.5 * sampleStd(xc) / keel::pow(n, 1.0 / 3.0);
        return hard ? binpickerbl(dmin, dmax, minx, maxx, bw, alloc)
                    : binpicker(minx, maxx, false, 0.0, bw, alloc);
    }
    case HistBinMethod::Fd: {
        keel::f64 bw;
        if (xc.size() > 1) {
            const keel::f64 iq = std::max(iqrMatlab(xc, alloc), (dmax - dmin) / 10.0);
            bw = 2.0 * iq * keel::pow(n, -1.0 / 3.0);
        } else {
            bw = 1.0;
        }
        return hard ? binpickerbl(dmin, dmax, minx, maxx, bw, alloc)
                    : binpicker(minx, maxx, false, 0.0, bw, alloc);
    }
    case HistBinMethod::Sturges: {
        const keel::i64 nb = std::max(1L, keel::as<keel::i64>(keel::ceil(keel::log2(n) + 1.0)));
        if (hard) return linspaceEdges(minx, maxx, nb, alloc);
        const keel::f64 bw = (maxx - minx) / keel::as<keel::f64>(nb);
        return keel::is_finite(bw) ? binpicker(minx, maxx, false, 0.0, bw, alloc)
                                 : binpicker(minx, maxx, true, keel::as<keel::f64>(nb), bw, alloc);
    }
    case HistBinMethod::Sqrt: {
        const keel::i64 nb = std::max(1L, keel::as<keel::i64>(keel::ceil(keel::sqrt(n))));
        if (hard) return linspaceEdges(minx, maxx, nb, alloc);
        const keel::f64 bw = (maxx - minx) / keel::as<keel::f64>(nb);
        return keel::is_finite(bw) ? binpicker(minx, maxx, false, 0.0, bw, alloc)
                                 : binpicker(minx, maxx, true, keel::as<keel::f64>(nb), bw, alloc);
    }
    default:
        return binpicker(minx, maxx, false, 0.0, 0.0, alloc);
    }
}

} // namespace

Value histcountsAutoEdges(const Value &x, const HistBinSpec &spec,
                          keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Allocator *alloc = scratch.allocator();

    // Finite-data extremes (MATLAB excludes Inf/NaN from the range).
    const keel::usize nx = x.numel();
    keel::Array<keel::f64> xc(alloc);
    xc.reserve(nx);
    bool any = false;
    keel::f64 dmin = 0.0, dmax = 0.0;
    for (keel::usize k = 0; k < nx; ++k) {
        const keel::f64 v = x.elemAsDouble(k);
        if (!keel::is_finite(v)) continue;
        xc.push_back(v);
        if (!any) { dmin = dmax = v; any = true; }
        else { if (v < dmin) dmin = v; if (v > dmax) dmax = v; }
    }

    keel::Array<keel::f64> edges(alloc);

    if (spec.hasBinLimits) {
        const keel::f64 lo = spec.limLo, hi = spec.limHi;
        if (spec.hasNumBins) {
            const keel::i64 n = std::max(1L, keel::as<keel::i64>(spec.numBins));
            edges.resize(keel::as<keel::usize>(n) + 1);
            const keel::f64 w = (hi - lo) / keel::as<keel::f64>(n);
            for (keel::i64 i = 0; i < n; ++i) edges[i] = lo + keel::as<keel::f64>(i) * w;
            edges[n] = hi;
        } else if (spec.hasBinWidth) {
            const keel::f64 bw = std::max(spec.binWidth, (hi - lo) / 65536.0);
            appendColon(edges, lo, bw, hi);
            if (edges.empty() || edges.back() < hi || edges.size() == 1) edges.push_back(hi);
        } else {
            keel::Array<keel::f64> xr(alloc);
            for (keel::f64 v : xc) if (v >= lo && v <= hi) xr.push_back(v);
            keel::f64 rmin = lo, rmax = hi;
            if (!xr.empty()) {
                rmin = *std::min_element(xr.begin(), xr.end());
                rmax = *std::max_element(xr.begin(), xr.end());
            }
            edges = applyMethod(spec.method, xr, lo, hi, true, rmin, rmax, alloc);
        }
    } else if (spec.hasNumBins) {
        const keel::i64 n = std::max(1L, keel::as<keel::i64>(spec.numBins));
        if (!any) { edges.resize(keel::as<keel::usize>(n) + 1);
                    for (keel::i64 i = 0; i <= n; ++i) edges[i] = keel::as<keel::f64>(i); }
        else edges = binpicker(dmin, dmax, true, keel::as<keel::f64>(n),
                               (dmax - dmin) / keel::as<keel::f64>(n), alloc);
    } else if (spec.hasBinWidth) {
        const keel::f64 bw = spec.binWidth;
        if (!any) { edges = keel::Array<keel::f64>({0.0, bw}, alloc); }
        else {
            const keel::f64 xrange = dmax - dmin;
            keel::f64 binWidth = bw;
            keel::f64 leftEdge = binWidth * keel::floor(dmin / binWidth);
            keel::i64 nbins = std::max(1L, keel::as<keel::i64>(keel::ceil((dmax - leftEdge) / binWidth)));
            const keel::f64 maxbins = 65536.0;
            if (keel::as<keel::f64>(nbins) > maxbins) {
                nbins = keel::as<keel::i64>(maxbins);
                binWidth = xrange / (maxbins - 1.0);
                leftEdge = binWidth * keel::floor(dmin / binWidth);
                if (dmax <= leftEdge + (nbins - 1) * binWidth) {
                    binWidth = xrange / maxbins;
                    leftEdge = dmin;
                }
            }
            edges.resize(keel::as<keel::usize>(nbins) + 1);
            for (keel::i64 i = 0; i <= nbins; ++i) edges[i] = leftEdge + keel::as<keel::f64>(i) * binWidth;
            if (edges.back() < dmax) edges.back() = dmax;
        }
    } else {  // BinMethod (default Auto)
        if (!any)
            edges = (spec.method == HistBinMethod::Integers)
                        ? keel::Array<keel::f64>({-0.5, 0.5}, alloc)
                        : keel::Array<keel::f64>({0.0, 1.0}, alloc);
        else
            edges = applyMethod(spec.method, xc, dmin, dmax, false, dmin, dmax, alloc);
    }

    Value e = Value::matrix(1, edges.size(), ValueType::DOUBLE, mr);
    std::copy(edges.begin(), edges.end(), e.doubleDataMut());
    return e;
}

Value discretize(const Value &x, const Value &edges, keel::Allocator *mr)
{
    validateEdges(edges, "discretize");
    auto r = createLike(x, ValueType::DOUBLE, mr);
    const keel::f64 *e = edges.doubleData();
    const keel::f64 *p = x.doubleData();
    keel::f64 *dst = r.doubleDataMut();
    const keel::usize n = x.numel();
    const keel::usize nBins = edges.numel() - 1;

    keel::f64 step;
    if (edgesAreUniform(e, edges.numel(), step)) {
        const keel::f64 e0 = e[0];
        const keel::f64 eN = e[nBins];
        const keel::f64 invStep = 1.0 / step;
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 v = p[i];
            if (v >= e0 && v <= eN) {
                keel::usize bin;
                if (v == eN) {
                    bin = nBins - 1;
                } else {
                    bin = keel::as<keel::usize>((v - e0) * invStep);
                    if (bin >= nBins) bin = nBins - 1;
                }
                dst[i] = keel::as<keel::f64>(bin + 1); // 1-based
            } else {
                dst[i] = keel::kNaN;
            }
        }
        return r;
    }

    const keel::f64 e0 = e[0];
    const keel::f64 eN = e[nBins];
    const keel::usize nE = edges.numel();
    if (nBins <= 8) {
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 v = p[i];
            if (!(v >= e0 && v <= eN)) { dst[i] = keel::kNaN; continue; }
            if (v == eN) { dst[i] = keel::as<keel::f64>(nBins); continue; }
            keel::usize k = 0;
            while (k + 1 < nBins && e[k + 1] <= v) ++k;
            dst[i] = keel::as<keel::f64>(k + 1); // 1-based
        }
    } else {
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 v = p[i];
            if (!(v >= e0 && v <= eN)) { dst[i] = keel::kNaN; continue; }
            if (v == eN) { dst[i] = keel::as<keel::f64>(nBins); continue; }
            keel::usize lo = 0, hi = nE;
            while (lo < hi) {
                const keel::usize mid = lo + (hi - lo) / 2;
                if (e[mid] <= v) lo = mid + 1;
                else hi = mid;
            }
            dst[i] = keel::as<keel::f64>(lo); // (lo-1)+1
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
    const keel::f64 *e = edges.doubleData();

    const keel::usize rows = keel::as<keel::usize>(x.dims().rows());
    const keel::usize cols = x.numel() == 0 ? 0 : x.numel() / std::max<keel::usize>(rows, 1);
    const keel::f64 *p = x.doubleData();

    // Row vector -> count along the row, return a 1 × nE row. Otherwise
    // (column vector / matrix) count each column -> nE × ncols.
    if (rows == 1) {
        Value r = Value::matrix(1, nE, ValueType::DOUBLE, mr);
        keel::f64 *dst = r.doubleDataMut();
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
    keel::f64 *dst = r.doubleDataMut();
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
    return Value::scalar(keel::as<keel::f64>(seen.size() + nanCount), alloc);
}


Value ismembertol(const Value &a, const Value &s, keel::f64 tol, keel::Allocator *mr)
{
    // Returns logical of size(a). For each a[i], true if there exists
    // s[j] with |a[i] - s[j]| ≤ tol * max(1, |a|, |s|). Naive O(|a||s|).
    auto r = createLike(a, ValueType::LOGICAL, mr);
    const keel::usize na = a.numel(), ns = s.numel();
    const keel::f64 *pa = a.doubleData();
    const keel::f64 *ps = s.doubleData();
    for (keel::usize i = 0; i < na; ++i) {
        bool hit = false;
        for (keel::usize j = 0; j < ns; ++j) {
            if (nearlyEqualTol(pa[i], ps[j], tol)) { hit = true; break; }
        }
        r.logicalDataMut()[i] = hit ? 1 : 0;
    }
    return r;
}

Value uniquetol(const Value &x, keel::f64 tol, keel::Allocator *mr)
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
    auto vals = keel::Array<keel::f64>(scratch.allocator());
    vals.reserve(n);
    const keel::f64 *p = x.doubleData();
    keel::f64 dataScale = 0.0;
    for (keel::usize i = 0; i < n; ++i) {
        vals.push_back(p[i]);
        if (!keel::is_nan(p[i])) {
            const keel::f64 m = keel::abs(p[i]);
            if (m > dataScale) dataScale = m;
        }
    }
    std::sort(vals.begin(), vals.end(),
              [](keel::f64 a, keel::f64 b) {
                  if (keel::is_nan(b)) return !keel::is_nan(a);
                  if (keel::is_nan(a)) return false;
                  return a < b;
              });
    const keel::f64 tolAbs = tol * dataScale;
    auto out = keel::Array<keel::f64>(scratch.allocator());
    out.reserve(n);
    for (keel::f64 v : vals) {
        if (keel::is_nan(v)) {
            // NaNs are unique to themselves -- always emit.
            out.push_back(v);
            continue;
        }
        if (out.empty() || keel::is_nan(out.back())
            || keel::abs(v - out.back()) > tolAbs) {
            out.push_back(v);
        }
    }
    return rowFromVec(out.data(), out.size(), mr);
}

} // namespace numkit::builtin
