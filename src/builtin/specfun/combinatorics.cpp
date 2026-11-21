// src/builtin/src/specfun/combinatorics.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/specfun.h>
#include <numkit/builtin/specfun.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/helpers.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>

#include "discrete_detail.hpp"

namespace numkit::builtin {

// ════════════════════════════════════════════════════════════════════════
// Number theory
// ════════════════════════════════════════════════════════════════════════


Value primes(double n, keel::Allocator *mr)
{
    if (!std::isfinite(n) || n < 2)
        return Value::matrix(1, 0, ValueType::DOUBLE, mr);
    const keel::u64 N = static_cast<keel::u64>(std::floor(n));
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u64> ps = keel::primes_up_to(N, scratch.allocator());

    auto out = Value::matrix(1, ps.size(), ValueType::DOUBLE, mr);
    double *dst = out.doubleDataMut();
    for (keel::usize i = 0; i < ps.size(); ++i)
        dst[i] = static_cast<double>(ps[i]);
    return out;
}

Value isprime(const Value &x, keel::Allocator *mr)
{
    if (x.type() == ValueType::COMPLEX)
        throw Error("isprime: complex inputs are not supported",
                     0, 0, "isprime", "", "numkit:isprime:complex");
    auto out = createLike(x, ValueType::LOGICAL, mr);
    keel::u8 *dst = out.logicalDataMut();
    const keel::usize N = x.numel();
    for (keel::usize i = 0; i < N; ++i) {
        keel::u64 u = 0;
        dst[i] = (isExactNonnegInt(x.elemAsDouble(i), u) && keel::is_prime_u64(u)) ? 1 : 0;
    }
    return out;
}

Value factor(double n, keel::Allocator *mr)
{
    keel::u64 u;
    if (!isExactNonnegInt(n, u))
        throw Error("factor: argument must be a non-negative integer scalar",
                     0, 0, "factor", "", "numkit:factor:badArg");
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u64> factors = keel::factor_u64(u, scratch.allocator());

    auto out = Value::matrix(1, factors.size(), ValueType::DOUBLE, mr);
    double *dst = out.doubleDataMut();
    for (keel::usize i = 0; i < factors.size(); ++i)
        dst[i] = static_cast<double>(factors[i]);
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Combinatorics
// ════════════════════════════════════════════════════════════════════════


Value perms(const Value &v, keel::Allocator *mr)
{
    if (v.type() == ValueType::COMPLEX)
        throw Error("perms: complex inputs are not supported",
                     0, 0, "perms", "", "numkit:perms:complex");
    if (v.isEmpty()) {
        return Value::matrix(1, 0, ValueType::DOUBLE, mr);
    }
    if (!v.dims().isVector())
        throw Error("perms: argument must be a vector",
                     0, 0, "perms", "", "numkit:perms:notVector");

    const keel::usize n = v.numel();
    if (n > static_cast<keel::usize>(kPermMaxN))
        throw Error("perms: numel(v) > 11 is not supported (n! is too large)",
                     0, 0, "perms", "", "numkit:perms:tooLarge");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto vals = keel::Array<double>(n, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i)
        vals[i] = v.elemAsDouble(i);

    keel::Array<double> cur(vals, scratch.allocator());
    std::sort(cur.begin(), cur.end(), [](double a, double b) { return a > b; });

    const keel::usize totalRows = static_cast<keel::usize>(permFactorial(static_cast<int>(n)));
    auto out = Value::matrix(totalRows, n, ValueType::DOUBLE, mr);
    double *dst = out.doubleDataMut();

    keel::usize row = 0;
    do {
        for (keel::usize c = 0; c < n; ++c)
            dst[c * totalRows + row] = cur[c];
        ++row;
    } while (std::prev_permutation(cur.begin(), cur.end()));

    return out;
}

Value factorial(const Value &n, keel::Allocator *mr)
{
    if (n.type() == ValueType::COMPLEX)
        throw Error("factorial: complex inputs are not supported",
                     0, 0, "factorial", "", "numkit:factorial:complex");
    auto out = createLike(n, ValueType::DOUBLE, mr);
    double *dst = out.doubleDataMut();
    const keel::usize N = n.numel();
    for (keel::usize i = 0; i < N; ++i)
        dst[i] = factorialDouble(n.elemAsDouble(i), "factorial");
    return out;
}

Value nchoosek(double n, double k, keel::Allocator *mr)
{
    if (!std::isfinite(n) || !std::isfinite(k))
        throw Error("nchoosek: arguments must be finite",
                     0, 0, "nchoosek", "", "numkit:nchoosek:badArg");
    if (n < 0 || k < 0 || n != std::floor(n) || k != std::floor(k))
        throw Error("nchoosek: arguments must be non-negative integers",
                     0, 0, "nchoosek", "", "numkit:nchoosek:badArg");
    if (k > n)
        throw Error("nchoosek: k must satisfy 0 ≤ k ≤ n",
                     0, 0, "nchoosek", "", "numkit:nchoosek:kTooLarge");

    return Value::scalar(keel::nchoosek(static_cast<keel::u64>(n), static_cast<keel::u64>(k)), mr);
}

// nchoosek(v, k) where v is a vector: all k-combinations of the elements of v,
// one per ROW, in lexicographic order of element indices (MATLAB R2025b):
// nchoosek([1 2 3 4],2) = [1 2;1 3;1 4;2 3;2 4;3 4]. k==0 -> 1x0; k==numel ->
// a single row of all elements. Result is DOUBLE (numeric input flattened).
Value nchoosekCombinations(const Value &v, double kd, keel::Allocator *mr)
{
    const keel::usize n = v.numel();
    if (!std::isfinite(kd) || kd < 0 || kd != std::floor(kd))
        throw Error("nchoosek: K must be a non-negative integer",
                     0, 0, "nchoosek", "", "numkit:nchoosek:badArg");
    const keel::usize k = static_cast<keel::usize>(kd);
    if (k > n)
        throw Error("nchoosek: K must satisfy 0 <= K <= numel(V)",
                     0, 0, "nchoosek", "", "numkit:nchoosek:kTooLarge");
    if (k == 0)
        return Value::matrix(1, 0, ValueType::DOUBLE, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto vd = keel::Array<double>(n, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) vd[i] = v.elemAsDouble(i);

    // Row count R = C(n,k) (computed with the symmetric product to limit error).
    const keel::usize kk = (k > n - k) ? n - k : k;
    double rd = 1.0;
    for (keel::usize i = 0; i < kk; ++i)
        rd = rd * static_cast<double>(n - i) / static_cast<double>(i + 1);
    const keel::usize R = static_cast<keel::usize>(std::round(rd));

    auto out = Value::matrix(R, k, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    auto idx = keel::Array<keel::usize>(k, scratch.allocator());
    for (keel::usize i = 0; i < k; ++i) idx[i] = i;
    keel::usize row = 0;
    while (true) {
        for (keel::usize j = 0; j < k; ++j) od[j * R + row] = vd[idx[j]];
        ++row;
        // Advance to the next lexicographic combination of indices.
        keel::usize i = k;
        while (i-- > 0) {
            if (idx[i] != n - k + i) {
                ++idx[i];
                for (keel::usize j = i + 1; j < k; ++j) idx[j] = idx[j - 1] + 1;
                break;
            }
            if (i == 0) { row = R; }   // exhausted (sentinel: stop outer loop)
        }
        if (row >= R) break;
    }
    return out;
}


Value colperm(const Value &s, keel::Allocator *mr)
{
    const auto &d = s.dims();
    if (d.ndim() > 2)
        throw Error("colperm: input must be 2-D", 0, 0, "colperm", "", "numkit:colperm:rank");
    const keel::usize rows = d.rows();
    const keel::usize cols = d.cols();
    if (cols == 0)
        return Value::matrix(1, 0, ValueType::DOUBLE, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    struct ColCount {
        keel::usize colIndex;
        keel::usize nnz;
    };
    auto counts = keel::Array<ColCount>(scratch.allocator());
    counts.reserve(cols);

    for (keel::usize c = 0; c < cols; ++c) {
        keel::usize nonzeros = 0;
        for (keel::usize r = 0; r < rows; ++r) {
            const keel::usize idx = c * rows + r;
            if (s.type() == ValueType::COMPLEX) {
                const auto val = s.complexData()[idx];
                if (val.real() != 0.0 || val.imag() != 0.0) ++nonzeros;
            } else {
                if (s.elemAsDouble(idx) != 0.0) ++nonzeros;
            }
        }
        counts.push_back({ c + 1, nonzeros });
    }

    std::stable_sort(counts.begin(), counts.end(), [](const ColCount &a, const ColCount &b) {
        return a.nnz < b.nnz;
    });

    auto out = Value::matrix(1, cols, ValueType::DOUBLE, mr);
    double *p = out.doubleDataMut();
    for (keel::usize i = 0; i < cols; ++i)
        p[i] = static_cast<double>(counts[i].colIndex);
    return out;
}

Value symrcm(const Value &s, keel::Allocator *mr)
{
    const auto &d = s.dims();
    if (d.ndim() != 2 || d.rows() != d.cols())
        throw Error("symrcm: S must be a square 2-D matrix", 0, 0, "symrcm", "", "numkit:symrcm:notSquare");
    const keel::usize n = d.rows();
    if (n == 0)
        return Value::matrix(1, 0, ValueType::DOUBLE, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::Array<keel::usize>> adj(scratch.allocator());
    adj.reserve(n);
    for (keel::usize i = 0; i < n; ++i)
        adj.push_back(keel::Array<keel::usize>(scratch.allocator()));

    for (keel::usize i = 0; i < n; ++i) {
        for (keel::usize j = 0; j < n; ++j) {
            if (i == j) continue;
            bool nonz = false;
            if (s.type() == ValueType::COMPLEX) {
                auto v1 = s.complexData()[j * n + i];
                auto v2 = s.complexData()[i * n + j];
                nonz = (v1.real() != 0.0 || v1.imag() != 0.0 || v2.real() != 0.0 || v2.imag() != 0.0);
            } else {
                nonz = (s.elemAsDouble(j * n + i) != 0.0 || s.elemAsDouble(i * n + j) != 0.0);
            }
            if (nonz)
                adj[i].push_back(j);
        }
    }

    // Sort adj[i] by degree (ascending), tie-break by node index
    for (keel::usize i = 0; i < n; ++i) {
        std::sort(adj[i].begin(), adj[i].end(), [&](keel::usize u, keel::usize v) {
            if (adj[u].size() != adj[v].size())
                return adj[u].size() < adj[v].size();
            return u < v;
        });
    }

    auto visited = keel::Array<keel::u8>(n, 0, scratch.allocator());
    auto order = keel::Array<keel::usize>(scratch.allocator());
    order.reserve(n);

    for (keel::usize startCand = 0; startCand < n; ++startCand) {
        if (visited[startCand]) continue;

        keel::usize start = startCand;
        for (keel::usize i = startCand + 1; i < n; ++i) {
            if (!visited[i] && adj[i].size() < adj[start].size()) {
                start = i;
            }
        }

        auto comp = keel::Array<keel::usize>(scratch.allocator());
        auto queue = keel::Array<keel::usize>(scratch.allocator());
        keel::usize head = 0;

        visited[start] = 1;
        queue.push_back(start);
        comp.push_back(start);

        while (head < queue.size()) {
            keel::usize u = queue[head++];
            for (keel::usize v : adj[u]) {
                if (!visited[v]) {
                    visited[v] = 1;
                    queue.push_back(v);
                    comp.push_back(v);
                }
            }
        }

        // Reverse the component BFS order
        std::reverse(comp.begin(), comp.end());
        for (keel::usize node : comp)
            order.push_back(node);
    }

    auto out = Value::matrix(1, n, ValueType::DOUBLE, mr);
    double *p = out.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i)
        p[i] = static_cast<double>(order[i] + 1);
    return out;
}

} // namespace numkit::builtin
