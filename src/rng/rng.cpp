// ops/src/rng.cpp
// Value-producing random generators + RngContext rng() control. Every generator
// draws from a caller-provided RngContext (the Engine owns one; engine.rng()) —
// no process-global stream, no mutex. A session shares ONE reproducible stream
// (MATLAB `rng(seed)`); two Engines are independent. Bit-identical with MATLAB
// R2025b's rng()+rand()+randn()+randperm() under 'twister' and legacy v4.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/hash_map.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/math.h>
#include <keel/random.h>
#include <keel/string.h>
#include <keel/time.h>
#include <numkit/ops/rng.hpp>

#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>

namespace numkit::ops {

// ────────────────────────────────────────────────────────────────────
// RngContext state control (seed lives inline in the header)
// ────────────────────────────────────────────────────────────────────

namespace {

// Serialise / restore MT19937 state as a text blob: "mt19937 <624 words> <idx>".
keel::String serializeState(const MatlabMT19937 &gen)
{
    keel::u32 state[MatlabMT19937::STATE_SIZE];
    int idx;
    gen.getState(state, idx);
    keel::String out;
    out.reserve(MatlabMT19937::STATE_SIZE * 11 + 24);
    out.append("mt19937");
    for (keel::usize i = 0; i < MatlabMT19937::STATE_SIZE; ++i)
        keel::format_into(out, " {}", state[i]);
    keel::format_into(out, " {}", idx);
    return out;
}

void deserializeState(MatlabMT19937 &gen, keel::StringRef blob)
{
    const char *p = blob.data();
    const char *end = p + blob.size();
    auto isSpace = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    auto nextToken = [&]() -> keel::StringRef {
        while (p < end && isSpace(*p)) ++p;
        const char *start = p;
        while (p < end && !isSpace(*p)) ++p;
        return keel::StringRef(start, static_cast<keel::usize>(p - start));
    };
    auto parseU32 = [](keel::StringRef tok, keel::u32 &out) -> bool {
        if (tok.empty()) return false;
        keel::usize pos = 0;
        if (tok[0] == '+') {
            pos = 1;
            if (pos == tok.size()) return false;
        }
        keel::u64 acc = 0;
        for (; pos < tok.size(); ++pos) {
            const char c = tok[pos];
            if (c < '0' || c > '9') return false;
            acc = acc * 10ULL + static_cast<keel::u64>(c - '0');
            if (acc > 0xFFFFFFFFULL) return false;
        }
        out = static_cast<keel::u32>(acc);
        return true;
    };
    auto parseInt = [](keel::StringRef tok, int &out) -> bool {
        if (tok.empty()) return false;
        keel::usize pos = 0;
        bool neg = false;
        if (tok[0] == '-' || tok[0] == '+') {
            neg = (tok[0] == '-');
            pos = 1;
            if (pos == tok.size()) return false;
        }
        keel::u64 acc = 0;
        for (; pos < tok.size(); ++pos) {
            const char c = tok[pos];
            if (c < '0' || c > '9') return false;
            acc = acc * 10ULL + static_cast<keel::u64>(c - '0');
            if (acc > 0x7FFFFFFFULL) return false;
        }
        out = neg ? -static_cast<int>(acc) : static_cast<int>(acc);
        return true;
    };

    if (nextToken() != "mt19937")
        throw Error("rng: malformed state blob",
                     0, 0, "rng", "", "numkit:rng:badState");
    keel::u32 state[MatlabMT19937::STATE_SIZE];
    for (keel::usize i = 0; i < MatlabMT19937::STATE_SIZE; ++i) {
        if (!parseU32(nextToken(), state[i]))
            throw Error("rng: malformed state blob",
                         0, 0, "rng", "", "numkit:rng:badState");
    }
    int idx = 0;
    if (!parseInt(nextToken(), idx) || idx < 0 || idx > static_cast<int>(MatlabMT19937::STATE_SIZE))
        throw Error("rng: malformed state blob",
                     0, 0, "rng", "", "numkit:rng:badState");
    bool anyNonZero = ((state[0] & 0x80000000UL) != 0);
    for (keel::usize i = 1; i < MatlabMT19937::STATE_SIZE && !anyNonZero; ++i)
        if (state[i] != 0)
            anyNonZero = true;
    if (!anyNonZero)
        throw Error("rng: invalid all-zero MT19937 state",
                     0, 0, "rng", "", "numkit:rng:badState");
    gen.setState(state, idx);
}

} // namespace

double RngContext::v5Normal()
{
    static constexpr double vt[65] = {
        0.3409450, 0.4573146, 0.5397793, 0.6062427, 0.6631691, 0.7136975, 0.7596125, 0.8020356,
        0.8417227, 0.8792102, 0.9148948, 0.9490791, 0.9820005, 1.0138492, 1.0447810, 1.0749254,
        1.1043917, 1.1332738, 1.1616530, 1.1896010, 1.2171815, 1.2444516, 1.2714635, 1.2982650,
        1.3249008, 1.3514125, 1.3778399, 1.4042211, 1.4305929, 1.4569915, 1.4834527, 1.5100122,
        1.5367061, 1.5635712, 1.5906454, 1.6179680, 1.6455802, 1.6735255, 1.7018503, 1.7306045,
        1.7598422, 1.7896223, 1.8200099, 1.8510770, 1.8829044, 1.9155831, 1.9492166, 1.9839239,
        2.0198431, 2.0571356, 2.0959930, 2.1366450, 2.1793713, 2.2245175, 2.2725186, 2.3239338,
        2.3795008, 2.4402218, 2.5075117, 2.5834658, 2.6713916, 2.7769942, 2.7769942, 2.7769942,
        2.7769942
    };
    const keel::u32 u32 = shr3Step();
    const keel::i32 hz = static_cast<keel::i32>(u32);
    const keel::u32 iz = u32 & 63u;
    const double x = static_cast<double>(hz) * vt[iz + 1] * 4.6566128730773926e-10;
    if (keel::abs(x) < vt[iz])
        return x;

    const double y = 0.4878992 * (vt[iz + 1] - keel::abs(x)) / (vt[iz + 1] - vt[iz]);
    const double u1 = 0.5 + static_cast<double>(static_cast<keel::i32>(shr3Step())) * 2.3283064365386963e-10;
    const double f0 = std::exp(-0.5 * vt[iz] * vt[iz]);
    const double f1 = std::exp(-0.5 * vt[iz + 1] * vt[iz + 1]);
    if (u1 < (std::exp(-0.5 * x * x) - f1) / (f0 - f1))
        return x;

    const double fv0 = std::exp(-0.5 * vt[0] * vt[0]);
    if ((1.0 - u1) < 12.37598359 * (std::exp(-0.5 * y * y) - fv0))
        return (x < 0.0) ? -y : y;

    for (;;) {
        const double u2 = 0.5 + static_cast<double>(static_cast<keel::i32>(shr3Step())) * 2.3283064365386963e-10;
        const double u3 = 0.5 + static_cast<double>(static_cast<keel::i32>(shr3Step())) * 2.3283064365386963e-10;
        const double tx = -std::log(u2) / 2.776994;
        if (-2.0 * std::log(u3) > tx * tx) {
            const double res = 2.776994 + tx;
            return (x < 0.0) ? -res : res;
        }
    }
}

double RngContext::nextNormal()
{
    if (legacyMode_)
        return legacyV4Randn_ ? v4Normal() : v5Normal();

    for (;;) {
        const keel::u32 w0 = gen_.genUint32();
        const keel::u32 w1 = gen_.genUint32();
        const keel::u32 k = (w1 >> 24) & 0xFFu;
        const keel::i64 u53 =
            static_cast<keel::i64>((static_cast<keel::u64>(w0 >> 3) << 24)
                                      | static_cast<keel::u64>(w1 & 0x00FFFFFFu))
            - (1LL << 52);
        const double u = static_cast<double>(u53) * (1.0 / 4503599627370496.0);
        const double z = u * keel::random_detail::kZigguratX[k + 1];
        if (keel::abs(z) < keel::random_detail::kZigguratX[k])
            return z;

        if (k == 255u) {
            constexpr double r = 0x1.d3bb48209ad36p+1; // kZigguratX[255]
            constexpr double invR = 1.0 / r;
            for (;;) {
                const double tx = -std::log(gen_.genRes53()) * invR;
                const double ty = -std::log(gen_.genRes53());
                if (ty + ty >= tx * tx)
                    return (u53 >= 0) ? (r + tx) : -(r + tx);
            }
        } else {
            if ((keel::random_detail::kZigguratF[k] - keel::random_detail::kZigguratF[k + 1]) * gen_.genRes53()
                < std::exp(-0.5 * z * z) - keel::random_detail::kZigguratF[k + 1]) {
                return z;
            }
        }
    }
}

void RngContext::shuffle()
{
    legacyMode_ = false;
    legacyV4Rand_ = false;
    legacyV5Rand_ = false;
    legacyV4Randn_ = false;
    const auto r = keel::os_random_u32();
    gen_.seed(r.is_ok() ? r.value() : static_cast<keel::u32>(keel::monotonic_ns()));
}

Value RngContext::state(keel::Allocator *mr) const
{
    auto blob = serializeState(gen_);
    auto s = Value::structure();
    s.field("Type")  = Value::fromString("twister", mr);
    s.field("State") = Value::fromString(blob, mr);
    return s;
}

void RngContext::restore(const Value &state)
{
    if (!state.isStruct())
        throw Error("rng: state must be a struct from rng()",
                     0, 0, "rng", "", "numkit:rng:notStruct");
    if (!state.hasField("State"))
        throw Error("rng: state struct missing .State field",
                     0, 0, "rng", "", "numkit:rng:noStateField");
    const auto &blob = state.field("State");
    if (!blob.isChar() && !blob.isString())
        throw Error("rng: .State must be a char array",
                     0, 0, "rng", "", "numkit:rng:badState");
    deserializeState(gen_, blob.toString());
    legacyMode_ = false;
    legacyV4Rand_ = false;
    legacyV5Rand_ = false;
    legacyV4Randn_ = false;
}

// ────────────────────────────────────────────────────────────────────
// Real-valued random (uniform / standard-normal)
// ────────────────────────────────────────────────────────────────────

Value rand(RngContext &rng, keel::usize rows, keel::usize cols, keel::usize pages, keel::Allocator *mr)
{
    auto m = (pages > 0) ? Value::matrix3d(rows, cols, pages, ValueType::DOUBLE, mr)
                         : Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    double *dst = m.doubleDataMut();
    const keel::usize n = m.numel();
    for (keel::usize i = 0; i < n; ++i)
        dst[i] = rng.nextUniform();
    return m;
}

Value randn(RngContext &rng, keel::usize rows, keel::usize cols, keel::usize pages, keel::Allocator *mr)
{
    auto m = (pages > 0) ? Value::matrix3d(rows, cols, pages, ValueType::DOUBLE, mr)
                         : Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    double *dst = m.doubleDataMut();
    const keel::usize n = m.numel();
    for (keel::usize i = 0; i < n; ++i)
        dst[i] = rng.nextNormal();
    return m;
}

Value randND(RngContext &rng, keel::ArrayRef<keel::usize> dims, keel::Allocator *mr)
{
    auto m = Value::matrixND(dims.data(), static_cast<int>(dims.size()), ValueType::DOUBLE, mr);
    double *dst = m.doubleDataMut();
    const keel::usize n = m.numel();
    for (keel::usize i = 0; i < n; ++i)
        dst[i] = rng.nextUniform();
    return m;
}

Value randnND(RngContext &rng, keel::ArrayRef<keel::usize> dims, keel::Allocator *mr)
{
    auto m = Value::matrixND(dims.data(), static_cast<int>(dims.size()), ValueType::DOUBLE, mr);
    double *dst = m.doubleDataMut();
    const keel::usize n = m.numel();
    for (keel::usize i = 0; i < n; ++i)
        dst[i] = rng.nextNormal();
    return m;
}

// ────────────────────────────────────────────────────────────────────
// Integer random
// ────────────────────────────────────────────────────────────────────

namespace {

void fillUniformInt(RngContext &rng, double *dst, keel::usize n, keel::i64 lo, keel::i64 hi)
{
    if (lo > hi)
        throw Error("randi: low bound must be <= high bound",
                     0, 0, "randi", "", "numkit:randi:badRange");
    const keel::u64 uSpan =
        static_cast<keel::u64>(hi) - static_cast<keel::u64>(lo) + 1ULL;
    const double span = (uSpan == 0ULL) ? 18446744073709551616.0
                                        : static_cast<double>(uSpan);
    for (keel::usize i = 0; i < n; ++i) {
        const double u = rng.nextUniform();
        dst[i] = static_cast<double>(lo) + std::floor(u * span);
    }
}

Value makeIntMatrix(RngContext &rng, keel::i64 lo, keel::i64 hi, keel::usize rows, keel::usize cols,
                    keel::usize pages, keel::Allocator *mr)
{
    auto m = (pages > 0) ? Value::matrix3d(rows, cols, pages, ValueType::DOUBLE, mr)
                         : Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    fillUniformInt(rng, m.doubleDataMut(), m.numel(), lo, hi);
    return m;
}

} // namespace

Value randi(RngContext &rng, keel::i64 imax, keel::Allocator *mr)
{
    double v = 0.0;
    fillUniformInt(rng, &v, 1, 1, imax);
    return Value::scalar(v, mr);
}

Value randi(RngContext &rng, keel::i64 imax, keel::usize rows, keel::usize cols, keel::usize pages, keel::Allocator *mr)
{
    return makeIntMatrix(rng, 1, imax, rows, cols, pages, mr);
}

Value randi(RngContext &rng, keel::i64 imin, keel::i64 imax, keel::usize rows, keel::usize cols, keel::usize pages, keel::Allocator *mr)
{
    return makeIntMatrix(rng, imin, imax, rows, cols, pages, mr);
}

// ────────────────────────────────────────────────────────────────────
// Permutations
// ────────────────────────────────────────────────────────────────────
// 1-arg randperm(n)    : [~, p] = sort(rand(1, n)) — matches MATLAB's 1-arg form.
// 2-arg randperm(n, k) : inside-out / partial shuffle using nextUniform().

Value randperm(RngContext &rng, keel::usize n, keel::Allocator *mr)
{
    auto r = Value::matrix(1, n, ValueType::DOUBLE, mr);
    if (n == 0) return r;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    auto u = keel::Array<double>(n, scratch.allocator());
    auto idx = keel::Array<keel::usize>(n, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i) {
        u[i] = rng.nextUniform();
        idx[i] = i;
    }
    std::stable_sort(idx.begin(), idx.end(), [&](keel::usize a, keel::usize b) {
        return u[a] < u[b];
    });

    double *dst = r.doubleDataMut();
    for (keel::usize i = 0; i < n; ++i)
        dst[i] = static_cast<double>(idx[i] + 1);
    return r;
}

Value randperm(RngContext &rng, keel::usize n, keel::usize k, keel::Allocator *mr)
{
    if (k > n)
        throw Error("randperm: k must not exceed n",
                     0, 0, "randperm", "", "numkit:randperm:badK");
    auto r = Value::matrix(1, k, ValueType::DOUBLE, mr);
    if (k == 0) return r;

    double *dst = r.doubleDataMut();
    if (k == n) {
        dst[0] = 1.0;
        for (keel::usize i = 1; i < n; ++i) {
            const keel::usize j = static_cast<keel::usize>(
                std::floor(rng.nextUniform() * static_cast<double>(i + 1)));
            if (j != i)
                dst[i] = dst[j];
            dst[j] = static_cast<double>(i + 1);
        }
        return r;
    }

    if (4 * k < n) {
        keel::HashMap<keel::usize, keel::usize> H;
        H.reserve(k * 2);
        for (keel::usize i = 0; i < k; ++i) {
            const keel::usize m = n - i;
            const keel::usize j = 1 + static_cast<keel::usize>(
                std::floor(rng.nextUniform() * static_cast<double>(m)));
            const keel::usize *pJ = H.get(j);
            const keel::usize valJ = pJ ? *pJ : j;
            const keel::usize *pM = H.get(m);
            const keel::usize valM = pM ? *pM : m;
            dst[i] = static_cast<double>(valJ);
            H.put(j, valM);
        }
        return r;
    }

    keel::usize t = 0;
    double nrem = static_cast<double>(n);
    double krem = static_cast<double>(k);
    for (keel::usize i = 0; i < k; ++i) {
        const double v = 1.0 - rng.nextUniform();
        keel::usize s = 0;
        double quot = (nrem - krem) / nrem;
        while (quot > v) {
            ++s;
            nrem -= 1.0;
            quot *= (nrem - krem) / nrem;
        }
        t += s + 1;
        nrem -= 1.0;
        krem -= 1.0;
        const keel::usize j = static_cast<keel::usize>(
            std::floor(rng.nextUniform() * static_cast<double>(i + 1)));
        if (j != i)
            dst[i] = dst[j];
        dst[j] = static_cast<double>(t);
    }
    return r;
}

} // namespace numkit::ops
