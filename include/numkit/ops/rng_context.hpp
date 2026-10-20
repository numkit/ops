// ops/include/nk/ops/rng_context.hpp
//
// RngContext — the seedable random-number session state. Owns one MATLAB-style
// MT19937 stream and the rng() control surface (seed / shuffle / state save +
// restore). The Engine owns one RngContext (engine.rng()); every RNG-using
// function (rand/randn/randi/randperm + the toolbox *rnd samplers, comm noise,
// …) takes `RngContext &` and draws from it — so a single session shares ONE
// reproducible stream (MATLAB `rng(seed)` semantics) WITHOUT a process-global
// singleton or a mutex. Two Engines have independent streams.
//
// It models UniformRandomBitGenerator (forwards operator()/min/max/result_type
// to the owned MT19937), so the std distributions draw from it directly:
//   std::normal_distribution<double> d; double x = d(rng);
// An L0.5 ops type (Value + MatlabMT19937 only, no Engine).

#pragma once

#include <numkit/ops/matlab_mt19937.hpp>
#include <numkit/value/value.hpp>

#include <cmath>    // v4Normal: std::sqrt / std::log
#include <cstdint>
#include <cstring>
#include <memory_resource>
#include <random>   // RngContext models a URBG — callers draw via std::*_distribution(rng)

namespace numkit::ops {

class RngContext
{
public:
    // ── UniformRandomBitGenerator surface (forwarded to the owned stream) ──
    // Lets std::*_distribution and the value generators draw via `dist(rng)`.
    using result_type = MatlabMT19937::result_type;
    static constexpr result_type min() { return MatlabMT19937::min(); }
    static constexpr result_type max() { return MatlabMT19937::max(); }
    result_type operator()() { return gen_(); }

    /// MATLAB-canonical 53-bit double in (0, 1) — what `rand` uses.
    double genRes53() { return gen_.genRes53(); }

    /// Session-aware uniform in (0, 1): v4Uniform() or v5Uniform() in legacy mode, else genRes53().
    double nextUniform()
    {
        if (legacyMode_) {
            if (legacyV4Rand_) return v4Uniform();
            if (legacyV5Rand_) return v5Uniform();
        }
        return gen_.genRes53();
    }

    /// Session-aware standard normal N(0, 1): v4Normal() or v5Normal() in legacy mode, else
    /// MATLAB-exact 256-layer Marsaglia-Tsang Ziggurat on the MT19937 stream.
    double nextNormal();

    // ── MATLAB rng() control ──────────────────────────────────────────
    /// Seed the stream. `seed == 0` ≡ `rng('default')`.
    void seed(std::uint64_t seed)
    {
        legacyMode_ = false;
        legacyV4Rand_ = false;
        legacyV5Rand_ = false;
        legacyV4Randn_ = false;
        gen_.seed(static_cast<std::uint32_t>(seed));
    }
    /// `rng('shuffle')` — seed from a nondeterministic source.
    void shuffle();
    /// `s = rng()` — capture the stream state as a struct (.Type / .State).
    Value state(std::pmr::memory_resource *mr = nullptr) const;
    /// `rng(s)` — restore a state struct captured by state().
    void restore(const Value &s);

    // ── Legacy v4 & v5 generators (rand/randn('seed'|'state'|'twister', S)) ──
    // In MATLAB legacy mode, `rand` and `randn` maintain independent active
    // generators:
    //   - `rand` defaults to `'twister'` (MT19937) unless `rand('seed', s)`
    //     switches it to `v4Uniform()` (Park-Miller `16807*x mod (2^31-1)`
    //     on `v4RandState_`).
    //   - `randn` defaults to `'state'` (`shr3cong` 64-layer Ziggurat on
    //     `(v5Icng_, v5Jsr_)`) unless `randn('seed', s)` switches it to
    //     `v4Normal()` (polar Box-Muller on `v4RandnState_`).
    void enterLegacyMode()
    {
        if (!legacyMode_) {
            setLegacyV5Rand(0);
            setLegacyV5Randn(0);
        }
    }
    void setLegacyV4Rand(std::uint64_t seed)
    {
        enterLegacyMode();
        legacyMode_ = true;
        legacyV4Rand_ = true;
        legacyV5Rand_ = false;
        v4RandSeed_ = seed;
        v4RandState_ = seed ? seed * 65536ull : 1144108930ull;
    }
    void setLegacyV4Randn(std::uint64_t seed)
    {
        enterLegacyMode();
        legacyMode_ = true;
        legacyV4Randn_ = true;
        v4RandnSeed_ = seed;
        v4RandnState_ = seed ? seed * 65536ull : 1144108930ull;
    }
    void setLegacyV4(std::uint64_t seed)
    {
        setLegacyV4Rand(seed);
        setLegacyV4Randn(seed);
    }
    void setLegacyTwisterRand(std::uint64_t seed)
    {
        enterLegacyMode();
        legacyMode_ = true;
        legacyV4Rand_ = false;
        legacyV5Rand_ = false;
        gen_.seed(static_cast<std::uint32_t>(seed));
    }
    void setLegacyV5Rand(std::uint64_t seed)
    {
        const bool wasLegacy = legacyMode_;
        legacyMode_ = true;
        if (!wasLegacy) setLegacyV5Randn(0);
        legacyV4Rand_ = false;
        legacyV5Rand_ = true;
        std::uint32_t j = static_cast<std::uint32_t>(seed);
        if (j == 0u) j = 0x80000000u;
        v5SwbJ_ = j;
        for (std::size_t i = 0; i < 32; ++i) {
            double s = 0.0;
            double t = 0.5;
            for (int b = 0; b < 53; ++b) {
                j ^= (j << 13);
                j ^= (j >> 17);
                j ^= (j << 5);
                if ((j >> 19) & 1u)
                    s += t;
                t *= 0.5;
            }
            v5SwbZ_[i] = s;
        }
        v5SwbIdx_ = 0;
        v5SwbBorrow_ = 0.0;
    }
    void setLegacyV5Randn(std::uint64_t seed)
    {
        const bool wasLegacy = legacyMode_;
        legacyMode_ = true;
        if (!wasLegacy) setLegacyV5Rand(0);
        legacyV4Randn_ = false;
        v5Icng_ = 362436069u;
        const std::uint32_t s32 = static_cast<std::uint32_t>(seed);
        v5Jsr_ = (s32 == 0u) ? 521288629u : s32;
    }
    void setLegacyV5RandnState(std::uint32_t icng, std::uint32_t jsr)
    {
        const bool wasLegacy = legacyMode_;
        legacyMode_ = true;
        if (!wasLegacy) setLegacyV5Rand(0);
        legacyV4Randn_ = false;
        v5Icng_ = icng;
        v5Jsr_ = (jsr == 0u) ? 521288629u : jsr;
    }
    /// `rand('seed')` / `randn('seed')` — query the last legacy seed.
    std::uint64_t legacySeed() const noexcept { return v4RandSeed_; }
    std::uint64_t legacyRandSeed() const noexcept { return v4RandSeed_; }
    std::uint64_t legacyRandnSeed() const noexcept { return v4RandnSeed_; }
    std::uint32_t v5RandnIcng() const noexcept { return v5Icng_; }
    std::uint32_t v5RandnJsr() const noexcept { return v5Jsr_; }
    bool legacyV4() const noexcept { return legacyMode_ && (legacyV4Rand_ || legacyV4Randn_); }
    double v4Uniform()
    {
        v4RandState_ = 16807ull * v4RandState_ % 2147483647ull;
        return static_cast<double>(v4RandState_) / 2147483647.0;
    }
    double v5Uniform()
    {
        const std::size_t i = v5SwbIdx_;
        double x = v5SwbZ_[(i + 20) & 31] - v5SwbZ_[(i + 5) & 31] - v5SwbBorrow_;
        if (x < 0.0) {
            x += 1.0;
            v5SwbBorrow_ = 1.1102230246251565e-16; // 2^-53
        } else {
            v5SwbBorrow_ = 0.0;
        }
        v5SwbZ_[i] = x;
        v5SwbIdx_ = (i + 1) & 31;
        const std::uint32_t jPrev = v5SwbJ_;
        std::uint32_t j = jPrev;
        j ^= (j << 13);
        j ^= (j >> 17);
        j ^= (j << 5);
        v5SwbJ_ = j;
        std::uint64_t xb = 0;
        std::memcpy(&xb, &x, sizeof(xb));
        const std::uint64_t mask = (static_cast<std::uint64_t>(j & 0xFFFFFu) << 32)
                                 | static_cast<std::uint64_t>(jPrev);
        xb ^= mask;
        double r = 0.0;
        std::memcpy(&r, &xb, sizeof(r));
        return r;
    }
    const double *v5SwbZ() const noexcept { return v5SwbZ_; }
    std::size_t v5SwbIdx() const noexcept { return v5SwbIdx_; }
    double v5SwbBorrow() const noexcept { return v5SwbBorrow_; }
    std::uint32_t v5SwbJ() const noexcept { return v5SwbJ_; }
    double v4RandnUniform()
    {
        v4RandnState_ = 16807ull * v4RandnState_ % 2147483647ull;
        return static_cast<double>(v4RandnState_) / 2147483647.0;
    }
    double v4Normal()
    {
        for (;;) {
            const double v1 = 2.0 * v4RandnUniform() - 1.0;
            const double v2 = 2.0 * v4RandnUniform() - 1.0;
            const double s = v1 * v1 + v2 * v2;
            if (s > 0.0 && s < 1.0)
                return v1 * std::sqrt(-2.0 * std::log(s) / s);
        }
    }
    double v5Normal();

    /// Escape hatch: the underlying MT19937 (e.g. for getState/setState).
    MatlabMT19937 &engine() noexcept { return gen_; }
    const MatlabMT19937 &engine() const noexcept { return gen_; }

private:
    std::uint32_t shr3Step() noexcept
    {
        v5Icng_ = 69069u * v5Icng_ + 1234567u;
        v5Jsr_ ^= (v5Jsr_ << 13);
        v5Jsr_ ^= (v5Jsr_ >> 17);
        v5Jsr_ ^= (v5Jsr_ << 5);
        return v5Icng_ + v5Jsr_;
    }

    MatlabMT19937 gen_;   // default-constructed = rng('default')
    bool legacyMode_ = false;
    bool legacyV4Rand_ = false;
    bool legacyV5Rand_ = false;
    bool legacyV4Randn_ = false;
    double v5SwbZ_[32] = {};
    std::size_t v5SwbIdx_ = 0;
    double v5SwbBorrow_ = 0.0;
    std::uint32_t v5SwbJ_ = 0x80000000u;
    std::uint64_t v4RandSeed_ = 0;
    std::uint64_t v4RandState_ = 1144108930ull;
    std::uint64_t v4RandnSeed_ = 0;
    std::uint64_t v4RandnState_ = 1144108930ull;
    std::uint32_t v5Icng_ = 362436069u;
    std::uint32_t v5Jsr_ = 521288629u;
};

} // namespace numkit::ops
