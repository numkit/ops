// ops/include/numkit/ops/rng_context.hpp
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

#include <keel/core.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/ops/matlab_mt19937.hpp>
#include <numkit/value/value.hpp>


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
    keel::f64 genRes53() { return gen_.genRes53(); }

    /// Session-aware uniform in (0, 1): v4Uniform() or v5Uniform() in legacy mode, else genRes53().
    keel::f64 nextUniform()
    {
        if (legacyMode_) {
            if (legacyV4Rand_) return v4Uniform();
            if (legacyV5Rand_) return v5Uniform();
        }
        return gen_.genRes53();
    }

    /// Session-aware standard normal N(0, 1): v4Normal() or v5Normal() in legacy mode, else
    /// MATLAB-exact 256-layer Marsaglia-Tsang Ziggurat on the MT19937 stream.
    keel::f64 nextNormal();

    // ── MATLAB rng() control ──────────────────────────────────────────
    /// Seed the stream. `seed == 0` ≡ `rng('default')`.
    void seed(keel::u64 seed)
    {
        legacyMode_ = false;
        legacyV4Rand_ = false;
        legacyV5Rand_ = false;
        legacyV4Randn_ = false;
        gen_.seed(keel::as<keel::u32>(seed));
    }
    /// `rng('shuffle')` — seed from a nondeterministic source.
    void shuffle();
    /// `s = rng()` — capture the stream state as a struct (.Type / .State).
    Value state(keel::Allocator *mr = nullptr) const;
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
    void setLegacyV4Rand(keel::u64 seed)
    {
        enterLegacyMode();
        legacyMode_ = true;
        legacyV4Rand_ = true;
        legacyV5Rand_ = false;
        v4RandSeed_ = seed;
        v4RandState_ = seed ? seed * 65536ull : 1144108930ull;
    }
    void setLegacyV4Randn(keel::u64 seed)
    {
        enterLegacyMode();
        legacyMode_ = true;
        legacyV4Randn_ = true;
        v4RandnSeed_ = seed;
        v4RandnState_ = seed ? seed * 65536ull : 1144108930ull;
    }
    void setLegacyV4(keel::u64 seed)
    {
        setLegacyV4Rand(seed);
        setLegacyV4Randn(seed);
    }
    void setLegacyTwisterRand(keel::u64 seed)
    {
        enterLegacyMode();
        legacyMode_ = true;
        legacyV4Rand_ = false;
        legacyV5Rand_ = false;
        gen_.seed(keel::as<keel::u32>(seed));
    }
    void setLegacyTwisterRandState(const keel::u32 state[MatlabMT19937::STATE_SIZE], keel::i32 idx)
    {
        enterLegacyMode();
        legacyMode_ = true;
        legacyV4Rand_ = false;
        legacyV5Rand_ = false;
        gen_.setState(state, idx);
    }
    void setLegacyV5Rand(keel::u64 seed)
    {
        const bool wasLegacy = legacyMode_;
        legacyMode_ = true;
        if (!wasLegacy) setLegacyV5Randn(0);
        legacyV4Rand_ = false;
        legacyV5Rand_ = true;
        keel::u32 j = keel::as<keel::u32>(seed);
        if (j == 0u) j = 0x80000000u;
        v5SwbJ_ = j;
        for (keel::usize i = 0; i < 32; ++i) {
            keel::f64 s = 0.0;
            keel::f64 t = 0.5;
            for (keel::i32 b = 0; b < 53; ++b) {
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
    void setLegacyV5RandState(const keel::f64 z[32], keel::usize idx, keel::f64 borrow, keel::u32 j)
    {
        const bool wasLegacy = legacyMode_;
        legacyMode_ = true;
        if (!wasLegacy) setLegacyV5Randn(0);
        legacyV4Rand_ = false;
        legacyV5Rand_ = true;
        for (keel::usize i = 0; i < 32; ++i)
            v5SwbZ_[i] = z[i];
        v5SwbIdx_ = idx & 31u;
        v5SwbBorrow_ = borrow;
        v5SwbJ_ = (j == 0u) ? 0x80000000u : j;
    }
    void setLegacyV5Randn(keel::u64 seed)
    {
        const bool wasLegacy = legacyMode_;
        legacyMode_ = true;
        if (!wasLegacy) setLegacyV5Rand(0);
        legacyV4Randn_ = false;
        v5Icng_ = 362436069u;
        const keel::u32 s32 = keel::as<keel::u32>(seed);
        v5Jsr_ = (s32 == 0u) ? 521288629u : s32;
    }
    void setLegacyV5RandnState(keel::u32 icng, keel::u32 jsr)
    {
        const bool wasLegacy = legacyMode_;
        legacyMode_ = true;
        if (!wasLegacy) setLegacyV5Rand(0);
        legacyV4Randn_ = false;
        v5Icng_ = icng;
        v5Jsr_ = (jsr == 0u) ? 521288629u : jsr;
    }
    /// `rand('seed')` / `randn('seed')` — query the last legacy seed.
    keel::u64 legacySeed() const noexcept { return v4RandSeed_; }
    keel::u64 legacyRandSeed() const noexcept { return v4RandSeed_; }
    keel::u64 legacyRandnSeed() const noexcept { return v4RandnSeed_; }
    keel::u32 v5RandnIcng() const noexcept { return v5Icng_; }
    keel::u32 v5RandnJsr() const noexcept { return v5Jsr_; }
    bool legacyV4() const noexcept { return legacyMode_ && (legacyV4Rand_ || legacyV4Randn_); }
    keel::f64 v4Uniform()
    {
        v4RandState_ = 16807ull * v4RandState_ % 2147483647ull;
        return keel::as<keel::f64>(v4RandState_) / 2147483647.0;
    }
    keel::f64 v5Uniform()
    {
        const keel::usize i = v5SwbIdx_;
        keel::f64 x = v5SwbZ_[(i + 20) & 31] - v5SwbZ_[(i + 5) & 31] - v5SwbBorrow_;
        if (x < 0.0) {
            x += 1.0;
            v5SwbBorrow_ = 1.1102230246251565e-16; // 2^-53
        } else {
            v5SwbBorrow_ = 0.0;
        }
        v5SwbZ_[i] = x;
        v5SwbIdx_ = (i + 1) & 31;
        const keel::u32 jPrev = v5SwbJ_;
        keel::u32 j = jPrev;
        j ^= (j << 13);
        j ^= (j >> 17);
        j ^= (j << 5);
        v5SwbJ_ = j;
        keel::u64 xb = keel::bit_cast<keel::u64>(x);
        const keel::u64 mask = (keel::as<keel::u64>(j & 0xFFFFFu) << 32)
                                 | keel::as<keel::u64>(jPrev);
        xb ^= mask;
        return keel::bit_cast<keel::f64>(xb);
    }
    const keel::f64 *v5SwbZ() const noexcept { return v5SwbZ_; }
    keel::usize v5SwbIdx() const noexcept { return v5SwbIdx_; }
    keel::f64 v5SwbBorrow() const noexcept { return v5SwbBorrow_; }
    keel::u32 v5SwbJ() const noexcept { return v5SwbJ_; }
    keel::f64 v4RandnUniform()
    {
        v4RandnState_ = 16807ull * v4RandnState_ % 2147483647ull;
        return keel::as<keel::f64>(v4RandnState_) / 2147483647.0;
    }
    keel::f64 v4Normal()
    {
        for (;;) {
            const keel::f64 v1 = 2.0 * v4RandnUniform() - 1.0;
            const keel::f64 v2 = 2.0 * v4RandnUniform() - 1.0;
            const keel::f64 s = v1 * v1 + v2 * v2;
            if (s > 0.0 && s < 1.0)
                return v1 * keel::sqrt(-2.0 * keel::log(s) / s);
        }
    }
    keel::f64 v5Normal();

    /// Escape hatch: the underlying MT19937 (e.g. for getState/setState).
    MatlabMT19937 &engine() noexcept { return gen_; }
    const MatlabMT19937 &engine() const noexcept { return gen_; }

private:
    keel::u32 shr3Step() noexcept
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
    keel::f64 v5SwbZ_[32] = {};
    keel::usize v5SwbIdx_ = 0;
    keel::f64 v5SwbBorrow_ = 0.0;
    keel::u32 v5SwbJ_ = 0x80000000u;
    keel::u64 v4RandSeed_ = 0;
    keel::u64 v4RandState_ = 1144108930ull;
    keel::u64 v4RandnSeed_ = 0;
    keel::u64 v4RandnState_ = 1144108930ull;
    keel::u32 v5Icng_ = 362436069u;
    keel::u32 v5Jsr_ = 521288629u;
};

} // namespace numkit::ops
