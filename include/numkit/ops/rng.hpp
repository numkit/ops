// ops/include/numkit/ops/rng.hpp
//
// Value-producing random generators (rand / randn / randi / randperm). Every
// one draws from a caller-provided RngContext — the seedable session stream the
// Engine owns (engine.rng()); there is NO process-global RNG and NO mutex. A
// single session shares one reproducible stream (MATLAB `rng(seed)` semantics);
// two Engines are independent. RngContext models UniformRandomBitGenerator, so
// the std distributions here draw from it directly (`dist(rng)`).
//
// An L0.5 ops primitive (Value + RngContext, no Engine). The user-facing
// rand/randn/randi/randperm/rng *builtins* (the CallContext wrappers that parse
// shape/type args) live in bundle and call these with engine.rng().

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>

#include <numkit/ops/rng_context.hpp>

#include <cstdint>

namespace numkit::ops {

// ── Real-valued random (uniform / standard-normal) ───────────────────
// `pages == 0` → 2-D matrix. Output is DOUBLE.
Value rand(RngContext &rng, keel::usize rows, keel::usize cols = 1,
           keel::usize pages = 0, keel::Allocator *mr = nullptr);
Value randn(RngContext &rng, keel::usize rows, keel::usize cols = 1,
            keel::usize pages = 0, keel::Allocator *mr = nullptr);
Value randND(RngContext &rng, keel::ArrayRef<keel::usize> dims,
             keel::Allocator *mr = nullptr);
Value randnND(RngContext &rng, keel::ArrayRef<keel::usize> dims,
              keel::Allocator *mr = nullptr);

// ── Integer random ───────────────────────────────────────────────────
Value randi(RngContext &rng, keel::i64 imax, keel::Allocator *mr = nullptr);
Value randi(RngContext &rng, keel::i64 imax, keel::usize rows, keel::usize cols,
            keel::usize pages = 0, keel::Allocator *mr = nullptr);
Value randi(RngContext &rng, keel::i64 imin, keel::i64 imax, keel::usize rows, keel::usize cols,
            keel::usize pages = 0, keel::Allocator *mr = nullptr);

// ── Permutations ─────────────────────────────────────────────────────
Value randperm(RngContext &rng, keel::usize n, keel::Allocator *mr = nullptr);
Value randperm(RngContext &rng, keel::usize n, keel::usize k,
               keel::Allocator *mr = nullptr);

// rng() control (seed / shuffle / state save+restore) lives on RngContext
// itself — see rng_context.hpp.

} // namespace numkit::ops
