// lang/include/numkit/lang/strings/scan_core.hpp
#pragma once

// Shared scan parse+shape core, exposed so the engine-coupled fscanf shim
// (relocated to the runtime layer in C6c-2b) can reuse the same machinery as
// the pure sscanf (which stays here in lang). scanfEmit itself is pure — no
// Engine — so it lives in the core-free lang layer; only the file-reading
// wrapper needed the engine, and that wrapper now lives in runtime.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>

#include <cstddef>
#include <string>

// SizeSpec is the read-size descriptor from the shared io_helpers infra
// (numkit::ops). Passed by const-ref here so this header only
// needs a forward declaration — the definition (io_helpers.hpp) is pulled in
// by the .cpp callers, not by every includer of this header.
namespace numkit::ops {
struct SizeSpec;
}

namespace numkit::builtin::detail {

// scanfCycle's run summary — matched-item count + input bytes consumed.
struct ScanfOut
{
    keel::usize count;
    keel::usize bytesConsumed;
};

// Common fscanf/sscanf body once the input buffer has been materialised:
// parse `input` per `fmt` (cyclically, bounded by `sz`), write the shaped
// result into outs[0] and (when nargout > 1) the count into outs[1].
void scanfEmit(const std::string &input, const std::string &fmt,
               const ::numkit::ops::SizeSpec &sz, keel::usize nargout,
               keel::MutArrayRef<Value> outs, keel::Allocator *mr, ScanfOut &r);

} // namespace numkit::builtin::detail
