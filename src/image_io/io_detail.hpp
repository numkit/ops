// toolboxes/image/src/io/io_detail.hpp
//
// Private (src-only) declarations shared between the engine-free compute in
// io.cpp and its CallContext register half in io_reg.cpp. NOT part of the
// public image API â€” both the native imread() and the VFS-backed imread_reg
// funnel pixel decoding through imreadFromBytes(), so it must be visible to
// both translation units.
//
// Phase 2b compute/register split â€” see project_layering_refactor memory.
#pragma once

#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/value/value.hpp>
namespace numkit::image {

// Lower-cased file extension (without the dot), or "" if none. Used by both
// the native and VFS imread/imwrite paths to dispatch on format.
inline keel::String lowerExt(keel::StringRef pathRef) {
    auto dot = pathRef.rfind('.');
    if (dot == keel::kInvalidIndex) return {};
    keel::StringRef tail = pathRef.substr(dot + 1);
    keel::String e;
    e.reserve(tail.size());
    for (keel::usize i = 0; i < tail.size(); ++i) {
        e.push_back(keel::to_ascii_lower(tail[i]));
    }
    return e;
}

// True if the byte buffer carries a TIFF magic header (II*. or MM.*, classic
// or BigTIFF).
bool isTiffBytes(keel::StringRef b);

// Decode an image from in-memory bytes â€” the single place pixels are produced.
Value imreadFromBytes(keel::StringRef bytes, keel::Allocator *mr);

} // namespace numkit::image
