// toolboxes/image/src/io/tga_codec.hpp
//
// In-tree Truevision TGA (.tga) decoder and encoder. Zero external dependencies.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/string.h>
#include <cstddef>
#include <cstdint>
#include <numkit/value/value.hpp>

namespace numkit::image {

/// Decode a TGA image from an in-memory byte buffer.
/// @return HÃ—W uint8 (grayscale) or HÃ—WÃ—3/4 uint8 (truecolor RGB/RGBA).
Value readTga(const keel::u8 *data, keel::usize len,
              keel::Allocator *mr = nullptr);

/// Encode an image Value to TGA format bytes.
/// @param A Image (HÃ—W grayscale, HÃ—WÃ—3 RGB, or HÃ—WÃ—4 RGBA uint8).
keel::String writeTgaToBytes(const Value &A);

/// Inspect a TGA header without decoding all pixels.
bool peekTga(const keel::u8 *data, keel::usize len,
             keel::u32 &W, keel::u32 &H,
             keel::u16 &bitsPerSample, keel::u16 &channels);

} // namespace numkit::image
