// toolboxes/image/src/io/bmp_codec.hpp
//
// In-tree Windows Bitmap (.bmp) decoder and encoder. Zero external dependencies.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <cstddef>
#include <cstdint>
#include <numkit/value/value.hpp>
#include <string>

namespace numkit::image {

/// Decode a BMP image from an in-memory byte buffer.
/// @return HÃ—W uint8 (grayscale) or HÃ—WÃ—3/4 uint8 (truecolor RGB/RGBA).
Value readBmp(const keel::u8 *data, keel::usize len,
              keel::Allocator *mr = nullptr);

/// Encode an image Value to BMP format bytes.
/// @param A Image (HÃ—W grayscale, HÃ—WÃ—3 RGB, or HÃ—WÃ—4 RGBA uint8).
std::string writeBmpToBytes(const Value &A);

/// Inspect a BMP header without decoding all pixels.
bool peekBmp(const keel::u8 *data, keel::usize len,
             keel::u32 &W, keel::u32 &H,
             keel::u16 &bitsPerSample, keel::u16 &channels);

} // namespace numkit::image
