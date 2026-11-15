// toolboxes/image/src/io/jpeg_codec.hpp
//
// In-tree Baseline JPEG (.jpg, .jpeg) decoder and encoder.
// Zero external dependencies.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <cstddef>
#include <cstdint>
#include <numkit/value/value.hpp>
#include <string>

namespace numkit::image {

/// Decode a JPEG image from an in-memory byte buffer.
/// @return HÃ—W uint8 (grayscale) or HÃ—WÃ—3 uint8 (truecolor RGB).
Value readJpeg(const keel::u8 *data, keel::usize len,
               keel::Allocator *mr = nullptr);

/// Encode an image Value to JPEG format bytes.
/// @param A        Image (uint8; HÃ—W grayscale or HÃ—WÃ—3 RGB).
/// @param quality  Quality level (1..100, default 90).
std::string writeJpegToBytes(const Value &A, int quality = 90);

/// Inspect a JPEG header without decoding pixels.
bool peekJpeg(const keel::u8 *data, keel::usize len,
              keel::u32 &W, keel::u32 &H,
              keel::u16 &bitsPerSample, keel::u16 &channels);

} // namespace numkit::image
