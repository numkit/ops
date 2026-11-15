// toolboxes/image/src/io/png_codec.hpp
//
// In-tree Portable Network Graphics (.png) decoder and encoder.
// Full 8-bit and 16-bit support with zero external dependencies.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/string.h>
#include <cstddef>
#include <cstdint>
#include <numkit/value/value.hpp>
#include <utility>

namespace numkit::image {

/// Decode a PNG image from an in-memory byte buffer.
/// @return HÃ—W uint8/uint16 (gray), HÃ—WÃ—3 uint8/uint16 (RGB), or HÃ—WÃ—4 uint8/uint16 (RGBA).
Value readPng(const keel::u8 *data, keel::usize len,
              keel::Allocator *mr = nullptr);

/// Two-output form for palette PNG images (`[A, map] = imread(...)`).
/// If image is not indexed, colormap will be empty.
std::pair<Value, Value>
readPngWithMap(const keel::u8 *data, keel::usize len,
               keel::Allocator *mr = nullptr);

/// Encode an image Value to PNG format bytes.
/// @param A      Image (uint8/uint16; 1, 3, or 4 channels).
/// @param level  Compression level (0..9, default 6).
keel::String writePngToBytes(const Value &A, int level = 6);

/// Inspect a PNG header without decoding pixel data.
bool peekPng(const keel::u8 *data, keel::usize len,
             keel::u32 &W, keel::u32 &H,
             keel::u16 &bitsPerSample, keel::u16 &channels);

} // namespace numkit::image
