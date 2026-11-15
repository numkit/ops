// toolboxes/image/src/io/pnm_codec.hpp
//
// In-tree Netpbm (.pnm, .ppm, .pgm, .pbm) decoder and encoder. Zero external dependencies.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <cstddef>
#include <cstdint>
#include <numkit/value/value.hpp>
#include <string>

namespace numkit::image {

/// Decode a PNM/PGM/PPM/PBM image from an in-memory byte buffer.
/// @return HÃ—W uint8/uint16 (grayscale) or HÃ—WÃ—3 uint8/uint16 (RGB).
Value readPnm(const keel::u8 *data, keel::usize len,
              keel::Allocator *mr = nullptr);

/// Encode an image Value to PNM (PGM/PPM) format bytes.
/// @param A Image (HÃ—W grayscale or HÃ—WÃ—3 RGB uint8/uint16).
std::string writePnmToBytes(const Value &A, const std::string &ext = "ppm");

/// Inspect a PNM header without decoding all pixels.
bool peekPnm(const keel::u8 *data, keel::usize len,
             keel::u32 &W, keel::u32 &H,
             keel::u16 &bitsPerSample, keel::u16 &channels);

} // namespace numkit::image
