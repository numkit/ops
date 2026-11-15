// toolboxes/image/src/io/tiff_codec.hpp
//
// In-tree TIFF and BigTIFF reader, writer, and peeker.
// Zero-dependency pure C++20 implementation.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <cstdint>
#include <numkit/value/value.hpp>
#include <utility>
#include <vector>

namespace numkit::image {

/// True if the byte buffer carries a TIFF magic header (II* / MM*, classic or BigTIFF).
bool isTiffBytes(keel::StringRef b);
bool isTiffBytes(const keel::u8 *data, keel::usize len);

/// Read a TIFF image (single page or specific page).
Value readTiff(keel::StringRef path, keel::Allocator *mr);
Value readTiff(keel::StringRef path, keel::u32 page, keel::Allocator *mr);
Value readTiff(std::vector<keel::u8> buf, keel::u32 page, keel::Allocator *mr);

/// Two-output form `[A, map] = readTiffWithMap(...)` for indexed/palette TIFFs.
std::pair<Value, Value> readTiffWithMap(keel::StringRef path, keel::u32 page, keel::Allocator *mr);
std::pair<Value, Value> readTiffWithMap(std::vector<keel::u8> buf, keel::u32 page, keel::Allocator *mr);

/// Peek TIFF first-IFD metadata without decoding raster data.
void peekTiff(keel::StringRef path, keel::u32 &W, keel::u32 &H, keel::u16 &bits, keel::u16 &channels);
void peekTiff(const std::vector<keel::u8> &buf, keel::u32 &W, keel::u32 &H, keel::u16 &bits, keel::u16 &channels);

/// Count number of IFDs (pages) in a TIFF file.
keel::u32 tiffNumPages(keel::StringRef path);

/// Write an image to disk or buffer as a TIFF.
void writeTiff(const Value &A, keel::StringRef path, keel::StringRef compression, bool appendMode);
std::vector<keel::u8> writeTiffToBytes(const Value &A, keel::StringRef compression, const std::vector<keel::u8> *existing);

} // namespace numkit::image
