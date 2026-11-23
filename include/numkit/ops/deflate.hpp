// ops/include/numkit/ops/deflate.hpp
//
// In-tree RFC 1951 Deflate/Inflate, RFC 1950 ZLIB wrapper, and
// CRC-32 / Adler-32 checksums. Zero external dependencies.
//
// Shared compression primitives for image (PNG, TIFF) and runtime (MAT v7).

#pragma once

#include <keel/array.h>
#include <keel/core.h>

namespace numkit::ops {

/// Compute IEEE 802.3 CRC-32 (used by PNG chunks).
keel::u32 crc32(const keel::u8 *data, keel::usize len);

/// Compute Adler-32 checksum (used by ZLIB RFC 1950).
keel::u32 adler32(const keel::u8 *data, keel::usize len);

/// Decompress raw RFC 1951 Deflate bitstream.
/// @param src               Compressed bytes.
/// @param len               Length of compressed bytes.
/// @param expectedSizeHint  Optional hint for output vector reservation.
/// @param maxOutputBytes    Maximum allowed uncompressed size (0 = unlimited).
/// @return                  Decompressed byte vector.
/// @throws                  numkit::Error on corrupted / truncated / oversized stream.
keel::Array<keel::u8> inflateRaw(const keel::u8 *src, keel::usize len,
                                 keel::usize expectedSizeHint = 0,
                                 keel::usize maxOutputBytes = 512ULL * 1024 * 1024,
                                 keel::Allocator *alloc = nullptr);

/// Compress uncompressed bytes using raw RFC 1951 Deflate.
/// @param src    Input bytes.
/// @param len    Length of input bytes.
/// @param level  Compression level: 0 = store uncompressed, 1..9 = LZ77 + Huffman.
/// @return       Compressed Deflate bitstream.
keel::Array<keel::u8> deflateRaw(const keel::u8 *src, keel::usize len,
                                 keel::i32 level = 6,
                                 keel::Allocator *alloc = nullptr);

/// Decompress RFC 1950 ZLIB stream (2-byte header + Deflate stream + 4-byte Adler-32).
/// @param src               ZLIB stream bytes.
/// @param len               Length of stream bytes.
/// @param expectedSizeHint  Optional output size hint.
/// @param maxOutputBytes    Maximum allowed uncompressed size (0 = unlimited).
/// @return                  Decompressed payload bytes.
/// @throws                  numkit::Error on bad header, corrupted data, oversized output, or Adler-32 mismatch.
keel::Array<keel::u8> zlibDecompress(const keel::u8 *src, keel::usize len,
                                     keel::usize expectedSizeHint = 0,
                                     keel::usize maxOutputBytes = 512ULL * 1024 * 1024,
                                     keel::Allocator *alloc = nullptr);

/// Compress payload into an RFC 1950 ZLIB stream.
/// @param src    Input payload.
/// @param len    Length of payload.
/// @param level  Compression level (0..9).
/// @return       Complete ZLIB stream (header + Deflate + Adler-32).
keel::Array<keel::u8> zlibCompress(const keel::u8 *src, keel::usize len,
                                   keel::i32 level = 6,
                                   keel::Allocator *alloc = nullptr);

} // namespace numkit::ops
