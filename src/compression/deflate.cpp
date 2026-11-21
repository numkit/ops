// ops/src/compression/deflate.cpp
//
// Thin PRIVATE adapter bridging numkit::ops compression primitives to keel::compress.

#include <keel/core.h>
#include <keel/ref.h>
#include <keel/array.h>
#include <numkit/ops/deflate.hpp>
#include <numkit/value/error.hpp>

#include "keel/compress.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace numkit::ops {

namespace {

keel::ArrayRef<keel::u8> toRef(const keel::u8 *data, keel::usize len) {
    return keel::ArrayRef<keel::u8>(data, len);
}

[[noreturn]] void throwInflateError(const keel::Error &err) {
    switch (err.code) {
        case keel::ErrorCode::EndOfFile:
            throw Error("deflate: unexpected end of stream",
                        0, 0, "inflateRaw", "", "numkit:deflate:truncated");
        case keel::ErrorCode::OutOfRange:
            throw Error("deflate: decompressed output exceeds maximum allowed size",
                        0, 0, "inflateRaw", "", "numkit:deflate:outputTooLarge");
        default:
            throw Error("deflate: invalid or corrupted Deflate bitstream",
                        0, 0, "inflateRaw", "", "numkit:deflate:corrupted");
    }
}

} // anonymous namespace

keel::u32 crc32(const keel::u8 *data, keel::usize len) {
    return keel::compress::crc32(toRef(data, len));
}

keel::u32 adler32(const keel::u8 *data, keel::usize len) {
    return keel::compress::adler32(toRef(data, len));
}

keel::Array<keel::u8> inflateRaw(const keel::u8 *src, keel::usize len,
                                 keel::usize expectedSizeHint,
                                 keel::usize maxOutputBytes,
                                 keel::Allocator *alloc) {
    keel::compress::InflateOptions opts;
    opts.expected_size_hint = expectedSizeHint;
    opts.max_output_bytes = maxOutputBytes;
    opts.allocator = alloc;

    auto res = keel::compress::inflate_raw(toRef(src, len), opts);
    if (!res) {
        throwInflateError(res.error());
    }
    return std::move(res.value());
}

keel::Array<keel::u8> deflateRaw(const keel::u8 *src, keel::usize len, int level,
                                 keel::Allocator *alloc) {
    return keel::compress::deflate_raw(toRef(src, len), level, alloc);
}

keel::Array<keel::u8> zlibDecompress(const keel::u8 *src, keel::usize len,
                                     keel::usize expectedSizeHint,
                                     keel::usize maxOutputBytes,
                                     keel::Allocator *alloc) {
    if (len < 6) {
        throw Error("zlib: stream too short (< 6 bytes)",
                    0, 0, "zlibDecompress", "", "numkit:zlib:truncated");
    }

    const keel::u8 cmf = src[0];
    const keel::u8 flg = src[1];
    if (((keel::as<keel::u32>(cmf) << 8) + flg) % 31 != 0) {
        throw Error("zlib: invalid header checksum",
                    0, 0, "zlibDecompress", "", "numkit:zlib:badHeader");
    }
    if ((cmf & 0x0F) != 8) {
        throw Error("zlib: unsupported compression method (only Deflate supported)",
                    0, 0, "zlibDecompress", "", "numkit:zlib:badMethod");
    }
    if ((flg & 0x20) != 0) {
        throw Error("zlib: preset dictionaries not supported",
                    0, 0, "zlibDecompress", "", "numkit:zlib:fdict");
    }

    keel::compress::InflateOptions opts;
    opts.expected_size_hint = expectedSizeHint;
    opts.max_output_bytes = maxOutputBytes;
    opts.allocator = alloc;

    auto inflated = keel::compress::inflate_raw(toRef(src + 2, len - 6), opts);
    if (!inflated) {
        throwInflateError(inflated.error());
    }

    auto &payload = inflated.value();
    const keel::u32 expectedAdler =
        (keel::as<keel::u32>(src[len - 4]) << 24) |
        (keel::as<keel::u32>(src[len - 3]) << 16) |
        (keel::as<keel::u32>(src[len - 2]) << 8)  |
        keel::as<keel::u32>(src[len - 1]);
    const keel::u32 actualAdler = keel::compress::adler32(payload.as_ref());
    if (expectedAdler != actualAdler) {
        throw Error("zlib: Adler-32 checksum mismatch",
                    0, 0, "zlibDecompress", "", "numkit:zlib:adlerMismatch");
    }

    return std::move(payload);
}

keel::Array<keel::u8> zlibCompress(const keel::u8 *src, keel::usize len, int level,
                                   keel::Allocator *alloc) {
    return keel::compress::zlib_compress(toRef(src, len), level, alloc);
}

} // namespace numkit::ops
