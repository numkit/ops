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
#include <vector>

namespace numkit::ops {

namespace {

std::vector<keel::u8> toStdVec(const keel::Array<keel::u8> &arr) {
    return std::vector<keel::u8>(arr.data(), arr.data() + arr.size());
}

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
    return keel::crc32(toRef(data, len));
}

keel::u32 adler32(const keel::u8 *data, keel::usize len) {
    return keel::adler32(toRef(data, len));
}

std::vector<keel::u8> inflateRaw(const keel::u8 *src, keel::usize len,
                                     keel::usize expectedSizeHint,
                                     keel::usize maxOutputBytes) {
    keel::InflateOptions opts;
    opts.expected_size_hint = expectedSizeHint;
    opts.max_output_bytes = maxOutputBytes;

    auto res = keel::inflate_raw(toRef(src, len), opts);
    if (!res) {
        throwInflateError(res.error());
    }
    return toStdVec(res.value());
}

std::vector<keel::u8> deflateRaw(const keel::u8 *src, keel::usize len, int level) {
    auto res = keel::deflate_raw(toRef(src, len), level);
    return toStdVec(res);
}

std::vector<keel::u8> zlibDecompress(const keel::u8 *src, keel::usize len,
                                         keel::usize expectedSizeHint,
                                         keel::usize maxOutputBytes) {
    if (len < 6) {
        throw Error("zlib: stream too short (< 6 bytes)",
                    0, 0, "zlibDecompress", "", "numkit:zlib:truncated");
    }

    const keel::u8 cmf = src[0];
    const keel::u8 flg = src[1];
    if (((static_cast<keel::u32>(cmf) << 8) + flg) % 31 != 0) {
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

    keel::InflateOptions opts;
    opts.expected_size_hint = expectedSizeHint;
    opts.max_output_bytes = maxOutputBytes;

    auto inflated = keel::inflate_raw(toRef(src + 2, len - 6), opts);
    if (!inflated) {
        throwInflateError(inflated.error());
    }

    const auto &payload = inflated.value();
    const keel::u32 expectedAdler =
        (static_cast<keel::u32>(src[len - 4]) << 24) |
        (static_cast<keel::u32>(src[len - 3]) << 16) |
        (static_cast<keel::u32>(src[len - 2]) << 8)  |
        static_cast<keel::u32>(src[len - 1]);
    const keel::u32 actualAdler = keel::adler32(payload.as_ref());
    if (expectedAdler != actualAdler) {
        throw Error("zlib: Adler-32 checksum mismatch",
                    0, 0, "zlibDecompress", "", "numkit:zlib:adlerMismatch");
    }

    return toStdVec(payload);
}

std::vector<keel::u8> zlibCompress(const keel::u8 *src, keel::usize len, int level) {
    auto res = keel::zlib_compress(toRef(src, len), level);
    return toStdVec(res);
}

} // namespace numkit::ops
