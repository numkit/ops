// ops/src/image_io/codecs_keel.cpp
//
// Thin PRIVATE adapters bridging NumKit's internal image codec headers
// (PNG, JPEG, BMP, TGA, PNM, TIFF) to keel::imagefile.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/array.h>
#include <keel/fs.h>
#include <keel/string.h>
#include "png_codec.hpp"
#include "jpeg_codec.hpp"
#include "bmp_codec.hpp"
#include "tga_codec.hpp"
#include "pnm_codec.hpp"
#include "tiff_codec.hpp"

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "keel/imagefile.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

namespace numkit::image {

namespace {

keel::ArrayRef<keel::u8> toRef(const keel::u8 *data, keel::usize len) {
    return keel::ArrayRef<keel::u8>(data, len);
}

keel::String toKeelString(const keel::Array<keel::u8> &arr) {
    return keel::String(keel::StringRef(reinterpret_cast<const char *>(arr.data()), arr.size()));
}

ValueType mapSampleType(keel::imagefile::SampleType st) {
    switch (st) {
        case keel::imagefile::SampleType::U8:  return ValueType::UINT8;
        case keel::imagefile::SampleType::U16: return ValueType::UINT16;
        case keel::imagefile::SampleType::U32: return ValueType::UINT32;
        case keel::imagefile::SampleType::I8:  return ValueType::INT8;
        case keel::imagefile::SampleType::I16: return ValueType::INT16;
        case keel::imagefile::SampleType::I32: return ValueType::INT32;
        case keel::imagefile::SampleType::F32: return ValueType::SINGLE;
        case keel::imagefile::SampleType::F64: return ValueType::DOUBLE;
    }
    return ValueType::UINT8;
}

template <typename T>
void transposeInterleavedToPlanar(const T *src, T *dst,
                                  keel::usize H, keel::usize W, keel::usize C) {
    const keel::usize plane = H * W;
    if (C == 1) {
        for (keel::usize r = 0; r < H; ++r) {
            const T *row = src + r * W;
            for (keel::usize c = 0; c < W; ++c) {
                dst[c * H + r] = row[c];
            }
        }
        return;
    }
    for (keel::usize r = 0; r < H; ++r) {
        const T *row = src + r * W * C;
        for (keel::usize c = 0; c < W; ++c) {
            const keel::usize dstBase = c * H + r;
            const T *px = row + c * C;
            for (keel::usize ch = 0; ch < C; ++ch) {
                dst[ch * plane + dstBase] = px[ch];
            }
        }
    }
}

Value keelImageToValue(const keel::imagefile::Image &img, keel::Allocator *mr,
                       bool promoteGaToRgba = false) {
    const keel::usize H = img.height;
    const keel::usize W = img.width;
    const keel::usize C = img.channels;

    if (promoteGaToRgba && C == 2) {
        const keel::usize plane = H * W;
        if (img.sample_type == keel::imagefile::SampleType::U16) {
            Value out = Value::matrix3d(H, W, 4, ValueType::UINT16, mr);
            auto *dst = out.uint16DataMut();
            const auto *src = reinterpret_cast<const keel::u16 *>(img.pixels.data());
            for (keel::usize r = 0; r < H; ++r) {
                for (keel::usize c = 0; c < W; ++c) {
                    const keel::u16 g = src[(r * W + c) * 2 + 0];
                    const keel::u16 a = src[(r * W + c) * 2 + 1];
                    const keel::usize idx = c * H + r;
                    dst[0 * plane + idx] = g;
                    dst[1 * plane + idx] = g;
                    dst[2 * plane + idx] = g;
                    dst[3 * plane + idx] = a;
                }
            }
            return out;
        }
        Value out = Value::matrix3d(H, W, 4, ValueType::UINT8, mr);
        auto *dst = out.uint8DataMut();
        const auto *src = img.pixels.data();
        for (keel::usize r = 0; r < H; ++r) {
            for (keel::usize c = 0; c < W; ++c) {
                const keel::u8 g = src[(r * W + c) * 2 + 0];
                const keel::u8 a = src[(r * W + c) * 2 + 1];
                const keel::usize idx = c * H + r;
                dst[0 * plane + idx] = g;
                dst[1 * plane + idx] = g;
                dst[2 * plane + idx] = g;
                dst[3 * plane + idx] = a;
            }
        }
        return out;
    }

    const ValueType vt = mapSampleType(img.sample_type);
    Value out = (C == 1)
        ? Value::matrix(H, W, vt, mr)
        : Value::matrix3d(H, W, C, vt, mr);

    switch (img.sample_type) {
        case keel::imagefile::SampleType::U8:
            transposeInterleavedToPlanar(img.pixels.data(), out.uint8DataMut(), H, W, C);
            break;
        case keel::imagefile::SampleType::U16:
            transposeInterleavedToPlanar(
                reinterpret_cast<const keel::u16 *>(img.pixels.data()),
                out.uint16DataMut(), H, W, C);
            break;
        case keel::imagefile::SampleType::U32:
            transposeInterleavedToPlanar(
                reinterpret_cast<const keel::u32 *>(img.pixels.data()),
                out.uint32DataMut(), H, W, C);
            break;
        case keel::imagefile::SampleType::I8:
            transposeInterleavedToPlanar(
                reinterpret_cast<const keel::i8 *>(img.pixels.data()),
                out.int8DataMut(), H, W, C);
            break;
        case keel::imagefile::SampleType::I16:
            transposeInterleavedToPlanar(
                reinterpret_cast<const keel::i16 *>(img.pixels.data()),
                out.int16DataMut(), H, W, C);
            break;
        case keel::imagefile::SampleType::I32:
            transposeInterleavedToPlanar(
                reinterpret_cast<const keel::i32 *>(img.pixels.data()),
                out.int32DataMut(), H, W, C);
            break;
        case keel::imagefile::SampleType::F32:
            transposeInterleavedToPlanar(
                reinterpret_cast<const keel::f32 *>(img.pixels.data()),
                out.singleDataMut(), H, W, C);
            break;
        case keel::imagefile::SampleType::F64:
            transposeInterleavedToPlanar(
                reinterpret_cast<const keel::f64 *>(img.pixels.data()),
                out.doubleDataMut(), H, W, C);
            break;
    }
    return out;
}

Value paletteToValue(const keel::imagefile::Image &img, keel::Allocator *mr,
                     bool emptyMatrixWhenNone) {
    if (!img.has_palette()) {
        return emptyMatrixWhenNone ? Value::matrix(0, 0, ValueType::DOUBLE, mr) : Value();
    }
    const keel::usize K = img.palette_entry_count();
    Value cmap = Value::matrix(K, 3, ValueType::DOUBLE, mr);
    keel::f64 *cd = cmap.doubleDataMut();
    const keel::f64 scale = (img.palette_bits == 16) ? (1.0 / 65535.0) : (1.0 / 255.0);
    for (keel::usize k = 0; k < K; ++k) {
        cd[0 * K + k] = keel::as<keel::f64>(img.palette[3 * k + 0]) * scale;
        cd[1 * K + k] = keel::as<keel::f64>(img.palette[3 * k + 1]) * scale;
        cd[2 * K + k] = keel::as<keel::f64>(img.palette[3 * k + 2]) * scale;
    }
    return cmap;
}

struct PackedImageBuffer {
    keel::u32 width = 0;
    keel::u32 height = 0;
    keel::u16 channels = 1;
    keel::imagefile::SampleType sampleType = keel::imagefile::SampleType::U8;
    keel::Array<keel::u8> bytes;

    keel::imagefile::ImageView view() const {
        return keel::imagefile::ImageView{
            .width = width,
            .height = height,
            .channels = channels,
            .sample_type = sampleType,
            .row_stride_bytes = 0,
            .pixels = keel::ArrayRef<keel::u8>(bytes.data(), bytes.size()),
        };
    }
};

PackedImageBuffer packStandardImage(const Value &A, bool allowU16, bool allow4Ch,
                                    const char *chErrMsg, const char *chErrId) {
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    if (H == 0 || W == 0) {
        throw Error("imwrite: empty image matrix",
                    0, 0, "imwrite", "", "numkit:imwrite:empty");
    }

    keel::usize C = 0;
    if (A.numel() == H * W)                  C = 1;
    else if (A.numel() == H * W * 3)         C = 3;
    else if (allow4Ch && A.numel() == H * W * 4) C = 4;
    else {
        throw Error(chErrMsg, 0, 0, "imwrite", "", chErrId);
    }

    PackedImageBuffer out;
    out.width = keel::as<keel::u32>(W);
    out.height = keel::as<keel::u32>(H);
    out.channels = keel::as<keel::u16>(C);

    const keel::usize plane = H * W;
    const bool is16Bit = allowU16 && (A.type() == ValueType::UINT16);
    if (is16Bit) {
        out.sampleType = keel::imagefile::SampleType::U16;
        out.bytes.resize(H * W * C * sizeof(keel::u16));
        auto *dst = reinterpret_cast<keel::u16 *>(out.bytes.data());
        const auto *src = A.uint16Data();
        for (keel::usize r = 0; r < H; ++r) {
            for (keel::usize c = 0; c < W; ++c) {
                for (keel::usize ch = 0; ch < C; ++ch) {
                    const keel::usize idx = (C == 1) ? (c * H + r) : (ch * plane + c * H + r);
                    dst[(r * W + c) * C + ch] = src[idx];
                }
            }
        }
    } else {
        out.sampleType = keel::imagefile::SampleType::U8;
        out.bytes.resize(H * W * C);
        auto *dst = out.bytes.data();
        if (A.type() == ValueType::UINT8) {
            const auto *src = A.uint8Data();
            for (keel::usize r = 0; r < H; ++r) {
                for (keel::usize c = 0; c < W; ++c) {
                    for (keel::usize ch = 0; ch < C; ++ch) {
                        const keel::usize idx = (C == 1) ? (c * H + r) : (ch * plane + c * H + r);
                        dst[(r * W + c) * C + ch] = src[idx];
                    }
                }
            }
        } else {
            for (keel::usize r = 0; r < H; ++r) {
                for (keel::usize c = 0; c < W; ++c) {
                    for (keel::usize ch = 0; ch < C; ++ch) {
                        const keel::usize idx = (C == 1) ? (c * H + r) : (ch * plane + c * H + r);
                        keel::f64 dv = A.elemAsDouble(idx);
                        if (dv < 0.0) dv = 0.0;
                        else if (dv > 255.0) dv = 255.0;
                        dst[(r * W + c) * C + ch] = keel::as<keel::u8>(keel::as<int>(dv));
                    }
                }
            }
        }
    }
    return out;
}

template <typename T>
void packPlanarToInterleavedTyped(const T *src, keel::u8 *dstBytes,
                                  keel::usize H, keel::usize W, keel::usize C) {
    auto *dst = reinterpret_cast<T *>(dstBytes);
    const keel::usize plane = H * W;
    for (keel::usize r = 0; r < H; ++r) {
        for (keel::usize c = 0; c < W; ++c) {
            for (keel::usize ch = 0; ch < C; ++ch) {
                const keel::usize idx = (C == 1) ? (c * H + r) : (ch * plane + c * H + r);
                dst[(r * W + c) * C + ch] = src[idx];
            }
        }
    }
}

PackedImageBuffer packTiffImage(const Value &A) {
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    if (H == 0 || W == 0) {
        throw Error("imwrite TIFF: empty image",
                    0, 0, "imwrite", "", "numkit:imwrite:empty");
    }

    keel::usize C = 0;
    if (A.numel() == H * W)          C = 1;
    else if (A.numel() == H * W * 3) C = 3;
    else if (A.numel() == H * W * 4) C = 4;
    else {
        throw Error("imwrite TIFF: only 1, 3, or 4 channels supported",
                    0, 0, "imwrite", "", "numkit:imwrite:tiffChannels");
    }

    PackedImageBuffer out;
    out.width = keel::as<keel::u32>(W);
    out.height = keel::as<keel::u32>(H);
    out.channels = keel::as<keel::u16>(C);

    switch (A.type()) {
        case ValueType::UINT16:
            out.sampleType = keel::imagefile::SampleType::U16;
            out.bytes.resize(H * W * C * 2);
            packPlanarToInterleavedTyped(A.uint16Data(), out.bytes.data(), H, W, C);
            break;
        case ValueType::UINT32:
            out.sampleType = keel::imagefile::SampleType::U32;
            out.bytes.resize(H * W * C * 4);
            packPlanarToInterleavedTyped(A.uint32Data(), out.bytes.data(), H, W, C);
            break;
        case ValueType::INT8:
            out.sampleType = keel::imagefile::SampleType::I8;
            out.bytes.resize(H * W * C * 1);
            packPlanarToInterleavedTyped(A.int8Data(), out.bytes.data(), H, W, C);
            break;
        case ValueType::INT16:
            out.sampleType = keel::imagefile::SampleType::I16;
            out.bytes.resize(H * W * C * 2);
            packPlanarToInterleavedTyped(A.int16Data(), out.bytes.data(), H, W, C);
            break;
        case ValueType::INT32:
            out.sampleType = keel::imagefile::SampleType::I32;
            out.bytes.resize(H * W * C * 4);
            packPlanarToInterleavedTyped(A.int32Data(), out.bytes.data(), H, W, C);
            break;
        case ValueType::SINGLE:
            out.sampleType = keel::imagefile::SampleType::F32;
            out.bytes.resize(H * W * C * 4);
            packPlanarToInterleavedTyped(A.singleData(), out.bytes.data(), H, W, C);
            break;
        case ValueType::DOUBLE:
            out.sampleType = keel::imagefile::SampleType::F64;
            out.bytes.resize(H * W * C * 8);
            packPlanarToInterleavedTyped(A.doubleData(), out.bytes.data(), H, W, C);
            break;
        default: {
            out.sampleType = keel::imagefile::SampleType::U8;
            out.bytes.resize(H * W * C);
            const keel::usize plane = H * W;
            for (keel::usize r = 0; r < H; ++r) {
                for (keel::usize c = 0; c < W; ++c) {
                    for (keel::usize ch = 0; ch < C; ++ch) {
                        const keel::usize idx = (C == 1) ? (c * H + r) : (ch * plane + c * H + r);
                        keel::f64 dv = A.elemAsDouble(idx);
                        if (dv < 0.0) dv = 0.0;
                        else if (dv > 255.0) dv = 255.0;
                        out.bytes[(r * W + c) * C + ch] = keel::as<keel::u8>(keel::as<int>(dv));
                    }
                }
            }
            break;
        }
    }
    return out;
}

keel::Array<keel::u8> loadFileBytes(keel::StringRef path, const char *who) {
    auto res = keel::fs_read_file(path);
    if (!res.is_ok()) {
        throw Error(keel::format("{}: cannot open '{}'", who, path),
                    0, 0, who, "", keel::format("numkit:{}:open", who));
    }
    return std::move(res.value());
}

keel::imagefile::TiffCompression parseTiffCompression(keel::StringRef s) {
    if (s.empty() || s == "none") return keel::imagefile::TiffCompression::None;
    if (s == "packbits")          return keel::imagefile::TiffCompression::PackBits;
    if (s == "lzw")               return keel::imagefile::TiffCompression::Lzw;
    if (s == "deflate")           return keel::imagefile::TiffCompression::Deflate;
    throw Error(keel::format("imwrite TIFF: unknown Compression '{}' (use none, packbits, lzw, or deflate)", s),
                0, 0, "imwrite", "", "numkit:imwrite:tiffCompression");
}

bool fillPeekInfo(const keel::Result<keel::imagefile::ImageInfo> &res,
                  keel::u32 &W, keel::u32 &H,
                  keel::u16 &bitsPerSample, keel::u16 &channels) {
    if (!res) return false;
    const auto &info = res.value();
    W = info.width;
    H = info.height;
    bitsPerSample = info.bits_per_sample;
    channels = info.channels;
    return true;
}

} // anonymous namespace

// ============================================================================
// PNG
// ============================================================================

std::pair<Value, Value> readPngWithMap(const keel::u8 *data, keel::usize len,
                                       keel::Allocator *mr) {
    keel::imagefile::ImageDecodeOptions opts;
    opts.expand_indexed = false;
    auto res = keel::imagefile::decode_png(toRef(data, len), opts);
    if (!res) {
        throw Error("imread: failed to decode PNG",
                    0, 0, "imread", "", "numkit:imread:png");
    }
    const auto &img = res.value();
    return {
        keelImageToValue(img, mr, /*promoteGaToRgba=*/true),
        paletteToValue(img, mr, /*emptyMatrixWhenNone=*/false)
    };
}

Value readPng(const keel::u8 *data, keel::usize len,
              keel::Allocator *mr) {
    return readPngWithMap(data, len, mr).first;
}

keel::String writePngToBytes(const Value &A, int level) {
    auto packed = packStandardImage(
        A, /*allowU16=*/true, /*allow4Ch=*/true,
        "imwrite: input must be HxW or HxWx{1,3,4}", "numkit:imwrite:shape");
    keel::imagefile::PngEncodeOptions opts;
    opts.compression_level = level;
    auto res = keel::imagefile::encode_png(packed.view(), opts);
    if (!res) {
        throw Error("imwrite: failed to encode PNG",
                    0, 0, "imwrite", "", "numkit:imwrite:png");
    }
    return toKeelString(res.value());
}

bool peekPng(const keel::u8 *data, keel::usize len,
             keel::u32 &W, keel::u32 &H,
             keel::u16 &bitsPerSample, keel::u16 &channels) {
    return fillPeekInfo(keel::imagefile::peek_png(toRef(data, len)), W, H, bitsPerSample, channels);
}

// ============================================================================
// JPEG
// ============================================================================

Value readJpeg(const keel::u8 *data, keel::usize len,
               keel::Allocator *mr) {
    auto res = keel::imagefile::decode_jpeg(toRef(data, len));
    if (!res) {
        throw Error("imread: failed to decode JPEG",
                    0, 0, "imread", "", "numkit:imread:jpeg");
    }
    return keelImageToValue(res.value(), mr);
}

keel::String writeJpegToBytes(const Value &A, int quality) {
    auto packed = packStandardImage(
        A, /*allowU16=*/false, /*allow4Ch=*/false,
        "imwrite: JPEG only supports 1 (grayscale) or 3 (RGB) channels",
        "numkit:imwrite:jpegChannels");
    keel::imagefile::JpegEncodeOptions opts;
    opts.quality = quality;
    auto res = keel::imagefile::encode_jpeg(packed.view(), opts);
    if (!res) {
        throw Error("imwrite: failed to encode JPEG",
                    0, 0, "imwrite", "", "numkit:imwrite:jpeg");
    }
    return toKeelString(res.value());
}

bool peekJpeg(const keel::u8 *data, keel::usize len,
              keel::u32 &W, keel::u32 &H,
              keel::u16 &bitsPerSample, keel::u16 &channels) {
    return fillPeekInfo(keel::imagefile::peek_jpeg(toRef(data, len)), W, H, bitsPerSample, channels);
}

// ============================================================================
// BMP
// ============================================================================

Value readBmp(const keel::u8 *data, keel::usize len,
              keel::Allocator *mr) {
    auto res = keel::imagefile::decode_bmp(toRef(data, len));
    if (!res) {
        throw Error("imread: failed to decode BMP",
                    0, 0, "imread", "", "numkit:imread:bmp");
    }
    return keelImageToValue(res.value(), mr);
}

keel::String writeBmpToBytes(const Value &A) {
    auto packed = packStandardImage(
        A, /*allowU16=*/false, /*allow4Ch=*/true,
        "imwrite: input must be HxW or HxWx{1,3,4}", "numkit:imwrite:shape");
    auto res = keel::imagefile::encode_bmp(packed.view());
    if (!res) {
        throw Error("imwrite: failed to encode BMP",
                    0, 0, "imwrite", "", "numkit:imwrite:bmp");
    }
    return toKeelString(res.value());
}

bool peekBmp(const keel::u8 *data, keel::usize len,
             keel::u32 &W, keel::u32 &H,
             keel::u16 &bitsPerSample, keel::u16 &channels) {
    return fillPeekInfo(keel::imagefile::peek_bmp(toRef(data, len)), W, H, bitsPerSample, channels);
}

// ============================================================================
// TGA
// ============================================================================

Value readTga(const keel::u8 *data, keel::usize len,
              keel::Allocator *mr) {
    auto res = keel::imagefile::decode_tga(toRef(data, len));
    if (!res) {
        throw Error("imread: failed to decode TGA",
                    0, 0, "imread", "", "numkit:imread:tga");
    }
    return keelImageToValue(res.value(), mr);
}

keel::String writeTgaToBytes(const Value &A) {
    auto packed = packStandardImage(
        A, /*allowU16=*/false, /*allow4Ch=*/true,
        "imwrite: input must be HxW or HxWx{1,3,4}", "numkit:imwrite:shape");
    auto res = keel::imagefile::encode_tga(packed.view());
    if (!res) {
        throw Error("imwrite: failed to encode TGA",
                    0, 0, "imwrite", "", "numkit:imwrite:tga");
    }
    return toKeelString(res.value());
}

bool peekTga(const keel::u8 *data, keel::usize len,
             keel::u32 &W, keel::u32 &H,
             keel::u16 &bitsPerSample, keel::u16 &channels) {
    return fillPeekInfo(keel::imagefile::peek_tga(toRef(data, len)), W, H, bitsPerSample, channels);
}

// ============================================================================
// PNM
// ============================================================================

Value readPnm(const keel::u8 *data, keel::usize len,
              keel::Allocator *mr) {
    auto res = keel::imagefile::decode_pnm(toRef(data, len));
    if (!res) {
        throw Error("imread: failed to decode PNM",
                    0, 0, "imread", "", "numkit:imread:pnm");
    }
    return keelImageToValue(res.value(), mr);
}

keel::String writePnmToBytes(const Value &A, keel::StringRef /*ext*/) {
    auto packed = packStandardImage(
        A, /*allowU16=*/true, /*allow4Ch=*/false,
        "imwrite: PNM only supports 1 (PGM) or 3 (PPM) channels",
        "numkit:imwrite:pnmChannels");
    keel::imagefile::PnmEncodeOptions opts;
    opts.subformat = keel::imagefile::PnmSubformat::Auto;
    auto res = keel::imagefile::encode_pnm(packed.view(), opts);
    if (!res) {
        throw Error("imwrite: failed to encode PNM",
                    0, 0, "imwrite", "", "numkit:imwrite:pnm");
    }
    return toKeelString(res.value());
}

bool peekPnm(const keel::u8 *data, keel::usize len,
             keel::u32 &W, keel::u32 &H,
             keel::u16 &bitsPerSample, keel::u16 &channels) {
    return fillPeekInfo(keel::imagefile::peek_pnm(toRef(data, len)), W, H, bitsPerSample, channels);
}

// ============================================================================
// TIFF & BigTIFF
// ============================================================================

bool isTiffBytes(const keel::u8 *data, keel::usize len) {
    return keel::imagefile::is_tiff_bytes(toRef(data, len));
}

bool isTiffBytes(keel::StringRef b) {
    return isTiffBytes(reinterpret_cast<const keel::u8 *>(b.data()), b.size());
}

std::pair<Value, Value> readTiffWithMap(keel::ArrayRef<keel::u8> buf, keel::u32 page,
                                        keel::Allocator *mr) {
    if (page == 0) {
        throw Error("imread: page index must be >= 1",
                    0, 0, "imread", "", "numkit:imread:tiffPage");
    }
    keel::imagefile::ImageDecodeOptions opts;
    opts.page = page - 1;
    opts.expand_indexed = false;
    auto res = keel::imagefile::decode_tiff(buf, opts);
    if (!res) {
        if (res.error() == keel::ErrorCode::OutOfRange) {
            throw Error(keel::format("imread: requested TIFF page {} out of range", page),
                        0, 0, "imread", "", "numkit:imread:tiffPage");
        }
        throw Error("imread: failed to decode TIFF",
                    0, 0, "imread", "", "numkit:imread:tiff");
    }
    const auto &img = res.value();
    return {
        keelImageToValue(img, mr),
        paletteToValue(img, mr, /*emptyMatrixWhenNone=*/true)
    };
}

std::pair<Value, Value> readTiffWithMap(keel::StringRef path, keel::u32 page,
                                        keel::Allocator *mr) {
    const auto buf = loadFileBytes(path, "imread");
    return readTiffWithMap(buf, page, mr);
}

Value readTiff(keel::ArrayRef<keel::u8> buf, keel::u32 page,
               keel::Allocator *mr) {
    return readTiffWithMap(buf, page, mr).first;
}

Value readTiff(keel::StringRef path, keel::u32 page,
               keel::Allocator *mr) {
    const auto buf = loadFileBytes(path, "imread");
    return readTiff(buf, page, mr);
}

Value readTiff(keel::StringRef path, keel::Allocator *mr) {
    return readTiff(path, 1u, mr);
}

void peekTiff(keel::ArrayRef<keel::u8> buf, keel::u32 &W,
              keel::u32 &H, keel::u16 &bits, keel::u16 &channels) {
    auto res = keel::imagefile::peek_tiff(buf);
    if (!res) {
        throw Error("imfinfo: failed to parse TIFF header",
                    0, 0, "imfinfo", "", "numkit:imfinfo:tiff");
    }
    const auto &info = res.value();
    W = info.width;
    H = info.height;
    bits = info.bits_per_sample;
    channels = info.channels;
}

void peekTiff(keel::StringRef path, keel::u32 &W, keel::u32 &H,
              keel::u16 &bits, keel::u16 &channels) {
    const auto buf = loadFileBytes(path, "imfinfo");
    peekTiff(buf, W, H, bits, channels);
}

keel::u32 tiffNumPages(keel::StringRef path) {
    const auto buf = loadFileBytes(path, "imfinfo");
    auto res = keel::imagefile::tiff_page_count(buf);
    if (!res) {
        throw Error("imfinfo: failed to count TIFF pages",
                    0, 0, "imfinfo", "", "numkit:imfinfo:tiff");
    }
    return res.value();
}

keel::Array<keel::u8> writeTiffToBytes(const Value &A, keel::StringRef compression,
                                       keel::ArrayRef<keel::u8> existing) {
    keel::imagefile::TiffEncodeOptions opts;
    opts.compression = parseTiffCompression(compression);
    auto packed = packTiffImage(A);

    auto res = keel::imagefile::encode_tiff(packed.view(), opts, existing);
    if (!res) {
        throw Error("imwrite TIFF: failed to encode TIFF",
                    0, 0, "imwrite", "", "numkit:imwrite:tiff");
    }
    return std::move(res.value());
}

void writeTiff(const Value &A, keel::StringRef path,
               keel::StringRef compression, bool appendMode) {
    keel::Array<keel::u8> existing;
    if (appendMode && keel::fs_exists(path)) {
        existing = loadFileBytes(path, "imwrite");
    }

    const keel::Array<keel::u8> buf = writeTiffToBytes(A, compression, existing);

    auto wres = keel::fs_write_file(path, buf);
    if (!wres.is_ok()) {
        throw Error(keel::format("imwrite TIFF: cannot open '{}' for write", path),
                    0, 0, "imwrite", "", "numkit:imwrite:open");
    }
}

} // namespace numkit::image
