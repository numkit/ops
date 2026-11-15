// ops/src/image_io/codecs_keel.cpp
//
// Thin PRIVATE adapters bridging NumKit's internal image codec headers
// (PNG, JPEG, BMP, TGA, PNM, TIFF) to keel::image.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/array.h>
#include <keel/fs.h>
#include "png_codec.hpp"
#include "jpeg_codec.hpp"
#include "bmp_codec.hpp"
#include "tga_codec.hpp"
#include "pnm_codec.hpp"
#include "tiff_codec.hpp"

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "keel/image.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace numkit::image {

namespace {

keel::ArrayRef<keel::u8> toRef(const keel::u8 *data, keel::usize len) {
    return keel::ArrayRef<keel::u8>(data, len);
}

std::string toStdString(const keel::Array<keel::u8> &arr) {
    return std::string(reinterpret_cast<const char *>(arr.data()), arr.size());
}

std::vector<keel::u8> toStdVec(const keel::Array<keel::u8> &arr) {
    return std::vector<keel::u8>(arr.data(), arr.data() + arr.size());
}

ValueType mapSampleType(keel::SampleType st) {
    switch (st) {
        case keel::SampleType::U8:  return ValueType::UINT8;
        case keel::SampleType::U16: return ValueType::UINT16;
        case keel::SampleType::U32: return ValueType::UINT32;
        case keel::SampleType::I8:  return ValueType::INT8;
        case keel::SampleType::I16: return ValueType::INT16;
        case keel::SampleType::I32: return ValueType::INT32;
        case keel::SampleType::F32: return ValueType::SINGLE;
        case keel::SampleType::F64: return ValueType::DOUBLE;
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

Value keelImageToValue(const keel::Image &img, keel::Allocator *mr,
                       bool promoteGaToRgba = false) {
    const keel::usize H = img.height;
    const keel::usize W = img.width;
    const keel::usize C = img.channels;

    if (promoteGaToRgba && C == 2) {
        const keel::usize plane = H * W;
        if (img.sample_type == keel::SampleType::U16) {
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
        case keel::SampleType::U8:
            transposeInterleavedToPlanar(img.pixels.data(), out.uint8DataMut(), H, W, C);
            break;
        case keel::SampleType::U16:
            transposeInterleavedToPlanar(
                reinterpret_cast<const keel::u16 *>(img.pixels.data()),
                out.uint16DataMut(), H, W, C);
            break;
        case keel::SampleType::U32:
            transposeInterleavedToPlanar(
                reinterpret_cast<const keel::u32 *>(img.pixels.data()),
                out.uint32DataMut(), H, W, C);
            break;
        case keel::SampleType::I8:
            transposeInterleavedToPlanar(
                reinterpret_cast<const keel::i8 *>(img.pixels.data()),
                out.int8DataMut(), H, W, C);
            break;
        case keel::SampleType::I16:
            transposeInterleavedToPlanar(
                reinterpret_cast<const keel::i16 *>(img.pixels.data()),
                out.int16DataMut(), H, W, C);
            break;
        case keel::SampleType::I32:
            transposeInterleavedToPlanar(
                reinterpret_cast<const keel::i32 *>(img.pixels.data()),
                out.int32DataMut(), H, W, C);
            break;
        case keel::SampleType::F32:
            transposeInterleavedToPlanar(
                reinterpret_cast<const float *>(img.pixels.data()),
                out.singleDataMut(), H, W, C);
            break;
        case keel::SampleType::F64:
            transposeInterleavedToPlanar(
                reinterpret_cast<const double *>(img.pixels.data()),
                out.doubleDataMut(), H, W, C);
            break;
    }
    return out;
}

Value paletteToValue(const keel::Image &img, keel::Allocator *mr,
                     bool emptyMatrixWhenNone) {
    if (!img.has_palette()) {
        return emptyMatrixWhenNone ? Value::matrix(0, 0, ValueType::DOUBLE, mr) : Value();
    }
    const keel::usize K = img.palette_entry_count();
    Value cmap = Value::matrix(K, 3, ValueType::DOUBLE, mr);
    double *cd = cmap.doubleDataMut();
    const double scale = (img.palette_bits == 16) ? (1.0 / 65535.0) : (1.0 / 255.0);
    for (keel::usize k = 0; k < K; ++k) {
        cd[0 * K + k] = static_cast<double>(img.palette[3 * k + 0]) * scale;
        cd[1 * K + k] = static_cast<double>(img.palette[3 * k + 1]) * scale;
        cd[2 * K + k] = static_cast<double>(img.palette[3 * k + 2]) * scale;
    }
    return cmap;
}

struct PackedImageBuffer {
    keel::u32 width = 0;
    keel::u32 height = 0;
    keel::u16 channels = 1;
    keel::SampleType sampleType = keel::SampleType::U8;
    std::vector<keel::u8> bytes;

    keel::ImageView view() const {
        return keel::ImageView{
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
    out.width = static_cast<keel::u32>(W);
    out.height = static_cast<keel::u32>(H);
    out.channels = static_cast<keel::u16>(C);

    const keel::usize plane = H * W;
    const bool is16Bit = allowU16 && (A.type() == ValueType::UINT16);
    if (is16Bit) {
        out.sampleType = keel::SampleType::U16;
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
        out.sampleType = keel::SampleType::U8;
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
                        double dv = A.elemAsDouble(idx);
                        if (dv < 0.0) dv = 0.0;
                        else if (dv > 255.0) dv = 255.0;
                        dst[(r * W + c) * C + ch] = static_cast<keel::u8>(static_cast<int>(dv));
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
    out.width = static_cast<keel::u32>(W);
    out.height = static_cast<keel::u32>(H);
    out.channels = static_cast<keel::u16>(C);

    switch (A.type()) {
        case ValueType::UINT16:
            out.sampleType = keel::SampleType::U16;
            out.bytes.resize(H * W * C * 2);
            packPlanarToInterleavedTyped(A.uint16Data(), out.bytes.data(), H, W, C);
            break;
        case ValueType::UINT32:
            out.sampleType = keel::SampleType::U32;
            out.bytes.resize(H * W * C * 4);
            packPlanarToInterleavedTyped(A.uint32Data(), out.bytes.data(), H, W, C);
            break;
        case ValueType::INT8:
            out.sampleType = keel::SampleType::I8;
            out.bytes.resize(H * W * C * 1);
            packPlanarToInterleavedTyped(A.int8Data(), out.bytes.data(), H, W, C);
            break;
        case ValueType::INT16:
            out.sampleType = keel::SampleType::I16;
            out.bytes.resize(H * W * C * 2);
            packPlanarToInterleavedTyped(A.int16Data(), out.bytes.data(), H, W, C);
            break;
        case ValueType::INT32:
            out.sampleType = keel::SampleType::I32;
            out.bytes.resize(H * W * C * 4);
            packPlanarToInterleavedTyped(A.int32Data(), out.bytes.data(), H, W, C);
            break;
        case ValueType::SINGLE:
            out.sampleType = keel::SampleType::F32;
            out.bytes.resize(H * W * C * 4);
            packPlanarToInterleavedTyped(A.singleData(), out.bytes.data(), H, W, C);
            break;
        case ValueType::DOUBLE:
            out.sampleType = keel::SampleType::F64;
            out.bytes.resize(H * W * C * 8);
            packPlanarToInterleavedTyped(A.doubleData(), out.bytes.data(), H, W, C);
            break;
        default: {
            out.sampleType = keel::SampleType::U8;
            out.bytes.resize(H * W * C);
            const keel::usize plane = H * W;
            for (keel::usize r = 0; r < H; ++r) {
                for (keel::usize c = 0; c < W; ++c) {
                    for (keel::usize ch = 0; ch < C; ++ch) {
                        const keel::usize idx = (C == 1) ? (c * H + r) : (ch * plane + c * H + r);
                        double dv = A.elemAsDouble(idx);
                        if (dv < 0.0) dv = 0.0;
                        else if (dv > 255.0) dv = 255.0;
                        out.bytes[(r * W + c) * C + ch] = static_cast<keel::u8>(static_cast<int>(dv));
                    }
                }
            }
            break;
        }
    }
    return out;
}

std::vector<keel::u8> loadFileBytes(const std::string &path, const char *who) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw Error(std::string(who) + ": cannot open '" + path + "'",
                    0, 0, who, "", std::string("numkit:") + who + ":open");
    }
    in.seekg(0, std::ios::end);
    const std::streamoff sz = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<keel::u8> buf(sz > 0 ? static_cast<keel::usize>(sz) : 0u);
    if (!buf.empty()) {
        in.read(reinterpret_cast<char *>(buf.data()), sz);
    }
    return buf;
}

keel::TiffCompression parseTiffCompression(const std::string &s) {
    if (s.empty() || s == "none") return keel::TiffCompression::None;
    if (s == "packbits")          return keel::TiffCompression::PackBits;
    if (s == "lzw")               return keel::TiffCompression::Lzw;
    if (s == "deflate")           return keel::TiffCompression::Deflate;
    throw Error("imwrite TIFF: unknown Compression '" + s + "' (use none, packbits, lzw, or deflate)",
                0, 0, "imwrite", "", "numkit:imwrite:tiffCompression");
}

bool fillPeekInfo(const keel::Result<keel::ImageInfo> &res,
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
    keel::ImageDecodeOptions opts;
    opts.expand_indexed = false;
    auto res = keel::decode_png(toRef(data, len), opts);
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

std::string writePngToBytes(const Value &A, int level) {
    auto packed = packStandardImage(
        A, /*allowU16=*/true, /*allow4Ch=*/true,
        "imwrite: input must be HxW or HxWx{1,3,4}", "numkit:imwrite:shape");
    keel::PngEncodeOptions opts;
    opts.compression_level = level;
    auto res = keel::encode_png(packed.view(), opts);
    if (!res) {
        throw Error("imwrite: failed to encode PNG",
                    0, 0, "imwrite", "", "numkit:imwrite:png");
    }
    return toStdString(res.value());
}

bool peekPng(const keel::u8 *data, keel::usize len,
             keel::u32 &W, keel::u32 &H,
             keel::u16 &bitsPerSample, keel::u16 &channels) {
    return fillPeekInfo(keel::peek_png(toRef(data, len)), W, H, bitsPerSample, channels);
}

// ============================================================================
// JPEG
// ============================================================================

Value readJpeg(const keel::u8 *data, keel::usize len,
               keel::Allocator *mr) {
    auto res = keel::decode_jpeg(toRef(data, len));
    if (!res) {
        throw Error("imread: failed to decode JPEG",
                    0, 0, "imread", "", "numkit:imread:jpeg");
    }
    return keelImageToValue(res.value(), mr);
}

std::string writeJpegToBytes(const Value &A, int quality) {
    auto packed = packStandardImage(
        A, /*allowU16=*/false, /*allow4Ch=*/false,
        "imwrite: JPEG only supports 1 (grayscale) or 3 (RGB) channels",
        "numkit:imwrite:jpegChannels");
    keel::JpegEncodeOptions opts;
    opts.quality = quality;
    auto res = keel::encode_jpeg(packed.view(), opts);
    if (!res) {
        throw Error("imwrite: failed to encode JPEG",
                    0, 0, "imwrite", "", "numkit:imwrite:jpeg");
    }
    return toStdString(res.value());
}

bool peekJpeg(const keel::u8 *data, keel::usize len,
              keel::u32 &W, keel::u32 &H,
              keel::u16 &bitsPerSample, keel::u16 &channels) {
    return fillPeekInfo(keel::peek_jpeg(toRef(data, len)), W, H, bitsPerSample, channels);
}

// ============================================================================
// BMP
// ============================================================================

Value readBmp(const keel::u8 *data, keel::usize len,
              keel::Allocator *mr) {
    auto res = keel::decode_bmp(toRef(data, len));
    if (!res) {
        throw Error("imread: failed to decode BMP",
                    0, 0, "imread", "", "numkit:imread:bmp");
    }
    return keelImageToValue(res.value(), mr);
}

std::string writeBmpToBytes(const Value &A) {
    auto packed = packStandardImage(
        A, /*allowU16=*/false, /*allow4Ch=*/true,
        "imwrite: input must be HxW or HxWx{1,3,4}", "numkit:imwrite:shape");
    auto res = keel::encode_bmp(packed.view());
    if (!res) {
        throw Error("imwrite: failed to encode BMP",
                    0, 0, "imwrite", "", "numkit:imwrite:bmp");
    }
    return toStdString(res.value());
}

bool peekBmp(const keel::u8 *data, keel::usize len,
             keel::u32 &W, keel::u32 &H,
             keel::u16 &bitsPerSample, keel::u16 &channels) {
    return fillPeekInfo(keel::peek_bmp(toRef(data, len)), W, H, bitsPerSample, channels);
}

// ============================================================================
// TGA
// ============================================================================

Value readTga(const keel::u8 *data, keel::usize len,
              keel::Allocator *mr) {
    auto res = keel::decode_tga(toRef(data, len));
    if (!res) {
        throw Error("imread: failed to decode TGA",
                    0, 0, "imread", "", "numkit:imread:tga");
    }
    return keelImageToValue(res.value(), mr);
}

std::string writeTgaToBytes(const Value &A) {
    auto packed = packStandardImage(
        A, /*allowU16=*/false, /*allow4Ch=*/true,
        "imwrite: input must be HxW or HxWx{1,3,4}", "numkit:imwrite:shape");
    auto res = keel::encode_tga(packed.view());
    if (!res) {
        throw Error("imwrite: failed to encode TGA",
                    0, 0, "imwrite", "", "numkit:imwrite:tga");
    }
    return toStdString(res.value());
}

bool peekTga(const keel::u8 *data, keel::usize len,
             keel::u32 &W, keel::u32 &H,
             keel::u16 &bitsPerSample, keel::u16 &channels) {
    return fillPeekInfo(keel::peek_tga(toRef(data, len)), W, H, bitsPerSample, channels);
}

// ============================================================================
// PNM
// ============================================================================

Value readPnm(const keel::u8 *data, keel::usize len,
              keel::Allocator *mr) {
    auto res = keel::decode_pnm(toRef(data, len));
    if (!res) {
        throw Error("imread: failed to decode PNM",
                    0, 0, "imread", "", "numkit:imread:pnm");
    }
    return keelImageToValue(res.value(), mr);
}

std::string writePnmToBytes(const Value &A, const std::string & /*ext*/) {
    auto packed = packStandardImage(
        A, /*allowU16=*/true, /*allow4Ch=*/false,
        "imwrite: PNM only supports 1 (PGM) or 3 (PPM) channels",
        "numkit:imwrite:pnmChannels");
    keel::PnmEncodeOptions opts;
    opts.subformat = keel::PnmSubformat::Auto;
    auto res = keel::encode_pnm(packed.view(), opts);
    if (!res) {
        throw Error("imwrite: failed to encode PNM",
                    0, 0, "imwrite", "", "numkit:imwrite:pnm");
    }
    return toStdString(res.value());
}

bool peekPnm(const keel::u8 *data, keel::usize len,
             keel::u32 &W, keel::u32 &H,
             keel::u16 &bitsPerSample, keel::u16 &channels) {
    return fillPeekInfo(keel::peek_pnm(toRef(data, len)), W, H, bitsPerSample, channels);
}

// ============================================================================
// TIFF & BigTIFF
// ============================================================================

bool isTiffBytes(const keel::u8 *data, keel::usize len) {
    return keel::is_tiff_bytes(toRef(data, len));
}

bool isTiffBytes(const std::string &b) {
    return isTiffBytes(reinterpret_cast<const keel::u8 *>(b.data()), b.size());
}

std::pair<Value, Value> readTiffWithMap(std::vector<keel::u8> buf, keel::u32 page,
                                        keel::Allocator *mr) {
    if (page == 0) {
        throw Error("imread: page index must be >= 1",
                    0, 0, "imread", "", "numkit:imread:tiffPage");
    }
    keel::ImageDecodeOptions opts;
    opts.page = page - 1;
    opts.expand_indexed = false;
    auto res = keel::decode_tiff(toRef(buf.data(), buf.size()), opts);
    if (!res) {
        if (res.error() == keel::ErrorCode::OutOfRange) {
            throw Error("imread: requested TIFF page " + std::to_string(page) + " out of range",
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

std::pair<Value, Value> readTiffWithMap(const std::string &path, keel::u32 page,
                                        keel::Allocator *mr) {
    return readTiffWithMap(loadFileBytes(path, "imread"), page, mr);
}

Value readTiff(std::vector<keel::u8> buf, keel::u32 page,
               keel::Allocator *mr) {
    return readTiffWithMap(std::move(buf), page, mr).first;
}

Value readTiff(const std::string &path, keel::u32 page,
               keel::Allocator *mr) {
    return readTiff(loadFileBytes(path, "imread"), page, mr);
}

Value readTiff(const std::string &path, keel::Allocator *mr) {
    return readTiff(path, 1u, mr);
}

void peekTiff(const std::vector<keel::u8> &buf, keel::u32 &W,
              keel::u32 &H, keel::u16 &bits, keel::u16 &channels) {
    auto res = keel::peek_tiff(toRef(buf.data(), buf.size()));
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

void peekTiff(const std::string &path, keel::u32 &W, keel::u32 &H,
              keel::u16 &bits, keel::u16 &channels) {
    peekTiff(loadFileBytes(path, "imfinfo"), W, H, bits, channels);
}

keel::u32 tiffNumPages(const std::string &path) {
    const auto buf = loadFileBytes(path, "imfinfo");
    auto res = keel::tiff_page_count(toRef(buf.data(), buf.size()));
    if (!res) {
        throw Error("imfinfo: failed to count TIFF pages",
                    0, 0, "imfinfo", "", "numkit:imfinfo:tiff");
    }
    return res.value();
}

std::vector<keel::u8> writeTiffToBytes(const Value &A, const std::string &compression,
                                           const std::vector<keel::u8> *existing) {
    keel::TiffEncodeOptions opts;
    opts.compression = parseTiffCompression(compression);
    auto packed = packTiffImage(A);

    keel::ArrayRef<keel::u8> appendRef{};
    if (existing && !existing->empty()) {
        appendRef = toRef(existing->data(), existing->size());
    }

    auto res = keel::encode_tiff(packed.view(), opts, appendRef);
    if (!res) {
        throw Error("imwrite TIFF: failed to encode TIFF",
                    0, 0, "imwrite", "", "numkit:imwrite:tiff");
    }
    return toStdVec(res.value());
}

void writeTiff(const Value &A, const std::string &path,
               const std::string &compression, bool appendMode) {
    std::vector<keel::u8> existing;
    const std::vector<keel::u8> *exptr = nullptr;
    if (appendMode && keel::fs_exists(path)) {
        existing = loadFileBytes(path, "imwrite");
        exptr = &existing;
    }

    const std::vector<keel::u8> buf = writeTiffToBytes(A, compression, exptr);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw Error("imwrite TIFF: cannot open '" + path + "' for write",
                    0, 0, "imwrite", "", "numkit:imwrite:open");
    }
    out.write(reinterpret_cast<const char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

} // namespace numkit::image
