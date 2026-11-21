// ops/src/image_io/io.cpp
//
// Image I/O facade (imread, imwrite, imfinfo) backed by keel::image.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/fs.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/image/io/io.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include "io_detail.hpp"
#include "bmp_codec.hpp"
#include "tga_codec.hpp"
#include "pnm_codec.hpp"
#include "png_codec.hpp"
#include "jpeg_codec.hpp"
#include "tiff_codec.hpp"

#include "keel/image.h"

#include <cstddef>
#include <cstdint>

namespace numkit::image {

namespace {

keel::StringRef formatStringFromKeel(keel::image::ImageFormat fmt) {
    switch (fmt) {
        case keel::image::ImageFormat::Png:  return "png";
        case keel::image::ImageFormat::Tiff: return "tif";
        case keel::image::ImageFormat::Jpeg: return "jpg";
        case keel::image::ImageFormat::Bmp:  return "bmp";
        case keel::image::ImageFormat::Gif:  return "gif";
        case keel::image::ImageFormat::Pnm:  return "pnm";
        case keel::image::ImageFormat::Tga:  return "tga";
        case keel::image::ImageFormat::Unknown:
            break;
    }
    return {};
}

Value readGifViaKeel(const keel::u8 *data, keel::usize len,
                     keel::Allocator *mr) {
    keel::image::ImageDecodeOptions opts;
    opts.expand_indexed = false;
    auto res = keel::image::decode_gif(keel::ArrayRef<keel::u8>(data, len), opts);
    if (!res) {
        throw Error("imread: failed to decode GIF",
                    0, 0, "imread", "", "numkit:imread:gif");
    }
    const auto &img = res.value();
    const keel::usize H = img.height;
    const keel::usize W = img.width;
    Value out = Value::matrix(H, W, ValueType::UINT8, mr);
    auto *dst = out.uint8DataMut();
    const auto *src = img.pixels.data();
    for (keel::usize r = 0; r < H; ++r) {
        for (keel::usize c = 0; c < W; ++c) {
            dst[c * H + r] = src[r * W + c];
        }
    }
    return out;
}

const char *colorTypeFromChannels(int c) {
    switch (c) {
        case 1: return "grayscale";
        case 2: return "grayscalealpha";
        case 3: return "truecolor";
        case 4: return "truecoloralpha";
        default: return "unknown";
    }
}

} // anonymous namespace

Value imreadFromBytes(keel::StringRef bytes, keel::Allocator *mr) {
    const auto *data = reinterpret_cast<const keel::u8 *>(bytes.data());
    const keel::usize len = bytes.size();

    if (len == 0) {
        throw Error("imread: empty image buffer",
                    0, 0, "imread", "", "numkit:imread:empty");
    }

    const keel::image::ImageFormat fmt = keel::image::sniff_image_format(keel::ArrayRef<keel::u8>(data, len));
    switch (fmt) {
        case keel::image::ImageFormat::Png:  return readPng(data, len, mr);
        case keel::image::ImageFormat::Tiff: return readTiff(keel::ArrayRef<keel::u8>(data, len), 1u, mr);
        case keel::image::ImageFormat::Jpeg: return readJpeg(data, len, mr);
        case keel::image::ImageFormat::Bmp:  return readBmp(data, len, mr);
        case keel::image::ImageFormat::Gif:  return readGifViaKeel(data, len, mr);
        case keel::image::ImageFormat::Pnm:  return readPnm(data, len, mr);
        case keel::image::ImageFormat::Tga:  return readTga(data, len, mr);
        case keel::image::ImageFormat::Unknown:
            break;
    }

    try { return readPng(data, len, mr); } catch (...) {}
    try { return readBmp(data, len, mr); } catch (...) {}
    try { return readJpeg(data, len, mr); } catch (...) {}
    try { return readTga(data, len, mr); } catch (...) {}

    throw Error("imread: failed to decode image - unsupported or unknown format",
                0, 0, "imread", "", "numkit:imread:unknownFormat");
}

Value imread(keel::StringRef path, keel::Allocator *mr) {
    auto res = keel::fs_read_file(path, mr);
    if (!res.is_ok()) {
        throw Error(keel::format("imread: failed to load '{}' - can't open", path),
                    0, 0, "imread", "", "numkit:imread:load");
    }
    const auto &buf = res.value();
    return imreadFromBytes(
        keel::StringRef(reinterpret_cast<const char *>(buf.data()), buf.size()), mr);
}

keel::String imwriteToBytes(const Value &A, keel::StringRef ext,
                            keel::Allocator *mr) {
    if (ext == "png") {
        return writePngToBytes(A, 6);
    }
    if (ext == "bmp") {
        return writeBmpToBytes(A);
    }
    if (ext == "tga") {
        return writeTgaToBytes(A);
    }
    if (ext == "jpg" || ext == "jpeg") {
        return writeJpegToBytes(A, 90);
    }
    if (ext == "pnm" || ext == "ppm" || ext == "pgm" || ext == "pbm") {
        return writePnmToBytes(A, ext);
    }
    if (ext == "tif" || ext == "tiff") {
        const auto buf = writeTiffToBytes(A, "none");
        return keel::String(keel::StringRef(reinterpret_cast<const char *>(buf.data()), buf.size()), mr);
    }

    throw Error(keel::format("imwrite: unsupported extension '{}' (supported: .png, .jpg, .bmp, .tga, .pnm, .tif)", ext),
                0, 0, "imwrite", "", "numkit:imwrite:ext");
}

void imwrite(const Value &A, keel::StringRef path, keel::Allocator *mr) {
    const keel::String ext = lowerExt(path);
    if (ext == "tif" || ext == "tiff") {
        writeTiff(A, path, "none", /*appendMode=*/false);
        return;
    }
    const keel::String bytes = imwriteToBytes(A, ext, mr);
    auto wres = keel::fs_write_file(
        path, keel::ArrayRef<keel::u8>(reinterpret_cast<const keel::u8 *>(bytes.data()), bytes.size()));
    if (!wres.is_ok()) {
        throw Error(keel::format("imwrite: cannot open '{}' for write", path),
                    0, 0, "imwrite", "", "numkit:imwrite:write");
    }
}

Value imfinfoFromBytes(keel::StringRef bytes, keel::StringRef filename,
                       keel::Allocator *mr) {
    const auto *data = reinterpret_cast<const keel::u8 *>(bytes.data());
    const keel::usize len = bytes.size();

    keel::u32 W = 0, H = 0;
    keel::u16 bits = 8, channels = 1;
    keel::String fmt = keel::String::from(formatStringFromKeel(
        keel::image::sniff_image_format(keel::ArrayRef<keel::u8>(data, len))), mr);
    if (fmt.empty()) fmt = lowerExt(filename);

    bool ok = false;
    if (fmt == "png") {
        ok = peekPng(data, len, W, H, bits, channels);
    } else if (fmt == "tif" || fmt == "tiff") {
        fmt = keel::String::from("tif", mr);
        peekTiff(keel::ArrayRef<keel::u8>(data, len), W, H, bits, channels);
        ok = true;
    } else if (fmt == "jpg" || fmt == "jpeg") {
        ok = peekJpeg(data, len, W, H, bits, channels);
    } else if (fmt == "bmp") {
        ok = peekBmp(data, len, W, H, bits, channels);
    } else if (fmt == "pnm" || fmt == "ppm" || fmt == "pgm" || fmt == "pbm") {
        ok = peekPnm(data, len, W, H, bits, channels);
    } else if (fmt == "tga") {
        ok = peekTga(data, len, W, H, bits, channels);
    } else if (fmt == "gif") {
        auto gifInfo = keel::image::peek_gif(keel::ArrayRef<keel::u8>(data, len));
        if (gifInfo) {
            W = gifInfo.value().width;
            H = gifInfo.value().height;
            bits = gifInfo.value().bits_per_sample;
            channels = gifInfo.value().channels;
            ok = true;
        }
    }

    if (!ok) {
        throw Error(keel::format("imfinfo: failed to parse header for '{}'", filename),
                    0, 0, "imfinfo", "", "numkit:imfinfo:read");
    }

    Value s = Value::structure(mr);
    s.field("Filename")         = Value::fromString(filename, mr);
    s.field("Format")           = Value::fromString(fmt, mr);
    s.field("Width")            = Value::scalar(double(W), mr);
    s.field("Height")           = Value::scalar(double(H), mr);
    s.field("NumberOfChannels") = Value::scalar(double(channels), mr);
    s.field("NumberOfSamples")  = Value::scalar(double(channels), mr);
    s.field("BitDepth")         = Value::scalar(double(bits * channels), mr);
    s.field("ColorType")        = Value::fromString(colorTypeFromChannels(channels), mr);
    s.field("FileSize")         = Value::scalar(double(bytes.size()), mr);
    return s;
}

Value imfinfo(keel::StringRef path, keel::Allocator *mr) {
    auto res = keel::fs_read_file(path, mr);
    if (!res.is_ok()) {
        throw Error(keel::format("imfinfo: failed to read '{}' - can't open", path),
                    0, 0, "imfinfo", "", "numkit:imfinfo:read");
    }
    const auto &buf = res.value();
    return imfinfoFromBytes(
        keel::StringRef(reinterpret_cast<const char *>(buf.data()), buf.size()), path, mr);
}

} // namespace numkit::image
