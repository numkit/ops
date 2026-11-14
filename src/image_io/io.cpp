// ops/src/image_io/io.cpp
//
// Image I/O facade (imread, imwrite, imfinfo) backed by keel::image.

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
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace numkit::image {

namespace {

std::string formatStringFromKeel(keel::ImageFormat fmt) {
    switch (fmt) {
        case keel::ImageFormat::Png:  return "png";
        case keel::ImageFormat::Tiff: return "tif";
        case keel::ImageFormat::Jpeg: return "jpg";
        case keel::ImageFormat::Bmp:  return "bmp";
        case keel::ImageFormat::Gif:  return "gif";
        case keel::ImageFormat::Pnm:  return "pnm";
        case keel::ImageFormat::Tga:  return "tga";
        case keel::ImageFormat::Unknown:
            break;
    }
    return {};
}

Value readGifViaKeel(const std::uint8_t *data, std::size_t len,
                     std::pmr::memory_resource *mr) {
    keel::ImageDecodeOptions opts;
    opts.expand_indexed = false;
    auto res = keel::decode_gif(keel::ArrayRef<keel::u8>(data, len), opts);
    if (!res) {
        throw Error("imread: failed to decode GIF",
                    0, 0, "imread", "", "numkit:imread:gif");
    }
    const auto &img = res.value();
    const std::size_t H = img.height;
    const std::size_t W = img.width;
    Value out = Value::matrix(H, W, ValueType::UINT8, mr);
    auto *dst = out.uint8DataMut();
    const auto *src = img.pixels.data();
    for (std::size_t r = 0; r < H; ++r) {
        for (std::size_t c = 0; c < W; ++c) {
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

Value imreadFromBytes(const std::string &bytes, std::pmr::memory_resource *mr) {
    const auto *data = reinterpret_cast<const std::uint8_t *>(bytes.data());
    const std::size_t len = bytes.size();

    if (len == 0) {
        throw Error("imread: empty image buffer",
                    0, 0, "imread", "", "numkit:imread:empty");
    }

    const keel::ImageFormat fmt = keel::sniff_image_format(keel::ArrayRef<keel::u8>(data, len));
    switch (fmt) {
        case keel::ImageFormat::Png:  return readPng(data, len, mr);
        case keel::ImageFormat::Tiff: return readTiff(std::vector<std::uint8_t>(bytes.begin(), bytes.end()), 1u, mr);
        case keel::ImageFormat::Jpeg: return readJpeg(data, len, mr);
        case keel::ImageFormat::Bmp:  return readBmp(data, len, mr);
        case keel::ImageFormat::Gif:  return readGifViaKeel(data, len, mr);
        case keel::ImageFormat::Pnm:  return readPnm(data, len, mr);
        case keel::ImageFormat::Tga:  return readTga(data, len, mr);
        case keel::ImageFormat::Unknown:
            break;
    }

    try { return readPng(data, len, mr); } catch (...) {}
    try { return readBmp(data, len, mr); } catch (...) {}
    try { return readJpeg(data, len, mr); } catch (...) {}
    try { return readTga(data, len, mr); } catch (...) {}

    throw Error("imread: failed to decode image - unsupported or unknown format",
                0, 0, "imread", "", "numkit:imread:unknownFormat");
}

Value imread(const std::string &path, std::pmr::memory_resource *mr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw Error("imread: failed to load '" + path + "' - can't open",
                    0, 0, "imread", "", "numkit:imread:load");
    }
    std::string bytes((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    return imreadFromBytes(bytes, mr);
}

std::string imwriteToBytes(const Value &A, const std::string &ext,
                           std::pmr::memory_resource * /*mr*/) {
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
        const auto buf = writeTiffToBytes(A, "none", nullptr);
        return std::string(buf.begin(), buf.end());
    }

    throw Error("imwrite: unsupported extension '" + ext +
                "' (supported: .png, .jpg, .bmp, .tga, .pnm, .tif)",
                0, 0, "imwrite", "", "numkit:imwrite:ext");
}

void imwrite(const Value &A, const std::string &path, std::pmr::memory_resource *mr) {
    const std::string ext = lowerExt(path);
    if (ext == "tif" || ext == "tiff") {
        writeTiff(A, path, "none", /*appendMode=*/false);
        return;
    }
    const std::string bytes = imwriteToBytes(A, ext, mr);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        throw Error("imwrite: cannot open '" + path + "' for write",
                    0, 0, "imwrite", "", "numkit:imwrite:write");
    }
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!f) {
        throw Error("imwrite: failed to write '" + path + "'",
                    0, 0, "imwrite", "", "numkit:imwrite:write");
    }
}

Value imfinfoFromBytes(const std::string &bytes, const std::string &filename,
                       std::pmr::memory_resource *mr) {
    const auto *data = reinterpret_cast<const std::uint8_t *>(bytes.data());
    const std::size_t len = bytes.size();

    std::uint32_t W = 0, H = 0;
    std::uint16_t bits = 8, channels = 1;
    std::string fmt = formatStringFromKeel(
        keel::sniff_image_format(keel::ArrayRef<keel::u8>(data, len)));
    if (fmt.empty()) fmt = lowerExt(filename);

    bool ok = false;
    if (fmt == "png") {
        ok = peekPng(data, len, W, H, bits, channels);
    } else if (fmt == "tif" || fmt == "tiff") {
        fmt = "tif";
        std::vector<std::uint8_t> buf(bytes.begin(), bytes.end());
        peekTiff(buf, W, H, bits, channels);
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
        auto gifInfo = keel::peek_gif(keel::ArrayRef<keel::u8>(data, len));
        if (gifInfo) {
            W = gifInfo.value().width;
            H = gifInfo.value().height;
            bits = gifInfo.value().bits_per_sample;
            channels = gifInfo.value().channels;
            ok = true;
        }
    }

    if (!ok) {
        throw Error("imfinfo: failed to parse header for '" + filename + "'",
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

Value imfinfo(const std::string &path, std::pmr::memory_resource *mr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw Error("imfinfo: failed to read '" + path + "' - can't open",
                    0, 0, "imfinfo", "", "numkit:imfinfo:read");
    }
    std::string bytes((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    return imfinfoFromBytes(bytes, path, mr);
}

} // namespace numkit::image
