// toolboxes/builtin/src/io_helpers.hpp
//
// Private helpers shared between fileio.cpp (fread/fwrite) and
// scan.cpp (fscanf/sscanf/textscan). Not a public header — lives
// in src/ and is only on the PRIVATE include path for toolboxes/builtin.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/optional.h>
#include <keel/format.h>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>

namespace numkit::ops {

// ── SizeSpec: fread / fscanf size-argument parsing ───────────────────
struct SizeSpec
{
    enum class Kind { Flat, Matrix };
    Kind kind;
    keel::usize limit;
    keel::usize rows;
    keel::usize cols;
    bool matrix() const { return kind == Kind::Matrix; }
};

inline SizeSpec parseReadSize(const Value &sz, const char *fn)
{
    if (sz.isScalar()) {
        keel::f64 s = sz.toScalar();
        if (std::isinf(s))
            return SizeSpec{SizeSpec::Kind::Flat, SIZE_MAX, 0, 0};
        if (s < 0 || !std::isfinite(s))
            throw Error(keel::format("{}: size must be Inf or a non-negative integer", fn));
        return SizeSpec{SizeSpec::Kind::Flat, static_cast<keel::usize>(s), 0, 0};
    }
    if (sz.numel() == 2) {
        keel::f64 r = sz(0);
        keel::f64 c = sz(1);
        if (r < 0 || !std::isfinite(r) || std::isinf(r))
            throw Error(keel::format("{}: rows in [m n] must be finite non-negative", fn));
        keel::usize rows = static_cast<keel::usize>(r);
        keel::usize cols;
        keel::usize limit;
        if (std::isinf(c)) {
            cols = SIZE_MAX;
            limit = SIZE_MAX;
        } else {
            if (c < 0 || !std::isfinite(c))
                throw Error(keel::format("{}: cols in [m n] must be finite non-negative or Inf", fn));
            cols = static_cast<keel::usize>(c);
            limit = rows * cols;
        }
        return SizeSpec{SizeSpec::Kind::Matrix, limit, rows, cols};
    }
    throw Error(keel::format("{}: size must be scalar, Inf, or a 2-element vector", fn));
}

inline Value shapeFreadOutput(const keel::f64 *values, keel::usize n,
                              SizeSpec sz, keel::Allocator *mr = nullptr)
{
    if (!sz.matrix()) {
        if (n == 0)
            return Value::matrix(0, 0, ValueType::DOUBLE, mr);
        Value M = Value::matrix(n, 1, ValueType::DOUBLE, mr);
        std::memcpy(M.doubleDataMut(), values, n * sizeof(keel::f64));
        return M;
    }
    keel::usize cols_out = (sz.cols == SIZE_MAX)
                          ? (sz.rows == 0 ? 0 : (n + sz.rows - 1) / sz.rows)
                          : sz.cols;
    if (sz.rows == 0 || cols_out == 0)
        return Value::matrix(sz.rows, cols_out, ValueType::DOUBLE, mr);
    Value M = Value::matrix(sz.rows, cols_out, ValueType::DOUBLE, mr);
    std::memcpy(M.doubleDataMut(), values, n * sizeof(keel::f64));
    return M;
}

struct PrecisionInfo
{
    int first;
    keel::usize second;
};

// ── Precision + endian parsing (fread / fwrite) ──────────────────────
// kind: 0 = unsigned, 1 = signed, 2 = float. byteSize in bytes.
inline keel::Optional<PrecisionInfo> parsePrecision(keel::StringRef raw)
{
    keel::StringRef s = raw;
    keel::usize arrow = s.find("=>");
    if (arrow != keel::kInvalidIndex) s = s.substr(0, arrow);
    s = s.trim();
    if (!s.empty() && s[0] == '*') s = s.substr(1).trim();
    keel::SmallString<64> p;
    for (char c : s) p.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

    if (p == "uint8" || p == "uchar" || p == "unsigned char" || p == "char" || p == "char*1")
        return PrecisionInfo{0, 1};
    if (p == "int8" || p == "schar" || p == "signed char" || p == "integer*1")
        return PrecisionInfo{1, 1};
    if (p == "uint16" || p == "ushort" || p == "unsigned short")
        return PrecisionInfo{0, 2};
    if (p == "int16" || p == "short" || p == "integer*2")
        return PrecisionInfo{1, 2};
    if (p == "uint32" || p == "uint" || p == "unsigned int" || p == "ulong" || p == "unsigned long")
        return PrecisionInfo{0, 4};
    if (p == "int32" || p == "int" || p == "long" || p == "integer*4")
        return PrecisionInfo{1, 4};
    if (p == "uint64")
        return PrecisionInfo{0, 8};
    if (p == "int64" || p == "integer*8")
        return PrecisionInfo{1, 8};
    if (p == "single" || p == "float" || p == "float32" || p == "real*4")
        return PrecisionInfo{2, 4};
    if (p == "double" || p == "float64" || p == "real*8")
        return PrecisionInfo{2, 8};
    return keel::kNullOpt;
}

// Returns true for big-endian, false for little-endian. 'native' → LE
// because every target we support (x86_64, ARM64, WASM) is LE.
inline bool parseEndian(keel::StringRef raw, const char *fn)
{
    keel::SmallString<32> lo;
    for (char c : raw) lo.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    if (lo == "b" || lo == "ieee-be" || lo == "ieee-be.l64") return true;
    if (lo == "l" || lo == "ieee-le" || lo == "ieee-le.l64"
        || lo == "n" || lo == "native")
        return false;
    throw Error(keel::format("{}: unsupported machine format '{}'", fn, raw));
}

inline void byteSwap(char *p, keel::usize n)
{
    for (keel::usize i = 0, j = n - 1; i < j; ++i, --j) {
        char tmp = p[i];
        p[i] = p[j];
        p[j] = tmp;
    }
}

} // namespace numkit::ops
