// toolboxes/builtin/src/io_helpers.hpp
//
// Private helpers shared between fileio.cpp (fread/fwrite) and
// scan.cpp (fscanf/sscanf/textscan). Not a public header — lives
// in src/ and is only on the PRIVATE include path for toolboxes/builtin.

#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/value/value.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>

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
            throw Error(std::string(fn) + ": size must be Inf or a non-negative integer");
        return SizeSpec{SizeSpec::Kind::Flat, static_cast<keel::usize>(s), 0, 0};
    }
    if (sz.numel() == 2) {
        keel::f64 r = sz(0);
        keel::f64 c = sz(1);
        if (r < 0 || !std::isfinite(r) || std::isinf(r))
            throw Error(std::string(fn) + ": rows in [m n] must be finite non-negative");
        keel::usize rows = static_cast<keel::usize>(r);
        keel::usize cols;
        keel::usize limit;
        if (std::isinf(c)) {
            cols = SIZE_MAX;
            limit = SIZE_MAX;
        } else {
            if (c < 0 || !std::isfinite(c))
                throw Error(std::string(fn)
                             + ": cols in [m n] must be finite non-negative or Inf");
            cols = static_cast<keel::usize>(c);
            limit = rows * cols;
        }
        return SizeSpec{SizeSpec::Kind::Matrix, limit, rows, cols};
    }
    throw Error(std::string(fn) + ": size must be scalar, Inf, or a 2-element vector");
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

// ── Precision + endian parsing (fread / fwrite) ──────────────────────
// kind: 0 = unsigned, 1 = signed, 2 = float. byteSize in bytes.
inline std::optional<std::pair<int, keel::usize>> parsePrecision(const std::string &raw)
{
    std::string p = raw;
    auto arrow = p.find("=>");
    if (arrow != std::string::npos) p = p.substr(0, arrow);
    while (!p.empty() && std::isspace(static_cast<unsigned char>(p.front()))) p.erase(0, 1);
    if (!p.empty() && p.front() == '*') p.erase(0, 1);
    while (!p.empty() && std::isspace(static_cast<unsigned char>(p.front()))) p.erase(0, 1);
    while (!p.empty() && std::isspace(static_cast<unsigned char>(p.back()))) p.pop_back();
    for (char &c : p) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (p == "uint8" || p == "uchar" || p == "unsigned char" || p == "char" || p == "char*1")
        return std::make_pair(0, keel::usize{1});
    if (p == "int8" || p == "schar" || p == "signed char" || p == "integer*1")
        return std::make_pair(1, keel::usize{1});
    if (p == "uint16" || p == "ushort" || p == "unsigned short")
        return std::make_pair(0, keel::usize{2});
    if (p == "int16" || p == "short" || p == "integer*2")
        return std::make_pair(1, keel::usize{2});
    if (p == "uint32" || p == "uint" || p == "unsigned int" || p == "ulong" || p == "unsigned long")
        return std::make_pair(0, keel::usize{4});
    if (p == "int32" || p == "int" || p == "long" || p == "integer*4")
        return std::make_pair(1, keel::usize{4});
    if (p == "uint64")
        return std::make_pair(0, keel::usize{8});
    if (p == "int64" || p == "integer*8")
        return std::make_pair(1, keel::usize{8});
    if (p == "single" || p == "float" || p == "float32" || p == "real*4")
        return std::make_pair(2, keel::usize{4});
    if (p == "double" || p == "float64" || p == "real*8")
        return std::make_pair(2, keel::usize{8});
    return std::nullopt;
}

// Returns true for big-endian, false for little-endian. 'native' → LE
// because every target we support (x86_64, ARM64, WASM) is LE.
inline bool parseEndian(const std::string &raw, const char *fn)
{
    std::string lo;
    lo.reserve(raw.size());
    for (char c : raw) lo += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (lo == "b" || lo == "ieee-be" || lo == "ieee-be.l64") return true;
    if (lo == "l" || lo == "ieee-le" || lo == "ieee-le.l64"
        || lo == "n" || lo == "native")
        return false;
    throw Error(std::string(fn) + ": unsupported machine format '" + raw + "'");
}

inline void byteSwap(char *p, keel::usize n)
{
    for (keel::usize i = 0, j = n - 1; i < j; ++i, --j) std::swap(p[i], p[j]);
}

} // namespace numkit::ops
