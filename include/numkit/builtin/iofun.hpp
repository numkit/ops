// src/builtin/include/numkit/builtin/iofun.hpp
//
// Pure C++ Formatted input/output and stream printing functions.
#pragma once

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <cstdio>
#include <numkit/value/value.hpp>

namespace numkit::builtin {

/// @addtogroup group_iofun
/// @{


/// @file
/// @ingroup group_iofun
/// @brief Formatted text I/O and display builtins.
///
/// Provides clean, engine-free C++ API for formatted string generation (`sprintf`),
/// stream printing (`fprintf`, `disp`), and text parsing (`sscanf`, `textscan`).

/// @brief Formats data into a character array or string according to format specifier `fmt` (`sprintf(fmt, ...)`).
/// @param fmt Format specifier string Value.
/// @param args Span of arguments to format.
/// @param mr Memory resource for allocations (nullptr for default).
/// @return Formatted character array or string Value.
/// @see fprintf, sscanf
Value sprintf(const Value &fmt, keel::ArrayRef<Value> args, keel::Allocator *mr = nullptr);

/// @brief Formats data into a C++ string according to format string `fmt`.
/// @param fmt Format string.
/// @param args Span of arguments to format.
/// @param mr Memory resource.
/// @return Formatted keel::String.
/// @see fprintf, disp
keel::String sprintf(keel::StringRef fmt, keel::ArrayRef<Value> args, keel::Allocator *mr = nullptr);

/// @brief Formats Value for terminal display via `disp`.
/// @param a Input value to format.
/// @return Formatted string.
keel::String dispFormat(const Value &a);

/// @brief Displays value to the standard output or specified stream (`disp(v)`).
/// @param v Value to display.
/// @param stream Target C file stream (defaults to stdout).
void disp(const Value &v, std::FILE *stream = stdout);

/// @brief Formats and writes data to an output stream (`fprintf(stream, fmt, ...)`).
/// @param stream Target C file stream.
/// @param fmt Format string.
/// @param args Span of arguments to format.
/// @param mr Memory resource.
/// @return Number of characters written.
/// @see sprintf, disp
int fprintf(std::FILE *stream, keel::StringRef fmt, keel::ArrayRef<Value> args, keel::Allocator *mr = nullptr);

/// @brief Reads formatted data from a string according to format specifier `fmt` (`sscanf(str, fmt)`).
/// @param str Source string to scan.
/// @param fmt Format specifier string.
/// @param mr Memory resource.
/// @return Scanned values array.
/// @see textscan, sprintf
Value sscanf(keel::StringRef str, keel::StringRef fmt, keel::Allocator *mr = nullptr);

/// @brief Reads formatted data from a string with multi-output support (`[A, count, errmsg, nextindex] = sscanf(...)`).
/// @param args Input arguments span `[str, fmt, size]`.
/// @param nargout Number of requested outputs.
/// @param outs Output spans to fill with results.
/// @param mr Memory resource.
void sscanf(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs, keel::Allocator *mr = nullptr);


/// @}
} // namespace numkit::builtin
