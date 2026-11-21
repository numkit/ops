// toolboxes/builtin/src/datatypes/strings/print.cpp

#include <keel/core.h>
#include <keel/string.h>
#include <numkit/builtin/strfun.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/shape_ops.hpp>
#include <numkit/value/error.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>

namespace numkit::builtin {

// ════════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════════

keel::String dispFormat(const Value &a)
{
    keel::String out;
    auto appendDouble = [&](double v) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6g", v);
        out.append(buf);
    };
    auto appendLL = [&](long long v) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%lld", v);
        out.append(buf);
    };
    auto appendUSize = [&](keel::usize v) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%zu", v);
        out.append(buf);
    };

    if (a.isChar()) {
        // Row vector (or 1×0): single line. Matrix: one row per line.
        auto d = a.dims();
        if (d.rows() <= 1) {
            out.append(a.toString());
        } else {
            for (keel::usize r = 0; r < d.rows(); ++r) {
                if (r > 0) out.push_back('\n');
                out.append(a.charRow(r));
            }
        }
    } else if (a.isEmpty()) {
        out.append("[]");
    } else if (a.type() == ValueType::DOUBLE) {
        if (a.isScalar()) {
            appendDouble(a.toScalar());
        } else {
            auto d = a.dims();
            if (d.rows() == 1) {
                out.push_back('[');
                for (keel::usize c = 0; c < d.cols(); ++c) {
                    if (c > 0) out.push_back(' ');
                    double v = a(0, c);
                    if (v == std::floor(v) && std::isfinite(v))
                        appendLL(keel::as<long long>(v));
                    else
                        appendDouble(v);
                }
                out.push_back(']');
            } else if (d.cols() == 1) {
                for (keel::usize r = 0; r < d.rows(); ++r) {
                    if (r > 0) out.push_back('\n');
                    double v = a(r, 0);
                    out.append("   ");
                    if (v == std::floor(v) && std::isfinite(v))
                        appendLL(keel::as<long long>(v));
                    else
                        appendDouble(v);
                }
            } else {
                const keel::usize R = d.rows(), C = d.cols();
                const int nd = d.ndim();
                const double *base0 = a.doubleData();
                forEachOuterPage(d, [&](keel::usize plin, const keel::usize *outerCoords) {
                    if (nd >= 3) {
                        out.append("(:,:");
                        for (int i = 2; i < nd; ++i) {
                            out.push_back(',');
                            appendUSize(outerCoords[i - 2] + 1);
                        }
                        out.append(") =\n");
                    }
                    const double *page = base0 + plin * R * C;
                    for (keel::usize r = 0; r < R; ++r) {
                        if (r > 0) out.push_back('\n');
                        out.append("   ");
                        for (keel::usize c = 0; c < C; ++c) {
                            double v = page[c * R + r];
                            out.push_back(' ');
                            if (v == std::floor(v) && std::isfinite(v))
                                appendLL(keel::as<long long>(v));
                            else
                                appendDouble(v);
                        }
                    }
                });
            }
        }
    } else if (a.isLogical()) {
        if (a.isScalar()) {
            out.push_back(a.toBool() ? '1' : '0');
        } else {
            auto d = a.dims();
            const keel::u8 *ld = a.logicalData();
            for (keel::usize r = 0; r < d.rows(); ++r) {
                if (r > 0) out.push_back('\n');
                out.append("   ");
                for (keel::usize c = 0; c < d.cols(); ++c) {
                    out.push_back(' ');
                    out.push_back(ld[d.sub2ind(r, c)] ? '1' : '0');
                }
            }
        }
    } else if (a.isStruct()) {
        for (auto entry : a.structFields()) {
            out.append("    ");
            out.append(entry.key);
            out.append(": ");
            out.append(entry.value.debugString());
            out.push_back('\n');
        }
    } else if (a.isCell()) {
        auto d = a.dims();
        out.push_back('{');
        appendUSize(d.rows());
        out.push_back('x');
        appendUSize(d.cols());
        out.append(" cell}");
    } else if (a.isComplex()) {
        if (a.isScalar()) {
            auto c = a.toComplex();
            if (c.real() != 0.0 || c.imag() == 0.0)
                appendDouble(c.real());
            if (c.imag() != 0.0) {
                if (c.real() != 0.0 && c.imag() > 0)
                    out.push_back('+');
                appendDouble(c.imag());
                out.push_back('i');
            }
        } else {
            out.append(a.debugString());
        }
    } else {
        out.append(a.debugString());
    }
    out.push_back('\n');
    return out;
}

} // namespace numkit::builtin
