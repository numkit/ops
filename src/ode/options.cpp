// toolboxes/ode/src/options.cpp
//
// odeset / odeget â€” options struct for the ODE solvers.
//
// Reference: Shampine & Reichelt, "The MATLAB ODE Suite", 1997.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/format.h>
#include <keel/string.h>
#include <numkit/ode/options.hpp>

#include <numkit/value/error.hpp>

namespace numkit::ode {

namespace {

// Canonical option names (MATLAB R2025b odeset). Case stays as MATLAB
// writes it; the lookup is case-insensitive.
const char *const kOptionNames[] = {
    "AbsTol", "BDF", "Events", "InitialSlope", "InitialStep",
    "Jacobian", "JPattern", "Mass", "MassSingular", "MaxOrder",
    "MaxStep", "MStateDependence", "MvPattern", "NonNegative",
    "NormControl", "OutputFcn", "OutputSel", "Refine", "RelTol",
    "Stats", "Vectorized",
};
constexpr keel::usize kNumOptions = sizeof(kOptionNames) / sizeof(kOptionNames[0]);

// Map case-insensitive name â†’ canonical capitalisation; return nullptr
// if not a recognised option.
const char *canonical_name(keel::StringRef raw)
{
    for (keel::usize i = 0; i < kNumOptions; ++i) {
        if (raw.equals_ignore_ascii_case(kOptionNames[i])) return kOptionNames[i];
    }
    return nullptr;
}

// Build the default options struct (all fields = []).
Value default_struct(keel::Allocator *mr)
{
    Value s = Value::structure(mr);
    for (keel::usize i = 0; i < kNumOptions; ++i) {
        s.field(kOptionNames[i]) = Value::Empty;
    }
    return s;
}

} // anonymous

Value odeset(const Value *args, keel::usize nargs,
             keel::Allocator *mr)
{
    // Start from defaults, optionally merge in an existing struct, then
    // process name-value pairs.
    Value out = default_struct(mr);
    keel::usize i = 0;
    // First-arg-is-struct cases.
    while (i < nargs && args[i].isStruct()) {
        const Value &src = args[i];
        for (keel::usize k = 0; k < kNumOptions; ++k) {
            const char *name = kOptionNames[k];
            if (src.hasField(name)) {
                out.field(name) = src.field(name);
            }
        }
        ++i;
    }
    // Name-value pairs.
    while (i + 1 < nargs) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error(keel::format("odeset: expected a string option name at arg {}", i + 1),
                        0, 0, "odeset", "", "numkit:odeset:badArg");
        const char *name = canonical_name(args[i].toString());
        if (!name)
            throw Error(keel::format("odeset: unknown option '{}'", args[i].toString()),
                        0, 0, "odeset", "", "numkit:odeset:badName");
        out.field(name) = args[i + 1];
        i += 2;
    }
    if (i < nargs)
        throw Error(keel::format("odeset: trailing unmatched argument at position {}", i + 1),
                    0, 0, "odeset", "", "numkit:odeset:unpaired");
    return out;
}

Value odeget(const Value &opts, const Value &name,
             const Value &default_v, keel::Allocator *mr)
{
    // Probed R2025b resolution model:
    //   1. [] options â†’ the default (or []).
    //   2. name must be a text scalar (char / 1x1 string; cells and
    //      numbers reject with the mustBeTextScalar class).
    //   3. An EXACT (case-sensitive) field of opts wins outright â€” even
    //      for non-option fields (odeget(o3,'Foo') returns o3.Foo).
    //   4. Otherwise name prefix-matches the canonical option list,
    //      case-insensitively: no match â†’ invalid name (even with a
    //      default given); ambiguous â†’ error; unique â†’ the canonical
    //      field, exact-case, and an absent or empty field falls back
    //      to the default.
    (void)mr;
    if (opts.isEmpty())
        return default_v.isEmpty() ? Value::Empty : default_v;
    if (!opts.isStruct())
        throw Error(keel::format("odeget: input 1 (options) must be an options struct "
                                 "created by odeset (got a {} array).",
                                 mtypeName(opts.type())),
                    0, 0, "odeget", "", "numkit:odeget:notStruct");
    if (!((name.isChar() && name.numel() >= 0)
          || (name.isString() && name.numel() == 1)))
        throw Error("odeget: input 2 (name) must be a character vector or "
                    "string scalar.",
                    0, 0, "odeget", "", "numkit:odeget:mustBeTextScalar");

    const keel::String raw = name.toString();

    // 3. Exact field first.
    if (opts.hasField(raw.c_str())) {
        const Value &v = opts.field(raw.c_str());
        if (!v.isEmpty()) return v;
        return default_v.isEmpty() ? Value::Empty : default_v;
    }

    // 4. Case-insensitive unambiguous prefix over the canonical names.
    const char *canon = nullptr;
    int matches = 0;
    if (raw.empty()) {
        matches = keel::as<int>(kNumOptions);   // every name matches ""
    } else {
        const keel::StringRef rawRef = raw.as_ref();
        for (keel::usize i = 0; i < kNumOptions; ++i) {
            const keel::StringRef optRef(kOptionNames[i]);
            if (optRef.size() >= rawRef.size()
                && optRef.substr(0, rawRef.size()).equals_ignore_ascii_case(rawRef)) {
                ++matches;
                canon = kOptionNames[i];
            }
        }
    }
    if (matches == 0)
        throw Error(keel::format("odeget: unrecognized option name '{}'.", raw),
                    0, 0, "odeget", "", "numkit:odeget:invalidName");
    if (matches > 1)
        throw Error(keel::format("odeget: '{}' is an ambiguous option prefix "
                                 "â€” use enough characters to be unambiguous.", raw),
                    0, 0, "odeget", "", "numkit:odeget:ambiguousName");

    if (opts.hasField(canon)) {
        const Value &v = opts.field(canon);
        if (!v.isEmpty()) return v;
    }
    return default_v.isEmpty() ? Value::Empty : default_v;
}

} // namespace numkit::ode
