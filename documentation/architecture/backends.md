# Vulkan Runtime Architecture

MuNet now has one public execution surface: `vulkan:<index>`.

## Runtime contract

The C++ runtime still uses the `Backend` type as the internal kernel contract, but
there is no multi-backend selection layer exposed to users. `BackendManager` owns a
single built-in Vulkan runtime registration and returns that runtime for Vulkan
devices.

`BackendSupport` is intentionally small: it reports whether a feature/dtype/shape
combination is supported and the preferred accumulation dtype. Unsupported
combinations throw explicit errors instead of flowing through a secondary policy matrix.

## Staging memory

Staging memory is an implementation detail used for scalar access, reference
metadata paths, and correctness-oriented helpers. It is not a public backend and
is not discovered or selected by users.

## Removed surfaces

- Runtime-loaded backend extension discovery.
- Alternate accelerator/runtime variants.
- Secondary policy matrices and telemetry counters for alternate execution paths.

## Current runtime set

- Built-in Vulkan runtime only.
