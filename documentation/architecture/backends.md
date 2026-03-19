# Backend Architecture

## Contract

Backends now expose a small coordination surface through `Backend` in
`src/core/backend.hpp`, while the executable kernels live behind composable
capability groups:

- allocation / transfer
- elementwise
- reduction
- BLAS / matmul
- shape / concat
- loss
- spatial ops
- normalization
- optimizer kernels
- random fill

`Backend::query_support(feature, dtype[, shape])` centralizes capability checks,
preferred accumulation dtype, and fallback policy so partial implementations can
participate safely.

## Registration

Use `BackendRegistry` for explicit backend registration, cache management, and
isolated tests. `BackendManager` now delegates to the default process-wide
registry, so existing call sites still work while tests can create their own
registries.

## Debug/Profiling Wrapper

`DebugBackend` remains a decorator. The default registry applies it as an
optional backend decoration step when debug/profile flags are enabled, rather
than coupling that behavior to backend factory selection.

## Current Backends

- CPU
- CUDA (optional)
- Vulkan (optional)

## Notes

Device selection is index-aware for CUDA and Vulkan through `Device.index`.
