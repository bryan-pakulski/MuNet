# Backend Architecture

## Contract surface

Backends implement the `Backend` contract in `src/core/backend.hpp` and expose
optional capability interfaces (allocation/transfer, elementwise, reduction,
BLAS, shape, loss, spatial, normalization, optimizer, random fill).

Capability checks are centralized through:

- `Backend::query_support(feature, dtype[, shape])`
- fallback policy + preferred accumulation dtype from `BackendSupport`

This allows partial backend implementations while keeping dispatch behavior
explicit and testable.

## Registration and caching

- `BackendRegistry` provides explicit registration and cache control.
- `BackendManager` delegates to the default process-wide registry.
- Tests may override backend factories with `BackendManager::register_backend(...)`
  and clear cache per device type/index for isolation.

## Dispatch interaction

`src/core/op_dispatch.*` owns policy resolution. Backends do not silently choose
fallback behavior themselves; they report support and the dispatch layer decides
whether to run backend-native or CPU fallback.

Current observability in dispatch includes:

- `dispatch.resolve.backend.<Op>`
- `dispatch.resolve.cpu_fallback.<Op>`
- `dispatch.fallback.reason.{dtype|shape|feature|policy}`
- accelerator CPU-fallback telemetry counters (programmatic snapshot/reset APIs)
- optional fail-fast via `MUNET_FAIL_FAST_ACCELERATOR_CPU_FALLBACK=1`

## Current backend set

- CPU (always)
- CUDA (optional build/runtime)
- Vulkan (optional build/runtime)

## Vulkan state ownership notes

Mutable Vulkan runtime state is backend-owned (instance/device/runtime struct
fields) rather than file-static mutable allocator/descriptor pools, which
reduces cross-instance and cross-test coupling.

## Test availability probing

Device inclusion in parameterized suites should use a real health check
(`real op + synchronize + copy back`) rather than just constructing
`Tensor({1}, device)`. Repository test utilities now follow this policy.

