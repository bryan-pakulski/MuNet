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

`src/core/op_dispatch.*` owns policy resolution. The public runtime is Vulkan-only:
backend capability checks either route to supported Vulkan kernels or return explicit
unsupported-operation errors.

Current observability in dispatch includes:

- `dispatch.resolve.backend.<Op>`
- `dispatch.resolve.vulkan_staging.<Op>`
- `dispatch.fallback.reason.{dtype|shape|feature|policy}`
- Vulkan telemetry snapshot/reset APIs
- optional fail-fast via `MUNET_FAIL_FAST_VULKAN_UNSUPPORTED=1`

## Current backend set

- Vulkan

## Vulkan state ownership notes

Mutable Vulkan runtime state is backend-owned (instance/device/runtime struct
fields) rather than file-static mutable allocator/descriptor pools, which
reduces cross-instance and cross-test coupling.

## Test availability probing

Device inclusion in parameterized suites should use a real health check
(`real op + synchronize + copy back`) rather than just constructing
`Tensor({1}, device)`. Repository test utilities now follow this policy.

