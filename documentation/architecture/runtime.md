# Runtime Architecture

## Core Components

- `Tensor`/`TensorImpl`: user-facing and internal tensor state/metadata.
- `Storage`: owns backend allocation and raw buffer handle.
- `Backend`: abstract interface for all kernels and memory operations.
- `BackendManager`: factory + cache for backend instances by `(DeviceType, index)`.
- `Autograd Engine`: backward graph execution for differentiable ops.

## Execution Flow

1. High-level API call (`Tensor`/`ops`) allocates output tensor.
2. Call is dispatched through backend interface (`Backend`).
3. If gradients are enabled, backward node is linked.
4. Optional trace/profiler metadata is recorded.

## Layering

- `munet_core`: tensor/runtime/backend/autograd primitives.
- `munet_training`: training-side APIs (`nn`, `optim`, loss wiring).
- `munet_inference`: inference-side API surface with eval-focused behavior.
