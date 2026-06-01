# Current Runtime Architecture

MuNet's runtime is intentionally small and Vulkan-only.

## Core components

- `Tensor` / `TensorImpl`: tensor metadata, storage ownership, and optional
  training/autograd metadata.
- `Storage`: owns the allocation returned by the runtime for a Vulkan device.
- `BackendManager`: returns the single built-in Vulkan runtime registration.
- `ops::resolve_dispatch(...)`: validates op metadata and decides between a
  supported Vulkan runtime call or a small reference metadata path for view-like
  tensor operations.
- `inference::Engine`: inference-only wrapper that disables autograd graph
  construction while running loaded modules.

## Execution flow

1. Tensor/op API selects operation metadata.
2. Dispatch validates dtype/shape support against the Vulkan runtime.
3. Supported ops execute through the Vulkan runtime contract.
4. Metadata-only ops such as reshape/narrow/transpose use direct reference paths.
5. Unsupported dtype/shape/feature combinations throw explicit errors.

## Training split

Training/autograd remains available through `munet_training` and Python training
APIs, but inference execution is isolated by `munet_inference` and
`inference::Engine` guards that disable grad recording during runs.

## Profiling

The profiler records dispatch stages and runtime operation labels. Use
`make perf-test` for operator min/avg/max baselines and profiler breakdowns.
