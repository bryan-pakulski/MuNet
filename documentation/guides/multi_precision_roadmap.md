# Multi-Precision Roadmap (Training + Inference)

This roadmap tracks MuNet multi-precision support across training and inference.

Legend:
- [x] completed
- [~] in progress
- [] not started

## Current implementation status

### Phase 0 — Foundation and API scaffolding

- [x] Extend `DataType` enum and `dtype_size()` to include:
  - `Float8E4M3FN`, `Float8E5M2`
  - `Float16`, `BFloat16`, `Float32`, `Float64`
  - `Int4`, `Int8`, `Int32`
- [x] Add dtype helper utilities:
  - `is_float_dtype(dt)`
  - `is_fp8(dt)`
  - `is_low_precision(dt)`
  - `accumulation_dtype(dt)`
  - `dtype_name(dt)`
- [x] Expose new dtypes in Python bindings.
- [x] Introduce inference precision policy API:
  - `LossScaleMode`
  - `PrecisionFallbackMode`
  - `PrecisionPolicy`
  - `EngineConfig::precision_policy`
  - `Engine::set_precision_policy(...)`, `Engine::precision_policy()`
- [x] Add unit tests for dtype helpers/sizing and inference precision policy plumbing.
- [~] Update docs to reflect implemented progress and next milestones.

### Phase 1 — Kernel dtype dispatch and fallback

- [~] Refactor CPU backend kernels to stop assuming `float*` in operator codepaths (started for core op paths in CPU backend).
- [~] Introduce typed dispatch for core ops (`add`, `mul`, `matmul`, `softmax`, `sum`, losses) (CPU fallback path now uses dtype-aware load/store + FP32 accumulation).
- [~] Add compute dtype vs storage dtype split in backend execution contracts (initial CPU compute-plan + dispatch context scaffolding added).
- [~] Implement fallback behavior (`error` vs `warn_and_upcast`) when kernels are unavailable for requested dtypes (CPU fallback mode now respects dispatch config).
- [~] Add explicit rules for integer quantized flows (`Int4`/`Int8`) with dequant/accumulate/requant boundaries (initial CPU rule added; exact packed formats/kernels still pending).

### Phase 2 — Mixed precision training

- [~] Add autocast context for forward op dispatch (C++ + Python) (initial autocast state + guard API added).
- [~] Add gradient scaling (`static`, `dynamic`) and overflow detection (initial GradScaler API + finite-check/unscale flow added).
- [] Keep optimizer state/master weights in FP32 while model tensors may be FP16/BF16.
- [] Enforce FP32 accumulation for numerically sensitive ops (norms, reductions, losses).
- [] Add training parity checks against FP32 baselines.

### Phase 3 — Inference precision runtime

- [~] Precision policy object exists in inference API.
- [] Wire `PrecisionPolicy` into actual engine runtime execution and cast planning.
- [] Add op capability registry by backend + dtype + accumulation mode.
- [] Add explicit inference modes:
  - FP16/BF16 full-graph mode
  - mixed precision per-op mode
  - FP8 serving path
  - INT8/INT4 quantized serving path
- [] Add calibration and quantization hooks in inference flow.

### Phase 4 — Serialization and portability

- [] Extend model serialization format to persist new dtypes (`FP8`, `BF16`, `INT8`, `INT4`, etc.).
- [] Persist quantization metadata (scale/zero-point or block scales).
- [] Persist precision policy metadata/version in artifacts.
- [] Backward-compatible load path for legacy FP32 checkpoints.

### Phase 5 — Validation and rollout

- [] Add correctness matrix tests per dtype and op family.
- [] Add end-to-end training checks for FP16/BF16 + scaler behavior.
- [] Add inference numerical regression checks for FP8/INT8/INT4 modes.
- [] Add performance dashboards (latency, throughput, memory) per dtype/backend.
- [] Add CI lanes for CPU-only and optional GPU dtype coverage.

## What’s done vs missing (short view)

### Done
- [x] API-level dtype scaffolding for floating-point + integer quantized types.
- [x] Inference precision policy data model and Python exposure.
- [x] Basic unit tests for new type metadata and policy plumbing.

### Missing
- [~] Real kernel implementations beyond FP32-fast-path assumptions (partial CPU coverage for selected core ops).
- [] Quantized arithmetic paths for `Int8`/`Int4` (packing, scale handling, kernels).
- [] Runtime cast planner/executor using `PrecisionPolicy`.
- [] Serialization schema updates for precision and quantization metadata.
- [] Accuracy/performance validation for lower-precision execution.

## Immediate next implementation steps

1. [~] Implement typed CPU dispatch for `add`, `matmul`, and `softmax` with FP32 accumulation.
2. [] Add `Tensor::to(dtype)` cast path and internal conversion helpers.
3. [] Add backend fallback adapter (`warn_and_upcast`) for unsupported dtype ops.
4. [] Add initial INT8 inference path for `Linear/MatMul` (weight-only first), then extend to INT4.
