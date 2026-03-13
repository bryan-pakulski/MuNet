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

- [x] Refactor CPU backend kernels to stop assuming `float*` in operator codepaths for Phase-1 core op set.
- [x] Introduce typed dispatch for core ops (`add`, `mul`, `matmul`, `softmax`, `sum`, losses) in CPU fallback path.
- [x] Add compute dtype vs storage dtype split in backend execution contracts (CPU compute-plan + dispatch context scaffold implemented).
- [x] Implement fallback behavior (`error` vs `warn_and_upcast`) when kernels are unavailable for requested dtypes (CPU fallback path).
- [x] Add explicit rules for integer quantized flows (`Int4`/`Int8`) with dequant/accumulate/requant boundaries in CPU fallback path.

### Phase 2 — Mixed precision training

- [~] Add autocast context for forward op dispatch (C++ + Python) (API + bindings + Tensor forward-op autocast integration for arithmetic/activation/loss, plus explicit `AutocastOp` policy table used by Tensor entrypoints (with scoped policy overrides via guard for controlled rollout/testing, including Python context helpers plus override snapshot/restore APIs); `layer_norm`, `upsample2d`, and `max_pool2d` are now enabled via policy while `conv2d`/`batch_norm` remain skip-by-default; broader module coverage still pending).
- [~] Add gradient scaling (`static`, `dynamic`) and overflow detection (GradScaler scaling/unscale path with finite-check overflow gating, dynamic growth/backoff behavior, Python bindings, and tests added; broader backend-specific overflow signals pending).
- [~] Keep optimizer state/master weights in FP32 while model tensors may be FP16/BF16 (initial FP32-master SGD/Adam utilities added).
- [~] Enforce FP32 accumulation for numerically sensitive ops (norms, reductions, losses) (implemented for `sum`, `mse_loss`, `cross_entropy`, and low-precision `softmax`/`log_softmax`/`layer_norm` CPU paths; broader op coverage pending).
- [~] Add training parity checks against FP32 baselines (deterministic single-step and multi-step FP32 vs FP32-master parity tests added in C++; Python/model-level loops still pending environment deps and broader model coverage).

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
- [] Expand default-enabled autocast policy for remaining skipped spatial/norm ops with per-op validation.
- [~] Add longer-horizon FP32-vs-AMP training parity checks (multi-step C++ parity loops added; broader model-level and Python parity loops pending test dependencies).
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
