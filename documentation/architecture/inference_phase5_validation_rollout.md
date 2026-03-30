# Inference Runtime Phase 5 Validation and Rollout

This document closes **Phase 5 - Validation, benchmarking, and rollout** for the inference/runtime separation effort.

## Phase 5 status

- [x] Focused automated validation covers the new separation guarantees
- [x] Benchmark suites are defined for cold-start, warm run, memory-policy, and rollout scenarios
- [x] Repository-side before/after or tradeoff notes are published in docs
- [x] Migration notes for deploy-facing inference APIs are documented
- [x] Phase status in the main runtime plan is updated

## Validation coverage landed

### 1. Inference-only build/link coverage

- `munet_inference_boundary_check` is now exercised directly by CTest.
- `munet_inference_baseline_cpu_smoke` is also exercised by CTest so the benchmark/tooling path stays build- and run-valid in CI.

### 2. Autograd isolation guarantees

The inference test suite already covers:

- gradients disabled during `Engine::compile(...)` / `run(...)`
- autograd-tracked inputs rejected by default
- grad-tracked outputs rejected from deploy execution paths

### 3. Shape / dtype / device contract enforcement

Focused tests cover:

- compile-time expected input/output shape contracts
- float16 preservation through engine/device flows
- deploy serialization manifest validation and runtime-only payload enforcement
- deploy-first `load_for_inference(...)` round-trips

### 4. Backend fallback behavior on minimal builds

The backend manager tests now remain part of the Phase 5 validation story:

- deploy-vs-training backend feature role checks
- constrained fallback policy assertions
- dispatch fallback reason/profile coverage

### 5. Deploy workflow coverage

Deploy serialization + run workflow is now validated by:

- `munet.load_for_inference(...)` tests
- `munet.inference.load_serialized(...)` tests
- serialized-model -> `inference.Engine` run coverage
- ONNX native-sequential -> serialized deploy artifact coverage

## Benchmark suites

Phase 5 standardizes the following benchmark scenarios around `munet_inference_baseline`:

### A. CPU-only edge / cold-start

```bash
./build/munet_inference_baseline \
  --device cpu --dtype float32 \
  --batch 8 --input-dim 32 --hidden-dim 64 --output-dim 16 \
  --warmup-runs 0 \
  --single-run-iters 20 \
  --batch-run-inputs 2 --batch-run-iters 10 \
  --prepared-input-cache-entries 0 \
  --prepared-input-cache-max-bytes 0
```

Observed in this environment:

- cold load wall: **0.0645 ms**
- compile: **0.8170 ms**
- steady single-run avg wall: **0.7350 ms**
- steady batch per-input wall: **1.1161 ms**

### B. CPU-only edge / lean-mode tradeoff

```bash
./build/munet_inference_baseline \
  --device cpu --dtype float32 \
  --batch 8 --input-dim 32 --hidden-dim 64 --output-dim 16 \
  --warmup-runs 0 \
  --single-run-iters 20 \
  --batch-run-inputs 2 --batch-run-iters 10 \
  --lean-mode true \
  --prepared-input-cache-entries 0 \
  --prepared-input-cache-max-bytes 0
```

Observed in this environment:

- cold load wall: **0.0264 ms**
- compile: **0.7036 ms**
- steady single-run avg wall: **0.9463 ms**
- steady batch per-input wall: **0.6016 ms**

Tradeoff note:

- lean mode improved cold-start and batched per-input cost in this CPU-only run
- the steady single-run average was slightly worse here, so the rollout guidance is to treat `lean_mode` as a deploy-policy knob rather than an unconditional throughput win

### C. Memory-policy / batch reuse scenario

```bash
./build/munet_inference_baseline \
  --device cpu --dtype float32 \
  --batch 8 --input-dim 32 --hidden-dim 64 --output-dim 16 \
  --warmup-runs 0 \
  --single-run-iters 20 \
  --batch-run-inputs 2 --batch-run-iters 10 \
  --prepared-input-cache-entries 8 \
  --prepared-input-cache-max-bytes 67108864 \
  --preallocate-batch-inputs true
```

Observed in this environment:

- cold load wall: **0.1427 ms**
- compile: **1.1040 ms**
- steady single-run avg wall: **0.7975 ms**
- steady batch per-input wall: **0.5921 ms**

Tradeoff note:

- this CPU-only environment showed the biggest benefit on repeated batched runs
- prepared-input cache counters remained zero because no host-to-device transfer cache is needed on CPU-only execution, so this scenario mostly reflects preallocated repeated-batch setup rather than transfer reuse

### D. Accelerator rollout suites

Use the same benchmark with deployment-target hardware for:

- `--device cuda --dtype float16`
- `--device vulkan --dtype float16`

Phase 5 keeps these suites documented even when the current validation environment is CPU-only; accelerator comparisons remain intentionally opt-in and hardware-specific.

## Migration notes

- **Deploy loading:** use `munet.load_for_inference(...)` or `munet.inference.load_serialized(...)` for deployment code; use `munet.load_checkpoint(...)` for training/checkpoint reconstruction.
- **Deploy artifacts:** serialized `.npz` artifacts are now validated as **runtime-only** payloads, and training/checkpoint-style keys are rejected during deploy loading.
- **ONNX packaging:** only native-sequential ONNX conversions should be promoted as deploy artifacts (`compile_onnx(..., output_path="model.npz")`); graph-runtime ONNX results remain development tooling.
- **Performance tests:** GPU comparison tests are now skipped cleanly unless both CUDA and Vulkan are available and `MUNET_RUN_PERF_TESTS=1` is set, which keeps CPU-only CI green without hiding the accelerator benchmark suite.

## Rollout conclusion

Phase 5 closes with:

- automated coverage for separation guarantees
- benchmark workflows that can be rerun on CPU-only and accelerator hardware
- explicit tradeoff documentation instead of assuming every runtime knob is a universal win
- a stable, intentionally minimal deploy boundary for future inference work
