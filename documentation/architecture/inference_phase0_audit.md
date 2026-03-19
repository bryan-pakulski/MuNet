# Inference Runtime Phase 0 Audit

This document begins **Phase 0 - Boundary audit and deploy baseline** for the inference/runtime separation effort.

It captures the current inference dependency inventory, classifies training/runtime coupling points, defines target build profiles, and documents the baseline benchmark workflow that can be run on CPU-only and accelerated systems.

## Phase 0 status

- [~] Phase 0 - Boundary audit and deploy baseline
- [x] Dependency inventory written
- [x] Coupling points classified
- [x] Target build profiles defined
- [x] Current blockers documented
- [ ] Baseline latency and memory measurements collected from accelerated hardware

## Minimum runtime surface for edge deployments

The current working definition of a minimal MuNet deploy runtime is:

- `munet_core` for tensor/storage/backend execution primitives.
- `munet_inference` for deploy-facing runtime contracts:
  - module loading
  - shape contracts
  - warmup/compile
  - run / run_batch
  - lightweight stats
- no optimizer dependency in the public surface
- no training-loop requirement in the public surface
- no mandatory Python/ONNX tooling requirement for the C++ runtime target
- observability and profiling remaining optional rather than required for inference correctness

## Dependency inventory and classification

| Area | Current dependency | Classification | Why it exists today | Phase 0 note |
| --- | --- | --- | --- | --- |
| `src/inference.hpp` | `core/module.hpp` | required shared runtime primitive | Inference modules reuse the shared module graph, parameter registration, and device/dtype transfer logic. | This is a valid shared boundary so long as `core::Module` remains training-agnostic. |
| `src/inference.hpp` | `autograd/engine.hpp` via `GradMode` | training-only leakage | The inference engine disables gradient tracking by reaching into autograd state directly. | This is the clearest public-header leak today and should eventually move behind a slimmer inference-safe guard. |
| `src/inference.hpp` | trace/profiler timing helpers from `core/util.hpp` | optional observability/debug feature | Compile/run phases emit trace ids, spans, timers, and profiler-backed memory counters. | Useful for diagnostics, but this should stay optional so hot-path deploy builds do not pay for it by default. |
| `src/inference.hpp` | `EngineStats` memory counters backed by `Profiler` | optional observability/debug feature | The engine reports current/peak memory during load/compile/run. | Good for benchmarking, but a lean runtime may need a lighter-weight or compile-time gated memory accounting path. |
| `src/inference.hpp` | `load(std::shared_ptr<core::Module>)` | required shared runtime primitive | The engine executes a shared module abstraction so serialized/native-converted models can run through one path. | Acceptable as long as the shared module API does not grow training-only assumptions. |
| `src/bindings.cpp` | Python `Engine.load(...)` accepts `std::shared_ptr<nn::Module>` | training-only leakage | The Python inference entrypoint is typed in terms of the training namespace even though the C++ engine accepts `core::Module`. | This is a concrete binding-layer coupling that should be removed in a future separation phase. |
| `src/bindings.cpp` | Python extension target links `munet_training` only | training-only leakage | The monolithic Python module exposes training, serialization, and inference from one compiled extension. | This blocks a truly inference-only Python package today. |
| serialization helpers in `src/bindings.cpp` | Full reconstruction for built-in `nn::*` module types | deploy convenience API | Deployment needs model loading, but the current ownership model is still centered on training-side module types. | Phase 4 should separate deploy artifact requirements from training/checkpoint concerns more cleanly. |
| `python_src/onnx_integration.py` | ONNX conversion, coverage reports, downloads, deprecated runtime wrapper | deploy convenience API + optional tooling | These helpers support conversion and inspection workflows around inference deployment. | Useful, but they expand the inference namespace with development-time tooling that may not belong in the leanest runtime package. |
| `CMakeLists.txt` target graph | `munet_core` still contains tensor/backends/autograd graph primitives in one library | required shared runtime primitive with future split pressure | The current build keeps autograd inside the shared core for simplicity. | This is the main build-graph blocker to a stricter inference-only linkage boundary. |

## Current blockers to a truly inference-only build target

1. **Public inference headers still include autograd state management.** `src/inference.hpp` pulls in `autograd/engine.hpp` to manipulate `GradMode`, which means the inference surface is not yet isolated from autograd internals.
2. **Python inference loading is typed against training modules.** In the bindings layer, `munet.inference.Engine.load(...)` currently accepts `nn::Module`, which exposes a training-centric surface in the inference namespace.
3. **The Python package is monolithic.** `pybind11_add_module(munet ...)` links against `munet_training`, so Python consumers cannot build or distribute an inference-only extension today.
4. **Core remains a broad shared layer.** `munet_core` still packages tensor execution, backend orchestration, and autograd graph primitives together, preventing a hard inference-only link boundary.
5. **Inference namespace includes developer tooling.** ONNX conversion/reporting/download helpers are valuable, but they enlarge the deploy surface and should eventually be separated from the minimal runtime package.

## Target build profiles

### 1. Minimal CPU edge runtime

For Raspberry Pi-class and other constrained CPU deployments:

- build CPU backend only
- link `munet_core` + `munet_inference`
- disable or minimize optional debug/profiler overhead where possible
- prefer serialized/native models plus pre-declared shape contracts
- avoid packaging training APIs, optimizers, or ONNX development utilities as required runtime dependencies

### 2. General-purpose desktop/server runtime

For workstation and standard server deployments:

- build CPU plus whichever local accelerators are available
- keep `munet_inference` as the primary deploy API
- allow serialization and lightweight observability features for integration and service debugging
- keep conversion/reporting utilities available, but treat them as tooling rather than core hot-path requirements

### 3. Accelerator-enabled runtime

For enterprise GPU and high-throughput deployments:

- enable CUDA and/or Vulkan at build time
- validate both cold-start and steady-state behavior on target accelerators
- focus on backend initialization cost, transfer cost, and shape-contract reuse
- use the baseline benchmark below to capture per-device latency and memory snapshots that can be compared back to CPU-only runs

## Baseline benchmark workflow

MuNet now includes a dedicated benchmark executable: `munet_inference_baseline`.

### What it measures

- cold load wall time
- load-to-device time
- eval transition time
- compile time
- compile input preparation time
- compile forward time
- compile warmup time
- steady-state single-run latency
- steady-state batch-run latency
- current and peak memory counters reported by the engine/profiler

### Build the benchmark

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target munet_inference_baseline -j
```

### CPU baseline example

```bash
./build/munet_inference_baseline \
  --device cpu \
  --dtype float32 \
  --batch 32 \
  --input-dim 256 \
  --hidden-dim 512 \
  --output-dim 128 \
  --warmup-runs 5 \
  --single-run-iters 50 \
  --batch-run-inputs 4 \
  --batch-run-iters 20
```

## CPU-only baseline snapshot from this environment

The benchmark was validated in this CPU-only environment with the following command:

```bash
./build/munet_inference_baseline --device cpu --dtype float32 --batch 8 --input-dim 32 --hidden-dim 64 --output-dim 16 --warmup-runs 0 --single-run-iters 3 --batch-run-inputs 2 --batch-run-iters 2
```

Observed output summary:

- cold load wall time: `0.1295 ms`
- compile time: `0.3194 ms`
- steady single-run average: `0.0987 ms`
- steady batch-run average: `0.3559 ms` for 2 inputs (`0.1779 ms` per input)
- engine/profiler memory counters: `0` current / `0` peak in this CPU sample

This establishes the repository-side baseline workflow and a reference CPU measurement. Accelerated device results are still needed from real target hardware.

### Accelerator baseline examples

CUDA:

```bash
./build/munet_inference_baseline --device cuda --dtype float16
```

Vulkan:

```bash
./build/munet_inference_baseline --device vulkan --dtype float16
```

### Output format

The executable prints a JSON payload containing:

- target device
- dtype
- build profile hint
- compiled input/output shapes
- cold load timings
- compile timings
- average steady-state single-run timings
- average steady-state batch-run timings
- current/peak memory counters

### What to send back in the next prompt

For accelerated hardware that is not available in this environment, please run the benchmark locally and send back:

- the exact command used
- the full JSON output
- any device-specific notes (GPU model, driver/runtime quirks, expected concurrency limits)

That will let the next phase update the plan with real baseline numbers instead of placeholders.
