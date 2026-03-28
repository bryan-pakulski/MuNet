# μNet

μNet is a lightweight C++ tensor + autograd framework with Python bindings.
It supports CPU execution out of the box and optional CUDA/Vulkan backends,
with centralized op dispatch, backend capability querying, and explicit fallback policy.

## Current repository state

- Core runtime (`munet_core`) is active and used by both training and inference surfaces.
- Training APIs (`nn`, `optim`, losses, autograd) are available in C++ and Python.
- Inference APIs and serialization flows are available (`munet_inference`, demos under `demos/inference/`).
- Backend dispatch now includes:
  - capability-based support checks,
  - fallback reason accounting,
  - accelerator→CPU fallback telemetry counters,
  - optional fail-fast mode for unexpected accelerator fallbacks.

## Architecture highlights

- **Backend registration/caching:** `BackendManager` + `BackendRegistry`.
- **Dispatch engine:** `src/core/op_dispatch.*` owns op metadata, policy checks, and fallback decisions.
- **Backend capability API:** `Backend::query_support(feature, dtype[, shape])`.
- **Vulkan backend state ownership:** mutable runtime/device state is backend-owned (not file-global mutable maps/pools).

## Diagnostics & safety env vars

- `MUNET_PROFILE=1` — profiler collection + summary.
- `MUNET_DEBUG=1` — debug logging/checks.
- `MUNET_LOG_LEVEL=0..3` — log verbosity.
- `MUNET_DISPATCH_DECISION_DUMP=1` — emit dispatch decision lines.
- `MUNET_FAIL_FAST_ACCELERATOR_CPU_FALLBACK=1` — throw immediately when a CUDA/Vulkan tensor is dispatched to CPU fallback.

## Build

### Requirements

- CMake 3.10+
- C++17 compiler
- Python 3.10+
- Optional: CUDA Toolkit (`nvcc`)
- Optional: Vulkan SDK (`glslc` in `PATH`)

### Common commands

```bash
make build-debug
make build-release
```

Or with raw CMake:

```bash
cmake -S . -B build
cmake --build build -j
```

## Test

```bash
make unit-test      # debug gtest binary
make py-test        # Python integration tests
make perf-test      # opt-in perf suite (sets MUNET_RUN_PERF_TESTS=1)
```

For selective C++ tests:

```bash
./build/debug/munet_tests --gtest_filter=*BackendManager*
```

## Docs

- Main docs index: `documentation/index.md`
- Architecture docs: `documentation/architecture/`
- Performance/profiling guide: `documentation/performance/profiling.md`

Serve docs locally:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

## Demos

- LLM demos: `demos/llm/`
- Vision demos: `demos/mnist/`, `demos/unet/`
- Inference demos: `demos/inference/`
- Feature demos: `demos/features/`

