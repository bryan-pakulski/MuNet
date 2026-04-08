# μNet

μNet is a lightweight C++ tensor + autograd framework with Python bindings.
It supports CPU, CUDA and Vulkan backends. The goal of the project is to interop with existing frameworks (torch / onnx)
and allow for mixed gpu training & inference.

## Setup / Runtime requirements
```bash
pip install munet_nn
```

μNet has provisioning enabled for accelerated support detection, you can install the accelerated backends by specifying the appropriate feature targeting:

vulkan:
`munet_nn["vk"]`

cuda 13:
`munet_nn["cu13_vk"]`

cuda 12:
`munet_nn["cu12_vk"]`

### CUDA runtime expectations

For CUDA-enabled extras (for example `cu12_vk` and `cu13_vk`), μNet expects CUDA runtime libraries
(including cuBLAS) to be installed at the system level. The Python package does **not** provision
CUDA runtime/cuBLAS via pip.

Ensure your CUDA library directories are discoverable by the dynamic linker (for example, via
`LD_LIBRARY_PATH`) before running GPU backends.

## Current repository state

- Core runtime (`munet_core`) is used by both training and inference surfaces.
- Training APIs (`nn`, `optim`, losses, autograd) are available in C++ and Python.
- Inference APIs and serialization flows are available (`munet_inference`, demos under `demos/inference/` and `demos/serialization/`).
- Backend dispatch which includes:
  - capability-based support checks,
  - fallback reason accounting,
  - accelerator→CPU fallback telemetry counters,
  - optional fail-fast mode for unexpected accelerator fallbacks.

## ENV Vars

- `MUNET_PROFILE=1` — profiler collection + summary on process exit or manual flush.
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

### Python publishing
There are two PyPi streams for μNet. Publishing is done via release tags, `dev*` tags will push to the test stream and `v*` tags to production.

#### Test stream
- https://test.pypi.org/project/munet-nn/

#### Prod stream
- https://pypi.org/project/munet-nn/

When building a release ensure that the version in `pyproject.toml` matches the tag.

## Tests

```bash
make unit-test      # debug gtest (C++)
make py-test        # Python integration test suite
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

See `demos/README.md` for the full categorized catalog:

- `demos/visual/` (object/semantic/instance segmentation categories)
- `demos/operators/`
- `demos/serialization/`
- `demos/inference/`
- `demos/multigpu/`
- `demos/transformers/`
