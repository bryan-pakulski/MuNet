# μNet

μNet is a lightweight C++ tensor + autograd framework with Python bindings.
The project is Vulkan-only: builds, wheels, runtime discovery, docs, demos, and tests target Vulkan as the single supported execution surface.

## Setup / Runtime requirements
```bash
pip install munet_nn
```

### Vulkan runtime expectations

μNet expects system Vulkan runtime/driver libraries (for example `libvulkan.so.1`) to be
installed and discoverable by the dynamic linker.

## Current repository state

- Core runtime (`munet_core`) is used by both training and inference surfaces.
- Training APIs (`nn`, `optim`, losses, autograd) are available in C++ and Python.
- Inference APIs and serialization flows are available (`munet_inference`, demos under `demos/inference/` and `demos/serialization/`).
- Backend dispatch uses capability-based support checks and reports unsupported Vulkan operations directly.
- Public device APIs expose only `DeviceType.VULKAN` and stable `vulkan:<index>` device strings.

## ENV Vars

- `MUNET_PROFILE=1` — profiler collection + summary on process exit or manual flush.
- `MUNET_DEBUG=1` — debug logging/checks.
- `MUNET_LOG_LEVEL=0..3` — log verbosity.
- `MUNET_DISPATCH_DECISION_DUMP=1` — emit dispatch decision lines.

## Build

### Requirements
- CMake 3.10+
- C++17 compiler
- Python 3.10+
- Vulkan SDK/runtime (`glslc` in `PATH` for shader workflows)

### Python publishing

Publishing produces Vulkan wheels only. The wheel workflow installs Vulkan development/runtime packages, validates that the package contains the core extension without removed integration helpers, smoke-tests Vulkan backend discovery, and publishes the same Vulkan-only artifact set to TestPyPI or PyPI.

- `dev*` tags publish Vulkan wheels to TestPyPI: https://test.pypi.org/project/munet-nn/
- `v*` tags publish Vulkan wheels to PyPI: https://pypi.org/project/munet-nn/

Before tagging a release, ensure that `pyproject.toml` has the intended version and that `.github/workflows/wheels.yml` remains Vulkan-only.

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
- `demos/transformers/`
