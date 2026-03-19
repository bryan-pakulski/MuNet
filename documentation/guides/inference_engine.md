# Inference Engine Guide

## Goal

Provide a stable deploy runtime with shape contracts, warmup/compile behavior, and diagnostics that stay separate from training/autograd internals.

## Compile contracts

Use `compile(...)` with optional shape contracts:

- `expected_input_shape`
- `expected_output_shape`

Use `-1` for dynamic dims:

- Dynamic batch MLP: `[-1, 4] -> [-1, 2]`
- Dynamic resolution conv: `[-1, 3, -1, -1] -> [-1, 2, -1, -1]`

## Inference/autograd boundary

`munet.inference.Engine` now enforces an inference-first execution path:

- `compile(...)` and `run(...)` temporarily disable `GradMode`, even if the loaded module still has trainable parameters.
- Inputs with `requires_grad=True` are rejected by default with a targeted deployment error.
- You can opt into `allow_autograd_inputs=True` only for debugging/inspection flows; MuNet still hard-fails if the resulting deployment path would surface a grad-tracked output.
- If a deployment path somehow produces a gradient-tracked output, the engine raises an error instead of silently leaking training behavior into inference.

## Low-overhead defaults and lean mode

MuNet now defaults the inference runtime toward lower overhead:

- `capture_profiler_memory` defaults to `False`
- trace ids / scoped trace contexts are only activated when observers, profiler mode, or debug logging are enabled
- `lean_mode=True` further favors predictable deploy execution by keeping optional runtime diagnostics off and skipping non-essential load-time diagnostics

For constrained devices, prefer:

- `eng.set_lean_mode(True)` in Python, or `EngineConfig::lean_mode = true` in C++
- `capture_profiler_memory=True` only when you are actively collecting memory diagnostics
- a bounded prepared-input cache (`prepared_input_cache_entries`, `prepared_input_cache_max_bytes`) when repeated host-to-device transfers must stay within a fixed memory budget
- `prepare_batch([...])` during warmup when you want to pre-populate prepared-input buffers before steady-state batched inference
- observers only when lifecycle event callbacks are required

## Strict vs non-strict checks

- strict mode validates compiled/expected shapes at runtime.
- non-strict mode allows mismatches (for experimentation).

## Observability hooks

The engine exposes lightweight lifecycle hooks without coupling deployment code to training internals:

- `set_observer(callback)` receives load / compile / run / error events.
- Each event includes:
  - event type
  - device
  - run index
  - input/output shapes when available
  - duration in milliseconds
  - profiler current/peak memory snapshots when enabled
  - a human-readable diagnostic message
- `EngineStats` also records compile/run timings, compiled shapes, profiler memory
  snapshots, per-run trace ids, and host-side phase timings for:
  - module load transfer/eval
  - compile input preparation / forward / warmup
  - run input preparation / forward / output validation
- `EngineEvent` now carries the active `trace_id` and `span` for compile/run
  lifecycle callbacks, making it possible to join observer output with
  profiler rows and debug logs from the same request.

For production-like profiling, keep `capture_profiler_memory=True` and combine
engine events with the process-level profiler (`MUNET_PROFILE=1`) when deeper
backend timing is required. If you also enable `MUNET_DEBUG=1`, log lines will
include the same `[trace_id=… span=…]` prefix used by profiler detail strings.

## ONNX native conversion direction

MuNet now follows a strict native-conversion flow:

1. `compile_onnx(...)` attempts to convert the full ONNX graph into a MuNet-native module.
2. If any node cannot be converted, conversion fails with a detailed unsupported-op report.

There is no runtime fallback path in conversion.

### Foundation added for runtime conversion

- A central `onnx_native_conversion_map()` reports known ONNX operator mappings and status:
  - `lowered`: converted to MuNet layers today.
  - `pass_through`: graph bookkeeping ops, no emitted layer.
  - `planned` / `unsupported`: not yet lowered.
- Lowering now uses a dispatch architecture (op -> lowering function), making it easier to add new operators without growing a single large conditional block.

### Current native-lowered operators

- `Gemm` -> `nn.Linear`
- `MatMul` (constant RHS) -> `nn.Linear`
- `Conv` (limited 2D case) -> `nn.Conv2d`
- `MaxPool` (2D symmetric case) -> `nn.MaxPool2d`
- `Relu` -> `nn.ReLU`
- `LeakyRelu` -> `nn.LeakyReLU`
- `Sigmoid` -> `nn.Sigmoid`
- `Tanh` -> `nn.Tanh`
- `Gelu` -> `nn.GELU`
- `Flatten` -> `nn.Flatten`
- `GlobalAveragePool` -> `nn.GlobalAvgPool2d`

### Recently completed mappings

- binary constant ops: `Add`, `Sub`, `Mul`, `Div`
- layout ops: `Reshape`, `Transpose`
- basic graph joins: `Concat`
- shape/index ops: `Squeeze`, `Expand`, `Tile`, `ConstantOfShape`, `Gather`

## YOLOv5 ONNX operator coverage check

You can inspect a downloaded ONNX model without compiling it natively:

```python
import munet
report = munet.inference.onnx_conversion_coverage_report("yolov5n.onnx")
print(report["unique_ops"])
print("unsupported:", report["coverage"]["unsupported"])
print("unmapped:", report["coverage"]["unmapped"])
```

To fetch the reference model used by tests/utilities:

```python
import munet
munet.inference.download_yolov5n_onnx("/tmp/yolov5n.onnx")
```

## Builder container for reproducible local builds

If local pybind11/Python toolchain setup is problematic, use the builder image:

```bash
./tools/build_in_docker.sh
```

This builds `docker/Dockerfile.builder` and runs a release CMake build inside
an Ubuntu 22.04 container with Python, CMake, pybind prerequisites, and ONNX tooling.

### YOLOv5n conversion status

The ONNX op set observed in `yolov5n.onnx` is now covered by the native
conversion map and strict native conversion path (`compile_onnx(...)`),
including:

- `Add`, `Cast`, `Concat`, `Constant`, `Conv`, `Floor`, `MaxPool`, `Mul`,
  `Pow`, `Reshape`, `Resize`, `Shape`, `Sigmoid`, `Slice`, `Split`,
  `Transpose`, `Unsqueeze`.

## Strict ONNX conversion policy

ONNX conversion now follows strict all-or-fail behavior:

1. **Success**: model is fully converted to MuNet native graph module.
2. **Failure**: conversion aborts and reports unsupported operators with:
   - unique unsupported op names
   - total unsupported node count
   - full per-op node counts

There is no runtime fallback during conversion.
