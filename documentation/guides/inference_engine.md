# Inference Engine Guide

## Goal

Provide a stable deploy runtime with shape contracts and warmup/compile behavior.

## Compile contracts

Use `compile(...)` with optional shape contracts:

- `expected_input_shape`
- `expected_output_shape`

Use `-1` for dynamic dims:

- Dynamic batch MLP: `[-1, 4] -> [-1, 2]`
- Dynamic resolution conv: `[-1, 3, -1, -1] -> [-1, 2, -1, -1]`

## Strict vs non-strict checks

- strict mode validates compiled/expected shapes at runtime.
- non-strict mode allows mismatches (for experimentation).

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
