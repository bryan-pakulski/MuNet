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

## ONNX Runtime + native conversion direction

The current two-path approach is acceptable and should be kept:

1. **Execution path**: `load_onnx(...)` uses ONNX Runtime as the compatibility fallback.
2. **Native path**: `compile_onnx(...)` lowers supported ONNX operators into MuNet modules.

This split is practical for incremental rollout because unsupported ops still run via ONNX Runtime while native lowering coverage increases over time.

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
- `Sigmoid` -> `nn.Sigmoid`
- `Tanh` -> `nn.Tanh`
- `Flatten` -> `nn.Flatten`

### Near-term recommended next mappings

- binary constant ops: `Add`, `Sub`, `Mul`, `Div`
- layout ops: `Reshape`, `Transpose`
- basic graph joins: `Concat`
