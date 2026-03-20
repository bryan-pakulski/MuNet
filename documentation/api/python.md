# Python API Reference (Guide)

This page summarizes key Python surfaces and points to usage patterns.

## Core tensors

- `munet.Tensor`
- device transfer via `.to(...)`
- autograd controls: `munet.no_grad`, `munet.enable_grad`

## NN modules

Common modules in `munet.nn` include:

- `Linear`, `Conv2d`, `BatchNorm2d`, `ReLU`, `Sigmoid`
- `GELU`, `Tanh`, `Dropout`, `LeakyReLU`
- `LayerNorm`, `Embedding`, `MultiHeadAttention`

## Inference API

- `munet.inference.EngineConfig`
- `munet.inference.Engine`
  - `load(module)`
- `munet.inference.load_serialized(path, device=...)`
- `munet.inference.load_weights_serialized(model, path, device=...)`
- `munet.inference.onnx_runtime_package_boundary()`
  - `compile(example_input, expected_input_shape=None, expected_output_shape=None)`
  - `run(input)`

`-1` in expected shapes means dynamic dimension (for example dynamic batch and/or resolution).

## Serialization

- `munet.save(model, path)`
- `munet.load(path)` (generic full reconstruction for supported built-ins)
- `munet.load(model, path)` / `munet.load_weights(model, path)` (generic weights-only restore)
- `munet.load_for_inference(path, device=...)`
- `munet.load_weights_for_inference(model, path, device=...)`
