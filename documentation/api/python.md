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
  - `compile(example_input, expected_input_shape=None, expected_output_shape=None)`
  - `run(input)`

`-1` in expected shapes means dynamic dimension (for example dynamic batch and/or resolution).

## Serialization

- `munet.save(model, path)`
- `munet.load(path)` (full reconstruction for supported built-ins)
- `munet.load(model, path)` / `munet.load_weights(model, path)` (weights-only)
