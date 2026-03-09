# C++ API Reference (Guide)

## Runtime

- `munet::Tensor`
- `munet::Device`
- `munet::Backend` + `BackendManager`

## Module hierarchy

- `munet::core::Module` (shared base)
- `munet::nn::Module` (training-oriented)
- `munet::inference::Module` and `munet::inference::Engine`

## Inference engine lifecycle

1. Construct `inference::Engine`
2. `load(module)`
3. `compile(example_input, expected_input_shape, expected_output_shape)`
4. `run(input)`

Shape contracts support wildcard `-1` dims for dynamic behavior.
