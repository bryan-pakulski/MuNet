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


## Training ergonomics

- `munet::core::Module::default_options()` exposes module device/dtype defaults used during construction and migration.
- `munet::optim::ParameterGroup` groups parameters with per-group learning-rate and optimizer-state policies.
- `munet::optim::OptimizerStatePolicy` makes model, master-weight, and optimizer-state dtypes explicit.
- `munet::amp::GradScaler` and `munet::amp::AutocastGuard` provide additive AMP primitives without requiring module-level special cases.
