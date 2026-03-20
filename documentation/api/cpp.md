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
2. Load either a module instance or a deploy artifact path
   - `load(module)`
   - `load("model.npz")`
3. `compile(example_input, expected_input_shape, expected_output_shape)`
4. `run(input)`

Shape contracts support wildcard `-1` dims for dynamic behavior.

## C++ deploy artifact loading

C++ now has a first-class deploy loader for the same `.npz` runtime artifacts described in the Python docs:

- `munet::inference::load_serialized("model.npz", device)`
- `munet::inference::load_weights_serialized(module, "model.npz", device)`
- `munet::inference::Engine::load("model.npz")`

This lets deploy code reconstruct supported built-in MuNet models directly from a serialized artifact without re-declaring the module structure in C++. Loaded modules are normalized for inference by validating the deploy manifest and forcing `eval()` before execution.

### Example

```cpp
#include "inference.hpp"

munet::inference::Engine engine;
engine.load("model.npz");
engine.compile(example_input, {-1, 3, -1, -1}, {-1, 1000});
auto output = engine.run(input);
```


## Training ergonomics

- `munet::core::Module::default_options()` exposes module device/dtype defaults used during construction and migration.
- `munet::optim::ParameterGroup` groups parameters with per-group learning-rate and optimizer-state policies.
- `munet::optim::OptimizerStatePolicy` makes model, master-weight, and optimizer-state dtypes explicit.
- `munet::amp::GradScaler` and `munet::amp::AutocastGuard` provide additive AMP primitives without requiring module-level special cases.
