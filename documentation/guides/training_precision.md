# Training precision and AMP flows

Phase 5 of the architecture refactor adds three training-facing building blocks that align with the dtype/backend work from earlier phases:

1. modules retain explicit default `TensorOptions` so construction and migration use predictable device/dtype behavior
2. optimizers accept parameter groups and explicit optimizer-state dtype policies
3. AMP primitives are surfaced through `munet::amp::GradScaler` and `munet::amp::AutocastGuard`

## Recommended precision policy

### FP32 baseline training

Use this when validating a new model, backend, or optimizer setup.

- Construct modules with default `TensorOptions{}` or an explicit fp32 dtype.
- Keep optimizer parameter groups on the default state policy.
- Skip autocast and leave `GradScaler` disabled.

```cpp
munet::nn::Linear linear(128, 64);
munet::optim::Adam opt(linear.parameters(), 1e-3f);
```

Why this remains the default:

- parameters, gradients, and optimizer state all stay in `Float32`
- backend fast paths remain available
- debugging numerical issues is simplest in this mode

### Mixed precision training

Use this when model parameters should live in `Float16` while optimizer state remains explicit.

```cpp
munet::TensorOptions fp16_options;
fp16_options.dtype = munet::DataType::Float16;

munet::nn::BatchNorm2d model(64, 1e-5f, 0.1f, fp16_options);

munet::optim::OptimizerStatePolicy state_policy;
state_policy.model_dtype = munet::DataType::Float16;
state_policy.master_weight_dtype =
    munet::optim::MasterWeightDTypePolicy::Float32;
state_policy.state_tensor_dtype =
    munet::optim::OptimizerStateTensorDTypePolicy::Float32;

munet::optim::ParameterGroup group(model.parameters(), 1e-3f, state_policy,
                                   "main");
munet::optim::Adam opt({group}, 1e-3f);
munet::amp::GradScaler scaler(true);

munet::amp::AutocastOptions autocast;
autocast.enabled = true;
autocast.compute_dtype = munet::DataType::Float16;
autocast.conversion_policy =
    munet::amp::AutocastConversionPolicy::PromoteInputs;

{
  munet::amp::AutocastGuard guard(autocast);
  munet::Tensor loss = model.forward(input).mse_loss(target);
  munet::Tensor scaled_loss = scaler.scale(loss);
  scaled_loss.backward();
}

scaler.unscale_(opt);
opt.step();
scaler.update();
opt.zero_grad();
```

Recommended policy details:

- model parameters may be stored in `Float16`
- normalization running statistics should remain accumulation-friendly buffers
- optimizer momentum/state tensors should be configured explicitly, usually `Float32`
- optional master weights belong to optimizer state, not module checkpoints
- autocast should only allow floating-point input promotion into the configured compute dtype

### Pure fp16 inference where safe

Use this only after validating numerics for a model family and backend.

- Convert modules explicitly with `module.to(DataType::Float16)` or construct them from fp16 defaults.
- Avoid optimizer or scaler usage.
- Keep autocast optional; for inference-only paths explicit module conversion is usually easier to reason about.

Recommended caveats:

- prefer `Float32` accumulation buffers for numerically sensitive normalization state
- verify backend kernel coverage before assuming all ops run natively in fp16
- benchmark with representative inputs because conversion overhead can erase expected gains

## Autocast policy

`munet::amp::AutocastGuard` is intentionally narrow in scope.

- `AutocastConversionPolicy::Strict` disallows implicit dtype changes.
- `AutocastConversionPolicy::PromoteInputs` allows floating-point inputs to be promoted into the configured compute dtype.
- `AutocastConversionPolicy::PromoteInputsAndOutputs` additionally permits returning outputs in that compute dtype.

Integral tensors are not implicitly converted by the autocast policy helpers.

## Optimizer state policy cheat sheet

`munet::optim::OptimizerStatePolicy` separates three questions:

- What dtype is the model logically training in? (`model_dtype`)
- Should the optimizer keep master weights? (`master_weight_dtype`)
- What dtype should momentum/state tensors use? (`state_tensor_dtype`)

That separation keeps AMP additive: modules own parameter/buffer dtype behavior, while optimizers own master-weight and momentum-state behavior.
