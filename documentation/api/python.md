# Python API Reference (Guide)

This guide summarizes the currently exposed Python surface in `munet`.

## Core module-level API

### Tensor creation & conversion

- `munet.zeros(shape, device=None, requires_grad=False, dtype=...)`
- `munet.ones(shape, device=None, requires_grad=False, dtype=...)`
- `munet.rand(shape, device=None, requires_grad=False, dtype=...)`
- `munet.from_numpy(ndarray)`
- `munet.copy_from_numpy(tensor, ndarray)`
- `munet.cat(tensors, dim=1)`
- `munet.matmul(a, b)`

### Capability + dispatch diagnostics

- `munet.supports(device, feature, dtype)`
- `munet.dispatch_policy_snapshot()`
- `munet.dispatch_decision_debug_dump(op_name, tensor)`
- `munet.fallback_telemetry_snapshot()`
- `munet.reset_fallback_telemetry()`

Related env flags:

- `MUNET_DISPATCH_DECISION_DUMP=1`
- `MUNET_FAIL_FAST_VULKAN_UNSUPPORTED=1`

### Profiling helpers

- `munet.print_profiler_stats()`
- `munet.reset_profiler()`

### Autograd mode helpers

- `munet.GradMode.is_enabled()`
- `munet.GradMode.set_enabled(bool)`
- `with munet.no_grad(): ...`
- `with munet.enable_grad(): ...`

## Devices and dtypes

- `munet.DataType` (`Float32`, `Float16`, `BFloat16`, `Int32`, `Int8`)
- `munet.Device(type, index=0)`
- `munet.TensorOptions` (`device`, `dtype`, `requires_grad`)

## Tensor API (`munet.Tensor`)

### Core

- construction: `Tensor(shape, device=..., dtype=..., requires_grad=False)`
- metadata: `shape`, `device`, `dtype`, `strides`, `storage_offset`,
  `is_contiguous`, `requires_grad`, `grad`, `name`
- conversion: `to(device)`, `to(dtype)`, `to_options(options)`
- structure/view: `reshape`, `permute`, `transpose`, `narrow`, `contiguous`
- value/data: `item()`, `numpy()`, `copy_from_numpy(...)`, `replace_(...)`

### Math + NN ops

- arithmetic: `+`, `-`, `*`, `/`, `@`, `matmul(other)`
- reductions: `sum()`, `mean(dim=-1, keepdim=False)`
- unary/activations: `relu`, `sigmoid`, `exp`, `log`, `sqrt`, `rsqrt`, `sin`,
  `cos`, `softmax`, `log_softmax`
- loss ops: `mse_loss(target)`, `cross_entropy(target)`
- spatial ops: `conv2d`, `max_pool2d`, `upsample2d`, `batch_norm`, `layer_norm`
- misc: `masked_fill`, `uniform_`, `fill_`, `step`, `all_reduce`

### Autograd

- `backward(retain_graph=False)`
- `backward(grad, retain_graph=False)`
- `detach()`, `clone()`, `zero_grad()`, `has_grad()`
- `register_gradient_hook(callable)`

## `munet.nn` module API

Base and common modules currently exposed include:

- `nn.Module`, `nn.Sequential`
- `nn.Linear`, `nn.Conv2d`, `nn.BatchNorm2d`
- `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.GELU`, `nn.LeakyReLU`, `nn.Dropout`
- `nn.LayerNorm`, `nn.RMSNorm`, `nn.Embedding`, `nn.MultiHeadAttention`
- `nn.MaxPool2d`, `nn.Upsample`, `nn.GlobalAvgPool2d`, `nn.Flatten`, `nn.Softmax`

Module management:

- `parameters()`, `named_parameters(prefix="")`, `named_modules(prefix="")`
- `train(mode=True)`, `eval()`, `is_training`
- `to(device)`, `to(dtype)`, `to_options(options)`
- `offload(device, layers=[...])`, `clear_offload()`, `offload_plan()`
- `freeze_offload_plan() -> dict[layer, "device:index"]`
- `apply_offload_plan(plan_dict)`
- `offload_plan(explain=True)` includes structured planner rationale payload per layer.
- `validate_offload_plan(sample_input) -> OffloadValidationReport`
- `set_offload_warnings(enabled=True)`
- `set_offload_warning_threshold_bytes(threshold_bytes)`
- `offload_telemetry_snapshot() -> OffloadTransferTelemetry`
- `reset_offload_telemetry()`
- `zero_grad()`

## `munet.optim` API

- `optim.Optimizer`
  - `step()`, `zero_grad()`, `grad_global_norm()`, `clip_grad_norm(max_norm)`
  - `apply_weight_decay(weight_decay)`
  - property: `lr`
- `optim.Adam(params, lr=..., beta1=..., beta2=..., eps=...)`
- `optim.SGD(params, lr=...)`

## `munet.inference` API

Lean Vulkan inference API:

- `inference.EngineConfig`
  - `device` (defaults to `DeviceType.VULKAN`)
  - `strict_shape_check`
  - `allow_autograd_inputs`
- `inference.EngineStats`
  - `loaded`, `prepared`, `compiled`
  - `runs`, `batch_runs`
  - `last_run_ms`, `compile_ms`
  - `compiled_input_shape`, `compiled_output_shape`
- `inference.Engine`
  - `set_device(device)`, `device()`
  - `set_strict_shape_check(enabled)`
  - `set_allow_autograd_inputs(enabled)`, `allow_autograd_inputs()`
  - `load(module)`
  - `prepare(input)`
  - `compile(example_input, expected_input_shape=None, expected_output_shape=None)`
  - `run(input)`, `run_batch(inputs)`
  - `is_loaded()`, `is_prepared()`, `is_compiled()`
  - `compiled_input_shape()`, `compiled_output_shape()`, `stats()`

The engine intentionally excludes observer callbacks, warmup runners, trace IDs,
and prepared-input caches. Keep those policies in application code around the
small Vulkan execution loop.

## Serialization surface

The Python module injects serialization helpers and metadata constants (for
example `save`, `load`, `load_weights`, `load_for_inference`,
`serialization_format_info`) during module initialization. See
`demos/serialization/munet/serialization_roundtrip_demo.py` for end-to-end usage.
