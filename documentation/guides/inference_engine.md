# Lean Vulkan inference engine

The inference engine is intentionally small: it owns one loaded module, targets one
Vulkan device, optionally records a shape contract, and runs tensors with autograd
disabled.  It no longer has observer hooks, trace scopes, warmup orchestration, or
prepared-input caches.  Those concerns belong outside the hot path so Vulkan
execution can stay predictable and easy to optimize.

## Runtime model

1. Build or load a module.
2. Create `munet.inference.Engine()`; the default device is `DeviceType.VULKAN`.
3. `load(module)` moves the module to the configured Vulkan device and switches it
   to eval mode.
4. Optionally call `compile(sample, expected_input_shape=None,
   expected_output_shape=None)` to capture a strict shape contract. Use `-1` in an
   expected shape dimension to allow that dimension to vary.
5. Call `run(tensor)` for one input or `run_batch([...])` for a simple sequential
   batch.

`prepare(tensor)` remains only as a convenience transfer/validation helper. It
returns the tensor on the engine device and marks the engine prepared; it does not
populate a cache or run warmup loops.

## Shape checks

`EngineConfig.strict_shape_check` defaults to `True`. After `compile(...)`, every
`run(...)` validates the prepared input shape and output shape against the captured
contracts. Disable strict checks only when the caller owns validation externally.

## Autograd boundary

`EngineConfig.allow_autograd_inputs` defaults to `False`. Inputs with
`requires_grad=True` are rejected before any device transfer, and forward execution
runs under inference mode so outputs do not attach to the autograd graph.

## Minimal Python example

```python
import munet_nn as mn

model = mn.nn.Sequential(mn.nn.Linear(4, 2), mn.nn.ReLU())
engine = mn.inference.Engine()
engine.load(model)

x = mn.Tensor([1, 4], device=mn.Device(mn.DeviceType.VULKAN, 0))
engine.compile(x, expected_input_shape=[-1, 4])
y = engine.run(x)
print(engine.stats().runs, y.shape())
```

## What moved out of the engine

Diagnostics and orchestration should be layered around the engine instead of inside
it:

- profiling remains available through the global profiler utilities;
- request tracing should be done by the application layer;
- warmup loops should call `run(...)` explicitly;
- repeated-input or staging caches should live in caller-owned code that can be
  tuned for a specific deployment.
