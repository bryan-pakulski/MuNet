# Lean Vulkan inference memory policy

The inference engine now avoids owning memory policy.  It does not keep a
prepared-input cache, pre-populate batches, or reuse caller-owned output
containers.  The core engine is only responsible for moving the current input to
the configured Vulkan device, running the loaded module with autograd disabled,
and enforcing any compiled shape contract.

## Why this is simpler

- The hot path has one transfer decision and one forward call.
- Cache sizing and eviction do not sit inside the generic runtime.
- Batch orchestration is a sequential loop over `run(...)`, making behavior easy
  to reason about before adding optimized Vulkan-specific batching later.
- Deployment-specific caches can live next to the serving code that understands
  request reuse patterns and memory budgets.

## Current API boundary

Keep these responsibilities in the engine:

- `load(module)`: move module to the Vulkan device and set eval mode.
- `prepare(input)`: validate and transfer one tensor to the engine device.
- `compile(sample, ...)`: capture optional input/output shape contracts.
- `run(input)` / `run_batch(inputs)`: execute with autograd disabled.

Keep these responsibilities outside the engine:

- request tracing and logging correlation;
- profiling collection policy;
- warmup loops;
- reusable staging/input caches;
- custom batch scheduling and output-container reuse.
