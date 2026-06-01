# Inference phase 5 validation rollout

Phase 5 now validates the lean Vulkan inference boundary instead of comparing
engine-owned policy toggles. The engine has no internal lean mode, warmup runner,
prepared-input cache, or output-container reuse API.

## Validation focus

1. `load(module)` moves the model to Vulkan and leaves it in eval mode.
2. Optional `compile(sample, ...)` records shape contracts and rejects mismatches.
3. `run(input)` rejects autograd inputs by default and executes with autograd
   disabled.
4. `run_batch(inputs)` is a transparent sequential wrapper over `run(...)`.
5. Profiling, request tracing, warmup loops, and staging caches are measured in
   external harnesses rather than inside `Engine`.

## Rollout guidance

- Keep the default runtime path small and deterministic.
- Add deployment-specific caches or warmup loops in serving code only when a
  benchmark proves the need.
- Use the global profiler and workload harnesses for optimization experiments;
  do not expand the engine API for speculative diagnostics.
- Treat any unsupported Vulkan behavior as an explicit deployment error.
