# Vulkan profiling

MuNet keeps profiling outside the lean inference engine. The engine reports only
minimal `EngineStats` counters and timing; detailed profiling remains available
through the global profiler and lower-level Vulkan/backend instrumentation.

## Reading profiler output

Profiler rows are grouped by operation label (for example module forwards,
transfers, and Vulkan backend phases). Use these rows to identify expensive
kernels or transfers, then add application-level request identifiers in your own
serving layer if you need per-request correlation.

## Recommended workflow

1. Keep the inference engine hot path simple: `load`, optional `compile`, then
   `run`/`run_batch`.
2. Enable profiler collection only around the scenario being measured.
3. Compare module-level rows with Vulkan backend rows to separate model cost from
   transfer/staging cost.
4. Move any request tracing, warmup loops, or cache experiments into the caller so
   engine measurements stay stable.

## Engine stats vs profiler stats

`EngineStats` is intentionally small: lifecycle state, run counts, compile/run
latency, and compiled shapes. It is suitable for smoke checks and simple health
reporting. Use the global profiler for detailed optimization work.
