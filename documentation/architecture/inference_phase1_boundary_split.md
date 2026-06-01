# Inference boundary split

The inference boundary has been reduced to a lean Vulkan execution wrapper around
`core::Module`.  The engine is intentionally not a general serving framework; it
is a small API that keeps the hot path clear for Vulkan optimization work.

## Kept in the engine

- `EngineConfig`: Vulkan device, strict shape checks, and the autograd-input gate.
- `EngineStats`: lifecycle booleans, run counts, compile/run timing, and compiled
  shapes.
- `Engine`: `load`, `prepare`, `compile`, `run`, and `run_batch`.

## Removed from the engine boundary

- observer event objects and callbacks;
- built-in warmup runners;
- trace-id ownership;
- profiler-memory capture toggles;
- prepared-input cache sizing and eviction;
- output-vector reuse APIs.

Applications can layer those policies around the engine when they need them. The
core runtime stays compact and Vulkan-specific.
