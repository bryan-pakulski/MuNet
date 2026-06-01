# Inference phase 2 runtime slimming

Runtime slimming is complete for the inference engine: diagnostic and memory-cache
policy has been moved out of the engine, leaving a compact Vulkan execution loop.

## Removed from the engine

- profiler-memory capture toggles;
- runtime mode switches;
- built-in warmup loops;
- trace-id ownership;
- prepared-input caches;
- batch output-container reuse APIs.

## Remaining hot path

`run(input)` validates the input, transfers it to the configured Vulkan device if
needed, executes the module with autograd disabled, validates a compiled shape
contract when present, and updates minimal stats.
