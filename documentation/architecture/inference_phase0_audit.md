# Inference phase 0 audit

The audit outcome is a deliberately lean Vulkan inference engine. Earlier plans
considered observer callbacks, internal trace ownership, profiler-memory toggles,
warmup loops, and input caches. Those policies have been removed from the engine
so optimization work starts from a small, predictable hot path.

## Engine-owned responsibilities

- Vulkan device selection.
- Module load/eval normalization.
- Optional shape-contract compilation.
- Autograd input rejection and inference-mode execution.
- Minimal lifecycle/run statistics.

## Caller-owned responsibilities

- Request tracing.
- Profiler collection policy.
- Warmup orchestration.
- Input/staging caches.
- Advanced batch scheduling.
