# Observability roadmap

The Vulkan-only runtime now keeps observability policy outside the inference
engine. The core engine should remain a small, predictable execution wrapper; rich
serving diagnostics can be layered on top without expanding the hot path.

## Current baseline

- Minimal `EngineStats` for load/prepare/compile/run state, counts, timing, and
  compiled shapes.
- Global profiler utilities for operation/backend measurements.
- Application-owned logging and request identifiers.

## Next steps

1. Keep the engine API stable and small while Vulkan kernels are optimized.
2. Add external profiling harnesses for representative workloads instead of
   embedding profiler policy in `Engine`.
3. Prototype request-level tracing in serving/demo code where request identity is
   available.
4. Revisit runtime diagnostics only when a concrete deployment need justifies the
   cost.
