# Runtime Architecture

## Core components

- `Tensor` / `TensorImpl`: user-facing tensor + internal metadata/storage links.
- `Storage`: backend allocation ownership + raw handle lifetime.
- `Backend` implementations: kernel and transfer execution.
- `BackendManager` / `BackendRegistry`: backend factory and cache lifecycle.
- `ops::resolve_dispatch(...)`: op metadata + capability + fallback policy gate.
- Autograd engine: backward graph execution when grad mode is enabled.

## Execution flow

1. High-level tensor/op API selects op metadata.
2. Dispatch resolves backend vs CPU fallback via capability query + policy.
3. Runtime executes selected path.
4. Optional autograd node wiring and tracing/profiling metadata are recorded.

## Dispatch fallback behavior

Fallback is explicit and observable:

- fallback reason classification (`dtype`, `shape`, `feature`, `policy`)
- structured fallback logs
- profiler rows for fallback decision paths
- accelerator fallback telemetry counters
- optional fail-fast (`MUNET_FAIL_FAST_ACCELERATOR_CPU_FALLBACK=1`)

This is intended to expose hidden fallback-induced flakiness/perf drift early.

## Layering

- `munet_core`: tensor/storage/backend/autograd/dispatch infrastructure.
- `munet_training`: training modules (`nn`, `optim`) over core.
- `munet_inference`: inference API surface over core with deployment-oriented
  flow (`load`, `compile`, `run`).

