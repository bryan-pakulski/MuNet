# Inference Runtime Phase 2 Runtime Slimming

This document starts **Phase 2 - Hot-path runtime slimming** for the inference/runtime separation effort.

The goal of this phase is to make low-overhead inference the default runtime behavior while keeping diagnostics and observability available as opt-in features.

## Phase 2 status

- [~] Phase 2 - Hot-path runtime slimming
- [x] Hot-path audit written
- [x] Optional runtime hooks gated off the default path
- [x] Lean-mode execution profile added
- [x] Benchmark support updated for lean-mode comparisons
- [ ] Host/device transfer path slimmed further
- [ ] Repeat-run benchmark deltas recorded across multiple hardware tiers

## Hot-path audit summary

The current inference hot path had three avoidable sources of runtime overhead before this phase started:

1. **Trace scopes were always created** for `compile(...)` and `run(...)`, even when no observer, profiler, or debug flow needed them.
2. **Profiler-backed memory capture defaulted on**, which meant repeated `Profiler` reads were happening even in plain deploy execution.
3. **Load diagnostics traversed parameters by default** to count `requires_grad` tensors, even though that detail is only useful for richer diagnostics.

## Phase 2 changes landed so far

### 1. Low-overhead defaults

The inference runtime now defaults `capture_profiler_memory` to `false`, which removes profiler memory reads from the default execution path.

Trace ids and scoped trace contexts are now only enabled when one of the following is true:

- an observer is registered
- process-level profiling is enabled
- debug logging is enabled

This means plain `load(...)`, `compile(...)`, and `run(...)` calls avoid trace setup work unless diagnostics are explicitly requested.

### 2. Lean mode

`EngineConfig` and `inference::Engine` now expose a **lean mode** profile.

Lean mode is intended for constrained or deployment-focused environments where the priority is predictable low overhead instead of extra runtime diagnostics.

Current lean-mode behavior:

- disables profiler-memory capture by default
- skips the trainable-parameter diagnostic count during `load(...)`
- keeps inference execution semantics the same (`compile`, `prepare`, `run`, `run_batch` still work as usual)

### 3. Benchmark support

`munet_inference_baseline` now accepts a `--lean-mode <0|1|false|true>` flag and reports the active lean-mode setting in its JSON payload.

This provides a repository-level way to compare:

- default runtime behavior
- lean-mode behavior
- explicit diagnostic/profiler settings

## Current CPU validation snapshot

The repository-side benchmark was rerun in this environment with both the default low-overhead path and explicit lean mode:

Default path:

```bash
./build/munet_inference_baseline --device cpu --dtype float32 --batch 8 --input-dim 32 --hidden-dim 64 --output-dim 16 --warmup-runs 0 --single-run-iters 3 --batch-run-inputs 2 --batch-run-iters 2
```

Lean mode:

```bash
./build/munet_inference_baseline --device cpu --dtype float32 --lean-mode true --batch 8 --input-dim 32 --hidden-dim 64 --output-dim 16 --warmup-runs 0 --single-run-iters 3 --batch-run-inputs 2 --batch-run-iters 2
```

Observed comparison summary:

- both runs kept trace ids and profiler-memory counters at zero on the default non-diagnostic path
- both runs preserved equivalent output-shape contracts and steady-state behavior
- lean mode kept the explicit deploy-safe diagnostic suppression contract visible in the benchmark output, which gives later phases a stable knob for constrained-device comparisons

## Current guidance

For the lowest-overhead deploy path today:

- keep `capture_profiler_memory` disabled unless you are collecting memory diagnostics
- avoid observers unless you need lifecycle events
- use `lean_mode=True` / `set_lean_mode(true)` for constrained or edge-style execution
- enable profiler/debug hooks only for targeted investigations

## Follow-on Phase 2 work

- slim host/device transfer behavior further where pre-positioned inputs or reusable staging buffers can avoid repeated copies
- capture benchmark deltas for lean mode versus diagnostic-heavy configurations on CPU and accelerator targets
- decide whether batched execution should gain a lighter-weight path that avoids per-item observer/event churn when batch-level reporting is enough
