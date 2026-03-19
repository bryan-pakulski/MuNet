# Inference Runtime Phase 1 Boundary Split

This document starts **Phase 1 - API and package boundary split** for the inference/runtime separation effort.

The goal of this phase is to make the public boundary between shared runtime, inference, and training APIs explicit enough that inference consumers do not accidentally depend on training-only headers or target definitions.

## Phase 1 status

- [~] Phase 1 - API and package boundary split
- [x] Target/header dependency audit written
- [x] Training-only public headers guarded behind `munet_training`
- [x] Inference-only build check added
- [x] Stable deploy API surface documented
- [ ] Python packaging split from the monolithic extension target
- [ ] Serialization/model-loading ownership fully moved to deploy-first contracts

## Target and header audit

| Target / surface | Intended ownership | Current Phase 1 rule |
| --- | --- | --- |
| `munet_core` | shared tensor/runtime/backend primitives | may expose tensor execution, device/dtype/state primitives shared by both products |
| `munet_inference` | deploy runtime contracts | may expose `inference.hpp`, deploy-safe module execution, shape contracts, runtime stats, observers, and baseline tooling |
| `munet_training` | training-only APIs | owns `nn.hpp`, `nn/*`, `optim.hpp`, loss wiring, and other train-step behavior |
| Python `munet` module | mixed surface today | remains monolithic for now, but inference bindings should avoid requiring training-only types where possible |

## Phase 1 changes landed

### 1. Training header guardrails

Training-only public headers are now explicitly guarded so they fail fast unless the consumer is compiled through `munet_training`:

- `src/nn/module.hpp`
- `src/optim.hpp`

This prevents inference-only targets from silently depending on training headers just because all headers are visible from the same include root.

### 2. Grad-mode dependency moved to a narrower shared primitive

`GradMode` now lives in `src/core/grad_mode.hpp`, which lets `src/inference.hpp` disable gradient tracking without including the full autograd engine header.

This narrows the inference public surface from “depends on autograd engine internals” to “depends on shared grad-state primitive”.

### 3. Inference-only build check

The repository now builds `munet_inference_boundary_check`, an inference-only executable linked against `munet_inference`.

Its purpose is simple:

- verify the inference target compiles without `MUNET_ENABLE_TRAINING`
- verify `inference.hpp` is sufficient for inference-only consumers
- provide a build-graph tripwire if training definitions leak into the inference target later

### 4. Python inference loading no longer names training types directly

The Python binding for `munet.inference.Engine.load(...)` now casts to the shared `core::Module` base instead of requiring `nn::Module` in the method signature.

That keeps the inference entrypoint aligned with the C++ runtime contract while preserving compatibility with existing `nn.Module` instances.

## Stable deploy API surface (current C++ contract)

The current deploy-facing API surface is:

- `inference::Module`
- `inference::EngineConfig`
- `inference::EngineEventType`
- `inference::EngineEvent`
- `inference::EngineStats`
- `inference::Engine`
  - `load(...)`
  - `compile(...)`
  - `prepare(...)`
  - `run(...)`
  - `run_batch(...)`
  - `set_device(...)`
  - `set_warmup_runs(...)`
  - `set_strict_shape_check(...)`
  - `set_allow_autograd_inputs(...)`
  - `set_capture_profiler_memory(...)`
  - `set_observer(...)`
  - `clear_observer()`
  - `stats()`

For Phase 1, new deploy functionality should prefer extending this surface rather than reaching into training-side namespaces.

## Ownership rules for future changes

When adding new functionality, use the following ownership rules:

### Put it in `munet_core` when it is:

- required by both training and inference
- tensor/backend/device/dtype/runtime state
- free of train-step semantics and optimizer assumptions

### Put it in `munet_inference` when it is:

- deploy/runtime execution behavior
- shape-contract, warmup, batching, observer, and runtime stat behavior
- model loading or execution policy that does not require training loops
- edge/server runtime configuration and benchmark tooling

### Put it in `munet_training` when it is:

- optimizer behavior
- gradient-scaling or train-step state
- training-only modules or train/eval mutation semantics
- checkpoint content that only exists for future training resumption

## Remaining Phase 1 gaps

- The Python extension is still built as one monolithic module and still needs a packaging split before the Python deploy story is truly inference-first.
- Serialization and model-loading flows still share training-defined module reconstruction paths that should be narrowed further in later phases.
- `munet_core` still contains autograd graph machinery, so Phase 1 improves the public boundary without fully splitting the internal build graph yet.
