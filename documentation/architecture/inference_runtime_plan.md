# Inference Runtime Separation Plan

This implementation plan defines the next project phase: splitting MuNet's inference engine into a lean deploy runtime that stays clearly separated from the training stack.

The target is a runtime that scales from Raspberry Pi-class edge devices up to enterprise GPU servers without carrying unnecessary training-side overhead into deployment flows.

## Status legend

- [ ] not started
- [~] in progress
- [x] completed

## Objectives

- Build a **trimmed deploy runtime** with the smallest practical dependency surface for inference-only workloads.
- Enforce a **hard architectural boundary** between shared execution primitives and training/autograd-only components.
- Keep **runtime overhead low by default**:
  - avoid hidden autograd work
  - avoid unnecessary allocations and copies
  - avoid debug/profiling cost unless explicitly enabled
  - avoid backend initialization that is irrelevant to the current deployment target
- Preserve **portability across hardware tiers**:
  - minimal CPU-only edge devices
  - mixed CPU/GPU workstation environments
  - large-scale accelerator-backed serving systems
- Make the new runtime easy to validate, benchmark, and package as an inference-first deliverable.

## Action Points

- [ ] Split deploy-critical runtime responsibilities from training-only services and codify that boundary in code organization, build targets, and docs.
- [ ] Reduce inference execution cost by removing or gating non-essential work on the hot path.
- [ ] Introduce deployment-oriented memory, initialization, and backend policies that favor deterministic low-overhead execution.
- [ ] Add phase-by-phase validation so every separation step preserves correctness, compatibility, and performance visibility.
- [ ] Keep the work documented in a way that allows progress tracking with explicit `[ ]`, `[~]`, and `[x]` status markers.

## Exit Criteria

- [ ] The inference runtime can be built and linked without depending on training-only APIs, optimizer code, or autograd execution machinery in its public surface.
- [ ] Inference hot paths have measurable overhead reductions for allocation churn, graph setup, and unnecessary runtime checks.
- [ ] CPU-only minimal builds remain functional for constrained hardware, while accelerator builds retain fast paths for higher-end deployments.
- [ ] Packaging and documentation clearly describe what belongs to `munet_core`, `munet_inference`, and `munet_training`.
- [ ] Benchmarks and tests prove that separation did not regress output correctness, shape enforcement, dtype fidelity, or supported deployment workflows.

## Phased Work Plan

### Phase 0 - Boundary audit and deploy baseline

**Status:** [x] completed

Current audit artifact: [Inference Runtime Phase 0 Audit](inference_phase0_audit.md).

#### Objectives

- Identify the exact training/runtime coupling points that still leak into inference flows.
- Establish a deploy-focused baseline before introducing deeper structural changes.
- Define what “minimal runtime” means for MuNet at the source, target, and package levels.

#### Action Points

- [x] Inventory current inference dependencies across:
  - `src/inference.hpp`
  - `src/core/*`
  - serialization/model loading paths
  - Python inference bindings
  - ONNX/native conversion flows
- [x] Classify each dependency as one of:
  - required shared runtime primitive
  - deploy convenience API
  - training-only leakage
  - optional observability/debug feature
- [~] Record current runtime costs for representative deployment paths:
  - cold load
  - compile/warmup
  - steady-state single-run latency
  - steady-state batched latency
  - peak/steady memory use
- [x] Define target build profiles for:
  - minimal CPU edge runtime
  - general-purpose desktop/server runtime
  - accelerator-enabled runtime
- [x] Document any current blockers that prevent a truly inference-only build target.

#### Exit Criteria

- [x] There is a written dependency and overhead inventory for the current inference engine.
- [x] Baseline latency/memory measurements exist for at least one CPU-only path and one accelerated path where available.
- [x] The project has an agreed definition of the minimum runtime surface that edge deployments must keep.

### Phase 1 - API and package boundary split

**Status:** [x] completed

Current boundary artifact: [Inference Runtime Phase 1 Boundary Split](inference_phase1_boundary_split.md).

#### Objectives

- Make the separation between shared runtime, inference APIs, and training APIs explicit in the public surface.
- Prevent training-side abstractions from being accidentally required by inference consumers.

#### Action Points

- [x] Audit `munet_core`, `munet_inference`, and `munet_training` headers/targets for unwanted transitive dependencies.
- [x] Move deploy-safe abstractions behind inference-oriented headers instead of exposing training-oriented includes by default.
- [x] Ensure inference-facing module/runtime interfaces depend only on:
  - tensor/runtime primitives
  - serialization/model execution contracts
  - backend capability discovery needed at deploy time
- [x] Introduce compile-time guards or build-time checks that fail if inference targets pull in training-only headers/symbols unintentionally.
- [x] Define a stable deploy API surface for:
  - model loading
  - shape contracts
  - warmup/prepare
  - run / run_batch
  - lightweight runtime stats
  - optional observers
- [x] Update docs so developers know where new functionality belongs before adding more coupling.

#### Exit Criteria

- [x] Inference headers and targets present a training-free public API surface.
- [x] Build graph checks exist for transitive dependency regressions.
- [x] Documentation clearly states the ownership boundary between core, inference, and training layers.

### Phase 2 - Hot-path runtime slimming

**Status:** [x] completed

Current runtime-slimming artifact: [Inference Runtime Phase 2 Runtime Slimming](inference_phase2_runtime_slimming.md).

#### Objectives

- Remove hidden costs from the steady-state inference path.
- Make low-overhead execution the default behavior, with diagnostics remaining opt-in.

#### Action Points

- [x] Audit `Engine::load`, `compile`, `prepare`, `run`, and `run_batch` for unnecessary work on repeat execution.
- [x] Minimize or cache:
  - backend discovery/lookup
  - shape validation setup that can be compiled once
  - repeated dtype/device checks that can be hoisted
  - metadata allocations for tracing/stat collection
- [x] Ensure autograd suppression remains explicit and cheap, with no graph-building residue on inference runs.
- [x] Gate debug/profiler hooks so the default runtime path pays near-zero cost when observability is disabled.
- [x] Add a “lean mode” execution profile for constrained devices that favors predictable memory use over convenience features.
- [x] Review host-device transfer behavior and remove avoidable copies in model load and input preparation paths.

#### Exit Criteria

- [x] Repeat inference runs show lower overhead in profiler/benchmark comparisons.
- [x] Optional diagnostics can be disabled without leaving measurable hot-path bookkeeping behind.
- [x] The lean runtime profile is documented and validated on constrained hardware assumptions.

### Phase 3 - Memory and backend policy optimization

**Status:** [x] completed

Current memory/backend artifact: [Inference Runtime Phase 3 Memory and Backend Policy](inference_phase3_memory_backend_policy.md).

#### Objectives

- Make runtime memory behavior predictable and efficient across edge and server deployments.
- Keep backend selection and initialization proportional to the actual deployment target.

#### Action Points

- [x] Add inference-focused memory policies for:
  - reusable input/output buffers
  - scratch/workspace reuse
  - optional preallocation during compile/warmup
  - bounded temporary allocations on CPU-only targets
- [x] Review backend manager behavior so inference startup does not eagerly initialize unused backends.
- [x] Separate required backend capabilities for deployment from training-only backend capabilities.
- [x] Define fallback policy for constrained systems:
  - CPU-only execution
  - selective fallback for unsupported accelerated ops
  - explicit failure where fallback would violate runtime guarantees
- [x] Evaluate whether static or reduced-feature builds should omit optional conversion, profiling, or debug facilities.
- [x] Document hardware-tier recommendations for edge, workstation, and enterprise deployments.

#### Exit Criteria

- [x] Runtime memory reuse policies are implemented and benchmarked.
- [x] Backend initialization is lazy/selective enough for minimal deployments.
- [x] Inference builds can target constrained systems without carrying unnecessary backend/runtime weight.

### Phase 4 - Serialization and model execution decoupling

**Status:** [~] in progress

#### Objectives

- Ensure deployed models can execute without inheriting training-time assumptions.
- Keep serialization artifacts and loading flows aligned with a lean runtime.

#### Action Points

- [x] Audit serialization metadata to separate deploy-required state from training/checkpoint-only state.
- [x] Ensure loaded inference modules default to eval-safe behavior without relying on training-mode conventions.
- [~] Define how exported/native-converted models are normalized for deploy execution:
  - fixed or dynamic shape contracts
  - dtype expectations
  - device placement policy
  - optional precompiled/warm state
- [ ] Review ONNX/native conversion helpers to keep conversion-time flexibility from bloating the runtime execution surface.
- [ ] Decide which conversion utilities stay in deploy packages versus development tooling.

#### Exit Criteria

- [ ] Deploy artifacts contain only runtime-required execution state for inference workflows.
- [ ] Model loading paths are documented as deploy-first rather than training-derived.
- [ ] Conversion and serialization tooling have a clear packaging boundary.

### Phase 5 - Validation, benchmarking, and rollout

**Status:** [ ] not started

#### Objectives

- Prove that the new runtime boundary is correct, lean, and maintainable.
- Roll out the new architecture without losing compatibility visibility.

#### Action Points

- [ ] Add focused tests for:
  - inference-only build/link coverage
  - autograd isolation guarantees
  - shape/dtype/device contract enforcement
  - backend fallback behavior on minimal builds
  - serialization/load/run deploy workflows
- [ ] Add benchmark suites for:
  - cold-start overhead
  - warm run latency
  - memory reuse effectiveness
  - CPU-only edge scenarios
  - GPU/accelerator server scenarios
- [ ] Track before/after metrics and publish them in docs.
- [ ] Add migration notes if any public inference APIs change while the runtime is slimmed down.
- [ ] Mark completed phases in this document as work lands.

#### Exit Criteria

- [ ] Automated tests cover the new separation guarantees.
- [ ] Benchmarks show clear runtime overhead improvements or document tradeoffs explicitly.
- [ ] The project can continue future inference work on top of a stable, intentionally minimal runtime boundary.

## Cross-cutting rules for all phases

- [ ] Prefer **shared runtime primitives** over duplicating execution logic, but do not let shared code grow training-only assumptions.
- [ ] Keep **observability optional** and default-off for deployment builds and hot paths.
- [ ] Treat **memory reuse and lazy initialization** as first-class runtime features, not later optimizations.
- [ ] Require **phase-local tests and benchmark updates** before marking a phase complete.
- [ ] Update this plan in-place as work starts and finishes so it remains the single progress tracker for runtime separation.
