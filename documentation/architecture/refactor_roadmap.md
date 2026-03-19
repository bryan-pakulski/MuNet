# MuNet Refactor Roadmap

This roadmap captures the next restructuring milestones needed to move MuNet from a promising research/demo codebase toward a more production-ready and hardened library.

The recent header split made the codebase easier to navigate, but the next stage needs to focus on execution architecture: dtype handling, op dispatch, backend capabilities, autograd safety, and training/inference ergonomics.

## Status legend

- [ ] not started
- [~] in progress
- [x] completed

## Current implementation status

- [x] Phase 0 - Baseline guardrails and inventory
- [x] Phase 1 - Dtype foundation and tensor options
- [x] Phase 2 - Op dispatch and op file decomposition
- [ ] Phase 3 - Backend capability split and registry cleanup
- [ ] Phase 4 - Autograd hardening
- [ ] Phase 5 - Module, optimizer, and training ergonomics
- [ ] Phase 6 - Inference and production hardening

## Primary goals

- Make **mixed precision and future dtype work** a planned extension point rather than a cross-cutting retrofit.
- Reduce the number of files and subsystems touched when adding a new op, kernel, or backend capability.
- Isolate responsibilities between **core tensor/storage**, **op dispatch**, **autograd**, **nn modules**, **optimizers**, and **inference**.
- Add guardrails so unsupported dtype/backend combinations fail clearly and early.
- Improve maintainability without breaking the current public API abruptly.

## Guiding principles

1. **Stabilize the core first.** Do not add AMP or fp16 training features until dtype and dispatch foundations exist.
2. **Prefer additive migration paths.** Keep compatibility wrappers where practical while moving implementation to new modules.
3. **Move behavior, not just headers.** A split file layout is useful only if execution responsibilities also become modular.
4. **Design for partial backend support.** New backends should be able to implement capability subsets and fallback cleanly.
5. **Make testability part of the refactor.** Every phase should ship with focused validation and clear exit criteria.

## Phase 0 - Baseline guardrails and inventory

### Objectives

- Establish a stable baseline before deep structural work begins.
- Document the current coupling points for dtype, backend dispatch, and autograd.
- Prevent regressions while the architecture is being reworked.

### Action points

- [x] Add an architecture decision log under `documentation/architecture/` for major refactor decisions.
- [x] Inventory all dtype-sensitive code paths:
  - tensor creation and scalar helpers
  - backend kernels
  - autograd accumulation paths
  - module parameter/buffer initialization
  - optimizers and serialization
- [x] Add focused tests for the current behavior of:
  - tensor dtype preservation
  - backend registration/fallback behavior
  - parameter/buffer device migration
  - autograd grad accumulation
- [x] Mark known technical debt explicitly in docs so future work can reference it.

### Exit criteria

- [x] Current architecture pain points are documented.
- [x] Regression coverage exists for the pieces that will be heavily refactored next.
- [x] Future phases can point to named design decisions instead of re-litigating fundamentals.

## Phase 1 - Dtype foundation and tensor options

### Why this phase comes first

Mixed precision support depends on having a real dtype model. Right now dtype exists mostly as metadata. That is not enough for fp16 storage, fp32 accumulation, or autocast.

### Objectives

- Introduce a proper dtype utility layer.
- Make dtype a first-class part of tensor creation and conversion.
- Eliminate float-only assumptions from the core tensor API.

### Action points

- [~] Replace the current lightweight dtype helpers with a richer core type system, for example:
  - `ScalarType` / `DataType`
  - `DTypeInfo`
  - `is_floating`, `is_integral`, `is_low_precision`
  - `promote_types(a, b)`
  - `accumulation_type(op, dtype)`
- [x] Introduce `TensorOptions` to carry:
  - device
  - dtype
  - requires_grad
  - future layout/memory format hooks
- [x] Add dtype-aware conversion APIs, e.g.:
  - `Tensor::to(Device)`
  - `Tensor::to(DataType)`
  - `Tensor::to(TensorOptions)`
- [~] Remove scalar helpers that hard-code `float` semantics from the public tensor interface.
- [~] Add typed scalar utility helpers for constants, fills, and host/device transfers.
  - Landed core scalar buffer conversion helpers and typed tensor fill/masked-fill paths.
- [x] Define policy for:
  - parameter dtype
  - buffer dtype
  - accumulation dtype
  - serialization dtype fidelity
  - Current implementation direction:
    - module parameters follow the explicitly requested `TensorOptions::dtype`
    - normalization running-stat buffers default to `accumulation_type(AccumulationOp::Normalization, dtype)`
    - module-wide `to(DataType)` / `to(TensorOptions)` conversions preserve parameter-vs-buffer `requires_grad` semantics
    - optimizer state follows `optimizer_state_type(parameter_dtype)`
    - serialization keeps tensor dtype fidelity during save/load and reconstructs built-in modules from saved dtype metadata
    - future fp32 master weights and grad-scaling metadata belong to optimizer/trainer checkpoints, not module-only save/load

### Validation checklist

- [x] Tensors can be created, cloned, moved, and copied while preserving dtype correctly.
- [x] Scalar creation and `item()`-style access work for all supported dtypes or fail explicitly when unsupported.
- [x] Promotion and accumulation rules are tested and documented.
  - Core dtype metadata (`DTypeInfo`), explicit accumulation policy coverage, and typed scalar regression coverage are now in place.

### Exit criteria

- [x] Dtype decisions live in one place, not spread through ops and kernels.
  - Accumulation, optimizer-state, backend-capability, autograd accumulation, and ONNX graph-runtime dtype handling now route through shared dtype helpers instead of isolated float32-only paths.
- [x] Future fp16 work no longer requires editing the tensor API shape itself.

Phase 1 follow-up items are complete. Any finer-grained per-op/per-shape dispatch or backend capability refinement belongs to Phase 2 / Phase 3 work rather than keeping Phase 1 open.

## Phase 2 - Op dispatch and op file decomposition

### Objectives

- Break up the monolithic op implementation surface.
- Separate op definitions, backend dispatch, and autograd node behavior.
- Make adding a new op a local change.

### Action points

- Split `ops.hpp` into per-op headers and implementation units, for example:
  - `core/ops/add.hpp`
  - `core/ops/matmul.hpp`
  - `core/ops/conv2d.hpp`
  - `core/ops/normalization.hpp`
- Move backward node implementations into dedicated autograd node files, e.g.:
  - `autograd/nodes/add_node.hpp`
  - `autograd/nodes/conv2d_node.hpp`
- Introduce an op dispatch layer that chooses implementation based on:
  - op id
  - device/backend
  - dtype
  - capability availability
- Keep the convenience tensor methods, but make them thin wrappers around the free-function op layer.
- Add a consistent registration pattern for op metadata and tracing.
- Minimize direct backend calls inside high-level op glue.

### Validation checklist

- [x] Introduced an op metadata/dispatch registry for the full op surface so backend + dtype capability checks are centralized.
- [x] Migrated all operators out of the monolithic legacy header into `core/ops/*` headers + implementation files and moved their backward nodes into `autograd/nodes/*`.
- [x] Removed the temporary legacy-op bridge after the migration was complete.
- [x] Convenience `Tensor` methods continue to forward through the free-function op layer.
- [x] An op can report unsupported backend/dtype combinations with a clear message.
- [x] Trace metadata remains intact after the split.
- [x] Adding a new op does not require editing `Tensor` internals beyond an optional forwarding convenience method.

### Exit criteria

- Op logic is modular.
- Backend dispatch is centralized.
- New kernels can be introduced without touching unrelated ops.

## Phase 3 - Backend capability split and registry cleanup

### Objectives

- Replace the single monolithic backend contract with composable capability groups.
- Improve support for partial backend implementations and clean fallbacks.
- Make backend registration more testable and explicit.

### Action points

- [x] Split backend responsibilities into capability interfaces such as:
  - allocation / transfer
  - elementwise math
  - reduction
  - BLAS/matmul
  - spatial ops
  - normalization
  - optimizer kernels
- [x] Add capability queries such as:
  - `supports(op, dtype)`
  - `supports(op, dtype, shape)` if shape-dependent kernels matter later
  - preferred accumulation dtype / preferred fallback policy
- [x] Introduce a `BackendRegistry` abstraction so registration, override, and test replacement are explicit.
- [x] Keep debug/profiling wrappers as decorators instead of baking them into backend selection paths.
- [x] Define fallback policy clearly:
  - CPU fallback
  - explicit unsupported error
  - opt-in conversion fallback for specific ops

### Phase 3 implementation update

- Capability groups now own backend responsibilities while `Backend` remains a thin coordination layer.
- Dispatch decisions consult backend support metadata before calling kernels, including fallback-policy reporting.
- Registry overrides can be exercised through isolated `BackendRegistry` instances or through `BackendManager` for integration coverage.

### Validation checklist

- [x] A backend can implement only a subset of capabilities and still participate safely.
- [x] Missing kernels are surfaced by capability checks rather than late runtime surprises.
- [x] Backend overrides are easy to test in isolation.

### Exit criteria

- [x] Backend extensions no longer require growing one giant interface every time.
- [x] Mixed backend coverage becomes manageable.

## Phase 4 - Autograd hardening

### Objectives

- Make autograd safer, more extensible, and more resilient to future AMP/training features.
- Reduce hidden global assumptions.

### Action points

- Turn the execution engine into an object-oriented service rather than a single static routine.
- Add version counters or equivalent in-place mutation safety checks.
- Add explicit `retain_graph` handling and clearer graph lifetime rules.
- Introduce gradient buffer reuse/pooling for intermediate accumulation.
- Separate edge metadata, dependency tracking, and execution scheduling more clearly.
- Design extension points for:
  - higher-order gradients
  - custom backward nodes
  - gradient hooks
  - unscale/overflow checks used by AMP

### Validation checklist

- In-place mutation errors surface clearly.
- Repeated backward behavior is predictable and documented.
- Gradient accumulation remains correct across multiple branches.

### Exit criteria

- Autograd is no longer a bottleneck for optimizer or AMP work.
- Safety rules are explicit instead of implicit.

## Phase 5 - Module, optimizer, and training ergonomics

### Objectives

- Make `nn` and `optim` align with the new dtype/backend architecture.
- Prepare for real mixed precision training workflows.

### Action points

- Add module construction/configuration patterns that can carry device and dtype defaults.
- Ensure parameter and buffer initialization no longer hard-code `Float32` unless explicitly requested.
- Introduce parameter groups in optimizers.
- Add optimizer state policies for:
  - model dtype
  - master weight dtype
  - momentum/state tensor dtype
- Design `GradScaler` / loss scaling interfaces.
- Add autocast guard APIs with a clear policy for allowed implicit conversions.
- Document recommended training flows for:
  - fp32 baseline
  - mixed precision training
  - pure fp16 inference where safe

### Validation checklist

- Modules can be created and migrated with predictable device/dtype behavior.
- Optimizer state behavior is explicit and tested.
- Mixed precision training primitives exist without leaking implementation details everywhere.

### Exit criteria

- AMP support becomes an additive feature, not a cross-cutting patch.
- Users can reason about parameter, gradient, and optimizer state dtypes clearly.

## Phase 6 - Inference and production hardening

### Objectives

- Align the inference surface with the refactored core without over-coupling it to training internals.
- Improve production-readiness and operability.

### Action points

- Separate inference-only graph execution concerns from training/autograd concerns more sharply.
- Formalize serialization compatibility and versioning expectations.
- Add stricter error handling and diagnostics for unsupported deployment paths.
- Add memory/performance test suites for representative model flows.
- Define observability hooks for profiling/debugging in production-like environments.

### Exit criteria

- Inference remains lean while still benefiting from the core refactor.
- Production diagnostics and compatibility policies are documented.

## Cross-phase workstreams

### Testing and CI

Each phase should add targeted tests, not just broad smoke coverage.

Recommended additions:

- dtype/unit tests
- backend capability tests
- op dispatch tests
- autograd safety tests
- serialization compatibility tests
- deterministic mixed precision tests once AMP lands

### Documentation

For every completed phase, update:

- architecture docs
- contributor guides
- add-new-op documentation
- backend authoring workflow
- mixed precision user guide once available

### Compatibility policy

- Keep current umbrella headers while internals move.
- Prefer deprecation periods over abrupt API removals.
- Document any intentional behavior changes at the end of each milestone.

## Suggested execution order

If we want the highest leverage path with the least future churn, the recommended order is:

1. Phase 0 - baseline guardrails
2. Phase 1 - dtype foundation
3. Phase 2 - op dispatch split
4. Phase 3 - backend capability split
5. Phase 4 - autograd hardening
6. Phase 5 - module/optimizer/AMP work
7. Phase 6 - inference/production hardening

## Near-term milestone summary

### Milestone A - Core typed foundation

- [~] Complete Phase 0 and Phase 1.
- Outcome: dtype-safe tensor core and test coverage for future fp16 work.

### Milestone B - Modular execution path

- [ ] Complete Phase 2 and Phase 3.
- Outcome: new ops and new backend kernels become local, testable changes.

### Milestone C - Safe training engine

- [ ] Complete Phase 4 and Phase 5.
- Outcome: mixed precision training becomes practical and maintainable.

### Milestone D - Hardened deployment story

- [ ] Complete Phase 6.
- Outcome: inference and deployment workflows become predictable and supportable.

## Definition of success

MuNet will be in a substantially better long-term state when:

- adding a new op requires touching a small, obvious set of files
- adding a new backend capability does not require bloating one central interface
- dtype behavior is centralized and documented
- fp16 / mixed precision support builds on existing abstractions instead of bypassing them
- training and inference surfaces are intentionally layered rather than accidentally entangled
