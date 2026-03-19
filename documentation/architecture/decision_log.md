# MuNet Architecture Decision Log

This log records the major refactor decisions that future phases should build on instead of re-deciding from scratch.

## How to use this log

- Create a new entry whenever a structural decision changes layering, extensibility, compatibility, or operational behavior.
- Keep entries short and practical: context, decision, consequences, and follow-up phases.
- Prefer append-only updates so later phases can trace why a trade-off was made.

---

## ADR-0001 - Preserve compatibility headers while internals move

- **Status:** Accepted
- **Related roadmap phases:** 0, 1, 2, 3

### Context

MuNet is actively restructuring internal layout (`core/`, `autograd/`, `nn/`, utility headers), but the public include surface should remain stable while the architecture is still moving.

### Decision

Keep lightweight compatibility/umbrella headers in place while implementation details move into more focused internal headers.

### Consequences

- Refactors can proceed without forcing immediate downstream include updates.
- The codebase gains a migration path instead of a flag day rename.
- Compatibility wrappers should be revisited once the core architecture is stable.

---

## ADR-0002 - Finish Phase 0 guardrails before treating Phase 1 as complete

- **Status:** Accepted
- **Related roadmap phases:** 0, 1

### Context

Some dtype/TensorOptions work landed early, but the roadmap still requires baseline inventory, explicit decision logging, and clearer regression/test coverage before the refactor can be considered controlled.

### Decision

Treat Phase 0 as the current priority and keep Phase 1 marked as in-progress until the remaining Phase 0 guardrails are documented and the phase is intentionally closed.

### Consequences

- The roadmap remains honest about current progress.
- Follow-up merge requests can target Phase 1 cleanly after Phase 0 closes.
- Documentation becomes the source of truth for what is done vs merely started.

---

## ADR-0003 - Dtype conversion is centralized in the tensor core before backend dispatch is generalized

- **Status:** Accepted
- **Related roadmap phases:** 1, 2, 3

### Context

Mixed precision and additional dtypes require a central policy for conversion and scalar behavior, but backend kernels and op dispatch are still heavily float-oriented.

### Decision

Centralize dtype helpers and tensor-level conversion behavior first, then move backend/op dispatch to capability-based and per-op systems in later phases.

### Consequences

- Early dtype work has one obvious place to evolve.
- Backend kernels remain a known hotspot until later phases land.
- Future dispatch work should reuse the dtype primitives added in the tensor/type layer rather than reintroducing per-backend policies.

---

## ADR-0004 - Typed scalar helpers live with core dtype metadata and tensor fills

- **Status:** Accepted
- **Related roadmap phases:** 1, 2

### Context

Several tensor helpers still assumed `float` semantics for scalar constants, including scalar expansion, masked fills, and host-side buffer conversions.

### Decision

Keep scalar read/write/conversion helpers in `src/types.hpp`, and route constant tensor fills through tensor-level typed scalar utilities instead of float-only shortcuts.

### Consequences

- Scalar conversion rules now extend from the same dtype policy layer as promotion and accumulation rules.
- Tensor helpers such as constant fills and CPU fallback paths can preserve non-float dtypes without ad hoc casts.
- Backend kernels are still largely float-specialized, but later phases can build on shared scalar primitives instead of duplicating conversions.

---

## ADR-0005 - Module constructor dtype defaults follow requested tensor options, while normalization buffers use accumulation dtype

- **Status:** Accepted
- **Related roadmap phases:** 1, 5

### Context

NN modules were still hard-coding `Float32` parameters and buffers, which made the new dtype/TensorOptions work stop at the tensor layer instead of reaching model construction.

### Decision

Allow core NN modules to accept `TensorOptions` for parameter construction, keep learnable parameters in the requested dtype, and initialize normalization running-stat buffers in `accumulation_type(AccumulationOp::Normalization, dtype)`.

### Consequences

- Module construction now participates in the same dtype policy as tensor creation.
- BatchNorm-style buffers can remain numerically safer than low-precision weights without inventing a separate API yet.
- Optimizers and serialization still need a follow-up phase to formalize how master/accumulation state should be stored and restored.

---

## ADR-0006 - Mixed-precision training preserves parameter dtype in checkpoints, while optimizer state and master/scaler state stay in fp32

- **Status:** Accepted
- **Related roadmap phases:** 1, 5, 6

### Context

Phase 1 needs one coherent answer for saved tensor dtypes, optimizer accumulation/state dtypes, and model reconstruction from serialized artifacts. The next mixed-precision stage also needs a decision for master weights and grad-scaling metadata before AMP-style training can expand beyond the current typed-state fallback.

### Decision

Preserve each tensor's own dtype in serialization, reconstruct built-in modules from saved parameter dtype metadata, and allocate optimizer state in `optimizer_state_type(parameter_dtype)` even when parameter storage remains lower precision.

When mixed-precision training grows beyond the current fallback path:

- low-precision trainable parameters keep their user-visible storage dtype in the model and in checkpoints
- optimizers may optionally maintain fp32 master weights for low-precision parameters, but those master weights are optimizer-owned state rather than part of the module parameter surface
- grad-scaling metadata is optimizer/trainer state and should be serialized alongside optimizer state, always in float32
- checkpoint loading restores:
  - model tensors in their saved dtype
  - optimizer state tensors in `optimizer_state_type(parameter_dtype)`
  - optional master weights only when the target optimizer/training policy enables them
- inference-only save/load paths must ignore optimizer/master/scaler state entirely

### Consequences

- Saving/loading no longer silently collapses tensors to float32.
- Low-precision parameters can keep float32 optimizer state without redefining the parameter API.
- Future master-weight and grad-scaling support now has a storage/serialization boundary: model checkpoints preserve model dtype fidelity, while trainer checkpoints own fp32 master/scaler state.
- Full backend/kernel support for mixed parameter/state/master-weight dtypes is still incomplete and should be made explicit in later phases.

---

## ADR-0007 - Coarse backend capability dtype policy stays centralized while dispatch is still maturing

- **Status:** Accepted
- **Related roadmap phases:** 1, 2, 3

### Context

The initial `BackendFeature` capability surface landed, but the coarse dtype policy for those capabilities was duplicated across CPU, CUDA, and Vulkan backend classes. That duplication would make the early capability surface drift before the later per-op dispatch split is ready.

### Decision

Keep the current coarse capability-vs-dtype policy in one shared helper in `src/core/backend.hpp`, and have concrete backends delegate to that helper until backend-specific or per-op dispatch rules need to diverge.

### Consequences

- Dtype gating for backend capabilities now has one obvious source of truth during Phase 1.
- Tests can assert the capability matrix directly without depending on a specific backend instance.
- Later phases can replace or refine the shared helper with backend-specific dispatch metadata without changing the public capability-query API.
