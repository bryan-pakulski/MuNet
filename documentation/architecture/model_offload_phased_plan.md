# Model Offload Roadmap (Phased)

This document defines an implementation plan and hard exit criteria for a
developer-facing model-offload API:

```python
model.offload(device_a, layers=[...])
model.offload(device_b, layers=[...])
out = model(x)  # automatic inter-device transitions
```

The goal is to keep user code simple while MuNet handles device placement,
transfers, and diagnostics.

---

## Scope and non-goals

### In scope
- Layer/module placement by device.
- Automatic boundary transfer between differently placed modules.
- Validation and diagnostics for placement plans.
- Optional auto-planner (heuristic) in later phase.
- Training + inference compatibility.

### Out of scope (initially)
- Distributed multi-process orchestration.
- Graph compiler-level partitioning.
- Cross-host / network collectives.

---

## Phase 1 — Manual placement + automatic boundary transfer

## Developer API (minimum)
- `model.offload(device, layers=[...])`
- `model.clear_offload()`
- `model.offload_plan() -> dict[path, device]`

## Core implementation
1. **Placement registry**
   - Add a model-owned mapping: `module_path -> Device`.
   - Paths align with `named_modules()` names.

2. **Forward boundary enforcement**
   - Before each child module forward, if input tensor device differs from that
     child’s assigned device, perform `x = x.to(assigned_device)`.
   - Preserve autograd edges for training.

3. **Parameter/buffer alignment**
   - On `offload(...)`, move module parameters/buffers to assigned device via
     existing module conversion paths.

4. **Deterministic behavior**
   - If no offload plan exists, current behavior remains unchanged.
   - Explicit, stable module ordering for containers (`Sequential` etc.).

## Phase 1 test requirements
- Unit: placement registry CRUD and path resolution.
- Unit: boundary transfer insertion on cross-device edges.
- Integration: training backward works across at least one mixed-device chain.
- Integration: inference parity against single-device baseline.

## Phase 1 docs + demos requirements
- Python API docs for `offload`, `clear_offload`, `offload_plan`.
- Debugging Playbook section: common offload errors and fixes.
- New demo:
  - `demos/multigpu/model_offload_manual_demo.py`
  - Show 2-stage model split across devices and normal `model(x)` usage.

## Phase 1 exit criteria (all required)
- [x] Offload API is public and documented.
- [x] Backward pass succeeds on cross-device boundaries.
- [x] Numerical parity (within tolerance) vs single-device baseline on reference model.
- [ ] No regression in existing unit/integration suites.
- [ ] Demo runs successfully on supported device pairs.

### Phase 1 exit criteria tracking (as of 2026-03-28)
- **Implemented and covered in code/tests/docs**
  - Offload API is exposed in bindings and documented.
  - Mixed-device backward + inference parity tests are present in
    `tests/test_offload_phase1.py`.
- **Left to confirm before declaring Phase 1 fully complete**
  1. **No-regression confirmation from full suite execution**
     - Ensure the full C++ and Python unit/integration suites run in CI and pass.
     - Python coverage now includes all `tests/*.py` via `make py-test`.
  2. **Demo validation on supported device pairs**
     - Run `demos/multigpu/model_offload_manual_demo.py` on at least one supported
       accelerator pair configuration and capture smoke-test evidence.

---

## Phase 2 — Validation tooling + transfer-cost warnings

## Developer API additions
- `model.validate_offload_plan(sample_input) -> ValidationReport`
- `model.set_offload_warnings(enabled=True)`

## Core implementation
1. **Plan checker**
   - Verify all referenced layers exist.
   - Verify required backend feature support per placed layer (dtype/op checks).
   - Verify tensor shape continuity at stage boundaries.

2. **Transfer diagnostics**
   - Track boundary transfer count, bytes, and directions per run.
   - Warn on expensive patterns (e.g., many small transfers, ping-ponging).

3. **Policy/report object**
   - Return structured warnings/errors with actionable guidance.
   - Integrate with existing profiler/dispatch diagnostics where possible.

## Phase 2 test requirements
- Unit: validator catches invalid/missing layer names.
- Unit: validator catches unsupported backend feature/dtype combinations.
- Integration: warnings emitted for synthetic ping-pong plan.
- Integration: telemetry counters reset/snapshot consistency.

## Phase 2 docs + demos requirements
- API reference for `ValidationReport` fields and warning classes.
- Playbook section: “offload plan validation and transfer hotspots.”
- Demo update:
  - Extend manual offload demo to show validator output and warning examples.

## Phase 2 exit criteria (all required)
- [ ] Validation report is stable, typed, and documented.
- [ ] Transfer warning thresholds configurable and documented.
- [ ] At least one CI job asserts validator behavior on known bad plans.
- [ ] Demo includes both valid and intentionally bad plan walkthroughs.

---

## Phase 3 — Auto planner (heuristic)

## Developer API additions
- `model.auto_offload(devices=[...], strategy="balanced", sample_input=...)`
- `model.offload_plan(explain=True)` includes planner rationale.

## Core implementation
1. **Cost model**
   - Estimate per-layer compute cost and activation/parameter memory.
   - Estimate transfer overhead between candidate boundaries.

2. **Heuristic planner**
   - Candidate strategies:
     - balanced latency
     - memory-first
     - transfer-minimized
   - Constrain to backend capability support and memory budgets.

3. **Plan freezing**
   - Planner returns explicit manual plan that can be persisted and reused.

## Phase 3 test requirements
- Unit: planner deterministic for fixed seed + fixed profile data.
- Unit: planner obeys hard constraints (unsupported op/dtype/device memory).
- Integration: planner plan executes and converges on reference training case.
- Benchmark: planner strategy improves selected metric vs naive split.

## Phase 3 docs + demos requirements
- Planner strategy guide (tradeoffs, knobs, caveats).
- Benchmark demo:
  - compare `manual`, `auto balanced`, `auto memory-first`.
- Export/import plan demo for reproducible deployment.

## Phase 3 exit criteria (all required)
- [ ] Planner emits valid executable plan for supported reference models.
- [ ] Planner rationale/explain output is user-visible and documented.
- [ ] Benchmarks show non-trivial gain for at least one strategy on reference workload.
- [ ] Users can persist and reapply plans without planner rerun.

---

## Cross-phase quality gates

These must hold at every phase boundary:
- Backward compatibility: existing non-offloaded model APIs unchanged.
- Observability: profiler/diagnostic breadcrumbs for transfers and placement.
- Safety: clear errors for unsupported plans (never silent incorrect fallback).
- Demo health: demos are runnable and covered by smoke checks.
- Documentation health: docs updated in same PR as behavior changes.

---

## Suggested CI gates by phase

- **Phase 1**
  - unit + integration + demo smoke for manual split.
- **Phase 2**
  - add validator negative-case suite and warning snapshot tests.
- **Phase 3**
  - add heuristic planner determinism + benchmark sanity thresholds.
