# Offload Planner Strategy Guide (Phase 3)

This guide describes the initial `model.auto_offload(...)` strategies and their
tradeoffs.

## API

```python
model.auto_offload(
    devices=[...],
    strategy="balanced",  # balanced | memory-first | transfer-minimized
    sample_input=...,
    memory_budgets_bytes={"cuda:0": 8_000_000_000}
)
plan = model.offload_plan(explain=True)
frozen = model.freeze_offload_plan()
model.apply_offload_plan(frozen)
```

`offload_plan(explain=True)` returns:
- `plan`: layer-path -> device
- `rationale`: layer-path -> planner rationale string

## Strategies

### `balanced`
- Greedy placement by lowest cumulative cost.
- Cost model includes parameter bytes + estimated activation bytes.
- Cost model includes a compute proxy (`param_bytes + activation_bytes/2`) and
  transfer proxy (boundary activation bytes).
- Good default for mixed latency/memory tradeoff.

### `memory-first`
- Greedy placement by lowest cumulative parameter bytes.
- Useful when memory pressure is the primary concern.
- Enforces optional per-device memory budgets when provided.

### `transfer-minimized`
- Keeps adjacent layers on same device whenever possible.
- Useful for reducing boundary transfer overhead.

## Current constraints

- Planner currently checks backend support using a conservative per-layer dtype
  capability gate.
- If no candidate device can satisfy a layer constraint, planning fails fast.
- If no candidate device can satisfy configured memory budget constraints,
  planning fails fast.

## Plan persistence

- `freeze_offload_plan()` exports an explicit manual plan (`layer -> "device:index"`).
- Persist this dictionary in JSON/YAML and reapply with `apply_offload_plan(...)`.
- Reapplying avoids planner reruns and is intended for reproducible deployment.

## Suggested workflow

1. Start with `balanced`.
2. Compare with `transfer-minimized` on transfer-heavy graphs.
3. Use `offload_plan(explain=True)` to review rationale.
4. Validate with `model.validate_offload_plan(sample_input)` before long runs.
