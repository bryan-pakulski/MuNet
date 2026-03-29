# Offload Planner Strategy Guide (Phase 3)

This guide describes the initial `model.auto_offload(...)` strategies and their
tradeoffs.

## API

```python
model.auto_offload(
    devices=[...],
    strategy="balanced",  # balanced | memory-first | transfer-minimized
    sample_input=...
)
plan = model.offload_plan(explain=True)
```

`offload_plan(explain=True)` returns:
- `plan`: layer-path -> device
- `rationale`: layer-path -> planner rationale string

## Strategies

### `balanced`
- Greedy placement by lowest cumulative cost.
- Cost model includes parameter bytes + estimated activation bytes.
- Good default for mixed latency/memory tradeoff.

### `memory-first`
- Greedy placement by lowest cumulative parameter bytes.
- Useful when memory pressure is the primary concern.

### `transfer-minimized`
- Keeps adjacent layers on same device whenever possible.
- Useful for reducing boundary transfer overhead.

## Current constraints

- Planner currently checks backend support using a conservative per-layer dtype
  capability gate.
- If no candidate device can satisfy a layer constraint, planning fails fast.

## Suggested workflow

1. Start with `balanced`.
2. Compare with `transfer-minimized` on transfer-heavy graphs.
3. Use `offload_plan(explain=True)` to review rationale.
4. Validate with `model.validate_offload_plan(sample_input)` before long runs.
