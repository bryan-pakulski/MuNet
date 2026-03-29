# Offload Planner Post-Phase-3 Follow-up Plan

This document tracks architecture items intentionally deferred after Phase 3.

## 1) Persistence compatibility contract (planned)

### Motivation
Current `freeze_offload_plan()` / `apply_offload_plan(...)` support enables
reproducible deployment, but there is not yet a formal serialized schema version
contract for long-term compatibility.

### Planned design
- Introduce a versioned plan envelope, for example:
  - `schema_version`
  - `model_fingerprint` (optional)
  - `layer_plan`
  - `planner_metadata`
- Define strict/relaxed apply modes:
  - strict: fail if layer set or device specs mismatch
  - relaxed: best-effort apply with warnings
- Define forward/backward compatibility policy and migration helper.

### Proposed milestones
- P1: schema and validation utility
- P2: loader migration hooks + warnings
- P3: docs + e2e persistence compatibility tests

## 2) Operator capability model broadening (planned)

### Motivation
Current auto-planner checks use conservative backend feature gates and dtype
checks, which are safe but can be too coarse for mixed-op graphs.

### Planned design
- Extend capability checks from generic backend gates to op-level requirements.
- Add optional per-layer op profile hints collected from dry-run tracing.
- Add planner constraint diagnostics that identify *which* op requirement blocked
  candidate placement.

### Proposed milestones
- C1: op capability catalog for core ops used in reference models
- C2: planner integration and richer explain payload
- C3: negative-case tests for unsupported op/backend combinations

## Tracking
- Status: planned
- Owner: runtime/offload
- Last updated: 2026-03-29
