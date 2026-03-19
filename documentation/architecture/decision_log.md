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
