# MuNet production readiness review (2026-03-26)

This review translates the current repo state into a concrete path to production.

## Executive summary

MuNet has a strong core architecture trajectory, but it is still in a transition stage from research/demo to hardened runtime. Existing roadmap docs and README already acknowledge this direction and leave key production items open.

## Top gaps to close before production

### 1) Reliability & correctness hardening

- Finish distributed primitives and multi-device correctness:
  - `all_reduce` is still unimplemented in CUDA backend.
  - Multi-GPU roadmap is documented but not complete.
- Complete dtype support parity in core math paths (notably fp16/bf16/int8 inference/training pathways), starting with matmul and reduction critical paths.
- Add stricter runtime checks and failure-domain isolation for memory, shape/stride, and backend transfer edge-cases.

## 2) Performance and memory management

- Replace current simple caching allocators with production-grade allocator strategy (arena/slab + fragmentation telemetry + high-watermark controls).
- Prioritize kernel maturity for CUDA/Vulkan hot paths (matmul, conv, attention) and add vendor-optimized routes where available.
- Build reproducible perf baselines and regression thresholds into default CI gates (not only opt-in local runs).

## 3) Inference productization

- Complete inference-runtime separation/lean packaging phase and keep training-only symbols isolated from deployment artifacts.
- Formalize model compile contract/versioning and load-time compatibility guarantees.
- Add cold-start/warm-start and memory budget SLO validation for target deploy environments.

## 4) Release engineering and supply-chain stability

- Introduce first-class install/distribution targets (CMake install/export, package metadata, semantic versioning).
- Make test execution hermetic (avoid tests that self-install dependencies or self-trigger builds).
- Lock dependency versions and build environments for deterministic release artifacts.

## 5) Security hardening

- Treat serialized model artifacts as untrusted input by default:
  - enforce artifact size limits,
  - enforce bounded parser behavior,
  - add fuzzing/property tests for serialization loaders,
  - and provide provenance/signing verification for deploy artifacts.

## 6) Observability and operability

- Extend current env-flag logging/profiling to structured telemetry:
  - stable metric names,
  - machine-readable logs,
  - trace/span export hooks,
  - and SLO-focused runtime counters (latency percentiles, OOM/fallback counters, backend error taxonomy).

## Suggested 90-day sequence

1. **Weeks 1-3 (Quality Gate Foundation):** CI matrix + hermetic tests + sanitizer jobs + packaging baseline.
2. **Weeks 3-6 (Inference Hardening):** complete runtime-separation milestones + serialization security controls + compatibility tests.
3. **Weeks 6-9 (Performance):** allocator rework + kernel tuning on top 3 operator families + perf regression gate.
4. **Weeks 9-12 (Operational Readiness):** observability export + release checklist + production canary criteria.

## Evidence pointers in repository

- README marks inference engine as the highest current priority and lists production improvements still needed (naive kernels, allocator limits, missing dtypes, limited error handling).
- Refactor roadmap still has Phase 5 in progress and Phase 7 not started (lean inference packaging/runtime separation pending).
- CPU backend currently uses a simple free-list allocator and no-op `all_reduce`.
- CUDA backend has `all_reduce` explicitly marked TODO.
- Performance tests are opt-in via env var and require both CUDA and Vulkan backends.
- Python tests currently self-install NumPy and can trigger local builds if import fails (non-hermetic behavior).
- Serialization loader currently reads full artifact bytes into memory before parse.
