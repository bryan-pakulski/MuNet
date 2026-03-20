# Inference Runtime Phase 3 Memory and Backend Policy

This document starts **Phase 3 - Memory and backend policy optimization** for the inference/runtime separation effort.

The goal of this phase is to make inference memory use bounded and deployment-oriented while keeping backend startup proportional to the devices actually used.

## Phase 3 status

- [x] Phase 3 - Memory and backend policy optimization
- [x] Prepared-input cache policy made configurable and bounded
- [x] Repeat-run cache stats exposed for benchmark/test validation
- [x] Backend initialization behavior reviewed
- [x] Deploy-vs-training backend capability split documented in code and tests
- [x] Constrained-system fallback policy documented in code and tests
- [x] Hardware-tier recommendations documented
- [x] Output/workspace reuse policy implemented
- [x] Packaging/build reduction for unused backends evaluated

## Phase 3 changes landed so far

### 1. Bounded prepared-input cache policy

`inference::Engine` now exposes a deployment-oriented prepared-input cache policy:

- `prepared_input_cache_entries`
- `prepared_input_cache_max_bytes`

This turns the transfer cache into an explicitly bounded memory feature instead of an unbounded convenience cache.

Current behavior:

- cache entries are capped by count and total bytes
- oversized inputs are not cached
- repeated `run_batch(...)` calls can still reuse transferred inputs when they fit inside the configured budget
- constrained deployments can set the entry count to `0` or use a very small byte budget to keep temporary memory bounded
- `prepare_batch(...)` can pre-populate reusable prepared-input buffers during compile/warmup flows
- `run_batch_into(...)` allows callers to reuse the batch-output container across repeated runs instead of reallocating the output vector each time

### 2. Runtime visibility

`EngineStats` now reports:

- prepared-input cache entry count
- prepared-input cache bytes
- cache hits / misses
- cache evictions

`munet_inference_baseline` also reports the active cache policy and cache statistics in its JSON output.

This gives Phase 3 a concrete way to validate whether memory reuse is helping repeat-run behavior without guessing from wall time alone.

### 3. Backend initialization review

The current backend manager already defers **backend instance construction** until `BackendManager::get(device)` is called, so unused devices are not initialized at process start.

The current evaluation is sufficient for Phase 3 because backend instance construction is lazy today, inference builds are already isolated from training headers/public APIs, and the remaining package-shape work has been pushed into later rollout/packaging phases rather than blocking the runtime-policy milestone.

### 4. Deploy-vs-training backend capability split

The backend feature surface now explicitly distinguishes deploy-runtime versus training-only features:

- deploy/runtime: math, activation, reduction, random fill, shape, loss, and spatial inference features
- training-only: `OptimizerStep`

This keeps the deploy/runtime classification visible in code and test coverage instead of leaving it implicit.

### 5. Constrained fallback policy

MuNet now has an explicit constrained-system policy encoded in backend fallback helpers:

- `CPUFallback` for deploy-safe math/activation/reduction/random features where CPU execution is acceptable
- `ExplicitUnsupported` for features where silent fallback would violate runtime expectations or hide material capability gaps (for example convolution/spatial acceleration gaps or optimizer functionality in an inference runtime)

### 6. Hardware-tier recommendations

- **Edge / constrained CPU-only**: keep `prepared_input_cache_entries` small (or `0`), favor `lean_mode=True`, and rely on CPU fallback only for deploy-safe math/reduction features.
- **Workstation / selective accelerator**: use bounded prepared-input cache plus `prepare_batch(...)` when repeated host-to-device transfers dominate request setup.
- **Enterprise / accelerator-heavy serving**: keep cache policy sized to the active request working set, use `run_batch_into(...)` to avoid repeat batch-output container allocations, and treat unsupported spatial/optimizer features as explicit deployment configuration errors rather than silent fallback.

## Current follow-on work

- carry deploy/runtime capability classes into later packaging/rollout work if the project introduces slimmer distribution targets
- deepen true tensor-storage reuse in future backend-specific work where operator APIs can safely accept caller-owned output storage
