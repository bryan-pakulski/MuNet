# Inference Runtime Phase 3 Memory and Backend Policy

This document starts **Phase 3 - Memory and backend policy optimization** for the inference/runtime separation effort.

The goal of this phase is to make inference memory use bounded and deployment-oriented while keeping backend startup proportional to the devices actually used.

## Phase 3 status

- [~] Phase 3 - Memory and backend policy optimization
- [x] Prepared-input cache policy made configurable and bounded
- [x] Repeat-run cache stats exposed for benchmark/test validation
- [~] Backend initialization behavior reviewed
- [ ] Output/workspace reuse policy implemented
- [ ] Packaging/build reduction for unused backends implemented

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

However, Phase 3 is not done yet because inference builds still link the compiled backend set into `munet_core`, and the deployment/runtime surface does not yet provide reduced-feature packaging for unused backends.

## Current follow-on work

- add reusable output/workspace policies where operator/backend support allows it
- evaluate preallocation flows for compile/warmup on repeated deployment shapes
- define reduced-feature or static inference build profiles that omit optional conversion/debug facilities where appropriate
- separate deploy-required backend capabilities from training-only/backend-development capabilities
- document hardware-tier recommendations for edge, workstation, and enterprise deployments
