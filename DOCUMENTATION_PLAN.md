# MuNet Documentation Plan

## Goals
- Make backend behavior, performance tuning, and extension workflows discoverable.
- Provide both API reference and architecture-level guidance.
- Add reproducible performance/debug playbooks.

## Phases
1. **Foundation**
   - Keep generated API reference (current pydoc) for all Python bindings.
   - Add a docs index page with navigation by audience: user, contributor, backend engineer.
2. **Architecture Guides**
   - Runtime architecture (Tensor/Storage/BackendManager/Autograd).
   - Training vs inference layering (`munet_core`, `munet_training`, `munet_inference`).
   - Backend lifecycle and registration pattern.
3. **Backend Authoring Guide**
   - Step-by-step for new op/kernels: interface, CPU/CUDA/Vulkan implementations, bindings, tests.
   - Validation checklist and perf sanity checklist.
4. **Performance & Profiling Guide**
   - How to use `MUNET_PROFILE`, `MUNET_DEBUG`, `MUNET_LOG_LEVEL`.
   - How to interpret profiler output and common bottleneck signatures.
   - Benchmark best practices (device-resident tensors, warmup, sync strategy).
5. **Modeling/User Guides**
   - Building models with `nn`, optimizers, save/load.
   - Device placement and mixed backend usage examples.

## Tooling Proposal
- Keep pydoc for API extraction initially.
- Add Markdown docs in `docs/` and publish static site via MkDocs (or Sphinx) in CI.
- Add doc linting in CI (links + markdown style) once docs baseline exists.

## Deliverables (Near-term)
- [ ] `docs/index.md`
- [ ] `docs/architecture/runtime.md`
- [ ] `docs/architecture/backends.md`
- [ ] `docs/performance/profiling.md`
- [ ] `docs/contributing/add_new_op.md`
