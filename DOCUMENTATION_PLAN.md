# MuNet Documentation Plan

## Goals
- Host an interactive, searchable docs site for both Python and C++ users.
- Make backend behavior, performance tuning, and extension workflows discoverable.
- Provide both API reference and architecture-level guidance.
- Add reproducible performance/debug playbooks and end-to-end deploy workflows.

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
- Use MkDocs Material for interactive site navigation + search.
- Keep pydoc for generated Python API extraction initially.
- Add a C++ API extraction step (Doxygen) and link it into the docs nav.
- Add Markdown docs in `documentation/` and publish static site in CI (`mkdocs build`).
- Add doc linting in CI (links + markdown style) once docs baseline exists.

## Deliverables (Near-term)
- [x] `documentation/index.md`
- [x] `documentation/architecture/runtime.md`
- [x] `documentation/architecture/backends.md`
- [x] `documentation/performance/profiling.md`
- [x] `documentation/contributing/add_new_op.md`
- [x] `documentation/getting-started/*`
- [x] `documentation/api/python.md`
- [x] `documentation/api/cpp.md`
- [x] `documentation/guides/*`
- [x] `documentation/tutorials/e2e_workflow.md`
- [x] `mkdocs.yml`
