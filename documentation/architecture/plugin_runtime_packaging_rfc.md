# RFC: Runtime-loaded backend plugins and single-wheel Python packaging

- **Status:** Proposed (Phase 0 architecture contract)
- **Date:** 2026-04-01
- **Owners:** MuNet runtime + packaging maintainers
- **Related roadmap:** `model_offload_phased_plan.md`, `refactor_roadmap.md`

## 1) Scope and goals

This RFC defines the compatibility contract for moving to:

1. A single `munet-nn` distribution containing `munet_nn._core` (CPU runtime + plugin loader).
2. Optional accelerator backends loaded at runtime as native plugins.
3. Extras that install dependency bundles only (`vk`, `cu12-vk`, `cu13-vk`) and do not select wheel variants.

Non-goals for this phase:

- Finalizing every plugin implementation detail.
- Shipping CUDA/Vulkan plugins in this change.

## 2) Packaging contract (public)

### 2.1 Distribution naming

- **Only** package name on PyPI: `munet-nn`.
- Import namespace remains `munet_nn`.

### 2.2 Extras policy

Extras are dependency bundles, not binary selectors:

- `munet-nn[vk]`: Vulkan runtime/tooling dependencies.
- `munet-nn[cu12-vk]`: CUDA 12 + Vulkan dependency bundle.
- `munet-nn[cu13-vk]`: CUDA 13 + Vulkan dependency bundle.

Rules:

- Installing an extra must never swap to a different core wheel artifact.
- Core install (`pip install munet-nn`) must remain CPU-usable.
- Missing optional runtimes must degrade gracefully with diagnostics.

### 2.3 Wheel content baseline

Core wheel ships:

- `munet_nn/__init__.py`
- `munet_nn/_core*.so`
- `munet_nn/_helpers/*`

Optional plugin binaries may be:

- included in wheel when legally/technically viable, or
- installed as runtime artifacts discoverable by the loader.

In both cases, import of `munet_nn` must not fail due to absent accelerator stack.

## 3) Core/plugin ABI contract (stable surface)

### 3.1 Plugin binary naming and discovery convention

Linux plugin soname/file convention:

- `libmunet_backend_<name>.so`

Initial canonical names:

- `libmunet_backend_vk.so`
- `libmunet_backend_cu12.so`
- `libmunet_backend_cu13.so`

Discovery order (highest to lowest priority):

1. Paths from `MUNET_BACKEND_PLUGIN_PATH` (path-list).
2. Directory adjacent to `munet_nn/_core*.so` in known plugin subdirs.
3. Platform default library search paths.

The loader must record all probe attempts and failures for diagnostics.

### 3.2 Required exported symbols

Each plugin must export **C ABI** symbols (exact names):

- `munet_backend_plugin_abi_version()` -> `uint32_t`
- `munet_backend_plugin_descriptor()` -> pointer to immutable descriptor struct
- `munet_backend_plugin_create(const munet_plugin_host_api*)` -> opaque plugin instance/handle
- `munet_backend_plugin_destroy(void*)`

The descriptor must include:

- backend key (`"vk"`, `"cu12"`, `"cu13"`),
- plugin build/runtime version strings,
- capability bitset,
- minimum/maximum compatible core ABI versions.

### 3.3 ABI versioning strategy

Define integer constants in core:

- `MUNET_PLUGIN_ABI_MAJOR`
- `MUNET_PLUGIN_ABI_MINOR`

Compatibility rule:

- plugin major must equal core major,
- plugin minor must be `<=` core minor,
- otherwise plugin is rejected deterministically.

Version bump policy:

- **Patch** changes: no ABI changes.
- **Minor** bump: additive ABI only (new optional fields/functions with guards).
- **Major** bump: breaking layout/signature/semantic contract; old plugins rejected.

Deprecation policy:

- any soon-to-be-removed ABI element must be supported for at least one minor release and marked deprecated in docs and diagnostic output.

## 4) Runtime behavior contract

### 4.1 CPU baseline

- CPU backend is always available when `munet_nn` imports successfully.
- Accelerator unavailability cannot break CPU execution paths.

### 4.2 Python probing APIs

Core runtime will expose:

- `munet_nn.list_available_backends() -> list[str]`
- `munet_nn.backend_status() -> dict`

`backend_status()` returns per-backend structured status:

- `discovered` (bool)
- `loadable` (bool)
- `active` (bool)
- `reason_code` (stable machine token)
- `detail` (human-readable actionable message)
- `plugin_path` (if discovered)
- `plugin_abi` and `core_abi`
- `runtime_dependencies` (missing/ok list)

### 4.3 Failure behavior and error-message matrix

| Scenario | Import `munet_nn` | `list_available_backends()` | `backend_status()` reason_code | User-facing guidance |
|---|---|---|---|---|
| No plugins installed | Success | `['cpu']` | `plugin_not_found` | Install plugin/runtime bundle or use CPU |
| Plugin found, ABI mismatch | Success | excludes backend | `abi_mismatch` | Upgrade/downgrade plugin or core to compatible ABI |
| Plugin found, missing driver/runtime | Success | excludes backend | `runtime_dependency_missing` | Install required driver/runtime package |
| Plugin found, init failure | Success | excludes backend | `plugin_init_failed` | Check `detail`, collect logs, validate driver/toolkit |
| Plugin healthy | Success | includes backend | `ok` | Backend available |
| Corrupt/invalid plugin file | Success | excludes backend | `invalid_plugin_binary` | Reinstall plugin artifact |

Contract rules:

- Import-time failures are reserved for core corruption, not optional backend absence.
- `reason_code` values are stable API and can be asserted in tests.
- `detail` text may evolve but must remain actionable.

## 5) Security and trust model

- Plugins are native code and treated as trusted local artifacts.
- Loader only opens files from explicit discovery locations.
- No network fetch/load at import time.
- Future hardening option: optional signature/hash verification gate.

## 6) Upgrade and migration policy

- Existing consumers importing `munet_nn` continue to work.
- One release cycle of compatibility shims is allowed for old helper paths/import assumptions.
- Release notes must include:
  - old vs new packaging layout,
  - extras semantics,
  - backend diagnostics troubleshooting table.

## 7) Phase-0 exit checklist mapping

This RFC satisfies the required Phase 0 approvals by defining:

- stable plugin ABI versioning strategy (Section 3.3),
- extras naming/mapping contract (Section 2.2),
- fallback behavior + actionable error matrix (Section 4.3).
