# Observability & Profiling Roadmap

This page turns the next round of profiling/debugging work into a concrete
execution plan. Each task below is written so it can be picked up as an
implementation milestone and reviewed against clear success criteria.

## 1. Per-module spans in forward passes

### Objectives

- Make host-side time attributable to logical modules/layers rather than only
  to backend ops or whole-engine phases.
- Support nested/hierarchical spans so complex models can be inspected at the
  block, sub-block, and leaf-module level.
- Keep naming stable enough that repeated runs can be compared directly.

### Action Points

- Introduce a scoped host-side module span around `Module::forward(...)`
  boundaries.
- Record hierarchical names such as:
  - `module.encoder.block_3.forward`
  - `module.encoder.block_3.attn.forward`
  - `module.decoder.head.forward`
- Reuse registered module names where available, and fall back to deterministic
  positional names for anonymous modules.
- Record both inclusive host time and call count in the existing profiler.
- Ensure spans can be correlated with inference engine phases when a module is
  executed during `compile(...)`, warmup, or `run(...)`.

### Exit Criteria

- A single inference request produces profiler rows for every named module in
  the executed forward path.
- Nested module names are stable across runs for the same model structure.
- The profiler summary makes it possible to identify the slowest module without
  inspecting backend-op rows.

### Enable

- Set `MUNET_PROFILE=1`.
- Keep `MUNET_LOG_LEVEL>=2` if additional textual span diagnostics are added.

### How to Parse It

- Start with the highest `%Total` entries under the `module.*` namespace.
- If `module.*` dominates while backend ops do not, the slowdown is likely in
  orchestration, data movement, validation, or module composition rather than
  raw kernel execution.
- Compare a slow parent span like `module.encoder.block_3.forward` against its
  children (`.attn.forward`, `.mlp.forward`) to isolate the hotspot.

## 2. Transfer-direction markers

### Objectives

- Make memory traffic direction explicit so transfer bottlenecks are immediately
  distinguishable from compute bottlenecks.
- Separate device movement from dtype conversion overhead.
- Make it clear whether traffic is host/device, device/device, or CPU-local.

### Action Points

- Split current generic movement markers into:
  - `transfer.h2d`
  - `transfer.d2h`
  - `transfer.d2d`
  - `transfer.cpu_copy`
  - `transfer.dtype_convert`
- Add shape and byte-size context to every transfer record.
- Ensure direction markers are emitted consistently from:
  - `Tensor::to(Device)`
  - `Tensor::to(DataType)`
  - backend copy paths
  - staging-buffer paths in GPU backends
- Keep backend-specific rows where useful (`vulkan.copy_d2h_*`, etc.) but
  ensure they roll up under directionally obvious top-level transfer markers.

### Exit Criteria

- Every explicit tensor/device move appears under a directional transfer row.
- CPU-only runs do not show GPU transfer markers.
- Mixed-device runs clearly reveal whether time is dominated by `h2d`, `d2h`,
  or `d2d` traffic.

### Enable

- Set `MUNET_PROFILE=1`.
- Optionally set `MUNET_DEBUG=1` when validating that expected transfers are
  occurring.

### How to Parse It

- Look at `transfer.*` rows before backend kernels if end-to-end latency is
  high.
- Heavy `transfer.h2d` often points to repeated uploads or missing caching.
- Heavy `transfer.d2h` often points to premature readbacks, logging, or CPU-side
  post-processing.
- Heavy `transfer.dtype_convert` suggests unnecessary precision churn rather
  than pure transport cost.

## 3. Fallback reason accounting

### Objectives

- Make fallback behavior explainable, not just visible.
- Distinguish *that* a fallback happened from *why* it happened.
- Support remediation by surfacing dtype, shape, and unsupported-feature causes.

### Action Points

- Extend dispatch accounting from path-only markers to reason-aware markers:
  - `dispatch.fallback.reason.dtype`
  - `dispatch.fallback.reason.shape`
  - `dispatch.fallback.reason.feature`
  - `dispatch.fallback.reason.policy`
- Add structured detail to debug logs for the triggering backend, feature,
  dtype, and relevant shape.
- Track counts per reason in addition to timing so frequent zero-cost fallbacks
  are still discoverable.
- Surface unsupported-feature names directly in the profiler/debug summary when
  possible.

### Exit Criteria

- A fallback-heavy run can be summarized by top fallback reasons without
  inspecting code.
- Dispatch diagnostics make it obvious whether the fix should be backend
  support, shape normalization, dtype normalization, or policy change.
- Repeated runs can be compared by fallback count and fallback time.

### Enable

- Set `MUNET_PROFILE=1` for timing/count visibility.
- Set `MUNET_DEBUG=1` to emit detailed textual explanations for each fallback.

### How to Parse It

- Start with `dispatch.resolve.*` to see how often fallback paths are chosen.
- Then inspect `dispatch.fallback.reason.*` counters/timers to determine root
  cause.
- If dtype reasons dominate, normalize precision earlier.
- If shape reasons dominate, inspect broadcasting/layout assumptions.
- If feature reasons dominate, prioritize backend feature implementation work.

## 4. Allocator and synchronization visibility

### Objectives

- Surface stalls caused by allocation strategy, buffer growth, synchronization,
  and queue starvation.
- Separate memory-management overhead from compute and dispatch overhead.
- Make backend wait behavior visible enough to guide pool sizing and batching.

### Action Points

- Add allocator markers for:
  - allocation reuse hits/misses
  - pool growth
  - large-allocation slow paths
  - deferred free flushes
- Add synchronization markers for:
  - explicit `synchronize()` calls
  - implicit waits inserted for timing or readback
  - queue idle / fence wait / event wait time
- Add queue starvation markers where a backend is blocked waiting for reusable
  command buffers, descriptors, or in-flight frames.
- Keep CPU and GPU backend markers consistent enough to compare host-side stall
  patterns across backends.

### Exit Criteria

- Slow runs make allocator churn and synchronization stalls visible in the
  profiler without special ad-hoc logging.
- It is possible to tell whether a regression is caused by compute, transfer,
  allocation churn, or synchronization/wait behavior.
- Vulkan/CUDA/CPU backends all emit backend-appropriate stall markers under a
  documented namespace.

### Enable

- Set `MUNET_PROFILE=1`.
- Use `MUNET_DEBUG=1` when validating allocator correctness or unexpected sync
  insertion.

### How to Parse It

- Large `allocator.*` time means pool policy or object lifetime is a stronger
  optimization target than math kernels.
- Large `sync.*` / `queue_wait.*` time means the backend is idle or blocked,
  often due to readbacks, forced timing syncs, or batching policy.
- Compare allocator and sync markers against backend-op rows to determine
  whether the device is busy or the host/runtime is the bottleneck.

## 5. Correlated trace IDs

### Objectives

- Reconstruct a single request/span across inference phases, dispatch, transfer,
  and backend execution.
- Support “why was this one request slow?” debugging, not just aggregate
  summaries.
- Make it possible to join profiler output with textual debug logs.

### Action Points

- Introduce a lightweight execution/span id generated at the start of
  `Engine::compile(...)` and `Engine::run(...)`.
- Propagate the id through:
  - inference phase profiler rows
  - dispatch markers
  - transfer markers
  - backend debug/profile wrapper markers
- Add optional log prefixes such as `[trace_id=1234 span=run.forward]`.
- Ensure nested spans carry parent/child relationships or consistent ancestry
  in their naming.

### Exit Criteria

- A single slow request can be followed end-to-end through engine, dispatch,
  transfer, and backend layers.
- Logs and profiler output can be correlated by the same trace/span id.
- Trace ids are cheap enough to leave enabled in debug/profiling builds.

### Enable

- Set `MUNET_PROFILE=1` for profiler correlation.
- Set `MUNET_DEBUG=1` to include trace ids in logs once implemented.

### How to Parse It

- Filter profiler/log output by a single trace id.
- Follow the request from `inference.run.*` to `dispatch.*` to `transfer.*` to
  backend rows.
- If one trace id shows abnormal time in one layer only, you have an immediate
  narrowing of the hotspot.

## Recommended implementation order

1. **Transfer-direction markers**
   - Lowest conceptual risk and immediately useful for existing slowdowns.
2. **Fallback reason accounting**
   - Builds directly on the dispatch markers already present.
3. **Allocator and synchronization visibility**
   - High value for backend/runtime regressions and complements transfer data.
4. **Per-module spans**
   - Best next step for model-level attribution once lower-level host costs are
     better understood.
5. **Correlated trace IDs**
   - Most powerful once the other namespaces exist and can be linked together.
