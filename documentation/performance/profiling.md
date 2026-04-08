# Profiling & Diagnostics

## Environment Flags

- `MUNET_PROFILE=1`: enable profiler collection + auto summary on process exit.
- `MUNET_DEBUG=1`: enable debug validation/logs.
- `MUNET_LOG_LEVEL=0..3`: control logging verbosity.
- `MUNET_DISPATCH_DECISION_DUMP=1`: emit structured dispatch decision lines.
- `MUNET_FAIL_FAST_ACCELERATOR_CPU_FALLBACK=1`: throw on CUDA/Vulkan tensor
  CPU fallback (useful for CI and flaky fallback detection).
- When these flags are **disabled**, MuNet takes a cheap fast path that avoids
  timer/string work in hot profiling call sites; runtime overrides are intended
  primarily for tests/debug harnesses rather than production toggling.

## Reading Profiler Output

- Focus first on `%Total` to find top contributors.
- Treat profiler rows as belonging to one of four buckets:
  - backend work (`add`, `matmul`, `conv2d`, etc.)
  - orchestration (`module.*`, `dispatch.resolve.*`, `inference.*`)
  - data movement (`transfer.*`)
  - runtime stalls (`allocator.*`, `sync.*`, `queue_wait.*`,
    `queue_starvation.*`)
- When tracing is active, each row’s detail string also carries
  `trace_id=<id>`, `span=<ancestry>`, and optionally `parent=<span>` so the same
  request can be followed across layers.
- Check `transfer.*` rows for data movement bottlenecks:
  - `transfer.h2d`
  - `transfer.d2h`
  - `transfer.d2d`
  - `transfer.cpu_copy`
  - `transfer.dtype_convert`
- Compare CPU vs GPU timings to identify launch/queue overhead.
- CPU backends now report `AvgGPU=0` consistently; GPU backends force a timing
  synchronization while profiling so kernel timings are actually populated.
- Use the new stack markers to localize overhead outside kernels:
  - `module.<path>.forward`: host-side module/layer forward spans.
  - `dispatch.resolve.backend.<Op>`: backend-supported op-dispatch resolution.
  - `dispatch.resolve.cpu_fallback.<Op>`: time spent deciding to fall back to CPU.
  - `dispatch.resolve.metadata_fallback.<Op>`: ops that intentionally bypass backend dispatch.
  - `dispatch.fallback.reason.*`: fallback counters/details grouped by
    `dtype`, `shape`, `feature`, or `policy`.
  - `allocator.<event>.<backend>`: allocation reuse/miss/growth/deallocation
    overhead by backend.
  - `sync.<event>.<backend>`: explicit or profiler-inserted synchronization.
  - `queue_wait.<event>.<backend>`: time blocked on queue/fence/frame
    availability.
  - `queue_starvation.<event>.<backend>`: backend had work to do but had to
    wait for reusable submission resources first.
  - `inference.load.*`, `inference.compile.*`, `inference.run.*`: host-side inference engine phases.

## Marker Namespaces

### Dispatch and fallback

- `dispatch.resolve.backend.<Op>` means the op stayed on the selected backend.
- `dispatch.resolve.cpu_fallback.<Op>` means dispatch decided to run on CPU
  instead.
- `dispatch.fallback.reason.*` tells you why that happened. The row count is the
  frequency; the attached detail string records the latest `op=`, `backend=`,
  `feature=`, `dtype=`, `shape=`, `reason=`, and `policy=` context.
- In addition to profiler rows, dispatch keeps in-process accelerator fallback
  counters (`fallback_telemetry_snapshot()` / `reset_fallback_telemetry()` in
  `op_dispatch`) so tests can assert fallback behavior directly.

### Trace/span correlation

- `Engine::compile(...)` and every `Engine::run(...)` now begin a fresh trace.
- Trace ancestry is encoded in span names such as:
  - `compile`
  - `compile.prepare_input`
  - `compile.forward`
  - `run`
  - `run.forward`
  - `run.validate_output`
- Any profiler row emitted while a trace is active appends that trace metadata to
  its detail string, including inference phases, module spans, transfers,
  dispatch rows, fallback rows, and backend wrapper markers.
- Debug/info logs emitted inside the same context automatically include prefixes
  like `[trace_id=17 span=run.forward]`.

### Allocator visibility

- `allocator.reuse_hit.<backend>`: an allocation request was satisfied from a
  reusable block/pool slot.
- `allocator.reuse_miss.<backend>`: the backend had to obtain a fresh
  allocation rather than reusing an existing one.
- `allocator.pool_growth.<backend>`: the backend (or the wrapper-visible pool
  high-water mark) had to grow capacity for the requested size class.
- `allocator.large_alloc_slow_path.<backend>`: a large allocation request that
  is more likely to bypass the fast path and should be scrutinized separately.
- `allocator.deallocate.<backend>`: host-side time spent returning a block to
  the backend or its deferred-free queue.
- `allocator.deferred_free_flush.<backend>`: deferred frees were retired and
  returned to the reusable pool after the backend confirmed prior work was done.

### Synchronization and queue visibility

- `sync.explicit.<backend>`: time spent inside an explicit `backend.synchronize()`
  call.
- `sync.implicit_timing.<backend>`: synchronization inserted by profiling/debug
  machinery to obtain kernel timings safely.
- `sync.readback_wait.<backend>`: waits introduced because host-visible reads
  required device work to finish first.
- `queue_wait.*.<backend>`: fence/event/frame waits where the runtime is blocked
  for in-flight work or reusable transfer resources.
- `queue_starvation.*.<backend>`: waits caused by running out of reusable
  command/descriptor/frame resources.

## Best Practices

- Keep benchmark tensors device-resident during loops.
- Batch scalar readbacks when collecting many metrics/loss values in one step.
  Instead of repeated `loss.item()` calls, use `batch_item_values(...)` to pack
  scalar tensors and perform one device→host transfer:

  ```cpp
  std::vector<ScalarValue> values =
      batch_item_values({loss, cls_loss, reg_loss, aux_metric});
  ```

  This is especially helpful on Vulkan/CUDA when profiler output is dominated by
  `transfer.d2h`, `sync.readback_wait.*`, or backend-specific D2H wait rows.
- Warm up before measuring.
- Use profile mode without debug for lower-overhead measurements, but note that
  GPU timing collection still synchronizes per profiled op so the reported
  timings favor observability over peak-throughput benchmarking.
- When a slowdown appears in `inference.*` or `dispatch.resolve.*` rather than a
  backend op row, the bottleneck is likely in orchestration, validation,
  fallback, or host-side data movement rather than kernel execution itself.
- Use `dispatch.fallback.reason.*` rows to answer *why* a fallback happened;
  the row count shows frequency and the attached detail string shows the last
  observed backend/feature/dtype/shape context for that reason.
- Use profile mode without debug for representative timing, then re-run with
  debug only when you need correctness checks or more textual context; debug can
  amplify sync overhead by design.
- When diagnosing a single outlier request, start from an
  `inference.run.*` or `inference.compile.*` row, copy its `trace_id`, then
  search profiler/log output for the same id.
- Compare `allocator.*` and `queue_wait.*` against backend op rows before
  optimizing kernels; if the stall markers dominate, kernel math is not the
  current bottleneck.
- Reuse the same tensor shapes when microbenchmarking allocator behavior;
  otherwise `allocator.pool_growth.*` will mostly reflect benchmark setup churn.
- When `module.*` dominates, the hotspot is likely at the model/layer
  composition level; compare parent module spans against child module spans to
  find the slowest block.

## What Results Actually Mean

- High backend-op time with low `allocator.*` / `sync.*` means the device is
  busy doing math; optimize kernels, fusion, batch sizing, or model structure.
- High `transfer.*` with low backend-op time means data movement or dtype churn
  is dominating; keep tensors device-resident longer or normalize dtype earlier.
- High `dispatch.resolve.*` or `dispatch.fallback.reason.*` means unsupported
  features/dtypes/shapes are bouncing work away from the intended backend.
- High `allocator.*` means memory lifecycle is expensive enough to matter:
  increase reuse, reduce shape churn, or revisit pool sizing.
- High `sync.*` means throughput is being limited by forced waits rather than
  compute saturation. Look for readbacks, explicit synchronization in the call
  path, or profiling-induced timing syncs.
- High `queue_wait.*` / `queue_starvation.*` means the backend could not submit
  or recycle work fast enough; investigate in-flight frame count, descriptor
  reuse policy, staging-buffer turnover, and batching cadence.
- If a single `trace_id` is slow only in one layer, you have immediate scope
  reduction:
  - slow only in `transfer.*` => transport/readback problem
  - slow only in `dispatch.*` => capability/fallback problem
  - slow only in backend rows => kernel/submission problem
  - slow only in `inference.*` / `module.*` => orchestration/model problem
- If CPU `AvgCPU` is high but `AvgGPU` stays near zero on a GPU backend, the
  slowdown is host/runtime overhead rather than kernel execution.

## Backend-Specific Notes

### Vulkan

When `MUNET_PROFILE=1` is enabled, Vulkan emits both generic stall namespaces
and Vulkan-specific rows to isolate queue/driver overhead that is not visible in
kernel timestamps:

- Generic rows include `allocator.*.vulkan`, `sync.*.vulkan`,
  `queue_wait.*.vulkan`, and `queue_starvation.*.vulkan`.
- Vulkan-specific rows include `vulkan.flush_batch`,
  `vulkan.wait_for_fence`, `vulkan.staging_wait_fences`,
  `vulkan.copy_d2h_wait_fence`, `vulkan.update_descriptors`,
  `vulkan.dispatch_encode`, and `vulkan.query_results`.

If these markers dominate `%Total` while GPU timings remain low, the slowdown is
likely CPU submission, allocator turnover, or synchronization overhead rather
than shader execution.

### CUDA and CPU

- CUDA now reports generic `allocator.*.cuda` and `sync.*.cuda` markers so you
  can separate pool reuse from fresh `cudaMalloc` growth and explicit device
  synchronization.
- CPU backends still report `AvgGPU=0`, but the shared wrapper emits the same
  `allocator.*.<backend>` / `sync.*.<backend>` namespaces during profiling so
  cross-backend host-side stall patterns stay comparable.

## Perf test skip diagnostics

GPU comparison tests in `tests/test_performance.cpp` require **both** CUDA and
Vulkan support. If they skip, the skip message now reports:

- compile-time backend availability (`compiled(cuda=..., vulkan=...)`)
- runtime device detection (`runtime(cuda=..., vulkan=...)`)

This helps quickly distinguish build-configuration problems from runtime device
visibility/driver problems.

## Trace-Centric Workflow

1. Find the slow request’s `trace_id` from `inference.compile.*`,
   `inference.run.*`, or an engine event.
2. Follow the same `trace_id` through:
   - `module.*`
   - `transfer.*`
   - `dispatch.resolve.*`
   - `dispatch.fallback.reason.*`
   - backend/allocator/sync/queue rows
3. If needed, re-run with `MUNET_DEBUG=1` and grep logs for the same
   `[trace_id=…]` prefix to align textual diagnostics with profiler samples.

## Next instrumentation roadmap

The next planned observability work is tracked in the
[Observability & Profiling Roadmap](observability_roadmap.md), including:

- per-module forward spans
- directional transfer markers
- fallback-reason accounting
- allocator/synchronization visibility ✅
- correlated trace/span ids ✅
