# Profiling & Diagnostics

## Environment Flags

- `MUNET_PROFILE=1`: enable profiler collection + auto summary on process exit.
- `MUNET_DEBUG=1`: enable debug validation/logs.
- `MUNET_LOG_LEVEL=0..3`: control logging verbosity.

## Reading Profiler Output

- Focus first on `%Total` to find top contributors.
- Check transfer ops (`to(...)`, `copy`) for data movement bottlenecks.
- Compare CPU vs GPU timings to identify launch/queue overhead.
- CPU backends now report `AvgGPU=0` consistently; GPU backends force a timing
  synchronization while profiling so kernel timings are actually populated.
- Use the new stack markers to localize overhead outside kernels:
  - `dispatch.resolve.backend.<Op>`: backend-supported op-dispatch resolution.
  - `dispatch.resolve.cpu_fallback.<Op>`: time spent deciding to fall back to CPU.
  - `dispatch.resolve.metadata_fallback.<Op>`: ops that intentionally bypass backend dispatch.
  - `inference.load.*`, `inference.compile.*`, `inference.run.*`: host-side inference engine phases.

## Best Practices

- Keep benchmark tensors device-resident during loops.
- Warm up before measuring.
- Use profile mode without debug for lower-overhead measurements, but note that
  GPU timing collection still synchronizes per profiled op so the reported
  timings favor observability over peak-throughput benchmarking.
- When a slowdown appears in `inference.*` or `dispatch.resolve.*` rather than a
  backend op row, the bottleneck is likely in orchestration, validation,
  fallback, or host-side data movement rather than kernel execution itself.


## Vulkan Host-Side Stall Markers

When `MUNET_PROFILE=1` is enabled, Vulkan now emits additional CPU-only profiler rows to isolate queue/driver overhead that is not visible in kernel timestamps:

- `vulkan.wait_for_fence`: time blocked waiting for the next in-flight frame fence.
- `vulkan.synchronize_wait_idle`: blocking time in explicit backend synchronize.
- `vulkan.copy_d2h_wait_idle`: forced wait when reading Vulkan buffers back to CPU.
- `vulkan.staging_wait_idle`: wait before staging-buffer reuse/growth.
- `vulkan.update_descriptors`: CPU time spent updating descriptor sets.
- `vulkan.dispatch_encode`: total host-side command encoding overhead per dispatch.
- `vulkan.flush_batch`: command-buffer end + queue submit overhead.
- `vulkan.query_results`: host cost of reading timestamp query results.

If these markers dominate `%Total` while GPU timings remain low, the slowdown is likely CPU submission/synchronization overhead rather than shader execution.
