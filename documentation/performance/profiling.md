# Profiling & Diagnostics

## Environment Flags

- `MUNET_PROFILE=1`: enable profiler collection + auto summary on process exit.
- `MUNET_DEBUG=1`: enable debug validation/logs.
- `MUNET_LOG_LEVEL=0..3`: control logging verbosity.

## Reading Profiler Output

- Focus first on `%Total` to find top contributors.
- Check transfer ops (`to(...)`, `copy`) for data movement bottlenecks.
- Compare CPU vs GPU timings to identify launch/queue overhead.

## Best Practices

- Keep benchmark tensors device-resident during loops.
- Warm up before measuring.
- Use profile mode without debug for lower-overhead measurements.


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
