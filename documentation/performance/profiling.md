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
