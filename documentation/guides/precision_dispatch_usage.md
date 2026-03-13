# Precision Dispatch & Guard Usage

This guide explains how to use MuNet's current precision-dispatch scaffolding and when to choose each dtype.

## Kernel dispatch guard usage

MuNet provides a thread-local dispatch configuration in `types.hpp`:

- `DTypeDispatchConfig`
- `DTypeDispatch`
- `DTypeDispatchGuard`
- `KernelFallbackMode`

### Example (C++)

```cpp
using namespace munet;

DTypeDispatchConfig cfg;
cfg.has_compute_dtype = true;
cfg.compute_dtype = DataType::Float16; // request compute dtype
cfg.fallback_mode = KernelFallbackMode::WarnAndUpcast;

{
  DTypeDispatchGuard guard(cfg);
  // Ops in this scope use this dispatch request.
  // If unsupported on current backend:
  //   - WarnAndUpcast => warn + fallback to float32 compute
  //   - Error         => throw
  Tensor z = a + b;
}
```

Notes:
- Scope is RAII-based and thread-local.
- Current Phase-1 backend support is primarily CPU fallback paths.

## Fallback modes

- `KernelFallbackMode::WarnAndUpcast`
  - best for iterative development and broad model coverage.
  - unsupported compute dtypes are upcast to FP32 with warning.

- `KernelFallbackMode::Error`
  - best for strict validation, benchmarking, and CI policy checks.
  - unsupported compute dtype requests fail immediately.

## DType guidance (current stage)

### Recommended

- `Float32`
  - default stable training/inference dtype.
  - best correctness baseline.

- `Float16` / `BFloat16`
  - use for mixed-precision experiments where memory pressure is high.
  - expect correctness-focused fallback behavior in places without native kernels.

### Experimental / scaffolding-only paths

- `Float8E4M3FN`, `Float8E5M2`
  - API-level and fallback scaffolding exists.
  - production-grade numeric kernels/packing are still pending.

- `Int8`, `Int4`
  - initial dequant/accumulate/requant fallback behavior is present in CPU paths.
  - packed formats, calibration flow, and optimized quantized kernels are pending.

## Practical recommendation right now

- For production-like correctness: run FP32 baseline first.
- For lower-precision validation: run with `WarnAndUpcast`, compare outputs/metrics.
- For readiness gating: switch to `Error` mode in tests to catch unsupported paths explicitly.
