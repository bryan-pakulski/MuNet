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


## Python usage

### Dispatch guard config

```python
import munet

cfg = munet.DTypeDispatchConfig()
cfg.has_compute_dtype = True
cfg.compute_dtype = munet.DataType.Float16
cfg.fallback_mode = munet.KernelFallbackMode.WarnAndUpcast

with munet.precision_dispatch(cfg):
    y = x + w
```

### Casting tensor storage dtype

```python
x_fp32 = munet.Tensor([4, 4], dtype=munet.DataType.Float32)
x_i8 = x_fp32.to_dtype(munet.DataType.Int8)
x_back = x_i8.to_dtype(munet.DataType.Float32)
```

### AMP helpers

```python
import munet

scaler = munet.amp.GradScaler()
with munet.amp.AutoCastGuard(munet.DataType.Float16):
    loss = model(x).mse_loss(target)

scaled_loss = scaler.scale(loss)
scaled_loss.backward()
stepped = scaler.step(optimizer, model.parameters())

# Optional: keep model in low precision with FP32 master weights
master_opt = munet.amp.FP32MasterSGD(model.parameters(), lr=1e-3)
# or Adam-style master optimizer
master_adam = munet.amp.FP32MasterAdam(model.parameters(), lr=1e-3)
```

## DType selection guidance

- **FP32**: baseline for correctness/debugging.
- **FP16/BF16**: preferred first mixed-precision trial dtypes.
- **FP8/INT4/INT8**: currently best treated as experimental until backend-native kernels and packing/calibration flows are completed.


## Should guard be exposed directly?

Short answer: **both low-level and high-level access are useful**.

- Expose low-level guard/config APIs (`DTypeDispatchConfig`, `DTypeDispatchGuard`) for advanced users, benchmarking, and backend bring-up.
- For normal training/inference usage, prefer higher-level wrappers/context managers (`precision_dispatch`, AMP helpers, and future policy-driven module/engine configs).

Recommended layering:
1. Low-level guard (expert/debug).
2. Python context wrapper (`precision_dispatch`) for day-to-day experimentation.
3. Framework-level policy objects (engine/trainer configs) as default ergonomic path.
