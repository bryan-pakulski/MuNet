# DType parity inventory (core math paths)

This inventory tracks dtype behavior across core ops with focus on Float16 pathways and backend coverage for CUDA/Vulkan.

## Legend
- **Native**: backend kernel executes requested dtype directly.
- **CPU fallback**: op dispatch falls back to CPU path when backend/dtype unsupported.
- **Conversion fallback**: op runs on CPU Float32 and casts results back to requested dtype.

## Current status after this change

| Op group | Dispatch fallback policy | Float16 on CUDA/Vulkan today | Notes |
|---|---|---|---|
| Elementwise (`add/sub/mul/div`, unary activations, softmax/log_softmax) | CPU fallback | CPU fallback / typed scalar path | Existing typed scalar fallback machinery handles this path. |
| Matmul / batched matmul | CPU fallback | CPU fallback (Float32 compute + cast back) + CPU backend has native Float16 matmul path | CUDA/Vulkan kernels still Float32-native only. |
| Conv2D | CPU fallback | CPU conversion fallback (to CPU Float32, compute, cast back) | Implemented in `conv2d.cpp`. |
| MaxPool2D / Upsample2D | CPU fallback | CPU conversion fallback (to CPU Float32, compute, cast back) | Implemented in `pooling.cpp`. |
| MSELoss / CrossEntropy | CPU fallback | CPU conversion fallback (to CPU Float32, compute, cast back) | Implemented in `loss.cpp`. |
| BatchNorm | CPU fallback | CPU conversion fallback for forward + backward paths | Float16 fallback now executes batchnorm forward/backward via CPU Float32 compute with cast-back. |
| LayerNorm | CPU fallback | CPU typed implementation | Already computes on CPU with typed scalar conversions. |

## Remaining parity work (next)

1. **BFloat16 / Int8 pathways**
   - `DataType` now includes `BFloat16` and `Int8` with scalar conversion helpers.
   - Initial fallback coverage tests were added for bf16/int8 in matmul/conv/loss paths.
   - Remaining: expand operator-level execution support beyond current fallback coverage and formalize quantized math semantics.
2. **Backend-native low precision kernels**
   - CUDA: native fp16/bf16/int8 kernels for matmul/conv/reduction.
   - Vulkan: native fp16 path where hardware/driver capabilities permit.
3. **Capability reporting**
   - Expand backend `query_support` reporting to distinguish native low-precision support from CPU conversion fallback availability.
