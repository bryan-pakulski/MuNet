# DType parity inventory (core math paths)


## Legend
- **Native**: backend kernel executes requested dtype directly.
- **Vulkan fallback**: op dispatch falls back to Vulkan path when backend/dtype unsupported.
- **Conversion fallback**: op runs on Vulkan Float32 and casts results back to requested dtype.

## Current status after this change

|---|---|---|---|
| Elementwise (`add/sub/mul/div`, unary activations, softmax/log_softmax) | Vulkan fallback | Vulkan fallback / typed scalar path | Existing typed scalar fallback machinery handles this path. |
| Conv2D | Vulkan fallback | Vulkan conversion fallback (to Vulkan Float32, compute, cast back) | Implemented in `conv2d.cpp`. |
| MaxPool2D / Upsample2D | Vulkan fallback | Vulkan conversion fallback (to Vulkan Float32, compute, cast back) | Implemented in `pooling.cpp`. |
| MSELoss / CrossEntropy | Vulkan fallback | Vulkan conversion fallback (to Vulkan Float32, compute, cast back) | Implemented in `loss.cpp`. |
| BatchNorm | Vulkan fallback | Vulkan conversion fallback for forward + backward paths | Float16 fallback now executes batchnorm forward/backward via Vulkan Float32 compute with cast-back. |
| LayerNorm | Vulkan fallback | Vulkan typed implementation | Already computes on Vulkan with typed scalar conversions. |

## Remaining parity work (next)

1. **BFloat16 / Int8 pathways**
   - `DataType` now includes `BFloat16` and `Int8` with scalar conversion helpers.
   - Initial fallback coverage tests were added for bf16/int8 in matmul/conv/loss paths.
   - Remaining: expand operator-level execution support beyond current fallback coverage and formalize quantized math semantics.
2. **Backend-native low precision kernels**
   - Vulkan: native fp16 path where hardware/driver capabilities permit.
3. **Capability reporting**
   - Expand backend `query_support` reporting to distinguish native low-precision support from Vulkan conversion fallback availability.
