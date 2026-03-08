# Add a New Operation

## Checklist

1. Add backend interface methods in `src/backend.hpp`.
2. Implement CPU/CUDA/Vulkan kernels.
3. Wire op in `src/ops.hpp` (+ backward node if differentiable).
4. Expose in `Tensor` API (`src/tensor.hpp` / `src/tensor.cpp`).
5. Add Python bindings in `src/bindings.cpp`.
6. Add unit tests in `tests/`.

## Tip

Use `BackendManager` registration and existing patterns in `ops.hpp` to keep
new ops consistent with existing graph/profiler behavior.
