# μNet: A lightweight C++ GPU agnostic AI framework for training & inference

μNet is a lightweight C++ AI framework with Python bindings.

It is designed to be GPU agnostic with the intended final goal of running on edge devices with limited resources.

It features a PyTorch-like API, making it familiar to use while handling low-level memory management and compute dispatch across CPU, CUDA, and Vulkan backends.

# Features

- **GPU Support**: Native support for **CUDA** (Nvidia) and **Vulkan** (Cross-vendor: AMD, Intel, Mobile GPUs).
- **Python Bindings**: Seamless integration with NumPy/Python via `pybind11`. Zero-copy memory sharing on CPU.
- **Easy to use**: Simple, PyTorch-like API for defining models and training.
- **Dynamic Autograd**: define-by-run automatic differentiation engine (DAG).
- **Cross Platform**: Built for Linux/Unix systems.
- **Model Parallelism**: tensors can exist on different devices within the same graph (e.g., Layer 1 on CPU, Layer 2 on Vulkan).
- **Shared Module Core**: `munet::core::Module` now owns parameter/buffer/module registration so training (`munet::nn`) and future inference libraries can share the same model graph foundation.
- **Library Split for Extensibility**:
  - `munet_core`: backend + tensor runtime shared by all products.
  - `munet_training`: training APIs (`nn`, optimizers) on top of core.
  - `munet_inference`: deploy/edge-focused inference APIs on top of core.
- **Backend Modularity**:
  - Debug/profiling interception now lives in a dedicated backend module (`munet_backend_debug`) instead of being embedded in tensor runtime code.
  - Backends are now factory-registered via `BackendManager::register_backend(...)`, making new backend integrations and swapping implementations simpler.

# Documentation
Documentation roadmap: see `DOCUMENTATION_PLAN.md`.
Documentation index: `documentation/index.md`.

## Interactive / searchable docs site

MuNet now includes a MkDocs config (`mkdocs.yml`) and a docs content tree under `documentation/` covering:

- Python API guide
- C++ API guide
- Inference + serialization guides
- Common design patterns and end-to-end tutorial

Run locally:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Build static site:

```bash
mkdocs build
```

## Generated Python API docs (legacy helper)

```bash
pip install pydoc
make doc
```


# Project Status (Current)

The project has a working core runtime + training stack with CPU/CUDA/Vulkan backends, autograd, and Python bindings.

**Current highest priority: build a lean inference engine/library.**

## Priority Order
1. **Inference Engine (P0)**
   - Add a deploy-first `munet_inference` runtime API for model loading, pre-allocation, warmup, and fast forward execution.
   - Current scaffold includes `compile(example_input, expected_input_shape, expected_output_shape)` with `-1` wildcard support for dynamic dims (e.g. dynamic batch/resolution) plus strict shape guards on `run(...)`.
   - Keep inference surface minimal and stable (no training/autograd dependencies in public API).
   - Add latency + memory benchmarks and regression checks for CPU/CUDA/Vulkan.
2. **Transformer Inference Readiness (P1)**
   - Optimize decoder inference path (causal attention, KV cache, efficient shape/layout handling).
   - Keep tiny LLM demos as smoke tests for inference APIs.
3. **RT-DETR Inference Path (P1)**
   - Add deploy-focused post-processing path and efficient multi-scale execution for detector heads.
4. **Training/Research Extras (P2)**
   - Continue optimizer/loss/kernel improvements once inference baseline is stable.

## What is already in place
- Modular targets: `munet_core`, `munet_training`, `munet_inference`.
- Backend factory and cache (`BackendManager`) plus optional debug/profiler wrapper.
- Core tensor ops needed for attention prototypes (`softmax`, `log_softmax`, `masked_fill`, `permute`).
- Initial inference runtime scaffold: `inference::Engine` with `load`, `prepare` (warmup), `run`, `run_batch`, and basic latency stats.

# Future Plans
- Transformer stack (LayerNorm + MultiHeadAttention + MLP) and a GPT3 LLM demo.
- Attention-ready tensor ops: softmax, log_softmax, transpose/permute, and masked_fill.
- Fused Kernels (i.e. conv2d calls 3 kernels -> conv, add (bias) and relu, can merge together)
- Adaptive Pooling

Serialization:
    - Save and Load of arbitrarily complex models i.e. Costum Modules and Layers extending munet.nn.Module
    - Preserve layers, paramers, buffers, skip connections etc.. as well as forward pass
    - Interop with torch models (save & load is supported for basic non graph models (Linear)) 
    - Interop with onnx models (save & load)

Multi-GPU RoadMap:

 1. Device Switching: Update CUDABackend and VulkanBackend to ensure they explicitly select the correct hardware index before executing any commands.
 2. Functional Collectives: Implement all_reduce in the Backend interface.
    - CUDA: Use ncclAllReduce.
    - Vulkan: Use timeline semaphores and cross-device buffer copies.
 3. Stream Management: Currently, you use the default stream (0) in CUDA. To allow the CPU to "fire and forget" kernels to multiple GPUs simultaneously, give each Device its own compute
   stream/queue.
 4. Data Parallel Wrapper: Create a nn::DataParallel modcule that:
    - Copies the model to all target devices.
    - Splits the input Tensor along the batch dimension.
    - Triggers the all_reduce on gradients automatically.

Production Ready Improvements:                                                                                                                                                          

 1. Performance (Kernels): Most kernels (especially in Vulkan/CUDA) are "naive." They don't use tiled memory, shared memory optimization, or vendor-tuned libraries like cuDNN or oneDNN.  
 2. Memory Management: Only using a simple caching allocator, but it lacks a memory-fragmentation strategy or a "Memory Arena."                                                               
 3. Missing Dtypes: Production requires BFloat16, Int8, Int4 (quantization), and Float16.                                                               
 4. Error Handling: There is limited validation for tensor strides, broadcast safety, or device-side out-of-memory errors.                                                                 

Inference Engine:

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Development

μNet is designed to be modular. Adding new functionality involves touching the backend interface, the specific implementations, and the autograd engine.

### Backend Registration (New)
Use `BackendManager::register_backend(DeviceType, factory)` to plug in a backend implementation without editing tensor dispatch logic.

```cpp
BackendManager::register_backend(DeviceType::CPU, [](Device d) {
    return std::make_shared<MyCPUBackend>(d.index);
});
```

### 1. Define the Interface
Modify `src/backend.hpp` to add the virtual function definitions for your new operation (both forward and backward if applicable).

```virtual void my_op(const Storage &in, Storage &out, int param) = 0;```

### 2. Implement the Kernels

Implement the specific logic for each backend:

 - CPU (`src/backend/cpu_backend.hpp`): Implement using parallel for loops or SIMD.
 - CUDA (`src/backend/cuda_backend.cu` & `src/backend/cuda_backend.hpp`): Write the CUDA kernel `(__global__ void ...)` and the host dispatch function.
 - Vulkan (`src/backend/vulkan_backend.cpp` & `src/backend/vulkan_backend.hpp`):
    1. Write the GLSL compute shader source string.
    2. In the constructor, compile it using createComputePipeline.
    3. Implement the dispatch logic using dispatch_kernel.

### 3. Register the Operation

Add the high-level logic in src/ops.hpp. This is where the Autograd magic happens.

 1. Create a `struct MyOpBackward : public Node` that defines how to calculate gradients.
 2. Create a function `inline Tensor my_op(...)` that:
    - Allocates the output tensor.
    - Calls `backend->my_op(...)`.
    - If `requires_grad` is true, creates the `MyOpBackward` node and links edges.

### 4. Expose to Tensor API

Add a method to the Tensor class in `src/tensor.hpp` and `src/tensor.cpp` that delegates to `ops::my_op`.

### 5. Python Binding

Finally, expose the method to Python in `src/bindings.cpp`.


```.def("my_op", &Tensor::my_op, py::arg("param"))```

## Debugging

You can enable different levels of debug / profiling via the following environment flags:

```
 python script.py                                Full speed. No overhead.
 MUNET_PROFILE=1 python script.py                Enables kernel/memory profiling and auto-prints a summary at process exit.
 MUNET_DEBUG=1 python script.py                  Enables verbose debug logs and NaN checks.
 MUNET_DEBUG=1 MUNET_PROFILE=1 python script.py  Full visibility: debug logs + profiler summaries.
 MUNET_LOG_LEVEL=0..3 python script.py           Control log verbosity (0=error, 1=warn, 2=info, 3=debug).
```

Profiler summaries are now printed automatically on shutdown when `MUNET_PROFILE=1`, even if `print_profiler_stats()` is not called explicitly from Python.

Profiling mode now avoids forced per-op synchronization unless debug mode is also enabled, so `MUNET_PROFILE=1` gives lower-overhead traces closer to real runtime behavior.

# Building

## Requirements

 - CMake 3.10+
 - C++17 Compiler
 - Optional: CUDA Toolkit (nvcc)
 - Optional: Vulkan SDK (glslc must be in PATH)
 - Python 3.10+ (and development headers)

## Build Steps

```
make build-release
```

# Testing

Run the C++ unit tests (GoogleTest) 

```
make unit-test
```

This will run across all available backends.

Run the Python integration tests:

```
make py-test
```

Run performance tests:
```
make perf-test
```

## Demos

LLM demos:
- `demos/llm/tiny_llm.py` — tiny character-level LM with token+position embedding, layer norm and MLP head.
- `demos/llm/decoder_block_demo.py` — decoder-block style LM with causal `nn.MultiHeadAttention` and residual MLP.
- `demos/llm/gpt3_multi_gpu_demo.py` — GPT3 style multi gpu demo

Computer vision demos:
- `demos/mnist/mnist.py` — simple segmentation-style toy training loop.
- `demos/unet/unet.py` — larger UNet-like segmentation workflow with visualization outputs.

Feature demos:
- `demos/features/transformer_ops_showcase.py` — quick forward-only showcase of `MultiHeadAttention`, `LayerNorm`, `GELU`, and `Dropout`.

Inference demos:
- `demos/inference/batch_forward_demo.py` — lean batch forward loop demo for deploy-style execution.
- `demos/inference/serialization_roundtrip_demo.py` — demonstrates full model save/load reconstruction and weights-only restore.
- `demos/inference/e2e_train_save_load_infer.py` — end-to-end flow: train -> save -> load -> compile -> inference.

> Note: when converting tensors to NumPy in demos, use `.detach()` (or `munet.no_grad()` + detach) to avoid buffer-access errors on tensors that require grad.


## Serialization

MuNet supports two serialization paths:

- **Full model file**: `munet.save(model, "model.npz")` then `restored = munet.load("model.npz")`
  - Reconstructs supported built-in model graphs without requiring the original Python class definition.
- **Weights-only restore**: `munet.load(existing_model, "model.npz")` (or `munet.load_weights(...)`)
  - Loads parameters/buffers into an already-defined model instance.

For a runnable example, see `demos/inference/serialization_roundtrip_demo.py`.
