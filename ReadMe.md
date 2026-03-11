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

## ONNX non-finite tracing (debugging NaN/Inf)

When running converted ONNX graphs, enable fail-fast non-finite tracing to stop at the **first** bad node output instead of seeing cascaded downstream failures.

```bash
MUNET_DEBUG=1 MUNET_ONNX_TRACE_NONFINITE=1 python demos/ort/convert.py
```

- `MUNET_DEBUG=1` enables backend-side debug checks/logging.
- `MUNET_ONNX_TRACE_NONFINITE=1` enables ONNX graph-runtime fail-fast checks.
- `MUNET_ONNX_TRACE_NONFINITE_INPUTS=1` (default on) includes per-input stats in the error (shape/dtype/finite count/min/max/mean).
- `MUNET_ONNX_POW_CLAMP_FINITE=1` (default on) clamps ONNX `Pow` overflow/NaN results into finite `float32` range to avoid cascade failures in normalization subgraphs.
- If you typed `NOFINITE`, use `MUNET_ONNX_TRACE_NONFINITE` (with `NONFINITE`).

The runtime will raise a descriptive error with op type, node name, output name, first bad index, bad-count summary, and (optionally) input/output stats so you can quickly locate the source op.

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
- Transformer stack (LayerNorm + MultiHeadAttention + MLP) and a tiny decoder-only LLM demo.
- Attention-ready tensor ops: softmax, log_softmax, transpose/permute, and masked_fill.
- Fused Kernels (i.e. conv2d calls 3 kernels -> conv, add (bias) and relu, can merge together)
- Adaptive Pooling

Serialization:
    - Save and Load of arbitrarily complex models i.e. Costum Modules and Layers extending munet.nn.Module
    - Preserve layers, paramers, buffers, skip connections etc.. as well as forward pass

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
 2. Memory Management: You use a simple caching allocator, but it lacks a memory-fragmentation strategy or a "Memory Arena."                                                               
 3. Missing Dtypes: You are essentially locked into Float32. Production requires BFloat16, Int8 (quantization), and Float16.                                                               
 4. Error Handling: There is limited validation for tensor strides, broadcast safety, or device-side out-of-memory errors.                                                                 

Inference Engine:

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Core Autograd Features                                                                                                                                                                 

 - In-place Operations: Your engine doesn't track versioning to prevent gradients from being calculated on modified data.                                                                 
 - Higher-Order Gradients: You cannot currently take the gradient of a gradient (needed for specialized GANs or MAML).                                                                    
 - Functionality: Missing detach_() (in-place) and retain_graph.                                                                                                                          

Optimization & Layers                                                                                                                                                                  

 - Advanced Optimizers: SGD and Adam are available; AdamW and RMSProp are still pending.                                                                                                             
 - Attention/Transformers: baseline `MultiHeadAttention` and `LayerNorm` exist; optimized fused attention kernels are still needed for production inference.                                                                                                     
 - Dropout: Essential for preventing overfitting in production models. ✅ Implemented in `munet.nn`.                                                                                     

Engineering Infrastructure                                                                                                                                                             

 - Data Loading: You need a Dataset and DataLoader with multi-threaded prefetching.                                                                                                       
 - Model Serialization: While you have .npz support, production usually requires a more robust format like Protobuf or ONNX export.                                                       
 - Lazy Execution: Currently, every op is dispatched immediately. Production frameworks often use a JIT (like TorchScript) to fuse kernels (e.g., ReLU(Add(x,y))) into a single GPU pass. 


Additional Layers & Operators:
1. Essential Tensor Operators

     - Division & Power: operator/, pow(), sqrt(), exp(), log(). (Crucial for custom loss functions and variance calculations).
     - Transposition/Permutation: transpose(dim1, dim2) and permute(dims). (Required for handling different data layouts and attention mechanisms). ✅ `transpose(dim1, dim2)` and `permute(dims)` implemented.
     - Mean & Variance: mean(dim), var(dim). (Currently you only have sum()).
     - Slice/Narrow: slice(dim, start, end). (Necessary for splitting tensors or taking sub-sections).

2. Core Neural Network Layers

     - Dropout: nn::Dropout. (Essential for preventing overfitting; requires a training flag to disable during inference). ✅ Implemented in `munet.nn`.
     - Global Average Pooling: nn::GlobalAvgPool2d. (Used in almost all modern CNNs before the final classifier). ✅ Implemented in `munet.nn`.
     - LeakyReLU: nn::LeakyReLU. (Standard improvement over basic ReLU to prevent "dying neurons"). ✅ Implemented in `munet.nn`.
     - Tanh: nn::Tanh. (Standard activation for Recurrent Neural Networks). ✅ Implemented in `munet.nn`.
     - GELU: nn::GELU. (Common Transformer MLP activation; fast approximation). ✅ Implemented in `munet.nn`.
     - LayerNorm: nn::LayerNorm. (The standard normalization layer for Transformers/NLP, which is easier to implement than BatchNorm for variable sequences). ✅ Implemented in `munet.nn` (CPU-fallback kernel with autograd support).

3. Advanced Modules

     - Embedding: nn::Embedding. (Mapping integer IDs to vectors; the foundation of all NLP models). ✅ Implemented in `munet.nn` (one-hot/probability input variant).
     - RNN/LSTM: nn::LSTM or nn::GRU. (To handle sequential or time-series data).
     - Padding Layers: nn::ZeroPad2d. (When you need padding outside of the convolution operation).

4. Mathematical Foundation (Optimizers & Loss)

     - Adam Optimizer: optim::Adam. (The industry standard. Requires tracking first and second moments: m and v).
     - BCEWithLogitsLoss: Binary Cross Entropy for multi-label classification.
     - NLLLoss: Negative Log Likelihood (often used with LogSoftmax).

5. Backend-Specific Kernels

     - Vectorized CPU Ops: Using AVX/SIMD for the CPUBackend to compete with the GPU backends on small batches.
     - Im2Col Convolution: Moving your Conv2d from the current "naive" nested loops to an im2col + GEMM approach for significantly higher performance on all backends.


6. Transformer / LLM Build-Out (Recommended Next Milestones)

     - Core layers to add first:
       - LayerNorm: `nn::LayerNorm` (per-token normalization used everywhere in Transformers).
       - Embedding: `nn::Embedding` (token + positional tables).
       - MultiHeadAttention: `nn::MultiHeadAttention` (QKV projections + scaled dot-product attention). ✅ Implemented (causal self-attention via tensor-op composition; backend-accelerated matmuls).
       - FeedForward block: `Linear -> GELU/SwiGLU -> Linear`.
     - Tensor operators to unlock attention workloads:
       - `softmax(dim)` and `log_softmax(dim)` (attention weights and stable losses). ✅ Implemented in Tensor API.
       - `transpose(dim1, dim2)` / `permute(dims)` (head and sequence layout transforms).
       - `masked_fill(mask, value)` (causal masking and padding masks). ✅ Implemented in Tensor API.
       - `sqrt`, `rsqrt`, `exp`, `log`, `pow` (attention scaling and normalization math).
     - Training-quality features for small LLMs:
       - `CrossEntropyLoss` (or `NLLLoss + LogSoftmax`) for token prediction.
       - Gradient clipping + weight decay options in optimizers.
       - Mixed precision support (`float16`/`bfloat16`) for memory and throughput.
     - Demo targets (incremental path):
       - Demo 1: Character-level language model (tiny corpus, CPU-safe).
       - Demo 2: Decoder-only Transformer block stack with causal mask.
       - Demo 3: Minimal text generation script (top-k / temperature sampling).
       - Demo 4: Optional tiny instruction-tuned chat example after baseline LM is stable.

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

## Demos

LLM demos:

- `demos/llm/tiny_llm.py` — tiny character-level LM with token+position embedding, layer norm and MLP head.
- `demos/llm/decoder_block_demo.py` — decoder-block style LM with causal `nn.MultiHeadAttention` and residual MLP.

Computer vision demos:

- `demos/mnist/mnist.py` — simple segmentation-style toy training loop.
- `demos/unet/unet.py` — larger UNet-like segmentation workflow with visualization outputs.

Feature demos (new):

- `demos/features/transformer_ops_showcase.py` — quick forward-only showcase of `MultiHeadAttention`, `LayerNorm`, `GELU`, and `Dropout`.
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

### Transformer Work Remaining
- Optimize `nn::MultiHeadAttention` with dedicated CUDA/Vulkan attention kernels (current implementation is tensor-op composition).
- Integer-index embedding gather path (avoid one-hot expansion for memory/perf).
- Add attention-adjacent ops still missing for scale/perf: `logsumexp`, fused causal mask-softmax, and efficient batched matmul layouts.
- Fused attention and layernorm backend kernels for CUDA/Vulkan performance.
- Add deeper multi-block decoder demo with KV-cache style autoregressive inference loop.
