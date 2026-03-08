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

Documentation can be generated using pydoc:

```
pip install pydoc
make doc
```

You can then open the generated `docs/index.html` in your browser.

# Future Plans
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

 - Advanced Optimizers: You only have SGD. You need Adam, AdamW, and RMSProp.                                                                                                             
 - Attention/Transformers: You lack optimized MultiHeadAttention or even a LayerNorm.                                                                                                     
 - Dropout: Essential for preventing overfitting in production models.                                                                                                                    

Engineering Infrastructure                                                                                                                                                             

 - Data Loading: You need a Dataset and DataLoader with multi-threaded prefetching.                                                                                                       
 - Model Serialization: While you have .npz support, production usually requires a more robust format like Protobuf or ONNX export.                                                       
 - Lazy Execution: Currently, every op is dispatched immediately. Production frameworks often use a JIT (like TorchScript) to fuse kernels (e.g., ReLU(Add(x,y))) into a single GPU pass. 


Additional Layers & Operators:
1. Essential Tensor Operators

     - Division & Power: operator/, pow(), sqrt(), exp(), log(). (Crucial for custom loss functions and variance calculations).
     - Transposition/Permutation: transpose(dim1, dim2) and permute(dims). (Required for handling different data layouts and attention mechanisms).
     - Mean & Variance: mean(dim), var(dim). (Currently you only have sum()).
     - Slice/Narrow: slice(dim, start, end). (Necessary for splitting tensors or taking sub-sections).

2. Core Neural Network Layers

     - Dropout: nn::Dropout. (Essential for preventing overfitting; requires a training flag to disable during inference).
     - Global Average Pooling: nn::GlobalAvgPool2d. (Used in almost all modern CNNs before the final classifier). ✅ Implemented in `munet.nn`.
     - LeakyReLU: nn::LeakyReLU. (Standard improvement over basic ReLU to prevent "dying neurons"). ✅ Implemented in `munet.nn`.
     - Tanh: nn::Tanh. (Standard activation for Recurrent Neural Networks). ✅ Implemented in `munet.nn`.
     - LayerNorm: nn::LayerNorm. (The standard normalization layer for Transformers/NLP, which is easier to implement than BatchNorm for variable sequences).

3. Advanced Modules

     - Embedding: nn::Embedding. (Mapping integer IDs to vectors; the foundation of all NLP models).
     - RNN/LSTM: nn::LSTM or nn::GRU. (To handle sequential or time-series data).
     - Padding Layers: nn::ZeroPad2d. (When you need padding outside of the convolution operation).

4. Mathematical Foundation (Optimizers & Loss)

     - Adam Optimizer: optim::Adam. (The industry standard. Requires tracking first and second moments: m and v).
     - BCEWithLogitsLoss: Binary Cross Entropy for multi-label classification.
     - NLLLoss: Negative Log Likelihood (often used with LogSoftmax).

5. Backend-Specific Kernels

     - Vectorized CPU Ops: Using AVX/SIMD for the CPUBackend to compete with the GPU backends on small batches.
     - Im2Col Convolution: Moving your Conv2d from the current "naive" nested loops to an im2col + GEMM approach for significantly higher performance on all backends.

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
