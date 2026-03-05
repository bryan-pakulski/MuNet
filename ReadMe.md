# μNet: A lightweight C++ GPU agnostic AI framework

μNet is a lightweight C++ AI framework with Python bindings. It is designed to be GPU agnostic with the intended final goal of running on edge devices with limited resources. It
features a PyTorch-like API, making it familiar to use while handling low-level memory management and compute dispatch across CPU, CUDA, and Vulkan backends.

# Features

- **GPU Support**: Native support for **CUDA** (Nvidia) and **Vulkan** (Cross-vendor: AMD, Intel, Mobile GPUs).
- **Python Bindings**: Seamless integration with NumPy/Python via `pybind11`. Zero-copy memory sharing on CPU.
- **Easy to use**: Simple, PyTorch-like API for defining models and training.
- **Dynamic Autograd**: define-by-run automatic differentiation engine (DAG).
- **Cross Platform**: Built for Linux/Unix systems.
- **Model Parallelism**: tensors can exist on different devices within the same graph (e.g., Layer 1 on CPU, Layer 2 on Vulkan).

# Supported Operations

## Core Engine
- **Tensor Operations**: `add`, `sub`, `mul`, `div`, `matmul`.
- **Memory Management**: Automatic reference counting and storage management.
- **Device Support**:
 - `DeviceType.CPU`
 - `DeviceType.CUDA`
 - `DeviceType.VULKAN`
- **Data Types**: Float32 (primary), Float16, Int32 support structure.

## Layers & Transformations
- **Convolution**: `Conv2d` (Forward & Backward) with stride and padding support.
- **Pooling**: `MaxPool2d` (Forward & Backward).
- **Normalization**: `BatchNorm2d` (Train & Eval modes, Running stats tracking).
- **Upsampling**: `Upsample2d` (Nearest neighbor).
- **Reshaping**: `view`/`reshape` operations.

## Activations
- **ReLU** (Rectified Linear Unit).

## Optimizers
- **SGD** (Stochastic Gradient Descent) with learning rate control.

## Loss Functions
- **MSE** (Mean Squared Error) via reduction operations.
- **Sum** (Scalar reduction).

# Future Plans

Multi-GPU RoadMap:

 1 Device Switching: Update CUDABackend and VulkanBackend to ensure they explicitly select the correct hardware index before executing any commands.
 2 Functional Collectives: Implement all_reduce in the Backend interface.
    • CUDA: Use ncclAllReduce.
    • Vulkan: Use timeline semaphores and cross-device buffer copies.
 3 Stream Management: Currently, you use the default stream (0) in CUDA. To allow the CPU to "fire and forget" kernels to multiple GPUs simultaneously, give each Device its own compute
   stream/queue.
 4 Data Parallel Wrapper: Create a nn::DataParallel modcule that:
    • Copies the model to all target devices.
    • Splits the input Tensor along the batch dimension.
    • Triggers the all_reduce on gradients automatically.


Additional Layers & Operators:
    1. Essential Tensor Operators

     • Division & Power: operator/, pow(), sqrt(), exp(), log(). (Crucial for custom loss functions and variance calculations).
     • Transposition/Permutation: transpose(dim1, dim2) and permute(dims). (Required for handling different data layouts and attention mechanisms).
     • Mean & Variance: mean(dim), var(dim). (Currently you only have sum()).
     • Broadcasting Logic: Upgrading existing math ops to handle tensors of different ranks (e.g., adding a bias vector [C] to an image batch [N, C, H, W]).
     • Slice/Narrow: slice(dim, start, end). (Necessary for splitting tensors or taking sub-sections).

    2. Core Neural Network Layers

     • Dropout: nn::Dropout. (Essential for preventing overfitting; requires a training flag to disable during inference).
     • Global Average Pooling: nn::GlobalAvgPool2d. (Used in almost all modern CNNs before the final classifier).
     • LeakyReLU: nn::LeakyReLU. (Standard improvement over basic ReLU to prevent "dying neurons").
     • Tanh: nn::Tanh. (Standard activation for Recurrent Neural Networks).
     • LayerNorm: nn::LayerNorm. (The standard normalization layer for Transformers/NLP, which is easier to implement than BatchNorm for variable sequences).

    3. Advanced Modules

     • Embedding: nn::Embedding. (Mapping integer IDs to vectors; the foundation of all NLP models).
     • RNN/LSTM: nn::LSTM or nn::GRU. (To handle sequential or time-series data).
     • Padding Layers: nn::ZeroPad2d. (When you need padding outside of the convolution operation).

    4. Mathematical Foundation (Optimizers & Loss)

     • Adam Optimizer: optim::Adam. (The industry standard. Requires tracking first and second moments: m and v).
     • BCEWithLogitsLoss: Binary Cross Entropy for multi-label classification.
     • NLLLoss: Negative Log Likelihood (often used with LogSoftmax).

    5. Backend-Specific Kernels

     • Vectorized CPU Ops: Using AVX/SIMD for the CPUBackend to compete with the GPU backends on small batches.
     • Im2Col Convolution: Moving your Conv2d from the current "naive" nested loops to an im2col + GEMM approach for significantly higher performance on all backends.

    Priority Implementation Order Recommendation:

     1 Adam Optimizer (Crucial for training stability).
     2 Dropout (Crucial for generalization).
     3 Transpose/Permute (Unlocks more complex model architectures).
     4 GlobalAvgPool2d (Allows you to build modern CNNs like ResNet).

# Development

μNet is designed to be modular. Adding new functionality involves touching the backend interface, the specific implementations, and the autograd engine.

### 1. Define the Interface
Modify `src/backend.hpp` to add the virtual function definitions for your new operation (both forward and backward if applicable).


`virtual void my_op(const Storage &in, Storage &out, int param) = 0;`

### 2. Implement the Kernels

Implement the specific logic for each backend:

 • CPU (src/backend/cpu_backend.hpp): Implement using parallel for loops or SIMD.
 • CUDA (src/backend/cuda_backend.cu): Write the CUDA kernel (__global__ void ...) and the host dispatch function.
 • Vulkan (src/backend/vulkan_backend.cpp):
    1 Write the GLSL compute shader source string.
    2 In the constructor, compile it using createComputePipeline.
    3 Implement the dispatch logic using dispatch_kernel.

### 3. Register the Operation

Add the high-level logic in src/ops.hpp. This is where the Autograd magic happens.

 1 Create a struct MyOpBackward : public Node that defines how to calculate gradients.
 2 Create a function inline Tensor my_op(...) that:
    • Allocates the output tensor.
    • Calls backend->my_op(...).
    • If requires_grad is true, creates the MyOpBackward node and links edges.

4. Expose to Tensor API

Add a method to the Tensor class in src/tensor.hpp and src/tensor.cpp that delegates to ops::my_op.

5. Python Binding

Finally, expose the method to Python in src/bindings.cpp.


 .def("my_op", &Tensor::my_op, py::arg("param"))

## Debugging

During development or runtime, you can enable debug logging by setting the `MUNET_DEBUG` environment variable:

`MUNET_DEBUG=1 python munet_script.py`

# Building

## Requirements

 • CMake 3.10+
 • C++17 Compiler
 • Optional: CUDA Toolkit (nvcc)
 • Optional: Vulkan SDK (glslc must be in PATH)
 • Python 3.10+ (and development headers)

## Build Steps

```
 mkdir build && cd build
 cmake ..
 make -j$(nproc)
```

# Testing

Run the C++ unit tests (GoogleTest) and Python integration tests:

```
 ./build/munet_tests
 python3 tests/test_python.py
```

# Runtime

The library is built as a shared object (munet.cpython-....so) in the build directory. To use it, simply import it in Python:

```
 import sys
 sys.path.append("path/to/build")
 import munet

 # Create a tensor on GPU
 dev = munet.Device(munet.DeviceType.VULKAN, 0)
 x = munet.Tensor([2, 2], device=dev, requires_grad=True)
```

# Examples

See the tests/ folder for functional examples:

 • test_python.py: Contains a full end-to-end training loop for a U-Net style segmentation model.
 • test_cuda.cpp: C++ implementation of MNIST training.
