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
- [ ] Implement Adam and RMSProp optimizers.
- [ ] Add Softmax and CrossEntropyLoss.
- [ ] Support for proper tensor broadcasting in arithmetic ops.
- [ ] SPIR-V Shader caching for faster Vulkan startup.
- [ ] ONNX model loading support.
- [ ] Winograd convolution algorithms for performance optimization.

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
