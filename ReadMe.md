# MuNet: A lightweight C++ GPU agnostic AI framework

MuNet is a lightweight C++ AI framework with Python bindings. It is designed to be GPU agnostic with the intended
final goal of running on edge devices with limited resources.

# Features

 • GPU Support: CUDA 
 • Python Bindings: Seamless integration with NumPy/Python.
 • Easy to use: Simple API for defining models and training.
 • Extensible: Support for custom layers and activations.

# Future Plans

 • ONNX Support: ONNX import & export.
 • Quantization: Quantization aware training / optimized inference.
 • Vulkan / OPENGL Kernels

# Supported Operations

## Core Engine

 • Tensor Backend: N-dimensional tensors with transparent CPU <-> GPU (CUDA) memory management.
 • Auto-Differentiation: Manual backpropagation implemented per layer (not a general autograd graph, but sufficient for layer stacks).
 • Python Bindings: Seamless integration with NumPy/Python.

## Layers & Transformations

 • Convolutional: Conv2D (with stride/padding), MaxPool2D, Upsample2D.
 • Dense: Linear (Fully Connected).
 • Normalization: BatchNorm2D (Train/Eval modes).
 • Regularization: Dropout.
 • Utility: Flatten, Concat, Skip Connection.

## Activations

 • ReLU, Softmax, Sigmoid.

## Optimizers

 • SGD, Adam (with weight decay).

## Loss Functions

 • MSELoss.
 • CrossEntropy (Standard).
 • SpatialCrossEntropy (for Segmentation).

# Development

Adding new functionality is easy:

1. Add a new class to the appropriate location under `src/`.
2. Implement custom gpu kernel in `src/munet.cu`. Add C++ hook as well within.
3. Add C++ function decleration to `src/kernels.hpp`.
4. Add python binding and expose in `src/bindings.cpp`.

There are plenty of examples that you can pull from the existing codebase.

## Building

### Requirements

- CMake
- CUDA (optional)
- Python 3.10+

### Build

```bash
./build.sh
```

This will build the C++ library and run unit tests.

### Runtime

The library is built as a shared object (`.so`) and can be loaded into a Python interpreter.

```python
import munet
```

# Examples

See the `demo/` directory for examples, included is:

- MNIST Classifier
- Unet Segmentation 
- Unet Segmentation (Complex)

