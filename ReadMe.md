# μNet: A lightweight C++ GPU agnostic AI framework

μNet is a lightweight C++ AI framework with Python bindings. It is designed to be GPU agnostic with the intended
final goal of running on edge devices with limited resources.

# Features

 • GPU Support: CUDA, VULKAN 
 • Python Bindings: Seamless integration with NumPy/Python.
 • Easy to use: Simple API for defining models and training.
 • Extensible: Support for custom layers and activations. 
 • DAG: Auto-differentiation of ops.
 • Cross Platform: Built for all platforms.

# Future Plans


# Supported Operations

## Core Engine


## Layers & Transformations


## Activations


## Optimizers


## Loss Functions


# Development

Adding new functionality is easy:

1. Add a new class to the appropriate location under `src/`.
2. Implement custom gpu kernels in `src/backend/...`. Add C++ hooks as well within.
3. Add C++ functions to backend in `src/backend/...`.
4. Add python binding and expose in `src/bindings.cpp`.

There are plenty of examples that you can pull from the existing codebase.

## Building

### Requirements

- CMake
- CUDA (optional)
- Vulkan (optional)
- Python 3.10+

### Build

```bash
./build.sh
```

This will build the C++ library.

### Testing

C++ & python Unit tests
```bash
./test.sh
```

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
- Simple object detection

