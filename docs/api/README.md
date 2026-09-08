# MuNet API guides

| Start here | What it covers |
|---|---|
| [Python API](python.md) | Model creation, training, inference, tensor/layer reference, checkpoints and export |
| [C++ API](cpp.md) | SDK installation, model loading, named/positional inference, Vulkan deployment and low-level graphs |
| [Model file contract](model-format.md) | The shared `.mnet` program format, optional embedded shaders and compatibility limits |

Run the guides' complete examples from the repository root:

```bash
make demo-python-api VULKAN=0
make demo-cpp VULKAN=0
```

MuNet supplies general tensors, layers, gradients, optimizers and native execution.
Architectures and data pipelines belong to applications/examples. The Python
library requires NumPy; a C++ application does not require Python, NumPy, ONNX
Runtime or PyTorch.

This usability pass makes **CPU the default** for Python compile/load/import and
the C++ `Model` API. Earlier prototype releases defaulted to Vulkan. Existing
explicit `device="vulkan"` / `"vulkan:N"` calls retain their behavior, including
an error if Vulkan cannot run. There is no automatic backend fallback.
