# MuNet API guides

| Start here | What it covers |
|---|---|
| [Python API](python.md) | Model creation, training, inference, tensor/layer reference, checkpoints and export |
| [C++ API](cpp.md) | SDK installation, model loading, named/positional inference, Vulkan deployment and low-level graphs |
| [Model file contract](model-format.md) | The shared `.mnet` program format, optional embedded shaders and compatibility limits |

Run the guides' complete examples from the repository root:

```bash
make demo-python-api
make demo-cpp
```

MuNet supplies general tensors, layers, gradients, optimizers and native execution.
Architectures and data pipelines belong to applications/examples. The Python
library requires NumPy; a C++ application does not require Python, NumPy, ONNX
Runtime or PyTorch.

**Vulkan is the default** for Python compile/train/load/import, the C++ `Model`
API, examples and Make targets. Export embeds deployment shaders by default.
Select `device="cpu"` (or `DEVICE=cpu` for Make) for an explicit reference fallback.
Use `VULKAN=0` only for a CPU-only build. Vulkan failures remain visible; there
is no silent backend switch.
