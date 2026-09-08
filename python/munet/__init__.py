from .core import Tensor, Parameter, Buffer, Result, Compiled, TensorSpec, compile, grad, devices, as_tensor, cat, stack, where
from . import nn, optim
from .nn import manual_seed
from .api import train_step, export

__all__ = ["Tensor", "Parameter", "Buffer", "Result", "Compiled", "TensorSpec", "compile",
           "grad", "devices", "as_tensor", "cat", "stack", "where", "nn", "optim",
           "manual_seed", "train_step", "export", "save", "load"]

__version__ = "0.3.0"

def save(program, path, *, include_vulkan=False):
    from .serialization import save
    return save(program, path, include_vulkan=include_vulkan)

def load(path, *, device="cpu", fuse=None):
    from .serialization import load
    return load(path, device=device, fuse=fuse)
