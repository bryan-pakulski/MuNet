from .core import Tensor, Parameter, Buffer, Result, compile, grad, devices, as_tensor, cat, stack, where
from . import nn, optim

__version__ = "0.3.0"

def save(program, path):
    from .serialization import save
    return save(program, path)

def load(path, *, device="vulkan", fuse=True):
    from .serialization import load
    return load(path, device=device, fuse=fuse)
