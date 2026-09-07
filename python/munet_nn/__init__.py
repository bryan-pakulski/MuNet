"""Retained import name for munet-nn, exposing the new 0.2 API.

This is an import alias, not an emulation of the removed 0.1 tensor API.
"""
import importlib as _importlib
import sys as _sys
from munet import Tensor, Parameter, Result, compile, grad, devices, save, load, nn, optim, __version__

for _name in ("core", "nn", "optim", "interop", "serialization"):
    globals()[_name] = _importlib.import_module("munet." + _name)
    _sys.modules[__name__ + "." + _name] = globals()[_name]


def __getattr__(name):
    if name == "swarm":
        return _importlib.import_module(__name__ + ".swarm")
    raise AttributeError(name)


__all__ = ["Tensor", "Parameter", "Result", "compile", "grad", "devices", "save", "load", "nn", "optim"]
