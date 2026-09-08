"""Retained import name for munet-nn, exposing the new 0.3 API.

This is an import alias, not an emulation of the removed 0.1 tensor API.
"""
import importlib as _importlib
import sys as _sys
from munet import Tensor, Parameter, Buffer, Result, compile, grad, devices, save, load, nn, optim, __version__

for _name in ("core", "nn", "optim", "interop", "serialization", "checkpoint", "models", "models.rtdetr", "models.rtdetr.backbone", "models.rtdetr.encoder", "models.rtdetr.decoder", "models.rtdetr.loss", "models.rtdetr.denoising", "models.rtdetr.training", "models.rtdetr.data"):
    _module = _importlib.import_module("munet." + _name)
    if "." not in _name: globals()[_name] = _module
    _sys.modules[__name__ + "." + _name] = _module


def __getattr__(name):
    if name == "swarm":
        return _importlib.import_module(__name__ + ".swarm")
    raise AttributeError(name)


__all__ = ["Tensor", "Parameter", "Result", "compile", "grad", "devices", "save", "load", "nn", "optim"]
