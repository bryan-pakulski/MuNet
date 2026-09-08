"""Small model-agnostic conveniences over the native graph interface."""
from pathlib import Path
import numpy as np
from .core import Compiled
from .nn import Module
from . import optim


def train_step(model, optimizer, loss_fn, *, device="vulkan", max_grad_norm=None, fuse=True):
    """Compile supervised (inputs, targets) -> scalar loss and one optimizer update.

    Sets model training mode when capturing. Use @munet.compile for multiple model
    inputs, custom objectives, EMA or other updates. Data sampling stays with the caller.
    """
    if not isinstance(model, Module): raise TypeError("train_step expects an nn.Module")
    if not callable(loss_fn): raise TypeError("loss_fn must be callable")
    if max_grad_norm is not None and (not np.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("max_grad_norm must be positive and finite")
    def update(inputs, targets):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        if max_grad_norm is not None: optim.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        return loss
    return Compiled(update, device=device, fuse=fuse)


def export(model, path, example_inputs, *, input_names=None, output_names=None, include_vulkan=True, fuse=True):
    """Export a callable/Module for Python or C++ inference without executing it.

    A single NumPy array means one input; a tuple/list means positional inputs.
    Module export temporarily selects eval mode and restores every module's mode.
    Export embeds precompiled SPIR-V by default and requires glslangValidator here.
    Set include_vulkan=False for a CPU-only artifact without a shader compiler.
    Returns the destination Path. Training updates are rejected.
    """
    from .serialization import save, validate_names
    if isinstance(model, Compiled):
        raise TypeError("export expects the underlying model/callable; use program.save() for an existing compiled program")
    args = tuple(example_inputs) if isinstance(example_inputs, (tuple, list)) else (example_inputs,)
    modes = [(m, m.training) for m in model.modules()] if isinstance(model, Module) else []
    try:
        if modes: model.eval()
        # Host-only graph capture: no model execution or Vulkan device allocation.
        # Deployment shaders are compiled by save() below.
        program = Compiled(model, device="cpu", fuse=fuse).prepare(*args)
        if program._plan.updates:
            raise ValueError("inference export cannot contain state/optimizer updates; export an eval-mode forward function")
        program._input_names = validate_names(input_names, len(args), "input")
        program._output_names = validate_names(output_names, len(program._plan.outputs), "output")
        save(program, path, include_vulkan=include_vulkan)
    finally:
        for module, training in modes: module.training = training
    return Path(path)
