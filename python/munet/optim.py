import math
from .core import _trace


class SGD:
    """SGD without momentum. Updates are compiled into the same device replay."""
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(dict.fromkeys(parameters))
        self.lr = float(lr)
        if not math.isfinite(self.lr) or self.lr < 0:
            raise ValueError("learning rate must be finite and nonnegative")

    def zero_grad(self):
        _trace().grads.clear()

    def step(self):
        ctx = _trace()
        if ctx.updates:
            raise RuntimeError("v0 supports one optimizer step per trace")
        if not ctx.backward_called:
            raise RuntimeError("call loss.backward() before optimizer.step()")
        for p in self.parameters:
            if p.grad is not None:
                update = p - self.lr * p.grad
                ctx.updates.append((p._resolve().value, update.value))
