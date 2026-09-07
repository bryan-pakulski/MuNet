"""Small native-training module surface. Only listed layers are implemented."""
import math
import numpy as np
from .core import Parameter


class Module:
    def __call__(self, *args): return self.forward(*args)
    def _named_children(self): return vars(self).items()

    def named_parameters(self):
        seen = set()
        def walk(value, prefix):
            if isinstance(value, Parameter):
                if id(value) not in seen:
                    seen.add(id(value))
                    yield prefix, value
            elif isinstance(value, Module):
                for name, child in value._named_children():
                    yield from walk(child, f"{prefix}.{name}" if prefix else name)
            elif isinstance(value, (list, tuple)):
                for i, child in enumerate(value):
                    yield from walk(child, f"{prefix}.{i}" if prefix else str(i))
        yield from walk(self, "")

    def parameters(self): return [p for _, p in self.named_parameters()]
    def state_dict(self): return {name: p.numpy() for name, p in self.named_parameters()}

    def load_state_dict(self, state):
        params = dict(self.named_parameters())
        if set(params) != set(state):
            raise ValueError(f"state keys differ; missing={set(params)-set(state)}, extra={set(state)-set(params)}")
        # Validate all values before making the first change.
        converted = {}
        for name, value in state.items():
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=np.float32)
            if value.shape != params[name].shape:
                raise ValueError(f"state shape mismatch for {name}")
            converted[name] = value
        for name, value in converted.items():
            params[name].assign(value)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, *, rng=None):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("layer dimensions must be positive")
        rng = np.random.default_rng() if rng is None else rng
        limit = 1 / math.sqrt(in_features)
        self.weight = Parameter(rng.uniform(-limit, limit, (out_features, in_features)))
        self.bias = Parameter(rng.uniform(-limit, limit, (out_features,))) if bias else None

    def forward(self, x):
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y


class Sequential(Module):
    def __init__(self, *layers): self.layers = list(layers)
    def _named_children(self): return ((str(i), layer) for i, layer in enumerate(self.layers))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ReLU(Module):
    def forward(self, x): return x.relu()


class Sigmoid(Module):
    def forward(self, x): return x.sigmoid()


class MSELoss(Module):
    def forward(self, prediction, target): return (prediction - target).square().mean()


class functional:
    @staticmethod
    def mse_loss(prediction, target): return (prediction - target).square().mean()
    @staticmethod
    def relu(x): return x.relu()
    @staticmethod
    def sigmoid(x): return x.sigmoid()
