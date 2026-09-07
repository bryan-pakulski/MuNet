"""A deliberately small PyTorch-style tracing surface. Numerical work is native."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import numpy as np
from . import _native

_active = ContextVar("munet_trace", default=None)


@dataclass
class _Trace:
    graph: object = field(default_factory=_native.Graph)
    parameters: dict = field(default_factory=dict)
    grads: dict = field(default_factory=dict)
    updates: list = field(default_factory=list)
    backward_called: bool = False


def _trace():
    ctx = _active.get()
    if ctx is None:
        raise RuntimeError("Tensor operations must run inside munet.compile(function)")
    return ctx


class Tensor:
    def __init__(self, graph, value):
        self.graph, self.value = graph, value

    def _resolve(self):
        if _trace().graph is not self.graph:
            raise RuntimeError("cannot mix tensors from different traces")
        return self

    @property
    def shape(self):
        return tuple(self.graph.shape(self.value))

    @property
    def ndim(self):
        return len(self.shape)

    def _binary(self, op, other):
        a, b = self._resolve(), as_tensor(other)
        return Tensor(a.graph, a.graph.op(op, [a.value, b.value]))

    def _unary(self, op, attrs=()):
        a = self._resolve()
        return Tensor(a.graph, a.graph.op(op, [a.value], attrs))

    def __add__(self, other): return self._binary("add", other)
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self._binary("sub", other)
    def __rsub__(self, other): return as_tensor(other) - self
    def __mul__(self, other): return self._binary("mul", other)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self._binary("div", other)
    def __rtruediv__(self, other): return as_tensor(other) / self
    def __matmul__(self, other): return self._binary("matmul", other)
    def __neg__(self): return self._unary("neg")
    def relu(self): return self._unary("relu")
    def sigmoid(self): return self._unary("sigmoid")
    def exp(self): return self._unary("exp")
    def log(self): return self._unary("log")
    def sqrt(self): return self._unary("sqrt")
    def square(self): return self * self

    @property
    def T(self): return self._unary("transpose")

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = list(shape)
        if shape.count(-1) == 1:
            known = int(np.prod([d for d in shape if d != -1]))
            if known <= 0 or int(np.prod(self.shape)) % known:
                raise ValueError("invalid inferred reshape")
            shape[shape.index(-1)] = int(np.prod(self.shape)) // known
        return self._unary("reshape", shape)

    def sum(self, dim=None, keepdim=False):
        axes = list(range(self.ndim)) if dim is None else [dim] if isinstance(dim, int) else list(dim)
        axes = [a + self.ndim if a < 0 else a for a in axes]
        result = self._unary("sum", axes)
        if not keepdim:
            result = result.reshape(tuple(d for i, d in enumerate(self.shape) if i not in axes))
        return result

    def mean(self, dim=None, keepdim=False):
        axes = list(range(self.ndim)) if dim is None else [dim] if isinstance(dim, int) else list(dim)
        count = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(dim, keepdim) / count

    def backward(self):
        ctx = _trace()
        if ctx.backward_called:
            raise RuntimeError("v0 supports one first-order backward call per compiled step")
        params = list(ctx.parameters)
        grads = ctx.graph.gradients(self._resolve().value, [ctx.parameters[p] for p in params])
        ctx.grads = {p: Tensor(ctx.graph, g) for p, g in zip(params, grads)}
        ctx.backward_called = True

    def __bool__(self):
        raise TypeError("data-dependent Python control flow cannot be traced; use static control flow")

    def numpy(self):
        raise RuntimeError("traced values cannot be read on the host; return the value from the compiled function")

    def item(self): return self.numpy().item()


class Parameter(Tensor):
    def __init__(self, data):
        self._array = np.array(data, dtype=np.float32, order="C", copy=True)
        if any(d <= 0 for d in self._array.shape):
            raise ValueError("empty parameters are unsupported")
        self._owner = None
        self._version = 0

    @property
    def shape(self): return self._array.shape

    def _resolve(self):
        ctx = _trace()
        if self not in ctx.parameters:
            data = self._snapshot()
            ctx.parameters[self] = ctx.graph.leaf("parameter", f"p{len(ctx.parameters)}", self.shape, data.ravel().tolist())
        return Tensor(ctx.graph, ctx.parameters[self])

    @property
    def grad(self): return _trace().grads.get(self)

    def _snapshot(self):
        if self._owner is not None:
            compiled, value = self._owner
            return compiled._plan.read(value)
        return self._array.copy()

    def numpy(self):
        if _active.get() is not None:
            raise RuntimeError("parameter host reads inside tracing would freeze values into the graph")
        return self._snapshot()

    def assign(self, data):
        if _active.get() is not None:
            raise RuntimeError("use an optimizer update inside a trace")
        data = np.asarray(data, dtype=np.float32)
        if data.shape != self.shape:
            raise ValueError("parameter shape mismatch")
        self._array = data.copy(order="C")
        self._owner = None
        self._version += 1


def as_tensor(value):
    if isinstance(value, Tensor):
        return value._resolve()
    ctx = _trace()
    data = np.asarray(value, dtype=np.float32)
    ident = ctx.graph.leaf("constant", f"c{len(ctx.graph.nodes())}", data.shape, data.ravel().tolist())
    return Tensor(ctx.graph, ident)


def grad(loss, tensors):
    """Return first-order symbolic gradients without applying an optimizer."""
    ctx = _trace()
    values = [t._resolve().value for t in tensors]
    return [Tensor(ctx.graph, g) for g in ctx.graph.gradients(loss._resolve().value, values)]


def _compile_shaders(sources):
    executable = os.environ.get("MUNET_GLSLANG") or shutil.which("glslangValidator")
    if not executable:
        raise RuntimeError("Vulkan JIT needs glslangValidator; install glslang-tools or set MUNET_GLSLANG")
    executable = str(Path(executable).resolve())
    version = subprocess.run([executable, "--version"], check=True, capture_output=True).stdout
    cache = Path(os.environ.get("MUNET_CACHE_DIR", Path.home() / ".cache" / "munet-next" / "spirv-v0"))
    cache.mkdir(parents=True, exist_ok=True)
    result = []
    for source in sources:
        key = hashlib.sha256(b"munet-v0-vulkan1.1\0" + version + source.encode()).hexdigest()
        target = cache / f"{key}.spv"
        if not target.is_file():
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                src, dst = Path(tmp) / "kernel.comp", Path(tmp) / "kernel.spv"
                src.write_text(source)
                proc = subprocess.run([executable, "-V", "--target-env", "vulkan1.1", "-S", "comp", str(src), "-o", str(dst)], capture_output=True, text=True)
                if proc.returncode:
                    raise RuntimeError(f"Vulkan shader compilation failed:\n{proc.stdout}\n{proc.stderr}")
                os.replace(dst, target)
        data = target.read_bytes()
        if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
            raise RuntimeError(f"invalid shader cache entry: {target}; remove it and compile again")
        result.append(np.frombuffer(data, dtype="<u4").tolist())
    return result


def _device(plan, name):
    if name == "cpu":
        return
    if name != "vulkan" and not name.startswith("vulkan:"):
        raise ValueError("device must be cpu, vulkan, or vulkan:<index>")
    if not _native.vulkan_built():
        raise RuntimeError("this build has no Vulkan runtime; rebuild with MUNET_VULKAN=ON")
    index = 0 if name == "vulkan" else int(name.split(":", 1)[1])
    if index < 0:
        raise ValueError("device index must be nonnegative")
    plan.enable_vulkan(_compile_shaders(plan.shaders()), index)


class Result:
    def __init__(self, compiled, value):
        self._compiled, self._value = compiled, value
        self._generation = compiled._generation

    @property
    def shape(self): return tuple(self._compiled._plan.graph.shape(self._value))

    def numpy(self):
        with self._compiled._lock:
            if self._generation != self._compiled._generation:
                raise RuntimeError("result storage was reused by a later call; call .numpy() before replay to retain a copy")
            return self._compiled._plan.read(self._value)

    def item(self): return self.numpy().item()


class Compiled:
    """One guarded static-shape specialization, with persistent device state."""
    def __init__(self, fn=None, *, device="vulkan", fuse=True):
        self.fn, self.device, self.fuse = fn, device, fuse
        self._plan = None
        self._generation = 0
        self._parameters = {}
        self._seen_versions = {}
        self._lock = threading.RLock()

    def _capture(self, args):
        ctx = _Trace()
        token = _active.set(ctx)
        try:
            symbolic = [Tensor(ctx.graph, ctx.graph.leaf("input", f"input_{i}", a.shape)) for i, a in enumerate(args)]
            outputs = self.fn(*symbolic)
            self._single = isinstance(outputs, Tensor)
            if not self._single and not isinstance(outputs, (tuple, list)):
                raise TypeError("compiled functions must return a Tensor or a tuple/list of Tensors")
            output_list = [outputs] if self._single else list(outputs)
            if not all(isinstance(x, Tensor) for x in output_list):
                raise TypeError("every compiled output must be a Tensor")
            ids = [t._resolve().value for t in output_list]
        finally:
            _active.reset(token)
        self._plan = _native.Plan(ctx.graph, ids, ctx.updates, self.fuse)
        self._parameters = ctx.parameters
        self._seen_versions = {p: p._version for p in ctx.parameters}
        self._input_specs = [a.shape for a in args]
        # Dead inputs still have caller guards, but are not uploaded or executed.
        nodes = ctx.graph.nodes()
        self._feed_indices = [int(nodes[i]["name"].split("_")[1]) for i in self._plan.inputs]
        try:
            _device(self._plan, self.device)
        except Exception:
            self._plan = None
            raise

    def __call__(self, *args):
        if _active.get() is not None:
            raise RuntimeError("nested compiled calls are not supported; call the underlying model while tracing")
        arrays = []
        for a in args:
            if hasattr(a, "detach"):
                raise TypeError("convert torch inputs explicitly with .detach().cpu().numpy(); zero-copy GPU interop is not implemented")
            a = np.asarray(a)
            if a.dtype != np.float32:
                raise TypeError(f"v0 supports float32 inputs, received {a.dtype}")
            arrays.append(a if a.flags.c_contiguous else np.ascontiguousarray(a))
        with self._lock:
            if self._plan is None:
                self._capture(arrays)
            if [a.shape for a in arrays] != self._input_specs:
                raise ValueError(f"shape guard failed: expected {self._input_specs}, got {[a.shape for a in arrays]}; create a new compiled specialization")
            for p, value in self._parameters.items():
                if p._version != self._seen_versions[p]:
                    self._plan.write(value, np.ascontiguousarray(p.numpy()))
                    self._seen_versions[p] = p._version
            self._plan.run([arrays[i] for i in self._feed_indices])
            self._generation += 1
            updated = {dst for dst, _ in self._plan.updates}
            for p, value in self._parameters.items():
                if value in updated:
                    p._version += 1
                    p._owner = self, value
                    self._seen_versions[p] = p._version
            values = [Result(self, v) for v in self._plan.outputs]
            return values[0] if self._single else tuple(values)

    def stats(self):
        if self._plan is None:
            return {"compiled": False}
        return {"compiled": True, "device": self._plan.device_name(), **self._plan.stats()}

    def synchronize(self):
        if self._plan is not None:
            with self._lock:
                self._plan.synchronize()


def compile(fn=None, *, device="vulkan", fuse=True):
    if fn is None:
        return lambda f: Compiled(f, device=device, fuse=fuse)
    return Compiled(fn, device=device, fuse=fuse)


def devices():
    """Enumerate Vulkan physical devices; absence is an explicit error, never a CPU fallback."""
    return _native.devices()
