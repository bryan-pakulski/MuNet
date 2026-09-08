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
from concurrent.futures import ThreadPoolExecutor

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
    optimizer_stepped: bool = False
    modules: dict = field(default_factory=dict)


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

    def contiguous(self): return self
    def abs(self): return self._unary("abs")
    def floor(self): return self._unary("floor")
    def tanh(self): return self._unary("tanh")
    def erf(self): return self._unary("erf")
    def sin(self): return self._unary("sin")
    def cos(self): return self._unary("cos")
    def gelu(self): return self._unary("gelu")
    def softplus(self): return self._unary("softplus")
    def detach(self): return self._unary("detach")
    def minimum(self, other): return self._binary("minimum", other)
    def maximum(self, other): return self._binary("maximum", other)
    def eq(self, other): return self._binary("eq", other)
    def __lt__(self, other): return self._binary("lt", other)
    def __le__(self, other): return self._binary("le", other)
    def __gt__(self, other): return self._binary("gt", other)
    def __ge__(self, other): return self._binary("ge", other)
    def __pow__(self, power):
        if power == 0: return self * 0 + 1
        if isinstance(power, int) and power > 0:
            result = self
            for _ in range(power - 1): result = result * self
            return result
        return (self.log() * power).exp()

    def clamp(self, min=None, max=None):
        result = self
        if min is not None: result = where(result < min, min, result)
        if max is not None: result = where(result > max, max, result)
        return result

    def permute(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)): axes = axes[0]
        return self._unary("permute", [a % self.ndim for a in axes])

    def transpose(self, dim0, dim1):
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return self.permute(axes)

    def unsqueeze(self, dim):
        if dim < 0: dim += self.ndim + 1
        if not 0 <= dim <= self.ndim: raise ValueError("unsqueeze axis out of range")
        shape = list(self.shape); shape.insert(dim, 1)
        return self.reshape(shape)

    def squeeze(self, dim=None):
        if dim is None: return self.reshape([s for s in self.shape if s != 1])
        dim %= self.ndim
        return self.reshape([s for d, s in enumerate(self.shape) if d != dim or s != 1])

    def flatten(self, start_dim=0, end_dim=-1):
        end_dim %= self.ndim
        start_dim %= self.ndim
        if start_dim > end_dim: raise ValueError("invalid flatten dimensions")
        return self.reshape(self.shape[:start_dim] + (int(np.prod(self.shape[start_dim:end_dim+1])),) + self.shape[end_dim+1:])

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)): shape = tuple(shape[0])
        if len(shape) < self.ndim: raise ValueError("expand rank is smaller than input")
        old = (1,) * (len(shape)-self.ndim) + self.shape
        return self._unary("broadcast", [old[i] if s == -1 else s for i, s in enumerate(shape)])

    def repeat(self, *repeats):
        if len(repeats) == 1 and isinstance(repeats[0], (list, tuple)): repeats = tuple(repeats[0])
        if len(repeats) < self.ndim or any(r <= 0 for r in repeats): raise ValueError("invalid repeat factors")
        old = (1,) * (len(repeats)-self.ndim) + self.shape
        a = self.reshape([v for size in old for v in (1, size)])
        a = a.expand([v for r, size in zip(repeats, old) for v in (r, size)])
        return a.reshape([r*size for r, size in zip(repeats, old)])

    def __getitem__(self, key):
        if not isinstance(key, tuple): key = (key,)
        if sum(k is Ellipsis for k in key) > 1: raise IndexError("multiple ellipses")
        used = sum(k is not None and k is not Ellipsis for k in key)
        if used > self.ndim: raise IndexError("too many indices")
        expanded = []
        for k in key:
            expanded.extend([slice(None)] * (self.ndim-used) if k is Ellipsis else [k])
        if not any(k is Ellipsis for k in key): expanded += [slice(None)] * (self.ndim-used)
        starts, sizes, steps, result_shape = [], [], [], []
        dim = 0
        for k in expanded:
            if k is None: result_shape.append(1); continue
            size = self.shape[dim]; dim += 1
            if isinstance(k, (int, np.integer)):
                k = int(k); k = k + size if k < 0 else k
                if not 0 <= k < size: raise IndexError("index out of range")
                starts.append(k); sizes.append(1); steps.append(1)
            elif isinstance(k, slice):
                a, b, c = k.indices(size); length = len(range(a,b,c))
                if not length: raise IndexError("empty slices are unsupported; use padded targets and masks")
                starts.append(a); sizes.append(length); steps.append(c); result_shape.append(length)
            else: raise TypeError("use gather/take for tensor indices")
        return self._unary("slice", starts+sizes+steps).reshape(result_shape)

    def split(self, sections, dim=0):
        dim %= self.ndim
        if isinstance(sections, int):
            if sections <= 0: raise ValueError("split size must be positive")
            sections = [min(sections,self.shape[dim]-i) for i in range(0,self.shape[dim],sections)]
        if sum(sections) != self.shape[dim] or any(s <= 0 for s in sections): raise ValueError("split sizes do not cover dimension")
        result, start = [], 0
        for size in sections:
            key = [slice(None)] * self.ndim; key[dim] = slice(start,start+size)
            result.append(self[tuple(key)]); start += size
        return tuple(result)

    def amax(self, dim=None, keepdim=False):
        axes = list(range(self.ndim)) if dim is None else [dim] if isinstance(dim,int) else list(dim)
        axes = [a % self.ndim for a in axes]
        result = self._unary("max", axes)
        return result if keepdim else result.reshape([s for d,s in enumerate(self.shape) if d not in axes])

    def softmax(self, dim=-1):
        values = (self-self.amax(dim,keepdim=True).detach()).exp()
        return values / values.sum(dim,keepdim=True).clamp(min=1e-30)

    def gather(self, dim, index): return operation("gather", self, index, attrs=[dim])
    def take(self, index, dim=0): return operation("take", self, index, attrs=[dim])
    def topk(self, k, dim=-1):
        ranks = self._unary("topk_rank", [dim, k])
        indices = operation("topk_indices", self, ranks, attrs=[dim, k])
        return self.gather(dim, indices), indices

    def backward(self):
        ctx = _trace()
        if ctx.backward_called:
            raise RuntimeError("v0 supports one first-order backward call per compiled step")
        params = [p for p in ctx.parameters if p.requires_grad]
        grads = ctx.graph.gradients(self._resolve().value, [ctx.parameters[p] for p in params])
        ctx.grads = {p: Tensor(ctx.graph, g) for p, g in zip(params, grads)}
        ctx.backward_called = True

    def __bool__(self):
        raise TypeError("data-dependent Python control flow cannot be traced; use static control flow")

    def numpy(self):
        raise RuntimeError("traced values cannot be read on the host; return the value from the compiled function")

    def item(self): return self.numpy().item()


class Parameter(Tensor):
    def __init__(self, data, requires_grad=True):
        self.requires_grad = bool(requires_grad)
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
            ctx.parameters[self] = ctx.graph.leaf_array("parameter", f"p{len(ctx.parameters)}", data)
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
    ident = ctx.graph.leaf_array("constant", f"c{ctx.graph.size}", np.ascontiguousarray(data) if data.ndim else data)
    return Tensor(ctx.graph, ident)


class Buffer(Parameter):
    """Persistent, non-differentiable model state, updated by a compiled step."""
    def __init__(self, data, persistent=True):
        super().__init__(data, requires_grad=False)
        self.persistent = bool(persistent)


def update_state(parameter, value):
    ctx = _trace()
    dst, src = parameter._resolve().value, as_tensor(value).value
    if parameter.shape != tuple(ctx.graph.shape(src)):
        raise ValueError("state update shape mismatch")
    ctx.updates[:] = [(d, s) for d, s in ctx.updates if d != dst]
    ctx.updates.append((dst, src))


def current_state(parameter):
    ctx = _trace()
    dst = parameter._resolve().value
    return Tensor(ctx.graph, next((s for d, s in ctx.updates if d == dst), dst))


def grad(loss, tensors):
    """Return first-order symbolic gradients without applying an optimizer."""
    ctx = _trace()
    values = [t._resolve().value for t in tensors]
    return [Tensor(ctx.graph, g) for g in ctx.graph.gradients(loss._resolve().value, values)]


def operation(kind, *values, attrs=()):
    xs = [as_tensor(x) for x in values]
    ctx = _trace()
    return Tensor(ctx.graph, ctx.graph.op(kind, [x.value for x in xs], list(attrs)))


def cat(tensors, dim=0): return operation("concat", *tensors, attrs=[dim])
def stack(tensors, dim=0): return cat([x.unsqueeze(dim) for x in tensors], dim)
def where(condition, a, b): return operation("where", condition, a, b)


def _flatten_outputs(value, leaves):
    if isinstance(value, Tensor):
        leaves.append(value); return ["tensor", len(leaves)-1]
    if isinstance(value, dict): return ["dict", [[k, _flatten_outputs(v,leaves)] for k,v in value.items()]]
    if isinstance(value, (tuple,list)): return ["tuple" if isinstance(value,tuple) else "list", [_flatten_outputs(v,leaves) for v in value]]
    if value is None or isinstance(value, (int,float,str,bool)): return ["constant", value]
    raise TypeError("compiled outputs must contain tensors and JSON-compatible metadata")


def _unflatten_outputs(tree, leaves):
    kind, value = tree
    if kind == "tensor": return leaves[value]
    if kind == "constant": return value
    if kind == "dict": return {k:_unflatten_outputs(v,leaves) for k,v in value}
    values = [_unflatten_outputs(v,leaves) for v in value]
    return tuple(values) if kind == "tuple" else values


def _compile_shaders(sources):
    executable = os.environ.get("MUNET_GLSLANG") or shutil.which("glslangValidator")
    if not executable:
        raise RuntimeError("Vulkan JIT needs glslangValidator; install glslang-tools or set MUNET_GLSLANG")
    executable = str(Path(executable).resolve())
    version = subprocess.run([executable, "--version"], check=True, capture_output=True).stdout
    cache = Path(os.environ.get("MUNET_CACHE_DIR", Path.home() / ".cache" / "munet-next" / "spirv-v0"))
    cache.mkdir(parents=True, exist_ok=True)
    def compile_one(source):
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
        return np.frombuffer(data, dtype="<u4").tolist()
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        return list(pool.map(compile_one, sources))


def _device(plan, name, spirv=None):
    if name == "cpu":
        return
    if name != "vulkan" and not name.startswith("vulkan:"):
        raise ValueError("device must be cpu, vulkan, or vulkan:<index>")
    if not _native.vulkan_built():
        raise RuntimeError("this build has no Vulkan runtime; rebuild with MUNET_VULKAN=ON")
    index = 0 if name == "vulkan" else int(name.split(":", 1)[1])
    if index < 0:
        raise ValueError("device index must be nonnegative")
    plan.enable_vulkan(_compile_shaders(plan.shaders()) if spirv is None else spirv, index)


class Result:
    def __init__(self, compiled, value):
        self._compiled, self._value = compiled, value
        self._generation = compiled._generation
        self._shape = tuple(compiled._plan.graph.shape(value))

    @property
    def shape(self): return self._shape

    def numpy(self):
        with self._compiled._lock:
            if self._generation != self._compiled._generation:
                raise RuntimeError("result storage was reused by a later call; call .numpy() before replay to retain a copy")
            return self._compiled._plan.read(self._value)

    def item(self): return self.numpy().item()

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("MuNet results require a host copy; use result.numpy()")
        return np.asarray(self.numpy(), dtype=dtype)


@dataclass(frozen=True)
class TensorSpec:
    """A named, fixed-shape FP32 input or flattened tensor output."""
    name: str
    shape: tuple
    dtype: str = "float32"


class Compiled:
    """One guarded static-shape specialization, with persistent device state."""
    def __init__(self, fn=None, *, device="vulkan", fuse=True):
        if fn is not None and not callable(fn):
            raise TypeError("compile expects a callable or nn.Module")
        if not isinstance(device, str) or not (device in ("cpu", "vulkan") or
                device.startswith("vulkan:") and device[7:].isascii() and device[7:].isdigit()):
            raise ValueError("device must be cpu, vulkan, or vulkan:<nonnegative index>")
        self.fn, self.device, self.fuse = fn, device, fuse
        self._plan = None
        self._generation = 0
        self._parameters = {}
        self._seen_versions = {}
        self._lock = threading.RLock()

    def _capture(self, args):
        # A mode change replaces the plan; detach state owned by the previous plan first.
        if self._plan is not None:
            for p, value in self._parameters.items():
                if p._owner is not None and p._owner[0] is self:
                    p._array = self._plan.read(value)
                    p._owner = None
        ctx = _Trace()
        token = _active.set(ctx)
        try:
            symbolic = [Tensor(ctx.graph, ctx.graph.leaf("input", f"input_{i}", a.shape)) for i, a in enumerate(args)]
            outputs = self.fn(*symbolic)
            self._single = isinstance(outputs, Tensor)
            output_list = []
            self._output_tree = _flatten_outputs(outputs, output_list)
            if not output_list: raise TypeError("compiled function must return at least one Tensor")
            ids = [t._resolve().value for t in output_list]
        finally:
            _active.reset(token)
        self._plan = _native.Plan(ctx.graph, ids, ctx.updates, self.fuse)
        self._parameters = ctx.parameters
        self._modules = ctx.modules
        self._requires_grad = {p:p.requires_grad for p in ctx.parameters}
        self._optimizers = getattr(ctx, "optimizers", [])
        self._seen_versions = {p: p._version for p in ctx.parameters}
        self._input_specs = [a.shape for a in args]
        # Dead inputs still have caller guards, but are not uploaded or executed.
        self._feed_indices = [int(ctx.graph.name(i).split("_")[1]) for i in self._plan.inputs]
        try:
            _device(self._plan, self.device)
        except Exception:
            self._plan = None
            raise

    def __call__(self, *args):
        if _active.get() is not None:
            raise RuntimeError("nested compiled calls are not supported; call the underlying model while tracing")
        arrays = self._arrays(args)
        with self._lock:
            self._prepare(arrays)
            for optimizer in getattr(self, "_optimizers", []): optimizer._sync_hyperparameters()
            for p, value in self._parameters.items():
                if p._version != self._seen_versions[p]:
                    self._plan.write(value, np.ascontiguousarray(p.numpy()))
                    self._seen_versions[p] = p._version
            self._plan.run([arrays[i] for i in self._feed_indices])
            self._generation += 1
            updated = {dst for dst, _ in self._plan.updates}
            for optimizer in getattr(self, "_optimizers", []): optimizer._sync_hyperparameters()
            for p, value in self._parameters.items():
                if value in updated:
                    p._version += 1
                    p._owner = self, value
                    self._seen_versions[p] = p._version
            values = [Result(self, v) for v in self._plan.outputs]
            return _unflatten_outputs(self._output_tree, values) if hasattr(self, "_output_tree") else values[0] if self._single else tuple(values)

    @staticmethod
    def _arrays(args):
        arrays = []
        for index, a in enumerate(args):
            if hasattr(a, "detach"):
                raise TypeError("convert torch inputs explicitly with .detach().cpu().numpy(); zero-copy GPU interop is not implemented")
            a = np.asarray(a)
            if a.dtype != np.float32:
                raise TypeError(f"input {index} requires float32, received {a.dtype}; use np.asarray(value, dtype=np.float32)")
            arrays.append(a if a.flags.c_contiguous else np.ascontiguousarray(a))
        return arrays

    def _prepare(self, arrays):
        if (self._plan is None or any(m.training != training for m, training in getattr(self, "_modules", {}).items())
                    or any(p.requires_grad != flag for p,flag in getattr(self,"_requires_grad",{}).items())):
            if self.fn is None:
                raise RuntimeError("a loaded program cannot be recaptured; load an export with the required shape")
            # Preparation can replace output storage even without executing a step.
            self._generation += 1
            self._capture(arrays)
        if [a.shape for a in arrays] != self._input_specs:
            raise ValueError(f"shape guard failed: expected {self._input_specs}, got {[a.shape for a in arrays]}; create a new compiled specialization")

    def prepare(self, *example_inputs):
        """Capture/compile without executing or applying optimizer updates; return self."""
        if _active.get() is not None:
            raise RuntimeError("cannot prepare a compiled program inside another trace")
        with self._lock:
            self._prepare(self._arrays(example_inputs))
        return self

    def predict(self, *inputs):
        """Execute and copy all tensor outputs to NumPy, preserving output containers."""
        def host(value):
            if isinstance(value, Result): return value.numpy()
            if isinstance(value, dict): return {k: host(v) for k, v in value.items()}
            if isinstance(value, tuple): return tuple(host(v) for v in value)
            if isinstance(value, list): return [host(v) for v in value]
            return value
        with self._lock:
            return host(self(*inputs))

    def save(self, path, *, include_vulkan=None):
        """Save a prepared program; Vulkan programs embed deployment shaders by default."""
        from .serialization import save
        save(self, path, include_vulkan=include_vulkan)

    @property
    def inputs(self):
        if self._plan is None: raise RuntimeError("call prepare(example_inputs) or execute the program first")
        names = getattr(self, "_input_names", [f"input_{i}" for i in range(len(self._input_specs))])
        return tuple(TensorSpec(n, tuple(s)) for n, s in zip(names, self._input_specs))

    @property
    def outputs(self):
        if self._plan is None: raise RuntimeError("call prepare(example_inputs) or execute the program first")
        names = getattr(self, "_output_names", [f"output_{i}" for i in range(len(self._plan.outputs))])
        return tuple(TensorSpec(n, tuple(self._plan.graph.shape(i))) for n, i in zip(names, self._plan.outputs))

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
