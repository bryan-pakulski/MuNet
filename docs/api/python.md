# Python API

MuNet is a graph compiler with a PyTorch-style model-building interface. Create
models with `nn.Module`, then run them through a compiled callable. NumPy arrays
are the host inputs and outputs. Tensor math, gradients and optimizer updates run
in the C++ backend.

**Start with `model.eval().compile().predict(x)` for inference and
`mu.train_step(model, optimizer, loss)` for ordinary supervised training.**
Use `@mu.compile` when you need a custom training step. MuNet does not currently
provide eager tensor math outside a trace.

## Setup and first inference

From a checkout, use `make setup`. For an installed wheel use
`python -m pip install /path/to/munet_nn-...whl`. See [installation](../install.md)
for platform requirements, published versions and Vulkan drivers.

Run source examples with `PYTHONPATH=python .venv/bin/python your_script.py`, or
run `make install` to install into `.venv` for use outside the checkout.
Both `import munet` and `import munet_nn` expose the same public API.

```python
import numpy as np
import munet as mu

mu.manual_seed(7)
model = mu.nn.Sequential(
    mu.nn.Linear(4, 16),
    mu.nn.ReLU(),
    mu.nn.Linear(16, 2),
)
x = np.ones((8, 4), dtype=np.float32)
predict = model.eval().compile()  # Vulkan is the default; create this once.
scores = predict.predict(x)      # Owned NumPy array, shape (8, 2).
print(scores.argmax(axis=-1))
print(model)                     # Readable layer structure.
```

Do not recreate the compiled object inside your batch loop. Its first call
captures the graph; later calls replay it with persistent parameter/device state.
Vulkan is selected automatically. Use `device="vulkan:1"` on `compile`,
`train_step`, `load` or model imports to select a different GPU. `mu.devices()` enumerates Vulkan device display names
in index order; it raises if the Vulkan loader/build is unavailable. CPU execution
and importing the library do not load a GPU driver. Select `device="cpu"`
explicitly for CPU fallback; a Vulkan failure never silently changes backends.
For CPU-only setup use `make setup VULKAN=0`, then pass `device="cpu"` in Python.
Make demos select CPU automatically when invoked with `VULKAN=0`.

## Train a classifier

`train_step` combines zeroing gradients, forward/loss, backward, optional clipping
and one optimizer update. Data loading and batching remain ordinary Python.

```python
optimizer = mu.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
step = mu.train_step(model, optimizer, mu.nn.CrossEntropyLoss(), max_grad_norm=1.0)

rng = np.random.default_rng(7)
for update in range(100):
    x = rng.normal(size=(8, 4)).astype(np.float32)
    labels = (x[:, 0] > 0).astype(np.float32)  # Class indices use FP32 storage.
    loss = step(x, labels).item()
    if update % 20 == 0:
        print(update, loss)

predict = model.eval().compile()
scores = predict.predict(x)
```

The model is set to training mode when the training step captures. Mode changes
between training and inference cause recapture where necessary. Create separate
compiled objects for training and inference; they share the current model state.
Do not mutate or execute the same parameters concurrently across these objects.

`train_step` takes one model input and one target array. For multiple model
inputs, several losses, EMA, or other custom updates, write the step explicitly:

```python
@mu.compile(device="vulkan")
def custom_step(x, target):
    optimizer.zero_grad()
    loss = mu.nn.functional.cross_entropy(model(x), target)
    loss.backward()
    mu.optim.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss

model.train()
loss = custom_step(x, labels).item()
```

For regression, substitute `mu.nn.MSELoss()`. For binary segmentation, use
`mu.nn.BCEWithLogitsLoss()` with target and logits of the same shape. These losses
take raw logits/predictions; do not apply softmax before cross entropy or sigmoid
before BCE with logits. For sequence logits `(batch, tokens, classes)`, use
`mu.nn.CrossEntropyLoss(dim=-1)` with `(batch, tokens)` targets.

`examples/python_api.py` is a complete runnable training/checkpoint/export tour:

```bash
make demo-python-api
```

## Define your own model

Store layers and parameters on a `Module`, and implement `forward`. MuNet discovers
state recursively, including `ModuleList`, `Sequential`, lists and dictionaries.
Parameters keep PyTorch-compatible layer layouts, for example linear weights
`(out_features, in_features)` and convolution weights `(out_channels, in_channels/groups, kh, kw)`.

```python
class Classifier(mu.nn.Module):
    def __init__(self):
        self.features = mu.nn.Sequential(
            mu.nn.Conv2d(1, 8, 3, padding=1),
            mu.nn.ReLU(),
            mu.nn.AvgPool2d(2),
            mu.nn.Flatten(),
        )
        self.head = mu.nn.Linear(8 * 14 * 14, 10)

    def forward(self, images):
        return self.head(self.features(images))

model = Classifier()
images = np.zeros((4, 1, 28, 28), np.float32)
logits = model.eval().compile().predict(images)
```

`mu.manual_seed(seed)` seeds subsequent layer initialization and dropout RNGs.
It does not reseed existing modules or NumPy data generators. Layers also accept
an explicit `rng=np.random.default_rng(seed)` where listed below.

## Compile, execute and inspect

| API | Behavior |
|---|---|
| `mu.compile(fn, *, device="vulkan", fuse=True)` | Create a `Compiled`; also works as `@mu.compile(...)` |
| `model.compile(*, device="vulkan", fuse=True)` | Compile a module; retains its train/eval mode |
| `program.prepare(*example_inputs)` | Capture/compile without execution or optimizer updates; returns the program |
| `program(*inputs)` | Execute; return `Result` tensors in the function's original output containers |
| `program.predict(*inputs)` | Execute and immediately copy every tensor output to NumPy |
| `program.inputs`, `program.outputs` | Tuples of `TensorSpec(name, shape, dtype="float32")`, available after prepare/execution/load |
| `program.stats()` | Compilation state, device name, run/dispatch/transfer counters and memory planning statistics |
| `program.synchronize()` | Wait for pending work on this program |
| `program.save(path, *, include_vulkan=None)` | Save a prepared/executed program; Vulkan programs embed shaders by default |
| `mu.train_step(model, optimizer, loss_fn, *, device="vulkan", max_grad_norm=None, fuse=True)` | Return a compiled supervised update callable |

`prepare` performs capture (including Python side effects during capture), but
does not execute the native graph. In particular, optimizer update counts and
model parameters do not advance. A graph may allocate optimizer state during
capture. Use it to separate compilation from a timed or latency-sensitive loop.

`predict` preserves dict/list/tuple nesting and constant metadata. For example:

```python
@mu.compile
def outputs(x):
    return {"scores": x.softmax(-1), "summary": (x.mean(), "example")}

result = outputs.predict(np.ones((2, 3), np.float32))
print(result["scores"].shape)  # (2, 3)
```

Ordinary `program(...)` returns borrowed `Result` objects. `result.numpy()`
performs an explicit host read and returns an owned array; `result.item()` reads
a one-element result; `np.asarray(result)` also makes a host copy. Read a borrowed
result before the next execution or recapture on the same program. Otherwise it
raises a stale-storage error. Prefer `predict()` when retaining output arrays.
`predict()` does not change model mode or strip updates from a training program.

Each compiled object has one static input specialization. To use a different batch
size or image/context shape, create another compiled object or export another
artifact. Inputs must be contiguous-compatible FP32 arrays; noncontiguous FP32
arrays are copied automatically. Convert other dtypes explicitly:

```python
x = np.asarray(raw_values, dtype=np.float32)
# For optional PyTorch interoperability:
# x = torch_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
```

Python control flow runs during capture. Branching on a tensor value is rejected;
use `mu.where` for tensor selection. Changing closure constants or model structure
requires a new compiled object. Mode and `requires_grad` changes are detected.
Nested compiled calls are rejected: call the underlying model inside a trace.

## Export, load and deploy

Export selects eval mode temporarily and restores each module's previous mode.
It captures without running inference or training updates. Supply representative
FP32 inputs to define the fixed signature:

```python
example = np.zeros((1, 4), np.float32)
model.export("classifier.mnet", example,
             input_names=["features"], output_names=["logits"])

inference = mu.load("classifier.mnet")
print(inference.inputs)
logits = inference.predict(example)
```

The preceding export assumes the introductory four-feature model. The same API
works with your own model and its own shapes. For a callable, use
`mu.export(fn, path, example_inputs, ...)`. A single array means one input; a
tuple/list contains positional input arrays. Names are optional, unique nonempty
strings; defaults are `input_0`, `output_0`, etc. Output names label flattened
tensor leaves in traversal order, while Python load preserves the original
container structure. `export` returns the destination `pathlib.Path`.

Export embeds shaders by default for C++ Vulkan deployment (equivalent to
`include_vulkan=True`):

```python
model.export("classifier-gpu.mnet", example, include_vulkan=True,
             input_names=["features"], output_names=["logits"])
```

The authoring machine needs `glslangValidator` (`glslang-tools` on Debian/Ubuntu),
or set `MUNET_GLSLANG` to its executable. Export captures a host graph without
executing the model and can prepare Vulkan shaders without a GPU. For an explicit
CPU-only export without a shader compiler, pass `include_vulkan=False`. The deployment application needs a Vulkan-enabled
SDK and a working driver, but no Python or shader compiler. Embedded kernels are
checked against the runtime's generated source; re-export after incompatible
compiler changes. See the [C++ guide](cpp.md).

`mu.save(program, path, *, include_vulkan=None)` is the function form of
`program.save`. With `include_vulkan=None`, Vulkan programs embed shaders and
CPU programs omit them; pass `True` or `False` to override. It preserves whatever
the compiled program does, including
training updates. `mu.export` is the inference-only path and rejects updates.
`mu.load(path, *, device="vulkan", fuse=None)` returns a `Compiled`. With `fuse=None`
it preserves the artifact's fusion setting (legacy files use `True`). Explicitly
changing fusion may require shader recompilation on Vulkan.

## Resume training and import weights

A model inference file is not the Python training loop. Save application state
separately with the data-only checkpoint API:

```python
from munet.checkpoint import save_state, load_state

save_state({"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": rng.bit_generator.state,
            "updates": 100}, "training.mnet")

state = load_state("training.mnet")
model.load_state_dict(state["model"])
optimizer.load_state_dict(state["optimizer"])
rng.bit_generator.state = state["rng"]
```

Recreate the same architecture and optimizer parameter order before loading.
Also save your scheduler, EMA, counters and data-sampling state when used. AdamW
has optimizer state; plain SGD currently has only a fixed learning rate and does
not expose `state_dict`. Checkpoint trees accept dictionaries, lists/tuples,
JSON scalar values and FP32 NumPy arrays; tuples reload as lists. No pickle or
arbitrary Python objects are loaded.

`model.load_state_dict(state, strict=True)` accepts matching NumPy arrays and
PyTorch tensors; strict mode checks missing/extra names and shapes. Model state
includes parameters and persistent buffers. It does not encode the Python model
class or its constructor configuration.

Optional ONNX/PyTorch graph conversion:

```python
from munet.interop import from_onnx, from_torch, to_onnx

# imported = from_onnx("source.onnx", device="vulkan")
# imported = from_torch(torch_model.eval(), (torch_example,), device="vulkan")
# imported.save("for-cpp.mnet", include_vulkan=True)
# to_onnx(inference, "classifier.onnx")
```

Install the `interop` or `torch` extra for conversion. These are authoring tools,
not native execution dependencies. Conversion supports a defined static FP32
subset, not arbitrary ONNX/PyTorch programs; unsupported operators/modes raise
`UnsupportedOperatorError`. `munet.interop.SUPPORTED_ONNX` lists accepted ONNX
operators. See [model interchange](../../README.md#conversion) for opset details.

## Module and layer reference

`Module` provides `train(mode=True)`, `eval()`, `parameters()`,
`named_parameters()`, `buffers()`, `named_buffers()`, `named_children()`,
`modules()`, `state_dict()`, `load_state_dict(state, strict=True)`,
`requires_grad_(enabled=True)` and `register_buffer(name, value, persistent=True)`.
`Parameter(data, requires_grad=True)` and `Buffer(data, persistent=True)` own FP32
state. Use `.assign(array)` to replace values without changing shape, and `.numpy()`
to copy current state to the host. Parameter `.grad` is available inside a trace
after backward; return it from the compiled function if you need to inspect it.

| Layer | Constructor / contract |
|---|---|
| `Linear` | `(in_features, out_features, bias=True, *, rng=None)`; operates on the last axis |
| `Conv2d` | `(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, *, rng=None)`; NCHW |
| `MaxPool2d` | `(kernel_size, stride=None, padding=0, ceil_mode=False)` |
| `AvgPool2d` | Same plus `count_include_pad=True` |
| `BatchNorm2d` | `(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)`; persistent running state |
| `FrozenBatchNorm2d` | `(num_features, eps=1e-5)`; fixed normalization state |
| `LayerNorm` | `(normalized_shape, eps=1e-5, elementwise_affine=True, bias=True)`; trailing dimensions |
| `Embedding` | `(num_embeddings, embedding_dim, padding_idx=None, *, rng=None)`; exact FP32 indices |
| `MultiheadAttention` | `(embed_dim, num_heads, dropout=0.0, bias=True, batch_first=True, *, rng=None)`; batch-first only |
| `Dropout` | `(p=0.5, *, rng=None)`; disabled in eval mode |
| `Sequential` | `(*layers)` or a single ordered mapping; forwards through each layer |
| `ModuleList` | `(layers=())`; iterable, indexable, `.append(layer)` |
| `Flatten` | `(start_dim=1, end_dim=-1)`; retains batch by default |
| `Identity`, `ReLU`, `Sigmoid`, `SiLU`, `GELU` | No constructor arguments |
| `MSELoss` | No arguments; mean squared error |
| `CrossEntropyLoss` | `(dim=1, reduction="mean")`; FP32 class indices, logits class axis selected by `dim` |
| `BCEWithLogitsLoss` | `(reduction="mean")`; elementwise binary targets |

For attention, `attention(query, key, value, attn_mask=None,
key_padding_mask=None, need_weights=False)` returns `(output, weights_or_none)`.
Nonzero mask values **block** attention; masks are not additive score biases.
Embedding/class indices must be exact integers in range, represented as FP32.
Keep every sequence position's attention row with at least one unmasked key.

`from munet.nn import functional as F` provides `mse_loss`, `cross_entropy`,
`binary_cross_entropy_with_logits`, `relu`, `sigmoid`, `silu`, `gelu`, `softmax`,
`conv2d`, `max_pool2d`, `avg_pool2d`, `grid_sample` and `interpolate`.
`F.cross_entropy(logits, targets, dim=1, reduction="mean")` and BCE accept
`reduction="none"`, `"sum"` or `"mean"`. `grid_sample` supports bilinear,
zero-padding, `align_corners=False`; `interpolate` currently supports nearest.

## Tensor operations inside a trace

`Tensor` is symbolic: callers normally receive one as a compiled-function argument,
not by invoking its internal graph/id constructor. `mu.as_tensor(value)` creates
a constant inside a trace. Shapes are positive static dimensions, rank at most
eight; scalars have shape `()`.

| Category | Operations |
|---|---|
| Arithmetic | `+`, `-`, `*`, `/`, `@`, unary `-`, `square`, `abs`, `minimum`, `maximum`, `clamp(min=None, max=None)` |
| Functions | `relu`, `sigmoid`, `tanh`, `gelu`, `softplus`, `exp`, `log`, `sqrt`, `erf`, `sin`, `cos`, `floor` |
| Reductions | `sum(dim=None, keepdim=False)`, `mean(...)`, `amax(...)`, `softmax(dim=-1)` |
| Shapes | `shape`, `ndim`, `reshape(*shape)`, `flatten(start_dim=0, end_dim=-1)`, `unsqueeze`, `squeeze`, `expand`, `repeat`, `contiguous` |
| Layout | `permute(*axes)`, `transpose(dim0, dim1)`; `.T` swaps a matrix's axes |
| Indexing | Integer/slice indexing, `split(sections, dim=0)`, `gather(dim, index)`, `take(index, dim=0)`, `topk(k, dim=-1)` |
| Comparisons | `eq`, `<`, `<=`, `>`, `>=`; return FP32 masks |
| Composition | `mu.cat(tensors, dim=0)`, `mu.stack(tensors, dim=0)`, `mu.where(condition, a, b)` |
| Gradients | Scalar `loss.backward()`, `mu.grad(loss, tensors)`, `detach()` |

Broadcasting is supported where the native operation allows it. Slicing uses
positive steps; advanced/boolean NumPy indexing is not a general API. `topk`
returns values and indices. Nonnegative integer powers use multiplication;
other powers use `exp(log(x) * power)` and require positive inputs. Tensor
`.numpy()`/`.item()` cannot read a symbolic value during capture.

## Optimizers and performance controls

| API | Supported parameters |
|---|---|
| `optim.SGD` | `(parameters, lr=0.01)`; plain SGD, no momentum/state dictionary |
| `optim.AdamW` | `(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`; parameter groups and state dictionaries |
| `optim.clip_grad_norm_` | `(parameters, max_norm, norm_type=2.0)`; invoke inside the compiled update |
| `optim.MultiStepLR` | `(optimizer, milestones, gamma=0.1)`; call `.step(epoch=None)` outside the trace; save/restore its state |
| `optim.ModelEMA` | `(model, decay=0.9999, warmups=2000)`; `.update()` inside a trace, `.copy_to(model)`, state dictionaries |

AdamW group values remain live between replays; the scheduler uses optimizer
parameter groups. SGD's learning rate is captured in the compiled graph: rebuild
that step to change it. Frozen parameters are excluded from updates.

On Vulkan, model state stays on the device between calls. Host inputs are uploaded;
`.item()`, `.numpy()`, `.predict()`, saving and state dictionaries read back data.
Batch work and avoid unnecessary host reads for throughput. `MUNET_CACHE_DIR`
selects the shader cache; `MUNET_GLSLANG` selects the compiler;
`MUNET_VULKAN_LIBRARY` selects the loader. These are process settings, not model
arguments. The current backend is FP32, static-shape and single-device per program;
there is no mixed precision, eager CUDA interop or zero-copy external device buffer API.

## Common mistakes

| Symptom | Fix |
|---|---|
| Calling `model(numpy_array)` raises | Create `model.eval().compile()` once, then call `.predict(array)` |
| Input dtype error | Use `np.asarray(value, dtype=np.float32)`, including index/label tensors |
| Shape guard fails on the last batch | Pad the final batch or create a second specialization for that shape |
| Output storage was reused | Call `.numpy()` before replay, or use `.predict()` |
| Save requires a prepared program | Use `model.export(path, example)` or `program.prepare(example).save(path)` |
| C++ rejects a checkpoint/update graph | Export the model's eval-mode forward graph, not the training step/state archive |
| Vulkan asks for a compiler at deployment | Export with `include_vulkan=True`, preserving its fusion setting |
| Python branch cannot use a Tensor | Use tensor operations/`where`, or move host decisions outside the trace |

See the [C++ API](cpp.md) for deployment and the [RT-DETR source example](../../examples/rtdetr/README.md)
for a larger model assembled from these same building blocks.
