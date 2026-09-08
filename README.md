# MuNet

A greenfield, experimental C++17 graph compiler for Vulkan training and inference, with a small PyTorch-style Python interface.

**MuNet is the building blocks for creating neural networks:** tensors, layers, automatic differentiation, optimizers, graph compilation, device execution and model interchange. The installed library has NumPy as its only Python runtime dependency. Model architectures, datasets and training recipes belong in examples or applications.

**Start here:** [Python API guide](docs/api/python.md) · [C++ inference API guide](docs/api/cpp.md) · [Install/setup](docs/install.md).
Use `make demo-python-api VULKAN=0` for a complete training/export tour, or
`make demo-cpp VULKAN=0` to train in Python and run the exported model in C++.

Version 0.3.0 adds the operations needed to build and train the [RT-DETR example](examples/rtdetr/README.md). That model is an acceptance test and reference application under `examples/rtdetr/`; it is not packaged in the library. Full COCO convergence and physical-device performance are not yet established. The runtime retains the `munet-nn` PyPI project and `munet_nn` import name; `import munet` is also supported. This is a breaking rewrite of the 0.1 API.

The C++ core owns graph validation, shape inference, symbolic reverse-mode differentiation, fusion, temporary-buffer planning, CPU reference execution, GLSL generation, and Vulkan execution. Python constructs models, captures a static graph once, invokes the shader compiler on cache misses, and exposes model conversion and serialization.

## What runs today

- Float32 tensors with positive static dimensions, rank up to eight.
- Convolution and pooling with backward, batched matrix multiplication, normalization, attention, differentiable bilinear sampling, indexing, TopK and broadcast arithmetic.
- PyTorch-style modules, persistent train/eval buffers, SGD, AdamW parameter groups, gradient clipping and EMA.
- A compiled forward/loss/backward/optimizer/EMA step and native assignment operation. Parameters remain in Vulkan memory between calls; `.item()`, `.numpy()`, saving, and explicit interop perform readbacks.
- Single-consumer scalar-operation fusion, including view expressions feeding matrix multiplication; temporary buffers reused after their final consumers.
- GLSL → SPIR-V compilation on demand, a persistent SPIR-V cache, Vulkan pipeline creation, and recorded command-buffer replay.
- Data-only `.mnet` save/load, including training state; native graph → ONNX → native graph conversion for the supported subset.
- Import of supported eval-mode PyTorch modules, including the pinned RT-DETR reference, through the modern ONNX exporter. Matching native layers accept PyTorch state dictionaries.
- An explicit CPU reference backend. Selecting Vulkan never silently runs native operators on the CPU.
- A disconnect-tolerant training swarm: one durable owner, C++ worker binaries, capability-based leases, cached offline computation, result retries, and sample-weighted MSE/SGD rounds.

Read the [RT-DETR example](examples/rtdetr/README.md), [installation and releases](docs/install.md), [the architecture and decisions](docs/architecture.md), [training swarm guide](docs/swarm.md), [RT-DETR acceptance plan](docs/rtdetr-acceptance.md), and [validation report](docs/validation.md).

## Install

After 0.3.0 is published, the existing PyPI project installs the library and both swarm commands:

```bash
python -m pip install --upgrade 'munet-nn>=0.3.0'
munet-node --version
munet-server --version
```

Release workflows build Linux x86-64/aarch64 wheels, standalone node/server archives, and a CMake SDK. Preview artifacts are available from the PR's `Wheels` run. See [installation and release details](docs/install.md).

## Build and run

For local development and testing, use the Makefile (Linux, Python 3.10+):

```bash
sudo apt-get install build-essential python3-dev python3-venv libvulkan-dev glslang-tools vulkan-validationlayers libcurl4-openssl-dev libssl-dev
make setup
make test
make test-vulkan
```

Use `make setup VULKAN=0` and `make test VULKAN=0` for a CPU-only build.
`make build` rebuilds native code after edits; `make smoke` runs a short training
example. Commands use `.venv` and the source tree automatically, without shell
activation. `make install` also installs the library and node/server commands
into `.venv` for use outside the checkout. See `make help` and the
[local development instructions](docs/install.md#local-development-with-make)
for prerequisites, targeted tests and configuration overrides.

On Ubuntu with a working Vulkan driver:

```bash
sudo apt-get install build-essential libvulkan-dev glslang-tools libcurl4-openssl-dev libssl-dev
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[interop,test]'
python examples/train_mlp.py --device vulkan --steps 200
```

`pip` builds the C++ extension using CMake and pybind11. The native runtime loads the host Vulkan loader on first GPU use; ONNX and PyTorch are conversion/development dependencies, not execution engines. Install the GPU vendor's Vulkan driver separately. Linux software testing can use `mesa-vulkan-drivers`.

For an explicit CPU-only build:

```bash
python -m pip install '.[interop,test]' -Ccmake.define.MUNET_VULKAN=OFF
python examples/train_mlp.py --device cpu
```

For manual source development without rebuilding a wheel:

```bash
python -m pip install cmake pybind11 numpy pytest onnx onnxruntime scipy torch onnxscript
python tools/build.py                  # add --cpu-only when needed
PYTHONPATH=python python -m pytest -q
MUNET_TEST_VULKAN=1 PYTHONPATH=python python -m pytest -q
# With vulkan-validationlayers installed:
PYTHONPATH=python python tools/test.py --vulkan-validation
```

The opted-in Vulkan tests fail when the driver/compiler is unavailable. They do not skip or select the CPU instead. `tools/build.py --cmake-arg=-DVulkan_INCLUDE_DIR=...` accepts custom SDK paths.

## RT-DETR example

The model, matching/loss recipe, image and COCO handling, training loop, and
acceptance tests live together under `examples/rtdetr/`. Set it up separately:

```bash
make setup-rtdetr
make test-rtdetr
PYTHONPATH=python .venv/bin/python -m examples.rtdetr.train --help
```

Use `from examples.rtdetr import RTDETR` from the repository root. There is no
`munet.models` or `munet_nn.models` API. See the [example guide](docs/rtdetr.md)
for training, checkpoint/resume, inference and evaluation commands. `make setup`
builds the library without downloading PyTorch or the RT-DETR reference; `make
test` installs the optional numerical-reference tools and runs library tests.

## Python training

```python
import numpy as np
import munet as mu

rng = np.random.default_rng(7)
model = mu.nn.Sequential(
    mu.nn.Linear(4, 16, rng=rng),
    mu.nn.ReLU(),
    mu.nn.Linear(16, 2, rng=rng),
)
optimizer = mu.optim.SGD(model.parameters(), lr=0.03)
train_step = mu.train_step(model, optimizer, mu.nn.MSELoss(), device="cpu")

x = rng.normal(size=(32, 4)).astype(np.float32)
y = rng.normal(size=(32, 2)).astype(np.float32)
for step in range(100):
    loss = train_step(x, y)
    if step % 20 == 0:
        print(loss.item())             # Explicit scalar readback.
print(train_step.stats())

predict = model.eval().compile()
print(predict.predict(x))             # Owned NumPy outputs.
model.export("model.mnet", x[:1])     # Fixed batch-one inference artifact.
print(mu.load("model.mnet").predict(x[:1]))
```

CPU is the default across Python compile/load/import and C++ inference. Select
`device="vulkan"` explicitly to execute on a GPU. This changes the prototype's
previous implicit Vulkan default. Use `@mu.compile` for custom training steps;
see the [Python guide](docs/api/python.md) for losses, layers, tensors, checkpoints,
named exports and a troubleshooting table.

The first call captures Python control flow and compiles the whole step. Subsequent calls replay the C++ execution plan. Python side effects inside the function happen only while tracing. Tensor-dependent Python branches are rejected. Input shapes and float32 dtype are guarded. Model structure and closure constants are static: create a new compiled function after changing them. Train/eval changes recapture automatically; AdamW group settings remain live between replays.

This initial API is tracing-only; it does not yet provide general eager tensor execution. Each compiled object has one specialization and one in-flight execution. A returned result borrows output storage: call `.numpy()` before the next invocation if you need to retain a copy. Stale results raise an error. Compiled calls are serialized; concurrent mutation of parameters shared across compiled objects is unsupported.

## C++ application inference

Export your trained Python model, then use the installed SDK directly:

```cpp
#include <munet/inference.hpp>

munet::Model model("model.mnet");
auto outputs = model.run({munet::Tensor{{1, 4}, {1.f, 2.f, 3.f, 4.f}}});
// outputs[0].data owns its FP32 values and survives later inference calls.
```

Link `MuNet::inference` through `find_package(MuNet CONFIG REQUIRED)`.
`make sdk VULKAN=0` installs a local SDK; `make demo-cpp VULKAN=0` builds and runs
the complete example. Vulkan exports can embed precompiled shaders with
`include_vulkan=True`, so the C++ deployment host needs neither Python nor a
shader compiler. See the [C++ guide](docs/api/cpp.md) for named inputs, device
selection, ownership, concurrency and compatibility limits.

## Conversion

```python
from munet.interop import to_onnx, from_onnx, from_torch

inference = mu.compile(model, device="vulkan")
prediction = inference(x).numpy()
mu.save(inference, "model.mnet")

restored = mu.load("model.mnet", device="vulkan")
to_onnx(restored, "model.onnx")
imported = from_onnx("model.onnx", device="vulkan")
mu.save(imported, "roundtrip.mnet")

# Optional: pip install '.[torch]'
# imported = from_torch(torch_model.eval(), (example_torch_tensor,), device="vulkan")
```

ONNX support is **default-domain opsets 13–18**, static float32 computation, and the operators listed in `munet.interop.SUPPORTED_ONNX`. Integer constants are allowed for shapes/axes; bounded indices/masks use exact float32 storage. Batched MatMul and general permutations are supported. Local functions are inlined. Unsupported nodes/modes, dynamic inputs, external tensor data, sparse tensors and ONNX training metadata raise explicit errors. See the detector guide for the precise contract.

Export produces a standard inference graph and preserves supported computation, not original Python classes or exact graph topology. Native training programs must be exported by compiling their trained model separately. A supported inference graph does not reconstruct the original training behavior of an arbitrary PyTorch model.

`mu.save(train_step, "step.mnet")` and `mu.load(...)` preserve compiled computation and current device state. `munet.checkpoint.save_state/load_state` supplies data-only state archives for application training loops; the RT-DETR example's `DetectorTrainer` uses it for model, optimizer, EMA, scheduler and RNG resume. This is a new versioned archive format; no reader for the previous MuNet format is included.

## Training swarm

Prepare a job from a native model and finite float32 arrays:

```python
from munet.swarm import create_job

create_job(model, x, y, "training-job", global_batch_size=32,
           micro_batch_size=4, epochs=10, lr=0.03, include_vulkan=True)
```

Build and launch native workers:

```bash
sudo apt-get install libcurl4-openssl-dev libssl-dev
cmake -S . -B build-node -DMUNET_PYTHON=OFF -DMUNET_SWARM_NODE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-node -j

# Supply the same MUNET_SWARM_TOKEN (at least 32 characters) to each process.
python -m munet.swarm training-job --port 8765
# In another terminal:
./build-node/munet-node --owner http://127.0.0.1:8765 --state node-a --device vulkan:0
```

The owner assigns chunks from the same checkpoint, combines gradients by sample count, and applies one SGD update per global batch. Nodes can disappear, reconnect, or restart using their state directories. Lost work is leased again; duplicate results never apply twice. Each node needs a full model replica. The current loss is MSE, and models must be deterministic and independent across samples.

The worker needs no Python or shader compiler. SPIR-V is prepared on the owner, and the driver compiles pipelines on each device. This Linux/POSIX prototype uses JSON transport and explicit gradient/checkpoint transfers; it has not established large-model or WAN performance. Longer offline local-SGD and fast multi-GPU collectives remain separate future modes. See [the swarm guide](docs/swarm.md) for TLS, deployment, restart rules, constraints and the runnable MLP example.

## Runtime limits

This version uses baseline float32 kernels, not tuned GEMM/convolution kernels. Matrix multiplication is a simple dot-product kernel; reductions are serial within each output invocation. Do not infer competitive throughput from the correctness results.

One device-local buffer holds the entire planned arena, with a host staging mirror. Graph/weight metadata also consumes host memory. `arena_bytes` reports only the planned device arena, not total process memory. Tensor-sized descriptors allow the total arena to exceed `maxStorageBufferRange`; individual tensors must fit it. Two-dimensional dispatches respect device workgroup limits. Host input arrays are copied into staging. One replay can be in flight, with a fence wait before its storage is reused. No CPU wait occurs between individual kernels.

Remaining work includes mixed precision, tiled/cooperative-matrix kernels, autotuning, general eager execution, asynchronous staging, fast multi-GPU collectives and physical platform validation. The detector has a bounded shape-specialization cache; switching cached programs synchronizes shared state through the host. The swarm still implements its narrow MSE/SGD contract, and does not yet train RT-DETR across workers. See the [detector guide](docs/rtdetr.md) and [coverage inventory](docs/rtdetr-coverage.json).

## Configuration

| Setting | Meaning |
|---|---|
| `device="vulkan:N"` | Select a physical Vulkan device by index from `mu.devices()` |
| `device="cpu"` | Explicit C++ reference execution |
| `fuse=False` | Disable expression fusion for numerical/debug comparisons |
| `MUNET_VULKAN_LIBRARY` | Optional explicit host Vulkan loader path |
| `MUNET_GLSLANG` | Absolute path to `glslangValidator` |
| `MUNET_CACHE_DIR` | Persistent SPIR-V cache directory |
| `MUNET_TEST_VULKAN=1` | Require Vulkan execution in the test matrix |
| `MUNET_SWARM_TOKEN` | Shared owner/node admission secret supplied through the environment |
| `MUNET_SWARM_NODE` | Path to a built worker executable for integration tests |
| `MUNET_TEST_SWARM=1` | Require native swarm integration tests; fail if the worker binary is missing |

The SPIR-V cache contains baseline Vulkan 1.1 modules keyed by generated source and shader compiler version. Device pipelines are built and retained per compiled object; their driver-specific cache is not persisted yet.
