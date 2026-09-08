# MuNet

A greenfield, experimental C++17 graph compiler for Vulkan training and inference, with a small PyTorch-style Python interface.

**Version 0.3.0 adds native RT-DETR v1 training and inference.** It includes the R50-vd model, detection losses, AdamW/EMA, resumable training and model interchange. This remains an experimental correctness baseline; full COCO convergence and physical-device performance are not yet established. The greenfield runtime retains the `munet-nn` PyPI project and `munet_nn` import name. The new API also uses `import munet`; this is a breaking rewrite of the 0.1 API.

The C++ core owns graph validation, shape inference, symbolic reverse-mode differentiation, fusion, temporary-buffer planning, CPU reference execution, GLSL generation, and Vulkan execution. Python constructs models, captures a static graph once, invokes the shader compiler on cache misses, and exposes model conversion and serialization.

## What runs today

- Float32 tensors with positive static dimensions, rank up to eight.
- Convolution and pooling with backward, batched matrix multiplication, normalization, attention, differentiable bilinear sampling, indexing, TopK and broadcast arithmetic.
- PyTorch-style modules, persistent train/eval buffers, SGD, AdamW parameter groups, gradient clipping and EMA.
- A compiled forward/loss/backward/optimizer/EMA step, including native Hungarian assignment and RT-DETR denoising and auxiliary losses. Parameters remain in Vulkan memory between calls; `.item()`, `.numpy()`, saving, and explicit interop perform readbacks.
- Single-consumer scalar-operation fusion, including view expressions feeding matrix multiplication; temporary buffers reused after their final consumers.
- GLSL → SPIR-V compilation on demand, a persistent SPIR-V cache, Vulkan pipeline creation, and recorded command-buffer replay.
- Data-only `.mnet` save/load, including training state; native graph → ONNX → native graph conversion for the supported subset.
- Import of supported eval-mode PyTorch modules, including the pinned RT-DETR reference, through the modern ONNX exporter. Matching native layers accept PyTorch state dictionaries.
- An explicit CPU reference backend. Selecting Vulkan never silently runs native operators on the CPU.
- A disconnect-tolerant training swarm: one durable owner, C++ worker binaries, capability-based leases, cached offline computation, result retries, and sample-weighted MSE/SGD rounds.

Read the [RT-DETR guide and runnable examples](docs/rtdetr.md), [installation and releases](docs/install.md), [the architecture and decisions](docs/architecture.md), [training swarm guide](docs/swarm.md), [RT-DETR acceptance plan](docs/rtdetr-acceptance.md), and [validation report](docs/validation.md).

## Install

After 0.3.0 is published, the existing PyPI project installs the library and both swarm commands:

```bash
python -m pip install --upgrade 'munet-nn>=0.3.0'
munet-node --version
munet-server --version
```

Release workflows build Linux x86-64/aarch64 wheels, standalone node/server archives, and a CMake SDK. Preview artifacts are available from the PR's `Wheels` run. See [installation and release details](docs/install.md).

## Build and run

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

For editable development without rebuilding a wheel:

```bash
python -m pip install cmake pybind11 numpy pytest onnx onnxruntime scipy Pillow torch onnxscript
python tools/fetch_rtdetr_reference.py
python tools/build.py                  # add --cpu-only when needed
PYTHONPATH=python python -m pytest -q
MUNET_TEST_VULKAN=1 PYTHONPATH=python python -m pytest -q
# With vulkan-validationlayers installed:
PYTHONPATH=python python tools/test.py --vulkan-validation
```

The opted-in Vulkan tests fail when the driver/compiler is unavailable. They do not skip or select the CPU instead. `tools/build.py --cmake-arg=-DVulkan_INCLUDE_DIR=...` accepts custom SDK paths.

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

@mu.compile(device="vulkan:0")
def train_step(x, target):
    optimizer.zero_grad()
    loss = mu.nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()
    return loss

x = rng.normal(size=(32, 4)).astype(np.float32)
y = rng.normal(size=(32, 2)).astype(np.float32)
for step in range(100):
    loss = train_step(x, y)
    if step % 20 == 0:
        print(loss.item())             # Explicit scalar readback.
print(train_step.stats())
```

The first call captures Python control flow and compiles the whole step. Subsequent calls replay the C++ execution plan. Python side effects inside the function happen only while tracing. Tensor-dependent Python branches are rejected. Input shapes and float32 dtype are guarded. Model structure and closure constants are static: create a new compiled function after changing them. Train/eval changes recapture automatically; AdamW group settings remain live between replays.

This initial API is tracing-only; it does not yet provide general eager tensor execution. Each compiled object has one specialization and one in-flight execution. A returned result borrows output storage: call `.numpy()` before the next invocation if you need to retain a copy. Stale results raise an error. Compiled calls are serialized; concurrent mutation of parameters shared across compiled objects is unsupported.

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

`mu.save(train_step, "step.mnet")` and `mu.load(...)` preserve compiled computation and current device state. Use `DetectorTrainer.save/load` for model, optimizer, EMA, scheduler and host RNG resume. This is a new versioned archive format; no reader for the previous MuNet format is included.

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
