# C++ inference API

Include `<munet/inference.hpp>` and link `MuNet::inference`. `munet::Model` loads
the same `.mnet` inference graph exported by Python, validates its signature,
and runs it on CPU or Vulkan. The application does not embed Python, NumPy,
PyTorch or ONNX Runtime. JSON/archive support is compiled into the SDK; its
third-party header is not part of your application's API.

## Build or install the SDK

Release SDK archives contain the static library, public headers and CMake package.
Extract one and use its directory as `CMAKE_PREFIX_PATH`. See
[release artifacts](../install.md#standalone-binaries-and-c-sdk) for platform/ABI
requirements. The source SDK needs C++17 or later and CMake 3.20 or later:

Installed SDKs include these guides under `share/doc/munet/docs/api` and the
example sources under `share/doc/munet/examples/cpp_inference`.

```bash
# Standalone CPU SDK: no Python or swarm/network dependencies.
cmake -S . -B build/sdk -DCMAKE_BUILD_TYPE=Release \
  -DMUNET_PYTHON=OFF -DMUNET_SWARM_NODE=OFF -DMUNET_VULKAN=OFF
cmake --build build/sdk --parallel 4
cmake --install build/sdk --prefix "$PWD/artifacts/sdk"
```

For Vulkan, set `-DMUNET_VULKAN=ON` and provide Vulkan development headers when
building. The deployed application needs the host Vulkan loader/driver. On Linux
the loader is opened on demand: CPU execution does not require a driver even
when the SDK was built with Vulkan enabled. The SDK is a C++ ABI package; use a
compatible compiler/standard library, or build it for your target toolchain.

The Make convenience targets use the project's development environment:

```bash
make sdk VULKAN=0       # Installs to artifacts/sdk; SDK_PREFIX can override it.
make test-sdk VULKAN=0  # Exports a fixture, compiles and runs an installed-SDK consumer.
make demo-cpp VULKAN=0  # Full Python-training → export → C++ inference example.
```

## Train/export in Python, run in C++

The complete example is under `examples/cpp_inference/`. It trains a four-feature
regressor in Python and produces `model.mnet` plus a Python reference prediction:

```bash
PYTHONPATH=python .venv/bin/python examples/cpp_inference/export_model.py
cmake -S examples/cpp_inference -B build/cpp-inference \
  -DCMAKE_PREFIX_PATH="$PWD/artifacts/sdk"
cmake --build build/cpp-inference --parallel 2
./build/cpp-inference/infer artifacts/cpp-inference/model.mnet cpu
cat artifacts/cpp-inference/model.expected.txt
```

Only model authoring/export uses Python. The resulting `infer` executable accepts
the `.mnet` file directly. Its two predictions should match the reference within
normal FP32 rounding tolerance. To export your own trained model:

```python
model.export("model.mnet", np.zeros((1, 4), np.float32),
             input_names=["features"], output_names=["prediction"])
```

Export captures eval-mode inference without executing it. Input shapes include
batch size and are fixed in the artifact. Image decoding, resizing, normalization,
tokenization and task-specific postprocessing belong to your application.

## Minimal application

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_application LANGUAGES CXX)
find_package(MuNet CONFIG REQUIRED)
add_executable(my_application main.cpp)
target_link_libraries(my_application PRIVATE MuNet::inference)
```

`main.cpp`, for the four-feature regressor exported above:

```cpp
#include <munet/inference.hpp>
#include <iostream>

int main() {
    try {
        munet::Model model("model.mnet");  // CPU by default; load once.
        munet::Tensor features{{1, 4}, {1.f, 2.f, 3.f, 4.f}};
        auto outputs = model.run_named({{"features", features}});
        for (float value : outputs.at("prediction").data)
            std::cout << value << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
```

You can also use `add_subdirectory(path/to/MuNet)` and link `MuNet::inference`.
Set `MUNET_PYTHON=OFF`/`MUNET_SWARM_NODE=OFF` before adding the subdirectory if
you need only the SDK. The same public include paths work in source and installed
builds. Existing consumers linking `MuNet::core` continue to work.

## Types and methods

All public inference types live in namespace `munet`.

| Type/member | Contract |
|---|---|
| `Shape` | `std::vector<int64_t>`; positive dimensions, rank ≤ 8; `{}` for a scalar |
| `Tensor` | Aggregate with `Shape shape` and `std::vector<float> data`; contiguous row-major FP32 |
| `TensorInfo` | `std::string name` and `Shape shape`; signature metadata |
| `ModelOptions::device` | `"cpu"` (default), `"vulkan"`, or `"vulkan:N"` |
| `ModelOptions::max_memory_bytes` | Default 2 GiB; independently bounds archive bytes and planned arena bytes, not total process memory |
| `Model(path, options={})` | Load/validate a model and initialize its selected backend; throws on error |
| `inputs() const`, `outputs() const` | `const std::vector<TensorInfo>&`; immutable signature metadata owned by the model |
| `run(const std::vector<Tensor>&)` | Return `std::vector<Tensor>` in output signature order |
| `run_named(const std::map<std::string, Tensor>&)` | Return `std::map<std::string, Tensor>` by output name |
| `stats() const` | `std::map<std::string, uint64_t>` of run/transfer/memory/kernel counters |
| `device_name() const` | Human-readable active backend/device name |
| `synchronize()` | Wait for this model's queued work |

The model is movable and noncopyable. Do not call methods on a moved-from model.
It owns its weights, execution plan and device state until destruction.

### Positional and named inputs

Use `inputs()`/`outputs()` to inspect the model before supplying buffers:

```cpp
for (const auto& input : model.inputs()) {
    std::cout << input.name << ':';
    for (auto dimension : input.shape) std::cout << ' ' << dimension;
    std::cout << '\n';
}
auto outputs = model.run({munet::Tensor{{1, 4}, {1.f, 2.f, 3.f, 4.f}}});
```

Positional order is Python's exported argument order. Named input keys must match
the signature exactly: missing, extra and misspelled names are errors. Shape and
element count are checked before execution, including inputs that compilation
determined were unused. Scalars have one data element and an empty shape.

Default names are `input_0`, `output_0`, etc. Set names explicitly during export
for an application contract. For a Python dict/tuple/list output, C++ returns its
**flattened tensor leaves** in traversal order. Constant Python metadata is not
returned. For example, `{"logits": logits, "extra": (boxes, "label")}` has two
tensor outputs; export `output_names=["logits", "boxes"]` to name them.

### Ownership, transfers and concurrency

`Tensor::data` owns its memory. Returned outputs stay valid after later calls,
model moves or model destruction. The API does not retain a caller's input
buffers. It copies host inputs into the native execution path and returns host
output copies. On Vulkan these imply uploads and readbacks; weights and
intermediates remain in the model's device arena between calls.

Calls on a single `Model` serialize internally, including collecting its outputs.
Use one model per independently concurrent execution stream and budget for each
instance's weights/arena. This API does not expose asynchronous futures, imported
`VkBuffer`s, external command buffers or a zero-copy video pipeline. `run` has
collected its outputs before returning; an extra `synchronize()` is normally
unnecessary after it.

## Vulkan deployment without a compiler

Generate a deployment artifact on the authoring machine:

```bash
PYTHONPATH=python .venv/bin/python examples/cpp_inference/export_model.py --include-vulkan
```

Then use a Vulkan-enabled SDK and select the device in your application:

```cpp
munet::ModelOptions options;
options.device = "vulkan:0";
munet::Model model("model.mnet", options);
```

`include_vulkan=True` bundles SPIR-V and matching kernel sources in the `.mnet`.
Export needs `glslangValidator`; deployment does not invoke an external compiler.
The same artifact can run on CPU. For GPU execution, the generated sources must
match the SDK compiler's plan exactly; an incompatible SDK reports that the model
needs re-export. Shader portability does not guarantee every model fits every
GPU's descriptor, buffer, workgroup and memory limits. The backend checks device
capabilities and reports unsupported workloads; it never silently runs them on CPU.

Use `munet::vulkan_built()` and `munet::vulkan_devices()` from `<munet/core.hpp>`
for explicit capability discovery. The latter may throw when the loader/driver
is unavailable. Linux `MUNET_VULKAN_LIBRARY` can select the loader before first
GPU use. Backend/device selection is per model.

## Errors and compatibility

Load errors include the model path and a reason. The loader rejects unsupported
versions/dtypes/operators, inconsistent shapes, malformed/duplicate ZIP entries,
bad checksums or NPY payloads, invalid input mappings and mismatched shaders.
It reads files in memory and never extracts archive paths. The supported ZIP/NPY
encoding is the one emitted by `munet.save/export`; repacking with compression is
not supported by the native loader.

`.mnet` training-state checkpoints and compiled programs containing updates are
rejected. Load a forward-only export for inference. To deploy an ONNX/PyTorch
model, convert with the optional Python importer and save its supported inference
graph; the C++ SDK does not import arbitrary ONNX directly.

The format currently bounds files at 2 GiB, graphs at 100,000 nodes, dependency
depth at 1,024 in the C++ loader, and individual tensors by the native 32-bit
indexing contract. `max_memory_bytes` can lower the archive/arena bounds; raising
it can allow a larger arena but does not remove format limits. Host copies,
graph metadata and Vulkan driver allocation add to total memory use. There are
no dynamic dimensions or mixed precision. See [the file contract](model-format.md).

## Low-level graph API

Advanced callers can construct graphs directly using `<munet/core.hpp>` and
`MuNet::core`. This exposes compiler primitives rather than a C++ neural-network
layer hierarchy. For example:

```cpp
munet::Graph graph;
auto x = graph.leaf("input", "features", {2});
auto bias = graph.leaf("constant", "bias", {2}, {0.5f, 1.f});
auto y = graph.op("add", {x, bias});
munet::Plan plan(graph, {y}, {}, true);
plan.run({{2.f, 3.f}});
auto values = plan.read(y);  // {2.5f, 4.f}, owned vector.
```

`Graph::leaf(kind, name, shape, data={})` accepts `input`, `parameter`, `constant`.
`Graph::op(kind, inputs, attrs={})` validates/infer shapes;
`Graph::gradients(loss, wrt)` constructs reverse-mode graph nodes.
`Plan(graph, outputs, updates, fuse=true)` compiles an execution plan. Updates
are pairs of destination parameter IDs and source value IDs.
`Plan::inputs` lists live input IDs in execution order. `run(feeds)` consumes that
order; `read(id)` returns output/parameter host data, and `write(id, values)` updates
parameter state. `enable_vulkan(spirv, device_index)` attaches compiled kernels.
`stats`, `device_name` and `synchronize` expose execution information.

Treat `Graph`/`Plan` fields as immutable once a plan is created. Direct `Plan`
callers own synchronization and output lifetime discipline; unlike `Model`,
`Plan` itself does not serialize concurrent callers. The graph operation/attribute
encoding is an advanced, evolving compiler interface. For normal application
deployment, use `Model` with an exported artifact and named tensor signatures.
