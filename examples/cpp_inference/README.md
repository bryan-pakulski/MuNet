# Train in Python, infer in C++

From the repository root:

```bash
make demo-cpp
```

This trains a tiny four-feature regressor, exports
`artifacts/cpp-inference/model.mnet`, installs a C++ SDK under `artifacts/sdk`,
builds `main.cpp` against `MuNet::inference`, and runs the standalone executable.
Compare its two printed values with `artifacts/cpp-inference/model.expected.txt`.

Vulkan is the default. Install Vulkan headers, a shader compiler and a driver.
The authoring step embeds shaders;
the C++ application needs the driver, but no Python runtime or shader compiler.
For the explicit CPU fallback use `make demo-cpp DEVICE=cpu`, or
`make demo-cpp VULKAN=0` to also disable Vulkan in the build.

`export_model.py` demonstrates training with `munet.train_step` and named model
export. `main.cpp` demonstrates loading once, supplying an owned FP32 tensor,
running named inputs/outputs, and reporting errors. Its four input values are
fixed for the demonstration; replace them with your application's preprocessing.

See the [Python guide](../../docs/api/python.md) and
[C++ guide](../../docs/api/cpp.md) for the full API and standalone build commands.
