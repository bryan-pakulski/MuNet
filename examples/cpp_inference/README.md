# Train in Python, infer in C++

From the repository root:

```bash
make demo-cpp VULKAN=0
```

This trains a tiny four-feature regressor, exports
`artifacts/cpp-inference/model.mnet`, installs a C++ SDK under `artifacts/sdk`,
builds `main.cpp` against `MuNet::inference`, and runs the standalone executable.
Compare its two printed values with `artifacts/cpp-inference/model.expected.txt`.

For Vulkan, install the build headers, shader compiler and a driver, then run
`make demo-cpp VULKAN=1 DEVICE=vulkan`. The authoring step embeds shaders;
the C++ application needs the driver, but no Python runtime or shader compiler.

`export_model.py` demonstrates training with `munet.train_step` and named model
export. `main.cpp` demonstrates loading once, supplying an owned FP32 tensor,
running named inputs/outputs, and reporting errors. Its four input values are
fixed for the demonstration; replace them with your application's preprocessing.

See the [Python guide](../../docs/api/python.md) and
[C++ guide](../../docs/api/cpp.md) for the full API and standalone build commands.
