# Shared inference file contract

`.mnet` program version 1 is an uncompressed ZIP archive containing
`manifest.json` and little-endian, row-major FP32 NumPy tensor files. The manifest
identifies `format: "munet-program"`, `version: 1`, `dtype: "float32"`, a
topologically ordered graph, output IDs, update pairs, caller input shapes and
the live-input mapping. Tensor leaves reference `tensors/<node-id>.npy`.

Python reads/writes the archive with standard-library ZIP/JSON support and NumPy.
The C++ inference reader contains its own bounded ZIP/NPY reader and bundled JSON
parser; it does not depend on a Python installation or execute Python code.

Optional signature/deployment metadata added by the usability API:

| Field | Meaning |
|---|---|
| `input_names` | Unique caller argument names, in `input_specs` order |
| `output_names` | Unique names for the flattened tensor outputs |
| `output_tree` | Python container/constant reconstruction; C++ validates it but returns tensor leaves only |
| `fuse` | Fusion setting used to generate the saved plan |
| `vulkan` | Ordered list of `{source: "shaders/N.comp", spirv: "shaders/N.spv"}` entries |

Legacy version-1 files without names use `input_N`/`output_N`; without `fuse`
they use fusion. Ordinary inference files remain CPU-loadable in C++. Missing
`vulkan` is an explicit GPU deployment error for C++; Python may JIT compile
ordinary inference files when a compiler is installed.

Embedded SPIR-V is executable GPU code and should come from a trusted authoring
pipeline. The SDK checks file integrity, SPIR-V framing and matching generated
kernel sources; these are compatibility/corruption checks, not a security sandbox
or a cryptographic signature proving that shader code implements that source.
Model artifacts are deployment inputs, like other executable model formats.

The native loader supports stored ZIP entries, Python's ZIP64 local tensor headers
and ZIP64 directory counts. It rejects compression, encryption, multi-disk files,
duplicate/overlapping entries and CRC failures. No archive entry is extracted.
NumPy payloads must use the canonical dtype/layout/shape headers emitted by the
MuNet writer, with exact payload byte counts.

Format bounds are 2 GiB of serialized content, 100,000 graph nodes and 32 MiB of
manifest data. Shader entries count toward archive limits. The C++ loader also
bounds dependency depth at 1,024 and allows an application to lower archive/arena
limits through `ModelOptions`. The format contract is FP32 with fixed positive
dimensions, rank at most eight. CPU and GPU have the same mathematical graph;
normal floating-point differences may occur across implementations/devices.

`munet-training-state` is a different data-only checkpoint format, also conventionally
named `.mnet`. It stores application state trees, not executable inference. Python
compiled training programs can also use `munet-program`, but contain updates;
the C++ inference loader intentionally rejects them. Use `munet.export` for deployment.
