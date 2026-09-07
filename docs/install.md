# Installation and releases

MuNet keeps the existing **`munet-nn` PyPI project**. `pip install munet_nn` and `pip install munet-nn` refer to that same project. The clean-slate implementation starts at 0.2.0; the old 0.1 tensor API, CUDA backends and archive format are removed. `import munet_nn as mu` exposes the new API, and `import munet as mu` is also supported. Both names share the same runtime and tensor classes.

## Install the library, node and server together

After a 0.2.0 release has been published:

```bash
python -m pip install --upgrade 'munet-nn>=0.2.0'
munet-node --version
munet-server --version
```

The release wheels include the C++ library extension, Python API, native `munet-node` executable and `munet-server` command. The node launcher replaces its Python process with the packaged C++ executable. The server command runs the Python coordinator. There is no separate node/server PyPI project or extra needed for swarm use.

The pipeline targets Linux x86-64 and aarch64, CPython 3.10–3.14, with a manylinux/glibc 2.28 baseline. It builds one CPU/Vulkan wheel per Python/platform combination. Other operating systems, musl/Alpine, 32-bit ARM and mobile packaging remain future targets. Architecture build coverage does not establish physical-GPU support or performance on that hardware.

CPU execution and owner startup do not require Vulkan installed. GPU execution requires a compatible **host Vulkan loader and driver**. Vulkan is loaded only when selected; an unavailable GPU raises an error without silently executing on the CPU. `MUNET_VULKAN_LIBRARY` can specify an explicit loader path before its first use.

For compiling new Vulkan graphs or preparing Vulkan swarm jobs, install `glslangValidator` on the authoring machine (`glslang-tools` on Debian/Ubuntu), or set `MUNET_GLSLANG`. Swarm nodes receive precompiled SPIR-V and do not need a shader compiler.

Optional Python tooling:

```bash
python -m pip install 'munet-nn[interop]>=0.2.0' # ONNX conversion
python -m pip install 'munet-nn[torch]>=0.2.0'   # PyTorch export/import tooling
python -m pip install 'munet-nn[vk]>=0.2.0'      # Retained ONNX/ONNX Runtime extra
```

The `vk` extra retains its previous tooling meaning; it does not install a GPU driver or select a different wheel. The old `cu12-vk` and `cu13-vk` extras are removed because this implementation has no CUDA backend.

## Standalone binaries and C++ SDK

Tagged releases publish these archives for each Linux architecture:

| Artifact | Contents | Entry point |
|---|---|---|
| `munet-node-VERSION-linux-ARCH.tar.gz` | C++ worker plus repaired libcurl/OpenSSL dependencies | `bin/munet-node` |
| `munet-server-VERSION-linux-ARCH.tar.gz` | Owner executable with bundled CPython, NumPy and native extension | `bin/munet-server` |
| `munet-sdk-VERSION-linux-ARCH.tar.gz` | Static C++ core, public header and CMake package | `MuNet::core` |
| `SHA256SUMS-linux-ARCH` | Checksums of the three matching archives | `sha256sum --check` |

Download the matching files from [GitHub Releases](https://github.com/bryan-pakulski/MuNet/releases), verify their checksums, and extract them into a directory you own. Add the extracted `bin` directory to `PATH`. Keep the archive's directory structure: the node uses relative library paths and the server needs its `_internal` directory. Neither standalone distribution requires Python to be installed separately. The server bundles Python; it is not a rewritten C++ coordinator.

For example, after downloading both node/server archives and their checksum file for version 0.2.0 and x86-64:

```bash
# Download the SDK archive too before checking the complete checksum file.
sha256sum --check SHA256SUMS-linux-x86_64
tar -xzf munet-node-0.2.0-linux-x86_64.tar.gz
tar -xzf munet-server-0.2.0-linux-x86_64.tar.gz
./munet-node-0.2.0-linux-x86_64/bin/munet-node --version
./munet-server-0.2.0-linux-x86_64/bin/munet-server --version
```

For the SDK, extract its archive and point CMake at that prefix:

```cmake
find_package(MuNet CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE MuNet::core)
```

Include `<munet/core.hpp>`. Configure with `-DCMAKE_PREFIX_PATH=/path/to/munet-sdk-0.2.0-linux-x86_64`. The SDK is a C++ ABI package; consumers need a compatible target/toolchain and C++17. Build from source when that contract differs from your system. Release CI compiles and runs an independent consumer using only the installed SDK.

## Run a swarm

Prepare a job using the Python API or `examples/swarm_mlp.py prepare JOB`. Set the same `MUNET_SWARM_TOKEN` of at least 32 characters in the owner and node environments, then:

```bash
munet-server JOB --port 8765
# Another terminal/process:
munet-node --owner http://127.0.0.1:8765 --state node-a --device vulkan:0
```

Use `--device cpu` for a CPU reference worker. Each worker gets its own persistent state directory. Remote owners require HTTPS or an explicitly configured private network with `--allow-http`. `MUNET_CA_BUNDLE` can supply a private CA certificate bundle; certificate and hostname verification remain enabled. The binary otherwise selects the host's standard Linux trust store. See [swarm.md](swarm.md) for retries, offline work, storage bounds and model constraints.

## Preview artifacts from a pull request

The `Wheels` workflow builds and smoke-tests artifacts on pull requests and pushes to `master`. Download the `wheels-linux-*` or `native-linux-*` artifacts from that workflow run. Install a matching preview wheel directly with `python -m pip install /path/to/munet_nn-0.2.0-...whl`. This is how to test the replacement before publishing; the existing PyPI release stays unchanged until a release tag is pushed.

## Publishing

The existing `.github/workflows/wheels.yml` path, `munet-nn` project identity and trusted-publishing jobs are retained. No new PyPI project, token secret or GitHub environment is introduced. The existing configured PyPI/TestPyPI trusted publisher must remain valid for this repository and workflow. Public PyPI attestations identify this workflow as the package's existing publisher. [Current PyPI project](https://pypi.org/project/munet-nn/), [PyPI trusted publisher configuration](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

Before releasing, update the matching versions in `pyproject.toml` and `python/munet/__init__.py`. CMake derives the native version from `pyproject.toml`. Run `python tools/release_version.py` and the validation gates.

| Tag | Example | Destination |
|---|---|---|
| `v<VERSION>` | `v0.2.0` | Existing production PyPI project and GitHub release downloads |
| `dev<VERSION>` | `dev0.2.1.dev1` | Existing TestPyPI project; package version must be `0.2.1.dev1` |

Tags must match the canonical package version exactly. The prior `v0.1.2a`/`0.1.2` mismatch is not carried forward. Pull requests and manual workflow runs build artifacts and never publish. Publishing requires the matching pushed tag plus successful numerical, fault-recovery, installed-wheel, frozen-server and C++ SDK gates. No package or GitHub release is published merely by merging the clean-slate PR.

## Source builds

```bash
sudo apt-get install build-essential cmake libvulkan-dev glslang-tools libcurl4-openssl-dev libssl-dev
python -m pip install .

# Explicit CPU-only build; source package still includes the node and server commands.
python -m pip install . -Ccmake.define.MUNET_VULKAN=OFF

# Native worker and SDK without Python build dependencies:
cmake -S . -B build-native -DMUNET_PYTHON=OFF -DMUNET_SWARM_NODE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-native -j
cmake --install build-native --prefix /path/to/install
```

To build standalone archives locally, use a clean Python 3.12 environment with a shared `libpython`, install the newly built/repaired wheel plus `pyinstaller==6.22.2`, `cmake` and `packaging`, then run `python tools/build_release.py --wheel /path/to/wheel.whl`. For distributable Linux binaries, use the workflow's matching manylinux container and its distribution-provided Python 3.12. The static `/opt/python` interpreters build wheels but cannot freeze the server. Building on a newer host raises the minimum glibc requirement. Each archive records its build host, version and architecture in `manifest.json`.
