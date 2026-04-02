# Build & Install

## Install with pip (recommended)

### Development (editable)

```bash
python -m pip install -e .
```

### Wheel build + local install

```bash
python -m pip install build
python -m build --wheel
python -m pip install dist/munet-*.whl
```

## Local CMake build (without pip packaging)

```bash
make build-release
```

## Run tests

```bash
make unit-test
make py-test
```

## Run docs locally (interactive search)

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Then open the local URL printed by MkDocs (usually `http://127.0.0.1:8000`).


## Optional runtime bundles (extras)

MuNet ships a single `munet-nn` wheel. Extras install dependency bundles only; they do **not** select a different wheel binary.

```bash
# Vulkan runtime/tooling dependency bundle
pip install "munet-nn[vk]"

# CUDA 12 + Vulkan dependency bundle
pip install "munet-nn[cu12-vk]"

# CUDA 13 + Vulkan dependency bundle
pip install "munet-nn[cu13-vk]"
```

If accelerator runtime dependencies are missing at execution time, `import munet_nn` still succeeds and `munet_nn.backend_status()` reports actionable diagnostics.
