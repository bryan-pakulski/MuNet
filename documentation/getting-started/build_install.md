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
