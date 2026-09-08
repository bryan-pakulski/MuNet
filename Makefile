.DEFAULT_GOAL := help
# Native builds install into the same source tree; serialize multi-target invocations.
.NOTPARALLEL:

PYTHON ?= python3
VENV ?= .venv
VULKAN ?= 1
BUILD_DIR ?= build/local
CMAKE_ARGS ?=
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cpu
DEVICE ?= $(if $(filter 0,$(VULKAN)),cpu,vulkan)
EXAMPLE_ARGS ?=

VENV_PYTHON := $(abspath $(VENV))/bin/python
RUN := env PYTHONPATH="$(CURDIR)/python" MUNET_SWARM_NODE="$(abspath $(BUILD_DIR))/munet-node" "$(VENV_PYTHON)"
TEST_VULKAN := $(if $(filter vulkan%,$(DEVICE)),1,0)
TEST_FLAGS := $(if $(filter 1,$(TEST_VULKAN)),--vulkan-validation,)
BUILD_FLAGS := --build-dir="$(BUILD_DIR)" --cmake-arg=-DMUNET_SWARM_NODE=ON --cmake-arg=-DMUNET_INSTALL_SDK=OFF
ifeq ($(VULKAN),0)
BUILD_FLAGS += --cpu-only
endif
ifeq ($(filter 0 1,$(VULKAN)),)
$(error VULKAN must be 0 or 1)
endif

.PHONY: help setup deps build test-deps rtdetr-deps setup-rtdetr reference test test-vulkan test-rtdetr test-rtdetr-vulkan smoke install clean
.PHONY: example-deps setup-examples test-examples test-examples-vulkan demo-mnist demo-segmentation demo-language-model

help:
	@printf '%s\n' \
	  'make setup           Create .venv and build Vulkan library/node (no PyTorch or model downloads)' \
	  'make setup VULKAN=0  Explicit CPU fallback; no Vulkan headers or driver' \
	  'make build           Rebuild native code after edits (incremental)' \
	  'make test            Install reference-test tools and run Vulkan/CPU/swarm tests' \
	  'make test-vulkan     CPU + Vulkan tests with API/synchronization validation (VULKAN=1)' \
	  'make setup-rtdetr    Set up the optional RT-DETR example and image dependencies' \
	  'make test-rtdetr     Run RT-DETR example acceptance tests with Vulkan validation' \
	  'make test-rtdetr-vulkan  Run example acceptance tests with Vulkan validation' \
	  'make setup-examples  Build and install optional image dependencies for the small examples' \
	  'make demo-mnist      Train a CNN on automatically downloaded MNIST' \
	  'make demo-segmentation  Train a small U-Net on generated shapes and masks (offline)' \
	  'make demo-language-model  Train a causal Transformer on downloaded Tiny Shakespeare' \
	  'make test-examples   Test small examples, checkpoint resume and inference (offline)' \
	  'make test-examples-vulkan  Run small example tests with Vulkan validation' \
	  'make smoke           Run the small MLP training example (Vulkan default; DEVICE=cpu fallback)' \
	  'make install         Install library and node/server commands into the virtual environment' \
	  'make clean           Clean native build outputs; keep dependencies, data and checkpoints' \
	  '' \
	  'Overrides: PYTHON, VENV, VULKAN=0|1, BUILD_DIR, CMAKE_ARGS, TORCH_INDEX_URL, DEVICE, EXAMPLE_ARGS' \
	  'Use the same overrides for later commands. See docs/install.md for system prerequisites.'

$(VENV_PYTHON):
	"$(PYTHON)" -m venv "$(VENV)"

# Keep the development dependencies aligned with pyproject.toml and ci.yml.
$(VENV)/.munet-deps: pyproject.toml Makefile | $(VENV_PYTHON)
	"$(VENV_PYTHON)" -m pip install --upgrade pip
	"$(VENV_PYTHON)" -m pip install 'cmake>=3.20' 'pybind11>=3.0' 'scikit-build-core>=0.10' 'numpy>=1.24' 'pytest>=8' 'onnx>=1.16'
	touch "$@"

$(VENV)/.munet-test-deps: pyproject.toml Makefile | deps
	"$(VENV_PYTHON)" -m pip install 'onnxruntime>=1.18' 'onnxscript>=0.3' 'scipy>=1.10'
	"$(VENV_PYTHON)" -m pip install 'torch>=2.6' --index-url "$(TORCH_INDEX_URL)"
	touch "$@"

$(VENV)/.munet-rtdetr-deps: examples/rtdetr/requirements.txt | deps
	"$(VENV_PYTHON)" -m pip install -r examples/rtdetr/requirements.txt
	touch "$@"

$(VENV)/.munet-example-deps: examples/requirements.txt | deps
	"$(VENV_PYTHON)" -m pip install -r examples/requirements.txt
	touch "$@"

deps: $(VENV)/.munet-deps
test-deps: $(VENV)/.munet-test-deps
rtdetr-deps: $(VENV)/.munet-rtdetr-deps
example-deps: $(VENV)/.munet-example-deps

setup: build
setup-rtdetr: build rtdetr-deps
setup-examples: build example-deps

build: deps
	"$(VENV_PYTHON)" tools/build.py $(BUILD_FLAGS) $(CMAKE_ARGS)

reference: $(VENV_PYTHON)
	"$(VENV_PYTHON)" examples/rtdetr/fetch_reference.py

test: build test-deps
	MUNET_TEST_VULKAN=$(TEST_VULKAN) $(RUN) tools/test.py $(TEST_FLAGS) --swarm

test-vulkan:
	@test "$(VULKAN)" = 1 || { printf '%s\n' 'Use make test-vulkan VULKAN=1 to build and test the Vulkan runtime.'; exit 1; }
	$(MAKE) build test-deps
	$(RUN) tools/test.py --vulkan-validation --swarm

test-rtdetr: build test-deps rtdetr-deps reference
	MUNET_TEST_VULKAN=$(TEST_VULKAN) $(RUN) tools/test.py $(TEST_FLAGS) examples/rtdetr/tests

test-rtdetr-vulkan:
	@test "$(VULKAN)" = 1 || { printf '%s\n' 'Use make test-rtdetr-vulkan VULKAN=1.'; exit 1; }
	$(MAKE) build test-deps rtdetr-deps reference
	$(RUN) tools/test.py --vulkan-validation examples/rtdetr/tests

test-examples: build example-deps
	MUNET_TEST_VULKAN=$(TEST_VULKAN) $(RUN) tools/test.py $(TEST_FLAGS) examples/tests

test-examples-vulkan:
	@test "$(VULKAN)" = 1 || { printf '%s\n' 'Use make test-examples-vulkan VULKAN=1.'; exit 1; }
	$(MAKE) build example-deps
	$(RUN) tools/test.py --vulkan-validation examples/tests

demo-mnist: setup-examples
	$(RUN) -m examples.mnist.train --device "$(DEVICE)" $(EXAMPLE_ARGS)

demo-segmentation: setup-examples
	$(RUN) -m examples.segmentation.train --device "$(DEVICE)" $(EXAMPLE_ARGS)

demo-language-model: build
	$(RUN) -m examples.language_model.train --device "$(DEVICE)" $(EXAMPLE_ARGS)

smoke: build
	$(RUN) examples/train_mlp.py --device "$(DEVICE)" --steps 20

# Normal installation for scripts outside this checkout; rerun after source edits.
install: deps
	"$(VENV_PYTHON)" -m pip install --no-build-isolation --no-deps . -Ccmake.define.MUNET_VULKAN=$(if $(filter 0,$(VULKAN)),OFF,ON)

clean:
	@if test -f "$(BUILD_DIR)/CMakeCache.txt"; then "$(VENV_PYTHON)" -m cmake --build "$(BUILD_DIR)" --target clean; fi
