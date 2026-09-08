.DEFAULT_GOAL := help
# Native builds install into the same source tree; serialize multi-target invocations.
.NOTPARALLEL:

PYTHON ?= python3
VENV ?= .venv
VULKAN ?= 1
BUILD_DIR ?= build/local
CMAKE_ARGS ?=
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cpu
DEVICE ?= cpu
SDK_PREFIX ?= artifacts/sdk
SDK_BUILD_DIR ?= build/sdk-local
SDK_CMAKE_ARGS ?=

VENV_PYTHON := $(abspath $(VENV))/bin/python
RUN := env PYTHONPATH="$(CURDIR)/python" MUNET_SWARM_NODE="$(abspath $(BUILD_DIR))/munet-node" "$(VENV_PYTHON)"
BUILD_FLAGS := --build-dir="$(BUILD_DIR)" --cmake-arg=-DMUNET_SWARM_NODE=ON --cmake-arg=-DMUNET_INSTALL_SDK=OFF
ifeq ($(VULKAN),0)
BUILD_FLAGS += --cpu-only
endif
ifeq ($(filter 0 1,$(VULKAN)),)
$(error VULKAN must be 0 or 1)
endif

.PHONY: help setup deps build test-deps rtdetr-deps setup-rtdetr reference test test-vulkan test-rtdetr test-rtdetr-vulkan smoke install clean
.PHONY: sdk test-sdk demo-python-api demo-cpp

help:
	@printf '%s\n' \
	  'make setup           Create .venv and build the library/node (no PyTorch or model downloads)' \
	  'make setup VULKAN=0  Set up without Vulkan headers or a GPU driver' \
	  'make build           Rebuild native code after edits (incremental)' \
	  'make sdk             Build/install the C++ SDK into artifacts/sdk (no Python runtime)' \
	  'make test-sdk        Compile and run an application against the installed SDK' \
	  'make demo-python-api Run the Python training/checkpoint/export API tour' \
	  'make demo-cpp        Train/export in Python and run inference in a C++ application' \
	  'make test            Install reference-test tools and run library CPU/swarm tests' \
	  'make test-vulkan     CPU + Vulkan tests with API/synchronization validation (VULKAN=1)' \
	  'make setup-rtdetr    Set up the optional RT-DETR example and image dependencies' \
	  'make test-rtdetr     Run RT-DETR example acceptance tests on CPU' \
	  'make test-rtdetr-vulkan  Run example acceptance tests with Vulkan validation' \
	  'make smoke           Run the small MLP training example (DEVICE=cpu or vulkan)' \
	  'make install         Install library and node/server commands into the virtual environment' \
	  'make clean           Clean native build outputs; keep dependencies, data and checkpoints' \
	  '' \
	  'Overrides: PYTHON, VENV, VULKAN=0|1, BUILD_DIR, CMAKE_ARGS, TORCH_INDEX_URL, DEVICE, SDK_PREFIX, SDK_BUILD_DIR, SDK_CMAKE_ARGS' \
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

deps: $(VENV)/.munet-deps
test-deps: $(VENV)/.munet-test-deps
rtdetr-deps: $(VENV)/.munet-rtdetr-deps

setup: build
setup-rtdetr: build rtdetr-deps

build: deps
	"$(VENV_PYTHON)" tools/build.py $(BUILD_FLAGS) $(CMAKE_ARGS)

# Standalone C++ build: no Python interface or swarm networking dependencies.
sdk: deps
	"$(VENV_PYTHON)" -m cmake -S . -B "$(SDK_BUILD_DIR)" -DCMAKE_BUILD_TYPE=Release -DMUNET_PYTHON=OFF -DMUNET_SWARM_NODE=OFF -DMUNET_INSTALL_SDK=ON -DMUNET_VULKAN=$(if $(filter 0,$(VULKAN)),OFF,ON) $(SDK_CMAKE_ARGS)
	"$(VENV_PYTHON)" -m cmake --build "$(SDK_BUILD_DIR)" --parallel 4
	"$(VENV_PYTHON)" -m cmake --install "$(SDK_BUILD_DIR)" --prefix "$(abspath $(SDK_PREFIX))"

test-sdk: build sdk
	MUNET_TEST_VULKAN=$(if $(filter vulkan%,$(DEVICE)),1,0) $(RUN) tools/smoke_sdk.py "$(SDK_PREFIX)"

demo-python-api: build
	$(RUN) examples/python_api.py --device "$(DEVICE)"

demo-cpp: build sdk
	$(RUN) examples/cpp_inference/export_model.py --device "$(DEVICE)" $(if $(filter vulkan%,$(DEVICE)),--include-vulkan,)
	"$(VENV_PYTHON)" -m cmake -S examples/cpp_inference -B build/cpp-inference -DCMAKE_PREFIX_PATH="$(abspath $(SDK_PREFIX))"
	"$(VENV_PYTHON)" -m cmake --build build/cpp-inference --parallel 2
	build/cpp-inference/infer artifacts/cpp-inference/model.mnet "$(DEVICE)"

reference: $(VENV_PYTHON)
	"$(VENV_PYTHON)" examples/rtdetr/fetch_reference.py

test: build test-deps
	MUNET_TEST_VULKAN=0 $(RUN) tools/test.py --swarm

test-vulkan:
	@test "$(VULKAN)" = 1 || { printf '%s\n' 'Use make test-vulkan VULKAN=1 to build and test the Vulkan runtime.'; exit 1; }
	$(MAKE) build test-deps
	$(RUN) tools/test.py --vulkan-validation --swarm

test-rtdetr: build test-deps rtdetr-deps reference
	MUNET_TEST_VULKAN=0 $(RUN) tools/test.py examples/rtdetr/tests

test-rtdetr-vulkan:
	@test "$(VULKAN)" = 1 || { printf '%s\n' 'Use make test-rtdetr-vulkan VULKAN=1.'; exit 1; }
	$(MAKE) build test-deps rtdetr-deps reference
	$(RUN) tools/test.py --vulkan-validation examples/rtdetr/tests

smoke: build
	$(RUN) examples/train_mlp.py --device "$(DEVICE)" --steps 20

# Normal installation for scripts outside this checkout; rerun after source edits.
install: deps
	"$(VENV_PYTHON)" -m pip install --no-build-isolation --no-deps . -Ccmake.define.MUNET_VULKAN=$(if $(filter 0,$(VULKAN)),OFF,ON)

clean:
	@if test -f "$(BUILD_DIR)/CMakeCache.txt"; then "$(VENV_PYTHON)" -m cmake --build "$(BUILD_DIR)" --target clean; fi
