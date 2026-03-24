SHELL := /bin/bash

CMAKE ?= cmake
CTEST ?= ctest
BUILD_JOBS ?= $(shell nproc)

SRC_DIR := .
BUILD_ROOT := build
BUILD_DEBUG := $(BUILD_ROOT)/debug
BUILD_RELEASE := $(BUILD_ROOT)/release
BUILD_ASAN := $(BUILD_ROOT)/asan
BUILD_GPU := $(BUILD_ROOT)/gpu-debug

.DEFAULT_GOAL := help

.PHONY: help \
	build build-debug build-release build-asan build-gpu \
	configure-debug configure-release configure-asan configure-gpu \
	unit-test test-debug test-release test-asan ctest-debug ctest-release ctest-asan \
	mem-test gpu-mem-test perf-test py-test \
	format doc clean clean-debug clean-release clean-asan clean-gpu \
	reconfigure-debug reconfigure-release reconfigure-asan reconfigure-gpu \
	docker-build

help:
	@echo "Targets:"
	@echo "  build            Build debug"
	@echo "  build-debug      Configure + build debug"
	@echo "  build-release    Configure + build release"
	@echo "  build-asan       Configure + build ASan/UBSan debug"
	@echo "  build-gpu        Configure + build debug for compute-sanitizer"
	@echo ""
	@echo "  unit-test        Run debug unit tests"
	@echo "  test-release     Run release unit tests"
	@echo "  test-asan        Run ASan unit tests"
	@echo "  ctest-debug      Run CTest in debug build"
	@echo "  ctest-release    Run CTest in release build"
	@echo "  ctest-asan       Run CTest in ASan build"
	@echo "  mem-test         Alias for test-asan"
	@echo "  gpu-mem-test     Run compute-sanitizer memcheck"
	@echo "  perf-test        Run performance tests from release build"
	@echo "  py-test          Run python tests"
	@echo "  format           Format code"
	@echo "  doc              Build docs"
	@echo "  clean            Remove all build directories"

define configure_build
	mkdir -p $(1)
	$(CMAKE) -S $(SRC_DIR) -B $(1) $(2)
endef

define build_dir
	$(CMAKE) --build $(1) -j $(BUILD_JOBS)
endef

configure-debug:
	$(call configure_build,$(BUILD_DEBUG),-DCMAKE_BUILD_TYPE=Debug)

configure-release:
	$(call configure_build,$(BUILD_RELEASE),-DCMAKE_BUILD_TYPE=Release)

configure-asan:
	$(call configure_build,$(BUILD_ASAN),-DCMAKE_BUILD_TYPE=Debug -DMUNET_ENABLE_ASAN=ON -DMUNET_ENABLE_UBSAN=ON)

configure-gpu:
	$(call configure_build,$(BUILD_GPU),-DCMAKE_BUILD_TYPE=Debug)

build: build-debug

build-debug: configure-debug
	$(call build_dir,$(BUILD_DEBUG))

build-release: configure-release
	$(call build_dir,$(BUILD_RELEASE))

build-asan: configure-asan
	$(call build_dir,$(BUILD_ASAN))

build-gpu: configure-gpu
	$(call build_dir,$(BUILD_GPU))

unit-test: test-debug

test-debug: build-debug
	./$(BUILD_DEBUG)/munet_tests

test-release: build-release
	./$(BUILD_RELEASE)/munet_tests

test-asan: build-asan
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
	./$(BUILD_ASAN)/munet_tests

ctest-debug: build-debug
	$(CTEST) --test-dir $(BUILD_DEBUG) --output-on-failure

ctest-release: build-release
	$(CTEST) --test-dir $(BUILD_RELEASE) --output-on-failure

ctest-asan: build-asan
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
	$(CTEST) --test-dir $(BUILD_ASAN) --output-on-failure

mem-test: test-asan

gpu-mem-test: build-gpu
	compute-sanitizer --tool memcheck ./$(BUILD_GPU)/munet_tests

perf-test: build-release
	MUNET_RUN_PERF_TESTS=1 ./$(BUILD_RELEASE)/munet_tests --gtest_filter=PerformanceTest.*

py-test: build-debug
	cd tests && python3 test_python.py && python3 test_pytorch_interop.py

format:
	find src tests -regex '.*\.\(cpp\|hpp\|cc\|cxx\|c\|h\|cu\)' -exec clang-format -style=file -i {} \;
	find src tests -regex '.*\.\(cpp\|hpp\|cc\|cxx\|c\|h\|cu\|txt\|md\|sh\)' -exec sed -i 's/[[:space:]]*$$//' {} \;
	find demos -regex '.*\.py' -exec black {} \;
	find demos -regex '.*\.py' -exec sed -i 's/[[:space:]]*$$//' {} \;

doc:
	mkdir -p docs
	pdoc ./munet -o ./docs

clean:
	rm -rf $(BUILD_ROOT)

clean-debug:
	rm -rf $(BUILD_DEBUG)

clean-release:
	rm -rf $(BUILD_RELEASE)

clean-asan:
	rm -rf $(BUILD_ASAN)

clean-gpu:
	rm -rf $(BUILD_GPU)

reconfigure-debug: clean-debug build-debug
reconfigure-release: clean-release build-release
reconfigure-asan: clean-asan build-asan
reconfigure-gpu: clean-gpu build-gpu

docker-build:
	./tools/build_in_docker.sh
