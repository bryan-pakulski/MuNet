cmake_minimum_required(VERSION 3.18)
project(MuNet_NN LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

option(MUNET_ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(MUNET_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(MUNET_BUILD_TESTS "Build C++ test targets" ON)
option(MUNET_BUILD_TOOLS "Build CLI/report tools" ON)
option(MUNET_BUILD_RUNTIME_PLUGINS "Build runtime backend plugin shared libraries" ON)

# --- Dependencies ---
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

if(MUNET_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# --- Check for CUDA ---
include(CheckLanguage)
check_language(CUDA)

set(MUNET_SOURCES
    src/tensor.cpp
    src/backend/backend_manager.cpp
    src/backend/plugin_loader.cpp
    src/core/op_dispatch.cpp
    src/core/kernel_fusion_planner.cpp
    src/core/ops/add.cpp
    src/core/ops/elementwise.cpp
    src/core/ops/matmul.cpp
    src/core/ops/activations.cpp
    src/core/ops/shape.cpp
    src/core/ops/conv2d.cpp
    src/core/ops/pooling.cpp
    src/core/ops/normalization.cpp
    src/core/ops/loss.cpp
    src/inference_serialization.cpp
)

set_source_files_properties(
    src/inference_serialization.cpp
    PROPERTIES COMPILE_DEFINITIONS MUNET_ENABLE_TRAINING=1
)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA Detected! Enabling GPU Acceleration.")
    add_compile_definitions(MUNET_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND MUNET_SOURCES src/backend/cuda_backend.cu)
else()
    message(WARNING "CUDA not found. Building CPU-only.")
endif()

# --- Check for Vulkan ---
find_package(Vulkan)
if(Vulkan_FOUND)
    message(STATUS "Vulkan Detected! Enabling Vulkan Backend.")
    add_compile_definitions(MUNET_USE_VULKAN)
    list(APPEND MUNET_SOURCES src/backend/vulkan_backend.cpp)
else()
    message(WARNING "Vulkan not found.")
endif()

# --- Shared compile/link options ---
add_library(munet_build_flags INTERFACE)

target_compile_features(munet_build_flags INTERFACE cxx_std_17)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(munet_build_flags INTERFACE -O3)
    # Only enable LTO if CUDA is NOT being used
    if(NOT CMAKE_CUDA_COMPILER)
        target_compile_options(munet_build_flags INTERFACE -flto=auto)
        target_link_options(munet_build_flags INTERFACE -flto=auto)
    endif()
endif()

if(MUNET_ENABLE_ASAN)
    target_compile_options(munet_build_flags INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(munet_build_flags INTERFACE -fsanitize=address)
endif()

if(MUNET_ENABLE_UBSAN)
    target_compile_options(munet_build_flags INTERFACE -fsanitize=undefined -fno-omit-frame-pointer)
    target_link_options(munet_build_flags INTERFACE -fsanitize=undefined)
endif()

# --- Backend Modules ---
add_library(munet_backend_debug OBJECT src/backend/debug_backend.cpp)
target_include_directories(munet_backend_debug PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(munet_backend_debug PUBLIC munet_build_flags)

# --- Library Layout ---
# 1) Core runtime
add_library(munet_core ${MUNET_SOURCES})
target_include_directories(munet_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_sources(munet_core PRIVATE $<TARGET_OBJECTS:munet_backend_debug>)
target_link_libraries(munet_core PUBLIC munet_build_flags)

if(Vulkan_FOUND)
    target_link_libraries(munet_core PUBLIC Vulkan::Vulkan)
endif()

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(munet_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

# 2) Training API
add_library(munet_training INTERFACE)
target_include_directories(munet_training INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(munet_training INTERFACE munet_core munet_build_flags)
target_compile_definitions(munet_training INTERFACE MUNET_ENABLE_TRAINING=1)

# 3) Inference API
add_library(munet_inference INTERFACE)
target_include_directories(munet_inference INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(munet_inference INTERFACE munet_core munet_build_flags)
target_compile_definitions(munet_inference INTERFACE MUNET_ENABLE_INFERENCE=1 MUNET_EDGE_DEPLOY=1)

set(MUNET_INFERENCE_TRAINING_HEADER_PROBE_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/inference_training_header_probe
)
file(MAKE_DIRECTORY ${MUNET_INFERENCE_TRAINING_HEADER_PROBE_DIR})

file(WRITE ${MUNET_INFERENCE_TRAINING_HEADER_PROBE_DIR}/CMakeLists.txt [=[
cmake_minimum_required(VERSION 3.18)
project(MuNetInferenceTrainingHeaderProbe LANGUAGES CXX)
add_executable(probe probe.cpp)
target_include_directories(probe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_definitions(probe PRIVATE MUNET_ENABLE_INFERENCE=1 MUNET_EDGE_DEPLOY=1)
]=])

file(WRITE ${MUNET_INFERENCE_TRAINING_HEADER_PROBE_DIR}/probe.cpp [=[
#include "nn.hpp"
#include "optim.hpp"
int main() { return 0; }
]=])

try_compile(
    MUNET_INFERENCE_REJECTS_TRAINING_HEADERS
    ${MUNET_INFERENCE_TRAINING_HEADER_PROBE_DIR}/build
    ${MUNET_INFERENCE_TRAINING_HEADER_PROBE_DIR}
    probe
    OUTPUT_VARIABLE MUNET_INFERENCE_TRAINING_HEADER_PROBE_OUTPUT
)

if(MUNET_INFERENCE_REJECTS_TRAINING_HEADERS)
    message(FATAL_ERROR
        "munet_inference boundary regression: training headers compiled under inference-only definitions.\n${MUNET_INFERENCE_TRAINING_HEADER_PROBE_OUTPUT}"
    )
endif()

# --- Python Bindings ---
pybind11_add_module(_core src/bindings.cpp)
target_link_libraries(_core PRIVATE munet_training munet_build_flags)

# Install extension module into package path for pip/wheel installs.
install(TARGETS _core
    LIBRARY DESTINATION munet_nn
    RUNTIME DESTINATION munet_nn
    ARCHIVE DESTINATION munet_nn
)


if(MUNET_BUILD_RUNTIME_PLUGINS)
    add_library(munet_backend_vk SHARED src/backend/plugins/plugin_vk.cpp)
    target_include_directories(munet_backend_vk PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    set_target_properties(munet_backend_vk PROPERTIES OUTPUT_NAME "munet_backend_vk")
    install(TARGETS munet_backend_vk
        LIBRARY DESTINATION munet_nn/plugins
        RUNTIME DESTINATION munet_nn/plugins
        ARCHIVE DESTINATION munet_nn/plugins
    )

    add_library(munet_backend_cu12 SHARED src/backend/plugins/plugin_cu12.cpp)
    target_include_directories(munet_backend_cu12 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    set_target_properties(munet_backend_cu12 PROPERTIES OUTPUT_NAME "munet_backend_cu12")
    install(TARGETS munet_backend_cu12
        LIBRARY DESTINATION munet_nn/plugins
        RUNTIME DESTINATION munet_nn/plugins
        ARCHIVE DESTINATION munet_nn/plugins
    )

    add_library(munet_backend_cu13 SHARED src/backend/plugins/plugin_cu13.cpp)
    target_include_directories(munet_backend_cu13 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    set_target_properties(munet_backend_cu13 PROPERTIES OUTPUT_NAME "munet_backend_cu13")
    install(TARGETS munet_backend_cu13
        LIBRARY DESTINATION munet_nn/plugins
        RUNTIME DESTINATION munet_nn/plugins
        ARCHIVE DESTINATION munet_nn/plugins
    )
endif()

# --- Unit Tests ---
if(MUNET_BUILD_TESTS)
    enable_testing()

    add_executable(munet_tests
        tests/unified_tests.cpp
        tests/test_tensor.cpp
        tests/test_autograd.cpp
        tests/test_spatial.cpp
        tests/test_loss.cpp
        tests/test_models.cpp
        tests/test_ops.cpp
        tests/test_optim.cpp
        tests/test_nn.cpp
        tests/test_inference.cpp
        tests/test_kernel_fusion_planner.cpp
        tests/test_backend_manager.cpp
        tests/test_performance.cpp
    )
    target_link_libraries(munet_tests PRIVATE munet_training munet_inference gtest_main munet_build_flags)
endif()

if(MUNET_BUILD_TOOLS)
    add_executable(munet_inference_baseline tools/inference_baseline.cpp)
    target_link_libraries(munet_inference_baseline PRIVATE munet_inference munet_build_flags)

    if(MUNET_BUILD_TESTS)
        add_test(
            NAME munet_inference_baseline_cpu_smoke
            COMMAND munet_inference_baseline
                    --device cpu
                    --dtype float32
                    --batch 4
                    --input-dim 16
                    --hidden-dim 32
                    --output-dim 8
                    --warmup-runs 0
                    --single-run-iters 1
                    --batch-run-inputs 2
                    --batch-run-iters 1
                    --prepared-input-cache-entries 0
                    --prepared-input-cache-max-bytes 0
        )
    endif()

    add_executable(munet_inference_boundary_check tools/inference_boundary_check.cpp)
    target_link_libraries(munet_inference_boundary_check PRIVATE munet_inference munet_build_flags)
    if(MUNET_BUILD_TESTS)
        add_test(NAME munet_inference_boundary_check COMMAND munet_inference_boundary_check)
    endif()

    add_executable(munet_dtype_coverage_report tools/dtype_coverage_report.cpp)
    target_link_libraries(munet_dtype_coverage_report PRIVATE munet_training munet_inference munet_build_flags)

    add_executable(munet_runtime_execution_coverage_report tools/runtime_execution_coverage_report.cpp)
    target_link_libraries(munet_runtime_execution_coverage_report PRIVATE munet_training munet_inference munet_build_flags)
endif()

if(MUNET_BUILD_TESTS)
    include(GoogleTest)
    gtest_add_tests(
        TARGET munet_tests
        TEST_PREFIX munet_
        TEST_LIST munet_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests
        TEST_SUFFIX .cpp
    )
endif()
