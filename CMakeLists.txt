### File: CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(MuNet LANGUAGES CXX)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# --- Dependencies ---
FetchContent_Declare(pybind11 GIT_REPOSITORY https://github.com/pybind/pybind11.git GIT_TAG v2.11.1)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.14.0)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --- Check for CUDA ---
include(CheckLanguage)
check_language(CUDA)

set(MUNET_SOURCES src/tensor.cpp)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA Detected! Enabling GPU Acceleration.")
    add_compile_definitions(MUNET_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Add our new CUDA backend file
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

# --- Core Library ---
add_library(munet_core ${MUNET_SOURCES})
target_include_directories(munet_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(Vulkan_FOUND)
    target_link_libraries(munet_core PUBLIC Vulkan::Vulkan)
endif()

if(CMAKE_CUDA_COMPILER)
		target_link_libraries(munet_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

# --- Python Bindings ---
pybind11_add_module(munet src/bindings.cpp)
target_link_libraries(munet PRIVATE munet_core)

# --- Unit Tests ---
enable_testing()
add_executable(munet_tests tests/unified_tests.cpp tests/test_tensor.cpp tests/test_autograd.cpp tests/test_spatial.cpp tests/test_loss.cpp)
target_link_libraries(munet_tests PRIVATE munet_core gtest_main)

include(GoogleTest)
gtest_discover_tests(munet_tests)
