cmake_minimum_required(VERSION 3.17)
project(MuNet CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Silence warnings
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

# Standard Flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(MUNET_USE_CUDA "Enable CUDA support" OFF)
option(MUNET_USE_ONNX "Enable ONNX export support" OFF)

find_package(OpenMP)

# Pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

if(MUNET_USE_CUDA)
    message(STATUS "Building with CUDA support")
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES native)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(MUNET_USE_CUDA)
endif()

if(MUNET_USE_ONNX)
    add_compile_definitions(MUNET_USE_ONNX)
endif()

set(HEADERS
    src/munet.hpp src/activation.hpp src/device.hpp src/dropout.hpp
    src/kernels.hpp src/layer.hpp src/loss.hpp src/model.hpp
    src/optimizer.hpp src/tensor.hpp src/transforms.hpp
)

# --- 1. Python Module ---
# FIX: Add src/munet.cu directly to the source list so it is compiled and linked.
pybind11_add_module(munet src/bindings.cpp src/munet.cu ${HEADERS})

target_include_directories(munet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(MUNET_USE_CUDA)
    # FIX: Disable Separable Compilation to prevent __cudaRegisterLinkedBinary errors
    # This allows munet.cu to be linked like a normal C++ object file.
    set_target_properties(munet PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    
    # FIX: Force CUDA linker to ensure cudart libraries are linked correctly
    set_target_properties(munet PROPERTIES LINKER_LANGUAGE CUDA)
    
    target_link_libraries(munet PRIVATE CUDA::cudart CUDA::cublas)
    target_include_directories(munet PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(munet PRIVATE OpenMP::OpenMP_CXX)
endif()

# --- 2. Test Executable ---
# FIX: Add src/munet.cu directly here as well.
add_executable(munet_test test/test.cpp src/munet.cu ${HEADERS})

target_include_directories(munet_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(MUNET_USE_CUDA)
    # FIX: Disable RDC and use CUDA linker
    set_target_properties(munet_test PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    set_target_properties(munet_test PROPERTIES LINKER_LANGUAGE CUDA)
    
    target_link_libraries(munet_test PRIVATE CUDA::cudart CUDA::cublas)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(munet_test PRIVATE OpenMP::OpenMP_CXX)
endif()
