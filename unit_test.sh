#!/bin/bash
set -e

echo "=== Building MuNet Tests ==="

BUILD_DIR="artifacts"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Check if nvcc is available to automatically enable CUDA
if command -v nvcc &> /dev/null; then
    echo "[INFO] CUDA found. Building with GPU support..."
    cmake .. -DMUNET_USE_CUDA=ON
else
    echo "[INFO] CUDA not found. Building CPU only..."
    cmake .. -DMUNET_USE_CUDA=OFF
fi

# Build
make -j$(nproc)

echo -e "\n=== Running MuNet Tests ==="
./munet_test
