#!/bin/bash
set -e

echo "=== Cleaning Code ==="
pushd src
  find . -regex '.*\.\(cpp\|hpp\|cc\|cxx\|c\|h\)' -exec clang-format -style=file -i {} \;
popd

echo "=== Building MuNet Tests ==="

BUILD_DIR="artifacts"
mkdir -p $BUILD_DIR
pushd $BUILD_DIR

# Check if nvcc is available to automatically enable CUDA
if command -v nvcc &> /dev/null; then
    echo "[INFO] CUDA found. Building with GPU support..."
    cmake .. -DMUNET_USE_CUDA=ON
else
    echo "[INFO] CUDA not found. Building CPU only..."
    cmake .. -DMUNET_USE_CUDA=OFF
fi

# Build
make -j$(nproc) munet

popd

echo "=== Copy artifacts ==="
cp $BUILD_DIR/munet.*.so demo/
cp $BUILD_DIR/munet.*.so test/

echo "=== Running tests ==="
./unit_test.sh
pushd test
  python test.py
popd
