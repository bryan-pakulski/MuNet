#/bin/bash
set -e

# 1. Clean Code
echo "=== Cleaning Code ==="
pushd src/
	find . -regex '.*\.\(cpp\|hpp\|cc\|cxx\|c\|h\|cu\)' -exec clang-format -style=file -i {} \;
popd
pushd tests/
	find . -regex '.*\.\(cpp\|hpp\|cc\|cxx\|c\|h\|cu\)' -exec clang-format -style=file -i {} \;
popd

# 2. Build
echo "=== Building MuNet ==="
mkdir -p build
pushd build/
	cmake ..
	make -j $(nproc)
popd
