#/bin/bash
set -e

function cleanup {
  local dir=$1
	find $1 -regex '.*\.\(cpp\|hpp\|cc\|cxx\|c\|h\|cu\)' -exec clang-format -style=file -i {} \;
  find $1 -regex '.*\.\(cpp\|hpp\|cc\|cxx\|c\|h\|cu\|txt\|md|\sh|\)' -exec sed -i 's/[[:space:]]*$//' {} \;
}

function cleanfile {
	local file=$1
	sed -i 's/[[:space:]]*$//' $file
}

# 1. Clean Code
echo "=== Cleaning Code ==="
cleanup src
cleanup tests
cleanup demos

cleanfile build.sh
cleanfile test.sh
cleanfile CMakeLists.txt
cleanfile ReadMe.md

# 2. Build
echo "=== Building MuNet ==="
mkdir -p build
pushd build/
	cmake .. -DCMAKE_BUILD_TYPE=Release
	make -j $(nproc)
popd
