"""Compile and run a consumer using only the installed CMake package and header."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def check(prefix):
    cmake = shutil.which("cmake") or str(Path(sys.executable).parent / "cmake")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.20)
project(consumer LANGUAGES CXX)
find_package(MuNet CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE MuNet::core)
''')
        (root / "main.cpp").write_text('''#include <munet/core.hpp>
int main() {
  munet::Graph graph;
  auto x=graph.leaf("input","x",{2});
  auto y=graph.op("add",{x,x});
  munet::Plan plan(graph,{y},{},true);
  plan.run({{1.f,2.f}});
  auto result=plan.read(y);
  return result==std::vector<float>{2.f,4.f}?0:1;
}
''')
        subprocess.run([cmake, "-S", str(root), "-B", str(root / "build"), "-DCMAKE_PREFIX_PATH=" + str(prefix)], check=True)
        subprocess.run([cmake, "--build", str(root / "build"), "--parallel", "2"], check=True)
        subprocess.run([str(root / "build/consumer")], check=True)
    print("Installed MuNet::core SDK consumer passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=Path)
    check(parser.parse_args().prefix.resolve())
