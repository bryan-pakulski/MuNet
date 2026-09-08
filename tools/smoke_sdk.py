"""Compile and run a consumer using only the installed CMake package and header."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def check(prefix):
    import numpy as np
    import munet as mu
    cmake = shutil.which("cmake") or str(Path(sys.executable).parent / "cmake")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.20)
project(consumer LANGUAGES CXX)
find_package(MuNet CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE MuNet::inference)
''')
        (root / "main.cpp").write_text('''#include <munet/core.hpp>
#include <munet/inference.hpp>
#include <cmath>
#include <iostream>
#include <utility>
int main(int argc, char** argv) {
  if(argc<2 || argc>3)return 2;
  try {
  munet::ModelOptions options;
  if(argc==3)options.device=argv[2];
  munet::Model loaded(argv[1], options);
  if(loaded.inputs().at(0).name!="features" || loaded.outputs().at(0).name!="prediction")return 3;
  auto first=loaded.run_named({{"features",{{1,2},{2.f,3.f}}}});
  auto second=loaded.run({{{1,2},{4.f,5.f}}});
  if(first.at("prediction").shape!=munet::Shape{1,2} || first.at("prediction").data!=std::vector<float>{5.f,7.f})return 4;
  if(second.at(0).data!=std::vector<float>{9.f,11.f})return 5;
  bool rejected=false;
  try { loaded.run({{{2,1},{1.f,2.f}}}); } catch(const std::exception&) { rejected=true; }
  if(!rejected)return 6;
  rejected=false;
  try { loaded.run_named({{"wrong",{{1,2},{1.f,2.f}}}}); } catch(const std::exception&) { rejected=true; }
  if(!rejected)return 7;
  auto moved=std::move(loaded);
  if(moved.stats().at("runs")!=2)return 8;
  munet::Graph graph;
  auto x=graph.leaf("input","x",{2});
  auto y=graph.op("add",{x,x});
  munet::Plan plan(graph,{y},{},true);
  plan.run({{1.f,2.f}});
  auto result=plan.read(y);
  return result==std::vector<float>{2.f,4.f}?0:1;
  } catch(const std::exception& e) { std::cerr<<e.what()<<"\\n";return 9; }
}
''')
        model_path = root / "model.mnet"
        import os
        vulkan = os.environ.get("MUNET_TEST_VULKAN", "1") == "1"
        mu.export(lambda x: x * 2 + 1, model_path, np.ones((1, 2), np.float32),
                  input_names=["features"], output_names=["prediction"], include_vulkan=vulkan)
        subprocess.run([cmake, "-S", str(root), "-B", str(root / "build"), "-DCMAKE_PREFIX_PATH=" + str(prefix)], check=True)
        subprocess.run([cmake, "--build", str(root / "build"), "--parallel", "2"], check=True)
        subprocess.run([str(root / "build/consumer"), str(model_path), "cpu"], check=True)
        if vulkan: subprocess.run([str(root / "build/consumer"), str(model_path)], check=True)
    print("Installed MuNet::inference model loading and MuNet::core graph consumer passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=Path)
    check(parser.parse_args().prefix.resolve())
