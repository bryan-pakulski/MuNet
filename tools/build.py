"""Local development build. Uses dependencies already installed in this environment."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import pybind11

parser = argparse.ArgumentParser()
parser.add_argument("--cpu-only", action="store_true")
parser.add_argument("--build-dir", default="build")
parser.add_argument("--cmake-arg", action="append", default=[])
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
build = root / args.build_dir
cmake = shutil.which("cmake") or str(Path(sys.executable).parent / "cmake")
subprocess.run([cmake, "-S", str(root), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                f"-DPython_EXECUTABLE={sys.executable}", f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
                f"-DMUNET_VULKAN={'OFF' if args.cpu_only else 'ON'}", *args.cmake_arg], check=True)
subprocess.run([cmake, "--build", str(build), "--parallel", "4"], check=True)
subprocess.run([cmake, "--install", str(build), "--prefix", str(root / "python")], check=True)
print("Run examples/tests with PYTHONPATH=python")
