"""Build relocatable node/server archives and an installable C++ SDK.

Run in a clean environment with the repaired cp312 wheel installed. Release CI
uses the matching manylinux_2_28 architecture container for the frozen server.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from release_version import ROOT, validate


def run(*args):
    subprocess.run([str(a) for a in args], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("release-dist"))
    parser.add_argument("--work", type=Path, default=Path("build-release"))
    parser.add_argument("--cmake-arg", action="append", default=[])
    args = parser.parse_args()
    version = validate()
    if importlib.metadata.version("munet-nn") != version:
        raise ValueError("install the just-built munet-nn wheel before freezing the server")
    arch = platform.machine()
    if sys.platform != "linux" or arch not in ("x86_64", "aarch64"):
        raise ValueError("release bundles currently target Linux x86_64 and aarch64")
    output, work = args.output.resolve(), args.work.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    stages = {}
    for kind in ("node", "server", "sdk"):
        name = f"munet-{kind}-{version}-linux-{arch}"
        stage = work / name
        if stage.exists():
            raise FileExistsError(f"use a clean work directory: {stage}")
        stage.mkdir()
        stages[kind] = stage
    # Preserve auditwheel's executable-relative RPATH and its hashed library names.
    with zipfile.ZipFile(args.wheel) as z:
        for info in z.infolist():
            parts = Path(info.filename).parts
            if ".." in parts or Path(info.filename).is_absolute():
                raise ValueError("unsafe wheel path")
            if info.is_dir():
                continue
            if info.filename == "munet/bin/munet-node" or parts[0].endswith(".libs"):
                target = stages["node"] / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(info))
                target.chmod((info.external_attr >> 16) & 0o777 or 0o644)
    (stages["node"] / "bin").mkdir()
    (stages["node"] / "bin/munet-node").symlink_to("../munet/bin/munet-node")
    run(stages["node"] / "bin/munet-node", "--version")
    frozen = work / "frozen"
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
               "--name", "munet-server", "--distpath", str(frozen), "--workpath", str(work / "pyinstaller"),
               "--specpath", str(work), "--log-level", "WARN"]
    for module in ("torch", "onnx", "onnxruntime", "pytest", "matplotlib", "scipy", "IPython"):
        command += ["--exclude-module", module]
    run(*command, ROOT / "tools/server_entry.py")
    shutil.copytree(frozen / "munet-server", stages["server"] / "bin")
    run(stages["server"] / "bin/munet-server", "--version")
    sdk_build = work / "sdk-build"
    cmake = shutil.which("cmake") or Path(sys.executable).parent / "cmake"
    run(cmake, "-S", ROOT, "-B", sdk_build, "-DCMAKE_BUILD_TYPE=Release", "-DMUNET_PYTHON=OFF",
        "-DMUNET_SWARM_NODE=OFF", "-DMUNET_INSTALL_SDK=ON", "-DCMAKE_INSTALL_LIBDIR=lib", *args.cmake_arg)
    run(cmake, "--build", sdk_build, "--parallel", "4")
    run(cmake, "--install", sdk_build, "--prefix", stages["sdk"])
    run(sys.executable, ROOT / "tools/smoke_sdk.py", stages["sdk"])
    run(sys.executable, ROOT / "tools/smoke_release.py", "--node", stages["node"] / "bin/munet-node",
        "--server", stages["server"] / "bin/munet-server")
    checksums = []
    for kind, stage in stages.items():
        docs = stage / "share/doc/munet"
        # CMake already installs the SDK's API guides and example sources here.
        docs.mkdir(parents=True, exist_ok=True)
        for source in ("README.md", "docs/install.md", "docs/swarm.md", "cpp/third_party/nlohmann/LICENSE.MIT"):
            shutil.copyfile(ROOT / source, docs / Path(source).name)
        (stage / "manifest.json").write_text(json.dumps({"version": version, "artifact": kind,
            "architecture": arch, "platform": "linux", "libc_build_host": platform.libc_ver(),
            "commit": os.environ.get("GITHUB_SHA", "local"),
            "entry": {"node": "bin/munet-node", "server": "bin/munet-server", "sdk": "lib/cmake/MuNet"}[kind]}, indent=2) + "\n")
        archive = output / (stage.name + ".tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(stage, arcname=stage.name)
        checksums.append(hashlib.sha256(archive.read_bytes()).hexdigest() + "  " + archive.name)
    (output / f"SHA256SUMS-linux-{arch}").write_text("\n".join(checksums) + "\n")
    print(f"Built and smoke-tested node, server and SDK in {output}")


if __name__ == "__main__":
    main()
