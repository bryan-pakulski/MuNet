"""Fail on missing package files, mismatched versions, or duplicate wheel identities."""
import argparse
from configparser import ConfigParser
from email.parser import BytesParser
from pathlib import Path
import tarfile
import zipfile
from packaging.utils import parse_wheel_filename
from release_version import validate


def check(directory, *, require_sdist=True):
    version = validate()
    wheels = sorted(directory.rglob("*.whl"))
    if not wheels:
        raise ValueError("no wheels produced")
    names = set()
    for wheel in wheels:
        if wheel.name in names:
            raise ValueError(f"duplicate wheel filename: {wheel.name}; backend variants must not collide")
        names.add(wheel.name)
        name, parsed_version, _, _ = parse_wheel_filename(wheel.name)
        if name != "munet-nn" or str(parsed_version) != version:
            raise ValueError(f"wrong package/version: {wheel.name}")
        if wheel.stat().st_size > 100 * 1024 * 1024:
            raise ValueError(f"wheel exceeds publishing size budget: {wheel.name}")
        with zipfile.ZipFile(wheel) as z:
            files = set(z.namelist())
            for required in ("munet/__init__.py", "munet_nn/__init__.py", "munet/cli.py",
                             "munet/swarm/__main__.py", "munet/swarm/owner.py", "munet/bin/munet-node",
                             "munet/models/rtdetr/__init__.py", "munet/models/rtdetr/LICENSE-RT-DETR", "munet/checkpoint.py"):
                if required not in files:
                    raise ValueError(f"{wheel.name} is missing {required}")
            if not any(n.startswith("munet/_native.") and n.endswith(".so") for n in files):
                raise ValueError("missing native extension")
            if not (z.getinfo("munet/bin/munet-node").external_attr >> 16) & 0o111:
                raise ValueError("packaged worker lost executable permissions")
            metadata = BytesParser().parsebytes(z.read(next(n for n in files if n.endswith(".dist-info/METADATA"))))
            if metadata["Name"] != "munet-nn" or metadata["Version"] != version:
                raise ValueError("wheel metadata does not match the project")
            entries = ConfigParser()
            entries.read_string(z.read(next(n for n in files if n.endswith(".dist-info/entry_points.txt"))).decode())
            for command in ("munet-node", "munet-server"):
                if command not in entries["console_scripts"]:
                    raise ValueError(f"missing installed command: {command}")
    sources = sorted(directory.rglob("munet_nn-*.tar.gz"))
    if require_sdist and len(sources) != 1:
        raise ValueError("exactly one source distribution is required")
    for source in sources:
        with tarfile.open(source) as archive:
            files = {str(Path(n).relative_to(Path(n).parts[0])) for n in archive.getnames()}
        for required in ("cpp/core.hpp", "cpp/swarm/node.cpp", "cpp/vulkan_loader.hpp", "CMakeLists.txt",
                         "cmake/MuNetConfig.cmake.in", "cpp/third_party/nlohmann/LICENSE.MIT", "tools/smoke_install.py",
                         "cpp/ops.cpp", "cpp/kernels/grid.inc", "cmake/kernel_sources.hpp.in"):
            if required not in files:
                raise ValueError(f"source distribution missing {required}")
    print(f"Validated {len(wheels)} munet-nn {version} wheels and {len(sources)} source distributions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--wheel-only", action="store_true")
    args = parser.parse_args()
    check(args.directory, require_sdist=not args.wheel_only)
