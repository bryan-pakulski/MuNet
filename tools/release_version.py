"""Validate the retained PyPI identity and release tag before publishing."""
import argparse
import os
from pathlib import Path
import re
import tomllib
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def validate(ref=""):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    if project["name"] != "munet-nn":
        raise ValueError("The PyPI project must remain munet-nn")
    version = project["version"]
    parsed = Version(version)
    if str(parsed) != version or parsed.local:
        raise ValueError("Use a canonical public PEP 440 version")
    runtime = re.search(r'^__version__ = "([^"]+)"', (ROOT / "python/munet/__init__.py").read_text(), re.M)
    if not runtime or runtime[1] != version:
        raise ValueError("Python runtime version and pyproject.toml differ")
    if ref.startswith("refs/tags/"):
        tag = ref.removeprefix("refs/tags/")
        if tag.startswith("dev"):
            if tag != "dev" + version or not parsed.is_devrelease:
                raise ValueError("TestPyPI tags must be dev<VERSION>, with a .devN package version")
        elif tag.startswith("v"):
            if tag != "v" + version or parsed.is_devrelease:
                raise ValueError("PyPI tags must exactly match v<VERSION>; dev releases use dev tags")
        else:
            raise ValueError("unsupported release tag")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF", ""))
    args = parser.parse_args()
    print(validate(args.ref))
