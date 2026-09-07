"""Installed command that replaces itself with the packaged native worker."""
import os
from pathlib import Path
import sys


def node():
    executable = Path(__file__).resolve().parent / "bin" / "munet-node"
    if not executable.is_file():
        raise SystemExit("This installation has no munet-node binary. Install a release wheel or build with MUNET_SWARM_NODE=ON.")
    os.execv(str(executable), [str(executable), *sys.argv[1:]])
