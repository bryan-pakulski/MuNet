"""Canonical swarm types are shared with munet; no duplicate runtime state."""
import importlib as _importlib
import sys as _sys
from munet.swarm import create_job, Owner, ProtocolError

for _name in ("artifacts", "bundle", "owner", "server"):
    _sys.modules[__name__ + "." + _name] = _importlib.import_module("munet.swarm." + _name)

__all__ = ["create_job", "Owner", "ProtocolError"]
