"""Bounded, content-addressed JSON artifacts; no Python code or pickle on the wire."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

MAX_ARTIFACT = 256 * 1024 * 1024
MAX_RESULT = 32 * 1024 * 1024
ABI = "munet-swarm-1"


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def decode(data):
    def invalid(value):
        raise ValueError(f"non-finite JSON number: {value}")
    return json.loads(data, parse_constant=invalid)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class Artifacts:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, key):
        if not isinstance(key, str) or not re.fullmatch("[0-9a-f]{64}", key):
            raise ValueError("invalid artifact hash")
        return self.directory / key

    def put(self, value):
        data = encode(value)
        if len(data) > MAX_ARTIFACT:
            raise ValueError("artifact exceeds 256 MiB prototype limit")
        key = digest(data)
        path = self.path(key)
        if not path.exists():
            atomic_write(path, data)
        return key

    def get(self, key):
        path = self.path(key)
        if path.stat().st_size > MAX_ARTIFACT:
            raise ValueError("oversized artifact")
        data = path.read_bytes()
        if digest(data) != key:
            raise ValueError("artifact checksum mismatch")
        return decode(data)


def tensor(array):
    return {"shape": list(array.shape), "data": array.ravel().tolist()}
