import os
from pathlib import Path
import subprocess
import sys
import pytest
import munet as mu
import munet_nn as retained


def test_retained_imports_share_runtime_types():
    from munet_nn.core import Parameter
    from munet_nn.nn import Linear
    from munet_nn.swarm.owner import Owner
    from munet.swarm import Owner as CanonicalOwner
    assert Parameter is mu.Parameter and Linear is mu.nn.Linear
    assert retained.Tensor is mu.Tensor and retained.nn is mu.nn
    assert retained.optim.SGD is mu.optim.SGD and Owner is CanonicalOwner


def test_library_import_is_independent_of_models_and_optional_frameworks():
    code = '''import importlib.abc
import importlib.util
import sys
class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"examples", "torch", "onnx", "onnxruntime", "PIL", "scipy", "pycocotools"}:
            raise AssertionError("library imported optional application/tooling dependency: " + fullname)
sys.meta_path.insert(0, BlockOptionalImports())
import munet
import munet_nn
import numpy as np
assert importlib.util.find_spec("munet.models") is None
assert importlib.util.find_spec("munet_nn.models") is None
model = munet_nn.nn.Sequential(munet_nn.nn.Linear(3, 4), munet_nn.nn.ReLU())
assert munet.compile(model, device="cpu")(np.ones((2, 3), np.float32)).shape == (2, 4)
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_cpu_import_and_execution_without_vulkan_loader():
    env = dict(os.environ, MUNET_VULKAN_LIBRARY="/missing/libvulkan.so.1")
    code = '''import numpy as np
import munet_nn as mu
x=np.ones((2,3),np.float32)
np.testing.assert_equal(mu.compile(lambda x:x*2,device="cpu")(x).numpy(),x*2)
import munet
if munet._native.vulkan_built():
    try:
        mu.devices()
    except RuntimeError as e:
        assert "Vulkan loader unavailable" in str(e)
    else:
        raise AssertionError("missing Vulkan loader was ignored")
'''
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)


def test_release_identity_and_exact_tag_validation():
    tool = Path(__file__).resolve().parents[1] / "tools/release_version.py"
    # Build/release tooling runs on Python 3.12, independently of the library's 3.10 baseline.
    if sys.version_info < (3, 11):
        pytest.skip("tomllib release tooling uses Python 3.11+")
    good = subprocess.run([sys.executable, str(tool), "--ref", "refs/tags/v" + mu.__version__], capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    for ref in ("refs/tags/v" + mu.__version__ + "a", "refs/tags/dev" + mu.__version__):
        bad = subprocess.run([sys.executable, str(tool), "--ref", ref], capture_output=True, text=True)
        assert bad.returncode != 0
