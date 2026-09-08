"""Runs against installed wheel contents, never the checkout's Python path."""
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sysconfig
import numpy as np

# Set before the first package import. A wheel must not require libvulkan for CPU work.
os.environ["MUNET_VULKAN_LIBRARY"] = "/not-installed-on-cpu-owner/libvulkan.so.1"
import munet as mu
import munet_nn as retained
from munet_nn.nn import Linear
from munet_nn.swarm.owner import Owner as RetainedOwner
from munet.swarm import Owner
from smoke_release import exercise

assert mu.__version__ == importlib.metadata.version("munet-nn") == retained.__version__
assert retained.Tensor is mu.Tensor and Linear is mu.nn.Linear and RetainedOwner is Owner
x = np.arange(6, dtype=np.float32).reshape(2, 3)
np.testing.assert_array_equal(retained.compile(lambda a: a * 2, device="cpu")(x).numpy(), x * 2)
if mu._native.vulkan_built():
    try:
        mu.devices()
    except RuntimeError as error:
        assert "Vulkan loader unavailable" in str(error)
    else:
        raise AssertionError("explicit missing Vulkan loader was ignored")
scripts = Path(sysconfig.get_path("scripts"))
exercise(scripts / "munet-node", scripts / "munet-server")

# Models live in examples; the wheel supplies reusable operations and layers.
assert importlib.util.find_spec('munet.models') is None
assert importlib.util.find_spec('munet_nn.models') is None
conv=mu.nn.Conv2d(1,2,3,padding=1)
image=np.ones((1,1,4,4),np.float32)
assert mu.compile(conv,device='cpu')(image).shape==(1,2,4,4)
