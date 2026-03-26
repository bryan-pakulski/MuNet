import os
import sys

import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for build_dir in (
    os.path.join(repo_root, "build", "debug"),
    os.path.join(repo_root, "build", "release"),
    os.path.join(repo_root, "build"),
):
    if os.path.isdir(build_dir):
        sys.path.insert(0, build_dir)

import munet


def test_munet_demo_smoke_forward_shape_and_finiteness():
    model = munet.nn.Sequential(
        munet.nn.Linear(8, 4),
        munet.nn.ReLU(),
        munet.nn.Linear(4, 2),
    )
    model.eval()

    x = np.random.randn(3, 8).astype(np.float32)
    y = np.array(model.forward(munet.from_numpy(x)).detach(), copy=False)

    assert list(y.shape) == [3, 2]
    assert np.isfinite(y).all()
