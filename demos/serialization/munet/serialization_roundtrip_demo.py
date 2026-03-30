import os
import tempfile
import numpy as np

import munet_nn as munet


def make_model():
    return munet.nn.Sequential(
        [
            munet.nn.Linear(6, 12),
            munet.nn.GELU(),
            munet.nn.Linear(12, 3),
        ]
    )


def main():
    x_np = np.random.randn(4, 6).astype(np.float32)
    x = munet.from_numpy(x_np)

    model = make_model()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model_full.npz")
        munet.save_checkpoint(model, path)

        # Full reconstruction from file (no original class definition required for supported built-ins).
        restored = munet.load_checkpoint(path, trusted=False)

        # Weights-only restore into an existing definition.
        target = make_model()
        munet.load_weights_checkpoint(target, path)

        with munet.no_grad():
            y0 = np.array(model.forward(x).detach(), copy=False)
            y1 = np.array(restored.forward(x).detach(), copy=False)
            y2 = np.array(target.forward(x).detach(), copy=False)

        print("serialization roundtrip demo")
        print("full restore max abs diff:", float(np.max(np.abs(y0 - y1))))
        print("weights-only max abs diff:", float(np.max(np.abs(y0 - y2))))


if __name__ == "__main__":
    main()
