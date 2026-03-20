import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../build"))
import munet


def build_model():
    model = munet.nn.Sequential(
        [
            munet.nn.Linear(16, 32),
            munet.nn.GELU(),
            munet.nn.Linear(32, 8),
        ]
    )
    model.eval()
    return model


def main():
    dev = munet.Device(munet.DeviceType.CPU, 0)
    model = build_model()
    model.to(dev)

    batches = [np.random.randn(4, 16).astype(np.float32) for _ in range(3)]

    with munet.no_grad():
        outputs = []
        for b in batches:
            t = munet.from_numpy(b).to(dev)
            y = model.forward(t).to(munet.Device(munet.DeviceType.CPU, 0)).detach()
            outputs.append(np.array(y, copy=False))

    print("Batch forward demo complete")
    for i, o in enumerate(outputs):
        print(f"batch[{i}] -> shape={o.shape}, max={o.max():.5f}, min={o.min():.5f}")


if __name__ == "__main__":
    main()
