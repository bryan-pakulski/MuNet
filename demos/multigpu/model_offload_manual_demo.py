#!/usr/bin/env python3
"""Manual model offload demo (Phase 1).

Shows:
1) building a simple Sequential model,
2) assigning layers to different devices via `model.offload(...)`,
3) training/inference without manually handling boundary transfers.
"""

from __future__ import annotations

import argparse
import numpy as np

import munet


CPU = munet.Device(munet.DeviceType.CPU, 0)


def detect_accelerators(max_index: int = 4):
    devices = []
    for dev_type in (munet.DeviceType.CUDA, munet.DeviceType.VULKAN):
        for idx in range(max_index):
            dev = munet.Device(dev_type, idx)
            try:
                a = munet.ones((1,), device=dev)
                b = munet.ones((1,), device=dev)
                c = a + b
                if float(c.to(CPU).item()) != 2.0:
                    raise RuntimeError("probe mismatch")
            except RuntimeError:
                continue
            devices.append(dev)
    return devices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--max-index", type=int, default=2)
    args = ap.parse_args()

    devices = detect_accelerators(args.max_index)
    if len(devices) < 2:
        print("Need >=2 accelerator devices for manual offload demo.")
        return

    d0, d1 = devices[0], devices[1]
    print("Using devices:", d0, d1)

    model = munet.nn.Sequential(
        munet.nn.Linear(4, 16),
        munet.nn.ReLU(),
        munet.nn.Linear(16, 8),
        munet.nn.ReLU(),
        munet.nn.Linear(8, 1),
    )

    # Offload early layers to d0 and later layers to d1.
    model.offload(d0, layers=["0", "1"])
    model.offload(d1, layers=["2", "3", "4"])
    print("offload_plan:", model.offload_plan())

    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 4)).astype(np.float32)
    true_w = np.array([[0.3], [-0.1], [0.5], [0.2]], dtype=np.float32)
    true_b = np.array([0.05], dtype=np.float32)
    y = x @ true_w + true_b

    for step in range(args.steps):
        xs = munet.from_numpy(x)
        ys = munet.from_numpy(y).to(d1)  # output layer is on d1
        pred = model(xs)
        loss = pred.mse_loss(ys)
        loss.backward()

        for p in model.parameters():
            p.step(args.lr)

        if step % 2 == 0 or step == args.steps - 1:
            print(f"step={step:03d} loss={float(loss.detach().to(CPU).item()):.6f}")

    with munet.no_grad():
        sample = munet.from_numpy(x[:4])
        out = model(sample).detach().to(CPU)
        print("sample_out:", np.array(out, copy=False).reshape(-1))


if __name__ == "__main__":
    main()
