#!/usr/bin/env python3
"""Phase 3 auto-offload strategy comparison demo."""

from __future__ import annotations

import argparse
import numpy as np

import munet

CPU = munet.Device(munet.DeviceType.CPU, 0)


def _first_accelerator(max_index: int = 4):
    for dev_type in (munet.DeviceType.CUDA, munet.DeviceType.VULKAN):
        for idx in range(max_index):
            dev = munet.Device(dev_type, idx)
            try:
                t = munet.ones((1,), device=dev)
                if float((t + t).to(CPU).item()) == 2.0:
                    return dev
            except RuntimeError:
                continue
    return None


def make_model():
    return munet.nn.Sequential(
        munet.nn.Linear(4, 16),
        munet.nn.ReLU(),
        munet.nn.Linear(16, 8),
        munet.nn.ReLU(),
        munet.nn.Linear(8, 1),
    )


def run_strategy(name: str, devices, sample):
    model = make_model()
    model.auto_offload(devices, strategy=name, sample_input=sample)
    explained = model.offload_plan(explain=True)
    out = model(sample).detach().to(CPU)
    print(
        f"[{name}] plan={explained['plan']} rationale={explained['rationale']} "
        f"mean_out={float(np.array(out, copy=False).mean()):.6f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-index", type=int, default=4)
    args = ap.parse_args()

    accel = _first_accelerator(args.max_index)
    if accel is None:
        print("Need at least one accelerator for auto_offload strategy demo.")
        return

    devices = [CPU, accel]
    sample = munet.from_numpy(np.random.randn(8, 4).astype(np.float32))
    run_strategy("balanced", devices, sample)
    run_strategy("memory-first", devices, sample)
    run_strategy("transfer-minimized", devices, sample)


if __name__ == "__main__":
    main()
