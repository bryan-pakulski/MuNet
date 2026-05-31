#!/usr/bin/env python3
"""Phase 3 benchmark demo comparing manual vs auto-offload strategies."""

from __future__ import annotations

import argparse
import numpy as np
import time

import munet_nn as munet

VULKAN_DEVICE = munet.Device(munet.DeviceType.VULKAN, 0)


def _first_vulkan_device(max_index: int = 4):
    for dev_type in (munet.DeviceType.VULKAN,):
        for idx in range(max_index):
            dev = munet.Device(dev_type, idx)
            try:
                t = munet.ones((1,), device=dev)
                if float((t + t).to(VULKAN_DEVICE).item()) == 2.0:
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


def _benchmark_forward(model, sample, iters: int = 40):
    start = time.perf_counter()
    for _ in range(iters):
        _ = model(sample)
    elapsed_s = time.perf_counter() - start
    return (elapsed_s * 1000.0) / float(iters)


def run_strategy(name: str, devices, sample):
    model = make_model()
    if name == "manual":
        # Naive baseline split: first half on Vulkan, second half on Vulkan device.
        model.offload(VULKAN_DEVICE, ["0", "1"])
        model.offload(devices[1], ["2", "3", "4"])
    else:
        model.auto_offload(devices, strategy=name, sample_input=sample)
    explained = model.offload_plan(explain=True)
    avg_ms = _benchmark_forward(model, sample)
    out = model(sample).detach().to(VULKAN_DEVICE)
    print(
        f"[{name}] plan={explained['plan']} rationale={explained['rationale']} "
        f"avg_ms={avg_ms:.4f} mean_out={float(np.array(out, copy=False).mean()):.6f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-index", type=int, default=4)
    args = ap.parse_args()

    vulkan_dev = _first_vulkan_device(args.max_index)
    if vulkan_dev is None:
        print("Need at least one Vulkan device for auto_offload strategy demo.")
        return

    devices = [VULKAN_DEVICE, vulkan_dev]
    sample = munet.from_numpy(np.random.randn(8, 4).astype(np.float32))
    run_strategy("manual", devices, sample)
    run_strategy("balanced", devices, sample)
    run_strategy("memory-first", devices, sample)


if __name__ == "__main__":
    main()
