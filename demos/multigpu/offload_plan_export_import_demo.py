#!/usr/bin/env python3
"""Export/import offload plan demo for reproducible deployment."""

from __future__ import annotations

import json
import numpy as np

import munet_nn as munet

CPU = munet.Device(munet.DeviceType.CPU, 0)


def make_model():
    return munet.nn.Sequential(
        munet.nn.Linear(4, 16),
        munet.nn.ReLU(),
        munet.nn.Linear(16, 8),
        munet.nn.ReLU(),
        munet.nn.Linear(8, 1),
    )


def first_accelerator(max_index: int = 4):
    for dev_type in (munet.DeviceType.CUDA, munet.DeviceType.VULKAN):
        for idx in range(max_index):
            dev = munet.Device(dev_type, idx)
            try:
                probe = munet.ones((1,), device=dev)
                if float((probe + probe).to(CPU).item()) == 2.0:
                    return dev
            except RuntimeError:
                continue
    return None


def main():
    accel = first_accelerator()
    if accel is None:
        print("Need one accelerator for export/import offload demo")
        return

    sample = munet.from_numpy(np.random.randn(8, 4).astype(np.float32))
    devices = [CPU, accel]

    planner_model = make_model()
    planner_model.auto_offload(devices, strategy="balanced", sample_input=sample)

    frozen = planner_model.freeze_offload_plan()
    print("Frozen plan JSON:")
    print(json.dumps(frozen, indent=2, sort_keys=True))

    deployed_model = make_model()
    deployed_model.apply_offload_plan(frozen)

    y = deployed_model(sample)
    print(f"Deployment output shape: {y.shape}")


if __name__ == "__main__":
    main()
