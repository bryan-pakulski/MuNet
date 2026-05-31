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

import munet_nn as munet


VULKAN_DEVICE = munet.Device(munet.DeviceType.VULKAN, 0)


def detect_vulkan_devices(max_index: int = 4):
    devices = []
    for dev_type in (munet.DeviceType.VULKAN,):
        for idx in range(max_index):
            dev = munet.Device(dev_type, idx)
            try:
                a = munet.ones((1,), device=dev)
                b = munet.ones((1,), device=dev)
                c = a + b
                if float(c.to(VULKAN_DEVICE).item()) != 2.0:
                    raise RuntimeError("probe mismatch")
            except RuntimeError:
                continue
            devices.append(dev)
    return devices


def can_transfer(src, dst) -> bool:
    try:
        probe = munet.ones((2,), device=src)
        moved = probe.to(dst)
        return float(moved.to(VULKAN_DEVICE).sum().item()) == 2.0
    except RuntimeError:
        return False


def pick_offload_pair(vulkan_devices):
    # Prefer true Vulkan↔Vulkan pairs when transfer works.
    for i in range(len(vulkan_devices)):
        for j in range(i + 1, len(vulkan_devices)):
            d0, d1 = vulkan_devices[i], vulkan_devices[j]
            if can_transfer(d0, d1):
                return d0, d1, "vulkan_pair"

    if vulkan_devices:
        return VULKAN_DEVICE, vulkan_devices[0], "vulkan_single"

    return None, None, "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--max-index", type=int, default=2)
    ap.add_argument(
        "--backend",
        choices=["vulkan"],
        default="vulkan",
        help="Vulkan backend selection policy (default: vulkan).",
    )
    args = ap.parse_args()

    devices = detect_vulkan_devices(args.max_index)
    if not devices:
        print(f"Need at least one Vulkan device for manual offload demo.")
        return

    d0, d1, mode = pick_offload_pair(devices)
    if d0 is None or d1 is None:
        print("Could not find a compatible device pair for boundary transfers.")
        return

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
    plan = model.offload_plan()
    print("offload_plan (layer -> device):", plan)
    by_device = {}
    for layer_name, device in plan.items():
        by_device.setdefault(str(device), []).append(layer_name)
    print("offload_plan summary (device -> layers):", by_device)
    sample_validate = munet.from_numpy(np.random.randn(2, 4).astype(np.float32))
    report = model.validate_offload_plan(sample_validate)
    print(
        "validate_offload_plan:",
        {
            "valid": report.valid,
            "errors": report.errors,
            "warnings": report.warnings,
            "estimated_boundaries": report.estimated_boundaries,
            "estimated_ping_pong_boundaries": report.estimated_ping_pong_boundaries,
        },
    )

    # Intentionally bad plan walkthrough (Phase 2): float16 linear on Vulkan.
    bad_opts = munet.TensorOptions()
    bad_opts.dtype = munet.DataType.Float16
    bad_model = munet.nn.Sequential(
        munet.nn.Linear(4, 8, options=bad_opts),
        munet.nn.ReLU(),
        munet.nn.Linear(8, 1, options=bad_opts),
    )
    bad_model.offload(VULKAN_DEVICE, layers=["0", "2"])
    bad_report = bad_model.validate_offload_plan(
        munet.from_numpy(np.random.randn(2, 4).astype(np.float16))
    )
    print(
        "validate_offload_plan (intentional bad plan):",
        {
            "valid": bad_report.valid,
            "errors": bad_report.errors,
            "warnings": bad_report.warnings,
        },
    )

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
            print(f"step={step:03d} loss={float(loss.detach().to(VULKAN_DEVICE).item()):.6f}")

    with munet.no_grad():
        sample = munet.from_numpy(x[:4])
        out = model(sample).detach().to(VULKAN_DEVICE)
        print("sample_out:", np.array(out, copy=False).reshape(-1))


if __name__ == "__main__":
    main()
