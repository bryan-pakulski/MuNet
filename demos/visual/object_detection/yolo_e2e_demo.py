#!/usr/bin/env python3
"""Standalone tiny YOLO-style object detection E2E demo."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import munet_nn as munet

CPU = munet.Device(munet.DeviceType.CPU, 0)


@dataclass
class Batch:
    images: np.ndarray
    targets: np.ndarray


def detect_accelerators(max_index: int = 4) -> list[munet.Device]:
    devices: list[munet.Device] = []
    for dev_type in (munet.DeviceType.CUDA, munet.DeviceType.VULKAN):
        for idx in range(max_index):
            dev = munet.Device(dev_type, idx)
            try:
                x = munet.ones((4,), device=dev, dtype=munet.DataType.Float32)
                y = munet.ones((4,), device=dev, dtype=munet.DataType.Float32)
                z = (x + y).to(CPU)
                if abs(float(z.sum().item()) - 8.0) > 1e-6:
                    raise RuntimeError("device self-check mismatch")
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "invalid device ordinal" in msg or "out of range" in msg:
                    break
                continue
            devices.append(dev)
    return devices


class ConvBNAct(munet.nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: int = 0):
        super().__init__()
        self.conv = munet.nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p)
        self.bn = munet.nn.BatchNorm2d(out_ch)
        self.act = munet.nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TinyYoloDetector(munet.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = munet.nn.Sequential(
            ConvBNAct(3, 16, 3, p=1),
            munet.nn.MaxPool2d(2, 2),
            ConvBNAct(16, 32, 3, p=1),
            munet.nn.MaxPool2d(2, 2),
            ConvBNAct(32, 64, 3, p=1),
            munet.nn.MaxPool2d(2, 2),
            ConvBNAct(64, 128, 3, p=1),
            ConvBNAct(128, 128, 3, p=1),
        )
        self.head = munet.nn.Conv2d(128, 5 + num_classes, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).permute([0, 2, 3, 1])


def make_synthetic_batch(batch_size: int, image_size: int, grid_size: int, num_classes: int, rng: np.random.Generator) -> Batch:
    images = np.zeros((batch_size, 3, image_size, image_size), dtype=np.float32)
    targets = np.zeros((batch_size, grid_size, grid_size, 5 + num_classes), dtype=np.float32)
    cell = image_size // grid_size
    yy, xx = np.indices((image_size, image_size))
    for b in range(batch_size):
        cls = int(rng.integers(0, num_classes))
        cx = int(rng.integers(cell, image_size - cell))
        cy = int(rng.integers(cell, image_size - cell))
        bw = int(rng.integers(cell // 2, cell * 2))
        bh = int(rng.integers(cell // 2, cell * 2))
        x0, x1 = max(0, cx - bw // 2), min(image_size, cx + bw // 2)
        y0, y1 = max(0, cy - bh // 2), min(image_size, cy + bh // 2)
        rect = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        color = rng.uniform(0.2, 1.0, size=(3,)).astype(np.float32)
        for c in range(3):
            images[b, c, rect] = color[c]
        gx, gy = min(grid_size - 1, cx // cell), min(grid_size - 1, cy // cell)
        targets[b, gy, gx, 0] = (cx - gx * cell) / float(cell)
        targets[b, gy, gx, 1] = (cy - gy * cell) / float(cell)
        targets[b, gy, gx, 2] = min(1.0, bw / float(image_size))
        targets[b, gy, gx, 3] = min(1.0, bh / float(image_size))
        targets[b, gy, gx, 4] = 1.0
        targets[b, gy, gx, 5 + cls] = 1.0
    images += rng.normal(scale=0.03, size=images.shape).astype(np.float32)
    return Batch(images=np.clip(images, 0.0, 1.0), targets=targets)


def allreduce_parameter_grads(models: Sequence[munet.nn.Module]):
    if len(models) <= 1:
        return
    for params in zip(*(m.parameters() for m in models)):
        for p in params:
            if p.has_grad():
                p.grad.all_reduce()
        scale = 1.0 / float(len(params))
        for p in params:
            if p.has_grad():
                p.grad.replace_(p.grad * scale)


def main() -> int:
    parser = argparse.ArgumentParser(description="MuNet tiny YOLO demo on CPU/CUDA/Vulkan")
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--device", choices=["cpu", "cuda", "vulkan"], default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-index", type=int, default=2)
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=2)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.mode == "single":
        device = {
            "cpu": munet.Device(munet.DeviceType.CPU, 0),
            "cuda": munet.Device(munet.DeviceType.CUDA, 0),
            "vulkan": munet.Device(munet.DeviceType.VULKAN, 0),
        }[args.device]
        model = TinyYoloDetector(args.num_classes).to(device)
        opt = munet.optim.Adam(model.parameters(), lr=args.lr)
        print(f"Running yolo single-device demo on {device}")
        for step in range(args.steps):
            batch = make_synthetic_batch(args.batch_size, args.image_size, args.grid_size, args.num_classes, rng)
            x = munet.from_numpy(batch.images).to(device)
            y = munet.from_numpy(batch.targets).to(device)
            opt.zero_grad()
            loss = model(x).mse_loss(y)
            loss.backward()
            opt.step()
            if step % max(1, args.log_every) == 0 or step == args.steps - 1:
                print(f"[single] step={step:03d} loss={float(loss.detach().to(CPU).item()):.6f}")
        return 0

    devices = detect_accelerators(max_index=args.max_index)
    if len(devices) < max(2, args.num_devices):
        print(f"Need >= {max(2, args.num_devices)} accelerator devices; found {len(devices)}")
        return 1
    devices = devices[: args.num_devices]
    os.environ["MUNET_ALLREDUCE_WORLD_SIZE"] = str(len(devices))
    os.environ["MUNET_ALLREDUCE_MODE"] = "host_fallback"
    os.environ["MUNET_ALLREDUCE_GROUP"] = "objdet_yolo_demo_group"
    os.environ["MUNET_ALLREDUCE_TIMEOUT_MS"] = "30000"
    models = [TinyYoloDetector(args.num_classes).to(d) for d in devices]
    opts = [munet.optim.Adam(m.parameters(), lr=args.lr) for m in models]
    per_replica = max(1, args.batch_size // len(devices))
    print(f"Running yolo multi-device demo on {[str(d) for d in devices]}")
    for step in range(args.steps):
        losses = []
        for dev, model, opt in zip(devices, models, opts):
            batch = make_synthetic_batch(per_replica, args.image_size, args.grid_size, args.num_classes, rng)
            x = munet.from_numpy(batch.images).to(dev)
            y = munet.from_numpy(batch.targets).to(dev)
            opt.zero_grad()
            loss = model(x).mse_loss(y)
            loss.backward()
            losses.append(float(loss.detach().to(CPU).item()))
        allreduce_parameter_grads(models)
        for opt in opts:
            opt.step()
        if step % max(1, args.log_every) == 0 or step == args.steps - 1:
            print(f"[multi ] step={step:03d} mean_loss={float(np.mean(losses)):.6f} losses={[round(v, 6) for v in losses]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
