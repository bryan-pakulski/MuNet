#!/usr/bin/env python3
"""Standalone tiny RTDETR-style object detection E2E demo."""

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
                z = (munet.ones((4,), device=dev) + munet.ones((4,), device=dev)).to(CPU)
                if abs(float(z.sum().item()) - 8.0) > 1e-6:
                    raise RuntimeError("device self-check mismatch")
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "invalid device ordinal" in msg or "out of range" in msg:
                    break
                continue
            devices.append(dev)
    return devices


def conv_act(in_ch: int, out_ch: int, k: int, s: int = 1, p: int = 0):
    return munet.nn.Sequential(
        munet.nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p),
        munet.nn.LeakyReLU(0.1),
    )


class TinyRTDETRDetector(munet.nn.Module):
    def __init__(self, num_classes: int, num_queries: int = 50, embed_dim: int = 128):
        super().__init__()
        self.num_queries = num_queries
        self.backbone = munet.nn.Sequential(
            conv_act(3, 32, 3, p=1),
            munet.nn.MaxPool2d(2, 2),
            conv_act(32, 64, 3, p=1),
            munet.nn.MaxPool2d(2, 2),
            conv_act(64, embed_dim, 3, p=1),
            munet.nn.MaxPool2d(2, 2),
        )
        self.token_proj = munet.nn.Linear(embed_dim, embed_dim)
        self.encoder_attn = munet.nn.MultiHeadAttention(embed_dim, 4, causal=False)
        self.decoder_attn = munet.nn.MultiHeadAttention(embed_dim, 4, causal=False)
        self.ln = munet.nn.LayerNorm(embed_dim)
        self.class_head = munet.nn.Linear(embed_dim, num_classes)
        self.bbox_head = munet.nn.Sequential(
            munet.nn.Linear(embed_dim, embed_dim),
            munet.nn.GELU(),
            munet.nn.Linear(embed_dim, 4),
        )

    def forward(self, x):
        feat = self.backbone(x)
        tokens = feat.permute([0, 2, 3, 1])
        b, h, w, c = tokens.shape
        memory = self.encoder_attn(self.token_proj(tokens.reshape([b, h * w, c])))
        q = min(self.num_queries, memory.shape[1])
        decoded = self.decoder_attn(self.ln(munet.cat([memory.narrow(1, 0, q), memory], dim=1))).narrow(1, 0, q)
        pred_boxes = self.bbox_head(decoded).sigmoid()
        pred_classes = self.class_head(decoded)
        objectness = munet.ones((b, q, 1), device=pred_boxes.device).to(pred_boxes.dtype)
        return munet.cat([pred_boxes, objectness, pred_classes], dim=2)


def make_synthetic_batch(batch_size: int, image_size: int, num_queries: int, num_classes: int, rng: np.random.Generator) -> Batch:
    images = np.zeros((batch_size, 3, image_size, image_size), dtype=np.float32)
    targets = np.zeros((batch_size, num_queries, 5 + num_classes), dtype=np.float32)
    for b in range(batch_size):
        cls = int(rng.integers(0, num_classes))
        cx = int(rng.integers(image_size // 4, 3 * image_size // 4))
        cy = int(rng.integers(image_size // 4, 3 * image_size // 4))
        bw = int(rng.integers(image_size // 8, image_size // 3))
        bh = int(rng.integers(image_size // 8, image_size // 3))
        yy, xx = np.indices((image_size, image_size))
        rect = (xx >= cx - bw // 2) & (xx < cx + bw // 2) & (yy >= cy - bh // 2) & (yy < cy + bh // 2)
        color = rng.uniform(0.2, 1.0, size=(3,)).astype(np.float32)
        for c in range(3):
            images[b, c, rect] = color[c]
        targets[b, 0, 0:4] = np.array([
            cx / float(image_size),
            cy / float(image_size),
            min(1.0, bw / float(image_size)),
            min(1.0, bh / float(image_size)),
        ], dtype=np.float32)
        targets[b, 0, 4] = 1.0
        targets[b, 0, 5 + cls] = 1.0
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
    parser = argparse.ArgumentParser(description="MuNet tiny RTDETR demo on CPU/CUDA/Vulkan")
    parser.add_argument("--mode", choices=["single", "multi"], default="single")
    parser.add_argument("--device", choices=["cpu", "cuda", "vulkan"], default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-index", type=int, default=2)
    parser.add_argument("--num-devices", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.mode == "single":
        device = {"cpu": munet.Device(munet.DeviceType.CPU, 0), "cuda": munet.Device(munet.DeviceType.CUDA, 0), "vulkan": munet.Device(munet.DeviceType.VULKAN, 0)}[args.device]
        model = TinyRTDETRDetector(args.num_classes, num_queries=args.num_queries).to(device)
        opt = munet.optim.Adam(model.parameters(), lr=args.lr)
        print(f"Running rtdetr single-device demo on {device}")
        for step in range(args.steps):
            batch = make_synthetic_batch(args.batch_size, args.image_size, args.num_queries, args.num_classes, rng)
            x = munet.from_numpy(batch.images).to(device)
            y = munet.from_numpy(batch.targets).to(device)
            opt.zero_grad()
            loss = model(x).mse_loss(y)
            loss.backward()
            opt.step()
            if step % max(1, args.log_every) == 0 or step == args.steps - 1:
                print(f"[single] step={step:03d} loss={float(loss.detach().item()):.6f}")
        return 0

    devices = detect_accelerators(max_index=args.max_index)
    if len(devices) < max(2, args.num_devices):
        print(f"Need >= {max(2, args.num_devices)} accelerator devices; found {len(devices)}")
        return 1
    devices = devices[: args.num_devices]
    os.environ["MUNET_ALLREDUCE_WORLD_SIZE"] = str(len(devices))
    os.environ["MUNET_ALLREDUCE_MODE"] = "host_fallback"
    os.environ["MUNET_ALLREDUCE_GROUP"] = "objdet_rtdetr_demo_group"
    os.environ["MUNET_ALLREDUCE_TIMEOUT_MS"] = "30000"
    models = [TinyRTDETRDetector(args.num_classes, num_queries=args.num_queries).to(d) for d in devices]
    opts = [munet.optim.Adam(m.parameters(), lr=args.lr) for m in models]
    per_replica = max(1, args.batch_size // len(devices))
    print(f"Running rtdetr multi-device demo on {[str(d) for d in devices]}")
    for step in range(args.steps):
        losses = []
        for dev, model, opt in zip(devices, models, opts):
            batch = make_synthetic_batch(per_replica, args.image_size, args.num_queries, args.num_classes, rng)
            x = munet.from_numpy(batch.images).to(dev)
            y = munet.from_numpy(batch.targets).to(dev)
            opt.zero_grad()
            loss = model(x).mse_loss(y)
            loss.backward()
            losses.append(float(loss.detach().item()))
        allreduce_parameter_grads(models)
        for opt in opts:
            opt.step()
        if step % max(1, args.log_every) == 0 or step == args.steps - 1:
            print(f"[multi ] step={step:03d} mean_loss={float(np.mean(losses)):.6f} losses={[round(v, 6) for v in losses]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
