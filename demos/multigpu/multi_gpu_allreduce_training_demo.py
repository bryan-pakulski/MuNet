#!/usr/bin/env python3
"""Multi-device data-parallel training demo with host-staged gradient all-reduce.

This demo:
1) Creates parameter replicas on multiple accelerator devices.
2) Runs per-replica forward/backward on batch shards.
3) Aggregates gradients through CPU (host staging) to emulate all-reduce.
4) Applies synchronized optimizer steps so replicas stay in sync.

Run:
    python demos/multigpu/multi_gpu_allreduce_training_demo.py --steps 20
"""

from __future__ import annotations

import argparse
import os
import threading
import numpy as np

import munet


CPU = munet.Device(munet.DeviceType.CPU, 0)


def detect_accelerators(max_index: int = 4):
    devices = []
    for dev_type in (munet.DeviceType.CUDA, munet.DeviceType.VULKAN):
        for idx in range(max_index):
            dev = munet.Device(dev_type, idx)
            try:
                # Probe with a real forward+backward op; allocation-only checks
                # can pass on devices that later fail during autograd kernels.
                x = munet.ones((2, 2), device=dev, requires_grad=True)
                w = munet.ones((2, 1), device=dev, requires_grad=True)
                y = x @ w
                y.sum().backward()
            except RuntimeError:
                continue
            devices.append(dev)
    return devices


def allreduce_gradients(param_replicas):
    threads = [threading.Thread(target=lambda p=p: p.grad.all_reduce()) for p in param_replicas]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Convert all-reduce sum to average for optimizer step parity.
    scale = 1.0 / float(len(param_replicas))
    for p in param_replicas:
        p.grad.replace_(p.grad * scale)


def make_model_replicas(devices):
    w_cpu = munet.Tensor((4, 1), device=CPU, dtype=munet.DataType.Float32, requires_grad=False)
    w_cpu.uniform_(-0.2, 0.2)
    b_cpu = munet.zeros((1,), device=CPU, dtype=munet.DataType.Float32, requires_grad=False)

    # Important: detach after `.to(dev)` so each replica is a leaf tensor on the
    # target device. Otherwise backward tries to traverse cross-device copy nodes.
    ws = [w_cpu.to(dev).detach() for dev in devices]
    bs = [b_cpu.to(dev).detach() for dev in devices]

    for w in ws:
        w.requires_grad = True
    for b in bs:
        b.requires_grad = True
    return ws, bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    devices = detect_accelerators()
    if len(devices) < 2:
        print("Need at least two accelerator devices (CUDA/Vulkan).")
        return

    # Match the all-reduce runtime knobs used by native backend tests.
    os.environ["MUNET_ALLREDUCE_WORLD_SIZE"] = str(len(devices))
    os.environ["MUNET_ALLREDUCE_MODE"] = "host_fallback"
    os.environ["MUNET_ALLREDUCE_GROUP"] = "python_demo_multigpu"

    devices = devices[:2]
    print("Using devices:", [str(d) for d in devices])

    ws, bs = make_model_replicas(devices)

    # Tiny synthetic regression dataset.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(128, 4)).astype(np.float32)
    true_w = np.array([[0.5], [-0.2], [0.7], [0.1]], dtype=np.float32)
    true_b = np.array([0.05], dtype=np.float32)
    y = x @ true_w + true_b

    for step in range(args.steps):
        # shard batch across replicas
        shard = len(x) // len(devices)
        losses = []

        for rank, dev in enumerate(devices):
            xs = munet.from_numpy(x[rank * shard : (rank + 1) * shard]).to(dev)
            ys = munet.from_numpy(y[rank * shard : (rank + 1) * shard]).to(dev)
            pred = (xs @ ws[rank]) + bs[rank]
            loss = pred.mse_loss(ys)
            loss.backward()
            losses.append(float(np.array(loss.to(CPU), copy=False)))

        allreduce_gradients(ws)
        allreduce_gradients(bs)

        for w in ws:
            w.step(args.lr)
        for b in bs:
            b.step(args.lr)

        # Check synchronization drift
        w0 = np.array(ws[0].to(CPU), copy=False)
        w1 = np.array(ws[1].to(CPU), copy=False)
        drift = np.abs(w0 - w1).max()
        if step % 5 == 0 or step == args.steps - 1:
            print(f"step={step:03d} mean_loss={np.mean(losses):.6f} max_param_drift={drift:.6e}")


if __name__ == "__main__":
    main()
