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
                # Keep probe lightweight and deterministic: run a real backend op,
                # force synchronize via copy-back, and validate value.
                a = munet.ones((1,), device=dev, dtype=munet.DataType.Float32)
                b = munet.ones((1,), device=dev, dtype=munet.DataType.Float32)
                c = a + b
                c_cpu = c.to(CPU)
                if float(c_cpu.item()) != 2.0:
                    raise RuntimeError("accelerator health-check produced incorrect value")
            except RuntimeError as exc:
                # If the index is out of range for this backend, no need to probe
                # higher indices of the same type.
                msg = str(exc).lower()
                if "invalid device ordinal" in msg or "out of range" in msg:
                    break
                continue
            devices.append(dev)
    return devices


def allreduce_gradients(param_replicas):
    errors = []

    def _run_all_reduce(param):
        try:
            param.grad.all_reduce()
        except Exception as exc:  # noqa: BLE001 - demo path, bubble up cleanly
            errors.append(exc)

    threads = [
        threading.Thread(target=_run_all_reduce, args=(p,)) for p in param_replicas
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise RuntimeError(f"all_reduce failed: {errors[0]}")

    # Convert all-reduce sum to average for optimizer step parity.
    scale = 1.0 / float(len(param_replicas))
    for p in param_replicas:
        p.grad.replace_(p.grad * scale)


def make_model_replicas(devices):
    # Initialize directly on each target device so mixed backend pairs
    # (e.g., CUDA + Vulkan) do not depend on cross-backend parameter copies.
    ws = []
    bs = []
    for dev in devices:
        w = munet.zeros((4, 1), device=dev, dtype=munet.DataType.Float32, requires_grad=True)
        w.uniform_(-0.2, 0.2)
        b = munet.zeros((1,), device=dev, dtype=munet.DataType.Float32, requires_grad=True)
        ws.append(w)
        bs.append(b)
    return ws, bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max-index", type=int, default=2,
                    help="Probe device indices in range [0, max_index).")
    args = ap.parse_args()

    devices = detect_accelerators(args.max_index)
    if len(devices) < 2:
        print("Need at least two healthy accelerator devices (CUDA/Vulkan).")
        return

    # Configure backend all-reduce rendezvous for this run. Explicitly set these
    # here so stale shell-level values do not cause participant-count mismatches.
    os.environ["MUNET_ALLREDUCE_WORLD_SIZE"] = str(len(devices))
    os.environ["MUNET_ALLREDUCE_MODE"] = "host_fallback"
    os.environ["MUNET_ALLREDUCE_GROUP"] = "python_demo_multigpu"
    os.environ["MUNET_ALLREDUCE_TIMEOUT_MS"] = "30000"

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
            losses.append(float(loss.detach().to(CPU).item()))

        allreduce_gradients(ws)
        allreduce_gradients(bs)

        for w in ws:
            w.step(args.lr)
        for b in bs:
            b.step(args.lr)

        # Check synchronization drift
        w0 = np.array(ws[0].detach().to(CPU), copy=False)
        w1 = np.array(ws[1].detach().to(CPU), copy=False)
        drift = np.abs(w0 - w1).max()
        if step % 5 == 0 or step == args.steps - 1:
            print(f"step={step:03d} mean_loss={np.mean(losses):.6f} max_param_drift={drift:.6e}")


if __name__ == "__main__":
    main()
