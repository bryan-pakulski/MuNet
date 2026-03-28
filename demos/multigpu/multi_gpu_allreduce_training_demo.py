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
import time
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


def max_tensor_drift(tensors):
    if len(tensors) < 2:
        return 0.0
    base = np.array(tensors[0].detach().to(CPU), copy=False)
    max_drift = 0.0
    for t in tensors[1:]:
        other = np.array(t.detach().to(CPU), copy=False)
        max_drift = max(max_drift, float(np.max(np.abs(base - other))))
    return max_drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--num-devices", type=int, default=2,
                    help="Number of accelerator replicas to train with.")
    ap.add_argument("--samples", type=int, default=1024,
                    help="Synthetic dataset size.")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="Global batch size split across replicas each step.")
    ap.add_argument("--max-index", type=int, default=2,
                    help="Probe device indices in range [0, max_index).")
    args = ap.parse_args()

    devices = detect_accelerators(args.max_index)
    if len(devices) < max(2, args.num_devices):
        print(f"Need at least {max(2, args.num_devices)} healthy accelerator devices (CUDA/Vulkan).")
        return

    devices = devices[:args.num_devices]
    # Configure backend all-reduce rendezvous for selected participants only.
    # This must use the sliced device list size (not total discovered devices),
    # otherwise rendezvous waits for non-participating devices and times out.
    os.environ["MUNET_ALLREDUCE_WORLD_SIZE"] = str(len(devices))
    os.environ["MUNET_ALLREDUCE_MODE"] = "host_fallback"
    os.environ["MUNET_ALLREDUCE_GROUP"] = "python_demo_multigpu"
    os.environ["MUNET_ALLREDUCE_TIMEOUT_MS"] = "30000"

    print("Using devices:", [str(d) for d in devices])

    ws, bs = make_model_replicas(devices)

    # Synthetic regression dataset.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(args.samples, 4)).astype(np.float32)
    true_w = np.array([[0.5], [-0.2], [0.7], [0.1]], dtype=np.float32)
    true_b = np.array([0.05], dtype=np.float32)
    y = x @ true_w + true_b

    per_replica = max(1, args.batch_size // len(devices))
    wall_start = time.perf_counter()

    for step in range(args.steps):
        # sample one global batch and shard it across replicas
        starts = rng.integers(0, max(1, len(x) - per_replica), size=len(devices))
        losses = []

        for rank, dev in enumerate(devices):
            start = int(starts[rank])
            end = start + per_replica
            xs = munet.from_numpy(x[start:end]).to(dev)
            ys = munet.from_numpy(y[start:end]).to(dev)
            pred = (xs @ ws[rank]) + bs[rank]
            loss = pred.mse_loss(ys)
            loss.backward()
            losses.append(float(loss.detach().to(CPU).item()))

        grad_drift_pre_w = max_tensor_drift([w.grad for w in ws])
        grad_drift_pre_b = max_tensor_drift([b.grad for b in bs])
        allreduce_gradients(ws)
        allreduce_gradients(bs)
        grad_drift_post_w = max_tensor_drift([w.grad for w in ws])
        grad_drift_post_b = max_tensor_drift([b.grad for b in bs])

        for w in ws:
            w.step(args.lr)
        for b in bs:
            b.step(args.lr)

        # Check synchronization drift
        w0 = np.array(ws[0].detach().to(CPU), copy=False)
        w1 = np.array(ws[1].detach().to(CPU), copy=False)
        drift = np.abs(w0 - w1).max()
        if step % 5 == 0 or step == args.steps - 1:
            losses_str = ", ".join(f"{v:.4f}" for v in losses)
            print(
                f"step={step:03d} "
                f"losses=[{losses_str}] mean_loss={np.mean(losses):.6f} "
                f"grad_drift_pre(w/b)=({grad_drift_pre_w:.3e}/{grad_drift_pre_b:.3e}) "
                f"grad_drift_post(w/b)=({grad_drift_post_w:.3e}/{grad_drift_post_b:.3e}) "
                f"max_param_drift={drift:.3e}"
            )

    elapsed = time.perf_counter() - wall_start
    learned_w = np.array(ws[0].detach().to(CPU), copy=False)
    learned_b = np.array(bs[0].detach().to(CPU), copy=False)
    print(
        f"done: devices={len(devices)} steps={args.steps} "
        f"samples/step={per_replica * len(devices)} elapsed_s={elapsed:.2f}\n"
        f"  learned_w={learned_w.reshape(-1)}\n"
        f"  learned_b={learned_b.reshape(-1)}\n"
        f"  true_w={true_w.reshape(-1)} true_b={true_b.reshape(-1)}"
    )


if __name__ == "__main__":
    main()
