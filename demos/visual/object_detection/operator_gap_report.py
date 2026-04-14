#!/usr/bin/env python3
"""Operator/feature gap report for MuNet object-detection demos.

The report maps architecture primitives to MuNet backend features and quickly
flags what is currently available vs. what still needs implementation work.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import munet_nn as munet


@dataclass(frozen=True)
class OpRequirement:
    primitive: str
    backend_feature: munet.BackendFeature | None
    required_for: str
    notes: str


REQS: tuple[OpRequirement, ...] = (
    OpRequirement("Conv2d", munet.BackendFeature.Convolution, "YOLO/FPN/RTDETR", "Core feature extractor + heads."),
    OpRequirement("BatchNorm2d", munet.BackendFeature.BatchNorm, "YOLO/FPN", "Training stability for CNN stacks."),
    OpRequirement("ReLU/LeakyReLU", munet.BackendFeature.UnaryActivation, "YOLO/FPN", "Non-linearity in backbone and neck."),
    OpRequirement("MaxPool2d", munet.BackendFeature.Pooling, "YOLO", "Downsampling path for tiny-YOLO variant."),
    OpRequirement("Upsample2d", munet.BackendFeature.Pooling, "FPN/RTDETR", "Top-down feature pyramid fusion."),
    OpRequirement("Concat", munet.BackendFeature.Concat, "FPN", "Skip-fusion between lateral + upsampled maps."),
    OpRequirement("Elementwise Add/Sub/Mul/Div", munet.BackendFeature.ElementwiseBinary, "YOLO/FPN/RTDETR", "Loss + coordinate/objectness transforms."),
    OpRequirement("Reduce Mean/Sum", munet.BackendFeature.Reduction, "YOLO/FPN/RTDETR", "Aggregate per-grid/per-query loss terms."),
    OpRequirement("Matmul", munet.BackendFeature.Matmul, "Detection heads + MHA", "Dense heads and attention projections."),
    OpRequirement("Softmax", munet.BackendFeature.Softmax, "RTDETR", "Attention score normalization."),
    OpRequirement("CrossEntropy / MSE", munet.BackendFeature.Loss, "Training", "Class/objectness + regression losses."),
    OpRequirement("Sigmoid/Exp/Log", munet.BackendFeature.UnaryActivation, "YOLO/RTDETR decode", "Decode bbox parameterization."),
    OpRequirement("NMS", None, "Postprocess", "Not a backend feature today; needs dedicated op/API."),
    OpRequirement("IoU/GIoU/DIoU/CIoU", None, "Training", "Loss primitives currently absent as first-class ops."),
    OpRequirement("Hungarian Matcher", None, "RTDETR training", "Set prediction assignment for one-to-one query matching."),
    OpRequirement("TopK query select", None, "RTDETR decode", "Efficient final candidate selection before postprocess."),
)


def resolve_device(kind: str) -> munet.Device:
    if kind == "cpu":
        return munet.Device(munet.DeviceType.CPU, 0)
    if kind == "cuda":
        return munet.Device(munet.DeviceType.CUDA, 0)
    if kind == "vulkan":
        return munet.Device(munet.DeviceType.VULKAN, 0)
    raise ValueError(f"Unknown device kind: {kind}")


def check_feature(device: munet.Device, feature: munet.BackendFeature | None) -> str:
    if feature is None:
        return "MISSING_OP"
    try:
        supported = munet.supports(device, feature, munet.DataType.Float32)
    except RuntimeError:
        return "BACKEND_UNAVAILABLE"
    return "OK" if supported else "FALLBACK_OR_MISSING"


def emit_report(device_names: Iterable[str]) -> int:
    print("== MuNet Object Detection Operator Gap Report ==")
    print("Legend: OK | FALLBACK_OR_MISSING | BACKEND_UNAVAILABLE | MISSING_OP")
    print()

    for name in device_names:
        device = resolve_device(name)
        print(f"--- Backend: {name} ({device}) ---")
        ok = 0
        degraded = 0
        missing = 0
        for req in REQS:
            status = check_feature(device, req.backend_feature)
            if status == "OK":
                ok += 1
            elif status == "MISSING_OP":
                missing += 1
            else:
                degraded += 1
            feature_name = req.backend_feature.name if req.backend_feature else "<explicit-op-needed>"
            print(f"[{status:>20}] {req.primitive:<26} feature={feature_name:<20} use={req.required_for:<12} note={req.notes}")

        print(f"summary: ok={ok} degraded={degraded} missing_explicit={missing}")
        print()

    print("Recommended next additions for production YOLO/FPN/RTDETR:")
    print("  1) Native NMS op (+ batched variant).")
    print("  2) IoU-family losses and bbox-overlap primitives.")
    print("  3) Grid/anchor decode helper kernels to reduce Python overhead.")
    print("  4) Hungarian matcher + TopK query selection utilities for RTDETR.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Report backend coverage for detection primitives.")
    parser.add_argument(
        "--devices",
        type=str,
        default="cpu,cuda,vulkan",
        help="Comma-separated backends to evaluate (cpu,cuda,vulkan).",
    )
    args = parser.parse_args()

    requested = [part.strip().lower() for part in args.devices.split(",") if part.strip()]
    valid = {"cpu", "cuda", "vulkan"}
    bad = [item for item in requested if item not in valid]
    if bad:
        raise SystemExit(f"Unsupported device names: {bad}. Valid={sorted(valid)}")

    return emit_report(requested)


if __name__ == "__main__":
    raise SystemExit(main())
