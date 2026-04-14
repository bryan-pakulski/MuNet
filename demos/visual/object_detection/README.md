# Object Detection Demos

This folder now contains a practical MuNet object detection path for:

- **YOLO-style dense detection** training/inference.
- **FPN-style (Vulkan-friendly) neck + head** training/inference.
- **RTDETR-style query detector** training/inference scaffold.
- **Single-device and multi-device** execution (CUDA + Vulkan + CPU fallback).

## Files

- `operator_gap_report.py`: audits detection-critical primitives and reports backend support status.
- `yolo_e2e_demo.py`: end-to-end tiny YOLO training/inference scaffold.
- `fpn_e2e_demo.py`: end-to-end tiny FPN training/inference scaffold.
- `rtdetr_e2e_demo.py`: end-to-end tiny RTDETR training/inference scaffold.

## Quickstart

```bash
python demos/visual/object_detection/operator_gap_report.py --devices cpu,cuda,vulkan
python demos/visual/object_detection/yolo_e2e_demo.py --mode single --device cpu --steps 10
python demos/visual/object_detection/rtdetr_e2e_demo.py --mode single --device cpu --steps 10
python demos/visual/object_detection/fpn_e2e_demo.py --mode multi --num-devices 2 --steps 10
```

## What is still missing for production COCO training

The demo intentionally uses a backend-portable MSE surrogate loss over YOLO-style grid targets. For full COCO-grade production training/inference, we should add:

1. **NMS ops** (single-image and batched variants).
2. **IoU-family loss ops** (IoU/GIoU/DIoU/CIoU).
3. **Anchor/grid decode kernels** (to avoid Python-side postprocess bottlenecks).
4. **Hungarian matcher + TopK query selection** (RTDETR training/inference path).
5. **Dataset pipeline glue** (COCO parser + augmentation + evaluation metrics such as mAP).

These are now clearly split from the existing operator set so we can implement backend kernels incrementally while keeping training loops and model assembly stable.
