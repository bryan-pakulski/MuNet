# RT-DETR example: training and inference

The source example under `examples/rtdetr/` builds RT-DETR v1 from MuNet's
reusable layers and operations. It contains a ResNet-vd backbone, the hybrid encoder,
and an iterative multiscale deformable decoder. `rtdetr_r50vd()` defaults to the
R50-vd architecture: 80 classes, 256 hidden channels, eight heads, six decoder
layers, 300 queries and 100 denoising queries. `rtdetr_r18vd()` is also available.
The source and numerical reference are pinned to
[official RT-DETR commit 29320b6](https://github.com/lyuwenyu/RT-DETR/tree/29320b6fd828f8e0987a71426cf2d961b09dfed7).
Adapted model files retain the upstream Apache-2.0 license and attribution.

The installed `munet-nn` library does not contain model implementations. Run this
example from the repository root. For local source development:

```bash
make setup-rtdetr                    # Add VULKAN=0 for CPU-only builds
source .venv/bin/activate
export PYTHONPATH="$PWD/python"
```

Alternatively, install a MuNet wheel into your environment and install
`examples/rtdetr/requirements.txt`; then run from the checkout root without the
`PYTHONPATH` override. Use `from examples.rtdetr import ...` for example APIs.

## Train a model

```python
import numpy as np
from examples.rtdetr import rtdetr_r50vd, DetectorTrainer

model = rtdetr_r50vd(num_classes=2, seed=7)
trainer = DetectorTrainer(model, device="vulkan", target_slots=32, seed=17)
images = np.random.default_rng(5).random((2, 3, 640, 640), dtype=np.float32)
targets = [
    {"labels": [1], "boxes": [[0.5, 0.5, 0.2, 0.3]]},
    {"labels": [], "boxes": np.empty((0, 4), np.float32)},
]
metrics = trainer.step(images, targets)
trainer.finish_epoch()
trainer.save("training.mnet")
trainer.load("training.mnet")
trainer.export("inference.mnet", images[:1], onnx_path="inference.onnx")
```

Images are contiguous float32 NCHW RGB in [0,1]. Both spatial dimensions must be
positive multiples of 32; feature maps must contain at least `num_queries`
positions. Targets use zero-based contiguous class labels and normalized
`cx,cy,width,height` boxes. Empty images are supported. Padding slots must cover
all objects and cannot exceed the query count; data is never silently truncated.
The native assignment solver supports at most 1,024 queries and 512 target slots.

A training replay includes Hungarian matching, VFL and L1/GIoU losses, independent
matching for decoder/encoder auxiliary losses, contrastive denoising losses,
backward, global L2 gradient clipping, AdamW, BatchNorm state and EMA. Matching is
a native rectangular Hungarian solver in both C++ and Vulkan. It has no SciPy or
CPU execution boundary on the Vulkan path. Ties may choose a different equally
optimal assignment; the solver uses float32 costs. The implementation prioritizes
correctness and is not a tuned large-assignment kernel.

The default optimizer groups reproduce the reference's learning rates and decay
rules: backbone LR 1e-5, other LR 1e-4, AdamW decay 1e-4 except encoder/decoder bias
and normalization parameters. The clip norm is 0.1, EMA decay is 0.9999 with a
2,000-update warmup, and the default scheduler milestone is 1,000 epochs, matching
the pinned 72-epoch recipe. LR and AdamW group settings are live values in an
existing compiled program. These are float32 training kernels; AMP is not needed
for correctness and is not implemented.

`DetectorTrainer` generates fresh denoising inputs on the host before every replay.
Its data-only checkpoint saves model parameters/buffers, AdamW moments and steps,
EMA, scheduler, RNG, epoch and step counters, model config and trainable selection.
Resume with the same constructor/configuration. Checkpoints use ZIP/JSON/NPY and
never pickle. Aggregate training-state archives are limited to 2 GiB; individual
arrays are bounded and validated before allocation. The example resumes at epoch
boundaries; arbitrary external dataloader/cursor state is the caller's responsibility.

Image, batch and target dimensions select a cached specialization. A fixed
`target_slots` reduces recompilation, with more padded work as a tradeoff. The
cache is bounded (two programs by default). Switching programs copies shared
state through the host; repeated execution of one shape keeps parameters,
gradients, optimizer state and EMA in device memory. Reading metrics, checkpointing
or exporting explicitly synchronizes and downloads the requested data.
Denoising groups use the actual maximum object count, independently of matcher
padding. Group/slot metadata also participates in the cache key: equal tensor
shapes can describe different denoising assignments and attention masks.

## COCO scripts and pretrained weights

After the example setup above (ONNX export additionally needs the `interop` extra
when using an installed wheel):

```bash
python -m examples.rtdetr.train --images coco/train2017 \
  --annotations coco/annotations/instances_train2017.json \
  --device vulkan --output runs/rtdetr --export-onnx
python -m examples.rtdetr.infer --checkpoint runs/rtdetr/last.mnet \
  --images picture.jpg --device vulkan --output detections.json
python -m pip install -r examples/rtdetr/requirements-eval.txt
python -m examples.rtdetr.evaluate --checkpoint runs/rtdetr/last.mnet \
  --images coco/val2017 --annotations coco/annotations/instances_val2017.json
```

`CocoDetection` maps sparse COCO category IDs to contiguous training labels,
clips boxes, excludes crowd/degenerate training annotations, converts RGB to
[0,1], and resizes. The example uses random horizontal flipping. Its transforms
are intentionally simpler than the official photometric/zoom/crop/multiscale
recipe. Full COCO convergence/AP has **not** been reproduced. Use a matching
external augmentation pipeline for an accuracy reproduction, and record its seed,
checkpoint and dataset hashes. `batches(multiscale=[...])` supports explicit
multiscale sizes, subject to the specialization transfer cost above.

Official pretrained weights are an explicit input, not an implicit download:

```python
from examples.rtdetr import load_torch_checkpoint
load_torch_checkpoint(model, "official_checkpoint.pth")
# Or: model.load_state_dict(torch_model.state_dict())
```

This optional reader uses `torch.load(..., weights_only=True)` on the CPU and
strictly validates state keys/shapes. It recognizes the official `model` and
`ema.module` layouts. The number of classes and architecture must match. Native
inference/training does not import PyTorch. Neither trained weights nor COCO data
are distributed with this package.

## Inference and interchange

`Predictor` compiles the model with native postprocessing. Predictions are the
highest sigmoid scores across query/class pairs, without NMS. Boxes are converted
to original-image `xyxy`; `PostProcessor` takes sizes in **width, height** order.
The user-facing predictor returns integer NumPy labels, float scores and boxes.
`coco_results` restores original category IDs and COCO `xywh` boxes.

The ONNX routes are native `.mnet` → standard ONNX and ONNX → native `.mnet`.
The default-domain opset 13–18 importer now handles detector convolution, pooling,
normalization, batched attention decomposition, bilinear GridSample, TopK,
gathers, slicing, concatenation and resize. Local ONNX functions are inlined.
Shape-only/constant subgraphs are folded at import; runtime model math executes
in the C++/Vulkan graph. `from_torch()` uses the modern ONNX exporter and has a
required test importing the pinned upstream detector.

This remains a static float32 inference contract: no dynamic input dimensions,
external tensor files, ONNX training graphs, arbitrary control flow or general
INT64 arithmetic. Shape constants can be integers; runtime indices and masks use
exact float32 integers within 2**24. ONNX export inserts standard integer/bool
casts where required. Unsupported operators, modes and dtypes raise errors.
Importing inference cannot reconstruct the original training loop or loss detach
semantics; construct the native model for training.

## Verification and remaining platform work

```bash
python -m pip install '.[test,torch]'
python -m pip install -r examples/rtdetr/requirements.txt
python examples/rtdetr/fetch_reference.py
PYTHONPATH=python python tools/test.py --swarm tests examples/rtdetr/tests
MUNET_TEST_VULKAN=1 PYTHONPATH=python python tools/test.py --vulkan-validation --swarm tests examples/rtdetr/tests
PYTHONPATH=python python -m examples.rtdetr.validate_training --device cpu
```

Library tests verify native primitive values/gradients; example tests verify full R50-vd default-width/six-layer
inference, full R50-backbone training with a compact decoder, denoising/auxiliary
losses, empty targets, checkpoints, postprocessing and ONNX/PyTorch interchange.
The standalone training gate uses all default R50 channels/layers/300 queries,
100 denoising queries and all trainable parameters, at 160×160 by default. Pass
`--size 640` to exercise the deployment resolution separately. Add `--plan-only`
to inspect its allocation plan without allocating the tensor arena or executing.
The default 2×3×640×640 training plan needs 5,327,510,016 arena bytes, plus its
staging mirror and host metadata; that size was planned but not executed locally.

Portable kernels are scalar/reduction baselines, not optimized GEMM/convolution
implementations. Cold shader/pipeline compilation can take minutes. The complete
arena and staging mirror must fit the device/host memory budget, with 64-bit arena offsets and 32-bit indices within each tensor. Tensor-sized Vulkan descriptors let the total arena exceed
`maxStorageBufferRange`; each individual tensor must still fit that limit.
Physical GPU/mobile/Raspberry Pi performance and full-resolution COCO training
remain unmeasured. The existing swarm continues to support its MSE/SGD contract;
distributed RT-DETR loss/BatchNorm semantics are a separate implementation step.
