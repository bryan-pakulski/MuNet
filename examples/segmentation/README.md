# Foreground segmentation and masks

`TinyUNet` demonstrates convolutional features, pooling, nearest upsampling and
skip concatenation. It predicts one foreground logit per pixel. The loss combines
binary cross entropy with soft Dice loss. This is binary semantic segmentation:
all shapes share a foreground class, including overlapping shapes.

```bash
make demo-segmentation
```

The default run generates 256 training and 64 independent validation images at
32x32, then trains for 150 updates with batch size 8. Generation needs no download.
`--size` must be even and at least 16. Metrics report global foreground IoU and
Dice at a 0.5 probability threshold. Inputs are RGB FP32 `[0, 1]` in NCHW layout.

Outputs in `artifacts/segmentation/` include `sample.png`, training/inference
archives and metrics. `masks.png` shows four columns: input, true mask, predicted
probability and the thresholded prediction overlaid on the input.

```bash
PYTHONPATH=python .venv/bin/python -m examples.segmentation.infer \
  --checkpoint artifacts/segmentation/training.mnet \
  --image artifacts/segmentation/sample.png --output artifacts/segmentation/prediction

make demo-segmentation EXAMPLE_ARGS='--resume artifacts/segmentation/training.mnet --steps 150'

# A different dataset/model shape, saved separately:
make demo-segmentation EXAMPLE_ARGS='--size 64 --width 8 --samples 512 --output artifacts/segmentation-64'
```

Inference resizes the input to the training resolution and writes `probability.png`
and `mask.png` into the selected output directory, resized back to the original
image dimensions. Binary masks use nearest-neighbor resizing. The trained demo
recognizes its generated shape distribution; natural-image segmentation requires
replacing the generator with paired RGB images/masks and retraining.

Vulkan is the default. Select `DEVICE=cpu` on Make commands or `--device cpu`
on Python commands for the explicit CPU fallback. See the
[example overview](../README.md) for setup, checkpoint rules and export contracts.
