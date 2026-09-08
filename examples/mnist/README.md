# MNIST digit classification

`DigitCNN` combines two convolution/ReLU/average-pooling stages with a linear
10-class head. Inputs are grayscale `N x 1 x 28 x 28` FP32 images in `[0, 1]`;
outputs are class logits. Training uses stable class-index cross entropy and
AdamW. Everything executes through MuNet's native runtime.

From the repository root:

```bash
make demo-mnist
# Defaults: 200 updates, batch 32, 10,000 training and 512 held-out images.
# More training and the full dataset:
make demo-mnist EXAMPLE_ARGS='--steps 1000 --limit 60000 --eval-samples 10000'
```

The [MNIST files](../DATASETS.md) download automatically and are cached. Outputs
under `artifacts/mnist/` include checkpoints, metrics, `sample.png` and
`predictions.png` (true and predicted labels). Validation reports cross entropy
and accuracy on the selected test subset before and after training.

```bash
PYTHONPATH=python .venv/bin/python -m examples.mnist.infer \
  --checkpoint artifacts/mnist/training.mnet --image artifacts/mnist/sample.png

# Continue a default run for another 200 updates, retaining AdamW and sampler state:
make demo-mnist EXAMPLE_ARGS='--resume artifacts/mnist/training.mnet --steps 200'

# Offline execution check, using generated seven-segment digits instead of MNIST:
make demo-mnist EXAMPLE_ARGS='--synthetic --steps 10 --limit 128 --eval-samples 32 --output artifacts/mnist-synthetic'
```

Inference accepts an image, converts/resizes it to grayscale 28x28, and prints
the predicted digit and ten probabilities. Use `--invert` for dark digits on a
light background. The model expects a single centered digit; preprocessing does
not detect/crop digits in natural photos.

Vulkan is the default. Select `DEVICE=cpu` on Make commands or `--device cpu`
on Python commands for the explicit CPU fallback. See the [example overview](../README.md) for setup,
native inference graph loading and checkpoint rules.
