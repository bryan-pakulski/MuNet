# Build models from MuNet layers

These are source examples: model definitions, datasets and application code live
here, outside the installed `munet` / `munet_nn` packages. Run commands from the
repository root. Only MuNet and NumPy perform training and inference; Pillow is
an optional dependency for reading and writing images.

| Example | Building blocks | Data | Result |
|---|---|---|---|
| [MNIST](mnist/README.md) | Convolution, ReLU, pooling, linear classifier | Automatically downloaded MNIST (~12 MB compressed) | Digit probabilities and labeled image grid |
| [Segmentation](segmentation/README.md) | Small U-Net, skip connections, upsampling, BCE + Dice loss | Generated RGB shapes with exact binary masks | Probability maps, masks and overlays |
| [Language model](language_model/README.md) | Embeddings, causal attention, LayerNorm, residual MLPs | Automatically downloaded Tiny Shakespeare (~1.1 MB), or your own text | Autoregressive character generation |
| [RT-DETR](rtdetr/README.md) | ResNet, hybrid encoder, deformable decoder, detection losses | Generated COCO smoke data or a COCO dataset | Detection training, inference and evaluation |
| [MLP](train_mlp.py) | Linear layers, ReLU, MSE, SGD | Generated regression data | Small introductory training loop |
| [Swarm MLP](swarm_mlp.py) | Compiled gradients, owner/worker rounds | Generated regression data | Distributed MSE/SGD training |

## Quick start

Install the [system prerequisites](../docs/install.md#local-development-with-make),
then choose an example:

```bash
make setup-examples
make demo-mnist
make demo-segmentation
make demo-language-model
```

These defaults build and execute on Vulkan using intentionally small models.
Select `DEVICE=cpu` for explicit CPU fallback, or `VULKAN=0` for a CPU-only build
and execution. Direct Python commands default to Vulkan; pass `--device cpu`
when using the CPU fallback. The first execution compiles the graph and, on
Vulkan, shaders. Kernel performance and physical GPU support are separate from
these learning examples. Select smaller `--steps`, `--batch-size` or `--width`
through `EXAMPLE_ARGS` for a shorter run.

Dataset downloads are automatic on first use, then verified and reused. Nothing
is downloaded during `make setup-examples` apart from build/Python dependencies.
See [dataset sources and caching](DATASETS.md). Offline smoke runs after setup:

```bash
make demo-mnist EXAMPLE_ARGS='--synthetic --steps 5 --limit 64 --eval-samples 16'
make demo-segmentation EXAMPLE_ARGS='--steps 5 --samples 32 --eval-samples 8'
make demo-language-model EXAMPLE_ARGS='--synthetic --steps 5 --context 8 --width 8 --heads 2 --layers 1 --sample-tokens 16'
make test-examples
```

Synthetic digits are seven-segment renderings, **not MNIST**. Generated text is
a repeated toy corpus for checking execution. Neither is a benchmark dataset.

## Reading and adapting the code

Each new example separates `model.py`, `data.py`, `train.py`, and an inference
entry point. [common.py](common.py) holds the small shared compiled training step,
class-index cross entropy, checkpoint handling and download helper. All use
AdamW, gradient clipping and fixed-shape FP32 batches. The Python loop samples
data and explicitly reads back a scalar loss per update; forward, backward and
optimizer arithmetic execute in the native compiled graph.

Every training command writes these files under `artifacts/<example>/`:

- `training.mnet`: data-only model, optimizer, sampler RNG, update count and configuration.
- `inference.mnet`: a separately compiled inference graph, checked by saving and reloading it.
- `metrics.json`: initial/final held-out metrics, configuration and runtime counters.
- Images or generated text, depending on the example.

`--resume PATH --steps N` performs **N additional updates**. Keep model, training
data, seed, learning rate and batch size unchanged; checkpoint validation rejects
incompatible settings. Datasets and outputs are ignored by git. Checkpoints are
saved at the end of a successful invocation; these short demos do not implement
periodic recovery from interruptions inside a run.

The exported graphs accept batch size one. MNIST and segmentation exports return
raw logits; the language model returns logits at every position in its fixed
context. For example, after the MNIST demo:

```bash
PYTHONPATH=python .venv/bin/python - <<'PY'
import numpy as np
from PIL import Image
import munet as mu

image = np.asarray(Image.open('artifacts/mnist/sample.png'), np.float32)[None, None] / 255
model = mu.load('artifacts/mnist/inference.mnet')
print('Predicted digit:', model(image).numpy().argmax(-1).item())
PY
```

`make test-examples` runs Vulkan validation and CPU reference checks by default,
as does `make test-examples-vulkan`. They are also included in CPU/Vulkan CI. Tests
cover learning, checkpoint continuation, native export, causal masking, dataset
integrity and offline command workflows. See [measured example results](VALIDATION.md).
