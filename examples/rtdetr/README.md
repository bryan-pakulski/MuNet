# RT-DETR built with MuNet

This example composes MuNet's layers and operations into RT-DETR v1. It includes
the ResNet-vd backbone, hybrid encoder, deformable decoder, matching and detection
losses, denoising, a training loop, checkpoint/resume, and COCO inference/evaluation
workflows. It is source code you can adapt for your own application.

The example is excluded from the `munet-nn` wheel. MuNet provides tensors,
convolution, normalization, attention, sampling, assignment, gradients, optimizers
and checkpoints; model architecture and detection-specific policy live here.

From the repository root, after installing the [system prerequisites](../../docs/install.md#local-development-with-make):

```bash
make setup-rtdetr                  # Library build plus Pillow; add VULKAN=0 for CPU only
PYTHONPATH=python .venv/bin/python -m examples.rtdetr.train --help
PYTHONPATH=python .venv/bin/python -m examples.rtdetr.infer --help
PYTHONPATH=python .venv/bin/python -m examples.rtdetr.evaluate --help
make test-rtdetr                   # CPU acceptance tests; installs reference-test dependencies
make test-rtdetr-vulkan            # CPU/Vulkan acceptance tests with validation
```

For use with an installed MuNet wheel, install `requirements.txt` here and run
the same modules from the checkout root without `PYTHONPATH=python`. For COCO AP
evaluation, additionally install `requirements-eval.txt`. ONNX conversion and
loading official PyTorch weights use MuNet's optional `interop` and `torch` extras.

```python
from examples.rtdetr import rtdetr_r50vd, DetectorTrainer

model = rtdetr_r50vd(num_classes=2)
trainer = DetectorTrainer(model, device="vulkan", target_slots=32)
```

The [complete guide](../../docs/rtdetr.md) covers data formats, training, resume,
export, inference, performance limits and reproduction commands. The model and
loss reference is pinned in `tests/rtdetr-reference.json`; fetch it with
`python examples/rtdetr/fetch_reference.py`. Adapted files retain their upstream
[Apache-2.0 license](LICENSE-RT-DETR) and [attribution](NOTICE).

This is a correctness example. Full COCO AP reproduction, tuned hardware
performance, mixed precision and distributed RT-DETR training remain open.
