# Small-example validation

Measured locally with the native CPU reference backend, Python 3.12 and seed 7.
These runs use the checked-in defaults and pinned dataset bytes. They demonstrate
learning and artifact creation, not tuned benchmark results or GPU performance.

| Example | Training | Evaluation | Initial → final |
|---|---|---|---|
| MNIST CNN | 200 updates, batch 32, first 10,000 real training images | First 512 test images | Accuracy 8.59% → 87.70%; cross entropy 2.3095 → 0.3677 |
| Shape U-Net | 150 updates, batch 8, 256 generated 32x32 images | 64 generated images with a separate seed | IoU 0.2140 → 0.9637; Dice 0.3525 → 0.9815 |
| TinyGPT | 200 updates, batch 8, context 32, width 32, 2 layers | Four fixed batches from the last 10% of Tiny Shakespeare | Cross entropy 4.4393 → 2.6853; perplexity 84.71 → 14.66 |

Reproduce from the repository root:

```bash
make demo-mnist VULKAN=0
make demo-segmentation VULKAN=0
make demo-language-model VULKAN=0
```

Each run writes its configuration and measurements to `metrics.json`, and verifies
native inference export by comparing reloaded execution to the trained model.
The 11 offline example tests pass on CPU, covering all three command workflows,
continued training with restored optimizer/RNG state, decreasing training losses,
future-token isolation, seeded text generation, dataset checksums/cache recovery,
IDX decoding and generated masks. Run `make test-examples VULKAN=0`.

The learning/resume/export and causal-attention tests also opt into Vulkan through
`make test-examples-vulkan VULKAN=1`; CI runs them with software Vulkan and
synchronization validation. Physical GPU throughput and cross-device convergence
have not been measured by this change.
