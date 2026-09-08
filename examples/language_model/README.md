# A small causal language model

`TinyGPT` demonstrates the pieces used in decoder language models: token and
position embeddings, pre-normalization Transformer blocks, masked multi-head
self-attention, residual MLPs and a vocabulary projection. It learns to predict
the next character using class-index cross entropy and AdamW.

This is a small model trained from scratch for teaching and runtime coverage.
It is not a pretrained assistant. The default short run learns character patterns;
fluent text is not an acceptance criterion.

```bash
make demo-language-model
```

The [Tiny Shakespeare corpus](../DATASETS.md) downloads automatically (~1.1 MB).
Defaults are 200 updates, batch 8, context 32, width 32, 4 heads and 2 layers.
The corpus has a contiguous 90/10 train/validation split. Evaluation uses four
fixed batches from validation and reports mean character cross entropy and
perplexity; it is a sampled estimate, not a pass over the entire validation split.

Outputs under `artifacts/language-model/` include model/optimizer checkpoints,
the vocabulary, configuration, native inference graph, metrics and `sample.txt`.

```bash
PYTHONPATH=python .venv/bin/python -m examples.language_model.generate \
  --checkpoint artifacts/language-model/training.mnet \
  --prompt 'First Citizen:' --tokens 200 --temperature 0.8 --top-k 20

make demo-language-model EXAMPLE_ARGS='--resume artifacts/language-model/training.mnet --steps 1000'

# Use your own UTF-8 file, with enough characters for both splits:
make demo-language-model EXAMPLE_ARGS='--text my-corpus.txt --steps 500 --output artifacts/my-language-model'

# Small offline smoke run:
make demo-language-model EXAMPLE_ARGS='--synthetic --steps 10 --context 8 --width 8 --heads 2 --layers 1 --sample-tokens 32 --output artifacts/language-model-synthetic'
```

Generation samples with a seeded NumPy RNG, temperature and optional top-k filtering
(`--top-k 0` uses the full vocabulary). Unknown prompt characters map to `<unk>`;
an emitted unknown token is displayed as the replacement character `�`.
Short prompts are right-padded and read at the last real position. The causal
mask blocks future positions, including padding. Long prompts use the most recent
context window. Generation recomputes the window per character; there is no KV
cache, subword tokenizer, pretrained-weight importer or instruction tuning here.

Model inputs are `N x context` token indices stored as exact FP32 values, matching
the current MuNet indexing contract. Outputs are `N x context x vocabulary` logits.
Width must be divisible by the number of heads. Increasing width, context and
layers increases CPU work and Vulkan graph/shader compilation cost.

Vulkan is the default. Select `DEVICE=cpu` on Make commands or `--device cpu`
on Python commands for the explicit CPU fallback. See the [example overview](../README.md) for setup and checkpoint rules.
