# Local GPT-3-Style Implementation Roadmap

> Note: **original GPT-3** used learned positional embeddings, LayerNorm, and GELU MLP blocks.
> **RMSNorm**, **RoPE**, and **SwiGLU** are not required for GPT-3 fidelity, but they are useful
> follow-on features if the goal is to evolve this demo toward more modern decoder architectures.

## Phase 1 — Baseline GPT-style local workflow

### Objectives
- [x] Provide a single Python entrypoint that can define a GPT-style decoder LM, train it locally, save/load it, and run generation/chat.
- [x] Keep the architecture compatible with current MuNet primitives so the flow works today on CPU and available accelerator backends.

### Action Points
- [x] Add a trainable decoder-only LM demo with token embeddings, learned positional embeddings, causal attention, residual blocks, LayerNorm, and output head.
- [x] Support sharded execution across multiple devices, with CPU fallback when GPUs are unavailable.
- [x] Add train / generate / chat CLI modes in one script.
- [x] Save weights plus JSON config so the same script can reconstruct the model for later inference.

### Exit Criteria
- [x] A user can run one script to train a small local model from text.
- [x] The script can reload saved weights/config and continue generation from a prompt.
- [x] The script can enter an interactive chat loop and produce responses from the trained model.

## Phase 2 — Modern decoder block options on current primitives

### Objectives
- [x] Add optional architectural upgrades that do **not** require new tensor primitives.
- [x] Make it easier to compare classic GPT-3 blocks versus more modern feed-forward variants.

### Action Points
- [x] Add optional SwiGLU feed-forward blocks using existing `sigmoid`, `mul`, and `Linear` support.
- [x] Add attention dropout and residual dropout knobs to the GPT demo for more stable local training.
- [x] Add a reusable sampling helper with temperature/top-k/top-p presets and repetition penalty.
- [x] Add lightweight evaluation metrics such as validation loss / perplexity logging and checkpoint selection.

### Exit Criteria
- [x] Users can switch between GELU and SwiGLU from the demo CLI.
- [x] The demo can report train/validation perplexity during training.
- [x] Sampling presets are exposed for deterministic and creative generation modes.

## Phase 3 — Missing primitives for RMSNorm and Rotary Embeddings

### Objectives
- [ ] Close the tensor/runtime gaps that block clean backend-accelerated RMSNorm and RoPE implementations.
- [ ] Keep new primitives available through Tensor API, autograd, Python bindings, and CPU/CUDA/Vulkan paths.

### Action Points
- [ ] Add reduction ops needed for RMSNorm, especially `mean(dim)` or an equivalent last-dimension reduction.
- [ ] Add reciprocal-root math (`rsqrt`) or scalar-power support so RMSNorm can avoid awkward graph workarounds.
- [ ] Add trigonometric unary ops (`sin`, `cos`) for rotary embedding tables.
- [ ] Add tensor slicing/select helpers (`slice`, `narrow`, or `split`) so RoPE can rotate head halves without host-side reshaping hacks.
- [ ] Add focused tests for the new math/shape ops across CPU and any compiled accelerator backends.
- [ ] Implement `nn::RMSNorm` and a reusable RoPE helper once the underlying ops exist.

### Exit Criteria
- [ ] RMSNorm runs end-to-end with autograd and backend coverage.
- [ ] RoPE can be applied inside attention without CPU fallbacks.
- [ ] The GPT demo can toggle LayerNorm vs RMSNorm and learned positions vs RoPE.

## Phase 4 — Full local train/inference quality and usability

### Objectives
- [ ] Make the model practical for longer local training runs and faster autoregressive inference.
- [ ] Reduce the gap between a teaching demo and a serious local experimentation stack.

### Action Points
- [ ] Add KV-cache support for incremental decoding so chat generation no longer recomputes the full prefix every step.
- [ ] Add optimizer/training utilities such as gradient clipping, weight decay, LR scheduling, and periodic checkpointing.
- [ ] Add dataset streaming / mini-batch loading from local text corpora instead of relying only on in-memory strings.
- [ ] Add mixed-precision and memory/perf measurement guidance for longer-context local runs.
- [ ] Add a reproducible inference command that loads a saved artifact and exposes a chat REPL suitable for local experimentation.

### Exit Criteria
- [ ] A user can train on a non-trivial local corpus with restarts/checkpoints.
- [ ] Chat inference is materially faster due to KV caching.
- [ ] The demo documents the practical limits for running larger local models on CPU / multi-GPU setups.
