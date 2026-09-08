# RT-DETR acceptance plan

## Fixed reference

Start with the official **RT-DETR v1 R50-vd** implementation and its `rtdetr_r50vd_6x_coco.yml` recipe, pinned at commit `29320b6fd828f8e0987a71426cf2d961b09dfed7`. Resolve and retain the included dataset, optimizer, dataloader, model, and runtime settings before a training comparison. Record the checkpoint hash, dataset split, image transforms, image size, batch size, seed, and framework versions in every result.

Reference: [official config](https://github.com/lyuwenyu/RT-DETR/blob/29320b6fd828f8e0987a71426cf2d961b09dfed7/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml).

Reproduce this baseline before modifying sampling, attention, losses, or the backbone. RT-DETRv2's optional discrete sampling is a different model choice; replacing v1's bilinear sampling would not establish parity with the selected baseline.

## Required operator families

| Area | Required behavior | Current foundation |
|---|---|---|
| Tensor semantics | FP32; indices/masks; view, broadcast, slice, concat, gather, accumulation; detach | Implemented; indices/masks use exact float32 integers, no general INT64 tensor arithmetic |
| Backbone | Conv2d/groups, pooling, BatchNorm buffers/gradients, activations | Implemented and compared with PyTorch |
| Hybrid encoder | Multiscale projections, resize, concat, FPN/PAN, positions, attention | Implemented native model |
| Decoder self-attention | Batched GEMM, stable masked softmax, views, LayerNorm, residuals | Implemented and compared with PyTorch |
| Deformable attention | Bilinear zeros/align_corners=False sampling and value/coordinate gradients | Implemented with portable gather-style adjoints |
| Query selection/refinement | TopK, gather, sigmoid/inverse sigmoid, detach | Implemented; explicit reference detach locations retained |
| Denoising | Fresh random label/box perturbations, embeddings, masks, target buckets | Implemented; host RNG is checkpointed |
| Criterion | VFL/focal/BCE, L1, GIoU, auxiliary/DN losses and normalizers | Implemented; pinned VFL recipe values/gradients tested |
| Matching | Rectangular linear assignment, empty targets | Native CPU/Vulkan Hungarian solver; float32 and bounded sizes |
| Training state | AdamW groups, scheduler, clipping, EMA, checkpoint/RNG | Implemented; mixed precision/loss scaling remain open |
| Evaluation/deployment | Preprocessing, box scaling, class top-k, COCO evaluation, model interchange | Implemented workflows; full COCO accuracy remains unmeasured |

The canonical machine-readable milestone/coverage snapshot is [rtdetr-coverage.json](rtdetr-coverage.json). This is a progress inventory, not a claim that an ONNX model containing one listed name is supported under every attribute/dtype.

## Two early technical gates

**Deformable sampling.** The reference core uses bilinear `grid_sample` with zero padding and `align_corners=False`. Implement and test the exact coordinate convention, border behavior, out-of-bounds samples, multiscale layout, and gradients with respect to both source values and sampling positions. Test non-square feature maps and samples near pixel/boundary transitions. Compare against CPU PyTorch and finite differences away from nondifferentiable boundaries. [Reference sampling core](https://github.com/lyuwenyu/RT-DETR/blob/29320b6fd828f8e0987a71426cf2d961b09dfed7/rtdetr_pytorch/src/zoo/rtdetr/utils.py).

Its backward pass accumulates contributions to shared feature pixels. Float atomic-add support cannot be a universal baseline assumption. Supply a portable accumulation route, such as gather-style adjoints or deterministic segmented accumulation, and gate any faster atomic path by capabilities and an explicit determinism policy. Benchmark this early: its cost can determine whether portable training is practical.

**Matching.** The reference creates a cost matrix, moves it to the CPU, and calls SciPy `linear_sum_assignment`. An initial correctness integration can use a declared host boundary for that exact solver, with its transfer and synchronization visible in profiling. It must not be described as fully resident GPU training. The final resident-training gate needs an exact device-side solver or another solution whose equivalence has been established. Approximate assignment would change the training algorithm. [Reference matcher](https://github.com/lyuwenyu/RT-DETR/blob/29320b6fd828f8e0987a71426cf2d961b09dfed7/rtdetr_pytorch/src/zoo/rtdetr/matcher.py).

## Milestones and exit criteria

| Gate | Deliverable | Exit criteria |
|---|---|---|
| M0 — working architecture | Dense model, graph AD, compiled SGD, Vulkan replay, model interchange | CPU and software-Vulkan numerical tests; PyTorch update parity; transfer accounting; round trips. **Implemented in this package.** |
| M1 — detector primitives | Conv forward/backward, pooling, normalization, batched GEMM, indexing, dtypes, train/eval buffers | Per-op forward/gradient comparisons; repeated-index accumulation; noncontiguous/view cases; no unreported CPU execution |
| M2 — attention correctness | Encoder/self-attention plus deformable sampling forward/backward | Exact layout/mask/boundary semantics, gradient checks, memory and latency measurements on physical GPUs |
| M3 — full model inference | Build RT-DETR in native Python modules and load a pinned PyTorch checkpoint | Compare intermediate tensors, final logits/boxes, ONNX Runtime results and COCO evaluation; no unsupported/fallback nodes |
| M4 — full training step | Exact criterion, matching, denoising, AdamW, normalization state, clipping and EMA as configured | Compare loss components, selected parameter gradients, updates, optimizer/buffer state and RNG progression for fixed batches; report matching boundary |
| M5 — reproduced training | Overfit a tiny set, run a fixed small-data comparison, then run the complete reference recipe | Comparable loss trajectory, deterministic checkpoint resume where requested, AP within predeclared tolerance based on baseline variance |
| M6 — portable release | Device-specific validation and optimized kernels | Desktop vendor matrix plus Adreno/Mali, Apple/MoltenVK and Pi tests; resource/precision capabilities reported; honest inference/training support matrix |
| M7 — multiple GPUs | Explicit data parallelism and gradient collectives | Single-/multi-device gradient equivalence, scaling measurements, bounded transfer/memory overhead, mixed-device behavior tested |

The 0.3 implementation supplies M1–M4 functionality, with explicit float32/index restrictions; see [the detector guide](rtdetr.md) for implemented and tested coverage. Full COCO recipe/AP reproduction (M5) and hardware performance (M6) remain open. M7 follows reliable single-device RT-DETR. The 0.2.0 [training swarm](swarm.md) already validates leased work and gradient aggregation for a small MSE/SGD dense model; it does not satisfy M7's RT-DETR, hardware, scaling or collective gates. Distributed detection losses need explicit global normalization, and BatchNorm needs a defined frozen/local/synchronized policy.

## Numerical and training protocol

Use fixed inputs and initial weights for primitive and single-step comparisons. Float32 smoke tests can start near `rtol=1e-4, atol=1e-5`, but tolerances must be chosen per operator and conditioning. Do not use one loose end-to-end tolerance to hide a systematic gradient error. Establish separate mixed-precision tolerances after a float32 baseline.

Check empty ground-truth images, one target, crowded images, border boxes, tiny boxes, varying class counts, and padded targets. Preserve detach locations in iterative box refinement and query selection. The reference decoder explicitly detaches some tensors; differentiating every forward edge would change the algorithm. [Reference decoder](https://github.com/lyuwenyu/RT-DETR/blob/29320b6fd828f8e0987a71426cf2d961b09dfed7/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py).

Compare every enabled loss term and its normalizer, including decoder/encoder auxiliary losses and denoising terms. Assignment and target selection are nondifferentiable decisions; validate their outcomes separately from the continuous gradients. [Reference criterion](https://github.com/lyuwenyu/RT-DETR/blob/29320b6fd828f8e0987a71426cf2d961b09dfed7/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_criterion.py).

For variable targets, prefer padded target tensors with masks and a small number of shape buckets. Treat multiscale training as guarded specialization, not a reason to recompile on every data value. Stateless random generation should key off seed, step, sample, and operation, so replay changes randomness correctly and checkpoints resume it.

Run a tiny overfit test before spending time on full COCO training. Then compare a fixed subset and the full recipe against the same framework/checkpoint settings. Define AP tolerance from repeated baseline runs and record seeds; a single apparently good run is not evidence of reproduced training.

## Performance protocol

Measure first-use compilation, warm model latency, complete training-step latency, input preprocessing/transfers, explicit matching time, host synchronization, peak live device memory, and output readbacks separately. Report throughput and latency distributions at the actual batch/image sizes. Preserve preprocessing and precision settings across comparisons.

Use PyTorch's appropriate native backend and at least one mature Vulkan inference implementation as comparisons. Do not compare a float16 or reduced-resolution MuNet path with a float32/full-resolution baseline without labeling the difference. Repeat mobile/Pi measurements after warm-up to observe thermal throttling and sustained performance.

The resident-training gate allows input uploads and requested results/checkpoints. Intermediate activations, gradients, and optimizer state must remain on the GPU. A CPU matcher or an unsupported operator is a recorded boundary, not an invisible fallback.

No physical-hardware throughput target is claimed. Full-model functionality now runs, while full COCO accuracy and physical-hardware performance remain separate acceptance gates.
