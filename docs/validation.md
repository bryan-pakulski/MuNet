# Validation report

RT-DETR now lives under `examples/rtdetr/` with its own model, workflows,
requirements and acceptance tests. The results below record the numerical work
that established the example; historical references to detector modules inside
the 0.3 wheel predate this separation. Current wheel validation requires generic
MuNet APIs, no model/example code, and NumPy as the only required Python dependency.
CI continues to execute both library and example tests.

## Library/example separation validation

- CPU library and relocated example tests: 72 passed, one expected Vulkan-only
  skip, and two ONNX Runtime tests deselected after automatic approval review
  blocked that dependency's telemetry request. Detector native/ONNX round trips
  still passed using ONNX's independent reference evaluator. CI retains the full
  ONNX Runtime checks.
- The rebuilt wheel contains 17 Python files, no model/example package, and
  NumPy as its sole required Python dependency. Source archives retain the
  complete example and upstream attribution while excluding Python caches.
- A fresh `make setup VULKAN=0` from the source archive built the library and
  worker without PyTorch, Pillow, ONNX Runtime, SciPy, COCO tools or an upstream
  model-reference download.
- In that clean environment, the installed wheel passed node/server training and
  recovery smoke checks. After separate example setup, the source RT-DETR example
  trained a compact model, resumed its checkpoint, exported native and ONNX
  inference, and reloaded native predictions using the installed wheel. PyTorch
  remained absent.
- All three example CLIs, reference checksum verification, package/source
  validation and actionlint passed. No model mathematics or native kernels were
  changed by this reorganization.

## RT-DETR 0.3 validation

The additional implementation was checked on the same CPU/software-Vulkan host.
The pinned official reference is fetched by SHA and each source file is checked
against `examples/rtdetr/tests/rtdetr-reference.json`. Reference adapters only remove registry/
distributed scaffolding, an unused torchvision import, and replace its float32
box-area helper with the same four-coordinate formula. Model/loss computations
remain the pinned upstream PyTorch code.

- 43 primitive/detector cases passed across CPU and software Vulkan, with API and
  synchronization validation enabled and zero validation error markers.
- The default-width, six-layer R50-vd inference test compares backbone features,
  hybrid-encoder features, logits and boxes with upstream (1×3×32×64, four queries).
- The R50 backbone training test includes a two-layer compact decoder, two images
  at 64×64, one empty target, fresh shared denoising inputs, all auxiliary loss
  terms and selected backbone/encoder/sampling/box-head/embedding gradients.
  Primitive sampling also has finite-difference checks. Tests avoid exact loss
  kinks, where a one-ULP coordinate change can legitimately change a derivative.
- The tensor-descriptor runtime passed a subsequent 52-case CPU/Vulkan regression
  run after adding deferred allocation and 64-bit arena offsets. A 6 GiB plan is
  tested without allocating its tensor arena. Full 640×640 R50 training plans
  successfully at 5,327,510,016 arena bytes; this is a planning-only check.
- A complete **default R50-vd** CPU training step used 80 classes, 256 hidden
  channels, eight heads, six layers, 300 queries and 100 DN queries, with a
  2×3×160×160 batch. All 387 participating trainable parameters had AdamW state;
  EMA advanced to step one. Loss was 38.91286 and the unclipped norm was 207.86143.
  The planned arena was 2,317,059,584 bytes and 21,268 kernels after fusion.
  First capture plus CPU execution took about 651 seconds under shared load.
  These numbers describe a correctness smoke, not a throughput benchmark.
- The compact-head fixed-batch learning test runs 100 AdamW updates and requires
  the final detection loss below 65% of its initial value. It is not COCO training.
- CPU training-state resume checks AdamW, BN, EMA, scheduler, host denoising RNG,
  trained-model export and `.mnet` reload over multiple subsequent steps.
- Both additional software-Vulkan workflow cases passed, including training-state
  resume and postprocessing, with no validation markers. A final CPU regression
  also checks denoising independence from matcher padding and state synchronization
  when switching between equal-shaped inputs with different group metadata.
- Complete compact-detector native → ONNX → native → `.mnet` round trips and the
  modern exporter import of the pinned upstream detector pass. Local independent
  ONNX execution uses its reference evaluator; ONNX Runtime is additionally
  required in CI because the local environment previously blocked its telemetry.
- 34 CPU runtime/packaging/swarm compatibility cases passed (the mixed-Vulkan
  swarm case is selected in the Vulkan CI job); 15 additional format/interop/
  packaging checks passed. Workflow YAML passed actionlint.
- The 0.3.0 Linux CPython 3.12 wheel built and installed successfully, includes the
  detector modules/license, and completed native node/server training, restart,
  checkpoint and headless-import smoke checks.

JUnit and generated validation reports are retained under `artifacts/validation`
in CI. Physical GPUs, 640×640 full training, full COCO AP, ARM correctness and
large-model swarm semantics are not established by these local checks. The
release CI matrix rebuilds Linux x86-64/aarch64 wheels and standalone binaries.

## Previous 0.2 foundation validation

Validated locally on 7 September 2026. **57 tests passed, 0 failed, 0 skipped**, including 18 swarm tests and three release-packaging tests. After adding a nineteenth swarm case for native TLS trust and hostname verification, a targeted run passed that case plus offline recovery and compiler-mismatch checks (3 passed). Vulkan API and synchronization validation were enabled; the captured output contained zero validation error markers. Loader diagnostics confirmed that `VK_LAYER_KHRONOS_validation` was loaded. Interoperability tests request that ONNX Runtime telemetry be disabled. A subsequent full-suite repeat was blocked by the execution environment when that dependency attempted a telemetry connection; the final worker changes were verified separately without importing it.

## Environment

| Component | Tested value |
|---|---|
| Host | Ubuntu 24.04, x86-64 |
| C++ compiler | GCC 13.3 |
| Python | 3.12.13 |
| Vulkan implementation | Mesa software Vulkan, `llvmpipe (LLVM 20.1.2, 256 bits)` |
| Vulkan loader / validation layers | 1.3.275 / 1.3.275.0 |
| GLSL compiler | glslang 14.0.0 |
| NumPy | 2.3.5 |
| PyTorch | 2.14.0+cpu |
| ONNX / ONNX Runtime | 1.22.0 / 1.29.0 |
| ONNX Script | 0.7.1 |

These are correctness results on a software Vulkan implementation. No physical NVIDIA, AMD, Intel, Apple, Adreno, Mali, or Raspberry Pi GPU was available for validation. Hardware compatibility and speed remain unverified. The release workflow targets native Linux x86-64 and aarch64 runners. Its remote run status is reported on the pull request; the local results here do not establish the manylinux 2.28 or aarch64 gates.

## What was checked

- Numerical forward results and gradients against finite differences and PyTorch.
- Matrix multiplication, view/broadcast derivatives, activation derivatives and reductions.
- Matching losses, gradients and parameter updates with PyTorch SGD over multiple steps.
- Shared parameter accumulation, cross-program parameter updates and state-dictionary compatibility.
- Fused/unfused execution with branches and reused temporary storage.
- Native input shape/dtype guards, scalar inputs, output-storage lifetime and malformed graph rejection.
- Parameter update snapshots, including simultaneous parameter swaps.
- `.mnet → ONNX → .mnet` round trips and independent ONNX Runtime results.
- Actual modern PyTorch export/import for a small eval-mode dense network.
- Deterministic SGD checkpoint save/load/resume.
- Explicit rejection of unsupported ONNX operators and dynamic inputs.
- Native archive version, graph topology, shape, entry and tensor-header validation, including oversized tensor headers and pickle-typed data.

During an 80-step measured training interval, the Vulkan test observed exactly one submission per step, only batch-input uploads, and zero additional readback bytes. Requested `.item()`, `.numpy()`, state-dictionary operations, exports and checkpoints do cause transfers, and are accounted for separately.

## Runnable MLP result

`examples/train_mlp.py --device vulkan --steps 200` trained a 4 → 16 → 2 dense network on a deterministic synthetic regression task:

| Measurement | Result |
|---|---:|
| Initial MSE | 3.5183215 |
| Final MSE | 0.0356060 |
| Training graph operations before fusion | 44 |
| Training dispatches per replay | 20 |
| Training device arena | 10,752 bytes |
| Inference graph operations before fusion | 7 |
| Inference dispatches per replay | 3 |
| Inference device arena | 3,840 bytes |

The example saved a trained inference program, an SGD training-step checkpoint and a standard ONNX graph. Reloaded native predictions matched within the example's `rtol=2e-5, atol=2e-5` tolerance. The example writes its serialized graphs under its selected output directory; generated models are not committed to the repository.

These counts show working fusion and memory planning. They are not a comparison with optimized vendor kernels. Device arena size excludes the staging mirror and host metadata. The full example's submission/readback totals include logging, checkpointing and conversion; the dedicated training test isolates the replay interval.

## Build/package checks

The Vulkan-enabled C++ library, Python extension and native worker built successfully. The release packaging checks also passed:

- A CPU/Vulkan `munet-nn` 0.2.0 CPython 3.12 wheel was built, repaired with auditwheel, installed in a clean environment, and exercised through `munet-node` and `munet-server` with a real training job and checkpoint reload.
- The retained `munet_nn` imports share types with `munet`; CPU import and execution succeed with the Vulkan loader deliberately unavailable.
- Wheel/source-distribution identity, contents, executable permissions and metadata passed the artifact validator and Twine.
- Standalone node and PyInstaller server archives completed the same training smoke check without a separate Python installation for either executable.
- A separate CMake consumer found the installed SDK through `find_package(MuNet)`, linked `MuNet::core`, and executed a graph.
- Both GitHub Actions workflows passed actionlint 1.7.12.

Local standalone archives were approximately 6.8 MiB (node), 35.5 MiB (server), and 0.09 MiB (SDK). The local host uses glibc 2.39, so these builds do not prove the release workflow's glibc 2.28 baseline. CI rebuilds and exercises the distributions in matching manylinux containers. See [installation and release details](install.md).

## Swarm checks and runnable result

The swarm cases include:

- Independent analytical Linear/MSE gradients and global-batch SGD updates, with uneven chunks and exact per-epoch sample coverage.
- Two native C++ workers training a 3 → 5 → 2 MLP against PyTorch SGD, including mixed CPU/software-Vulkan workers contributing to the same rounds. Final parameters matched at `rtol=2e-5, atol=2e-6` across three epochs, 69 accepted samples and nine optimizer steps.
- Worker capability filtering, throughput-based bundle sizing, lease renewal, expired work reassignment and a late original result winning safely.
- Four concurrent identical result submissions: one acceptance and three duplicate acknowledgements, with one optimizer update.
- Checkpoint/version, shape, data, sample-count, attempt, parameter-name and non-finite result rejection.
- An injected failure after writing the next checkpoint but before journal commit; owner restart resumed the previous committed state and accepted the retry once.
- A real native worker process killed after leasing; a second worker completed the same chunk after lease expiry.
- Native computation from prefetched files with the owner unavailable, durable outbox persistence, owner restart, lost response after commit, another owner restart, and idempotent delivery.
- Authenticated HTTP Range downloads, continuation of a partial cached artifact, and native compiler/source-hash mismatch rejection without a training contribution.
- Native HTTPS rejection of an untrusted certificate and a mismatched hostname, followed by successful training with an explicitly supplied private CA. Connection failures in `--once` mode return exit code 2.

Worker subprocesses ran with no `PYTHONPATH` and an invalid `MUNET_GLSLANG` path to verify that execution needs neither Python imports nor an on-node shader compiler. The separate CMake worker build disabled Python entirely.

`examples/swarm_mlp.py` also completed a ten-epoch 4 → 16 → 2 regression run with one CPU-only native binary and one Vulkan-enabled native binary. Both contributed work:

| Measurement | Result |
|---|---:|
| Initial full-dataset MSE | 1.8272933 |
| Final full-dataset MSE | 0.2468306 |
| Dataset size / epochs | 129 / 10 |
| Accepted sample contributions | 1,290 |
| Global batch / microbatch size | 32 / 4 |
| Committed optimizer updates | 50 |
| Accepted chunks across the two workers | 159 + 171 = 330 |

The run exported `artifacts/validated-swarm/trained.mnet` and `trained.onnx`. Reloaded native predictions matched the trained model. No Vulkan validation error markers appeared in worker logs. Elapsed time in the raw evidence is a local correctness observation with software Vulkan, not a physical-GPU or network scaling benchmark.

The swarm is currently a Linux/POSIX small-model MSE/SGD implementation. It has not validated real WAN drop patterns, physical GPUs, ARM/mobile packaging, remote TLS deployment, public/untrusted workers, large-model throughput, or RT-DETR distributed semantics. Normal computation is deterministic and sample-independent; it does not test BatchNorm, stochastic layers or detection-loss normalization.

## Evidence and reproduction

Generated logs and example models are build artifacts rather than source files. Running `tools/test.py` writes `artifacts/validation/tests.xml` and `tests.log`; GitHub Actions uploads those as `correctness-cpu` and `correctness-vulkan`. The targeted release-worker run also produced `release-worker-tests.xml` locally. The example commands below reproduce the model and training checks.

With the required dependencies and a Vulkan driver installed:

```bash
python tools/build.py --cmake-arg=-DMUNET_SWARM_NODE=ON
PYTHONPATH=python python tools/test.py --vulkan-validation --swarm
PYTHONPATH=python python examples/train_mlp.py --device vulkan --steps 200
```

Select the intended ICD/device explicitly when multiple implementations are installed. Software Vulkan is useful for continuous correctness testing; it is not evidence of physical-GPU performance.
