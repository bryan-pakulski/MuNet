# Debugging Playbook

This playbook is a practical checklist for investigating MuNet runtime/demo
issues, especially accelerator and dispatch problems.

---

## 0) Start with a clean, known-good build

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j
```

For Python runs, prefer the just-built module:

```bash
PYTHONPATH=build/debug python -c "import munet; print('ok')"
```

---

## 1) Python API mismatch errors

### Symptom: `AttributeError: module 'munet' has no attribute 'mse_loss'`

`mse_loss` is a **tensor method**, not a module function.

Use:

```python
loss = pred.mse_loss(target)
```

### Symptom: `AttributeError: 'munet.Tensor' object has no attribute 'matmul'`

MuNet now supports all PyTorch-style forms:

- `a @ b`
- `a.matmul(b)`
- `munet.matmul(a, b)`

If you still see this error, your runtime may be importing an older build.
Confirm:

```bash
PYTHONPATH=build/debug python - <<'PY'
import munet
print(hasattr(munet, "matmul"))
x = munet.ones((2,2)); y = munet.ones((2,2))
print((x @ y).shape, x.matmul(y).shape, munet.matmul(x, y).shape)
PY
```

---

## 2) Accelerator discovery is too optimistic

### Symptom

- Device appears in probe list, but fails later during backward/optimizer.

### Recommended check

Probe each device with a **real backend op + copy-back** (not allocation-only).
This catches many invalid devices early without over-constraining discovery.

```python
a = munet.ones((1,), device=dev)
b = munet.ones((1,), device=dev)
c = a + b
_ = c.to(munet.Device(munet.DeviceType.CPU, 0))
```

If logs show `invalid device ordinal` for higher indices, that is expected when
your probe range exceeds actual device count.

---

## 3) CUDA illegal memory access in multi-GPU demo

### Symptom

`RuntimeError: CUDA Error: an illegal memory access was encountered`

### Common root causes and fixes

1. **Asynchronous error reporting** hides the true failing op.
   - Re-run with:
     ```bash
     CUDA_LAUNCH_BLOCKING=1 python demos/multigpu/multi_gpu_allreduce_training_demo.py --steps 1
     ```

2. **Cross-device autograd graph edges** from `cpu_tensor.to(dev)` replicas.
   - Ensure per-device parameters are leaf tensors:
     ```python
     replica = cpu_param.to(dev).detach()
     replica.requires_grad = True
     ```

3. **One “available” GPU is not actually healthy for autograd kernels.**
   - Use the real forward+backward health probe in section 2.

4. **Silent fallback masking backend behavior.**
   - Temporarily enable fail-fast fallback:
     ```bash
     MUNET_FAIL_FAST_ACCELERATOR_CPU_FALLBACK=1 python ...
     ```

---

## 4) Unexpected accelerator→CPU fallback

Use both programmatic telemetry and log dumps:

```bash
MUNET_DISPATCH_DECISION_DUMP=1 python your_script.py
```

Python helpers:

- `munet.dispatch_policy_snapshot()`
- `munet.dispatch_decision_debug_dump(op_name, tensor)`
- `munet.fallback_telemetry_snapshot()`
- `munet.reset_fallback_telemetry()`

Suggested workflow:

1. `munet.reset_fallback_telemetry()`
2. Run a minimal repro
3. Inspect `munet.fallback_telemetry_snapshot()`
4. Enable `MUNET_FAIL_FAST_ACCELERATOR_CPU_FALLBACK=1` to catch first unexpected fallback with a stack trace

---

## 5) Multi-device all-reduce issues (host fallback mode)

For Python demos, ensure environment knobs are explicitly set:

```bash
MUNET_ALLREDUCE_MODE=host_fallback
MUNET_ALLREDUCE_WORLD_SIZE=<num_devices>
MUNET_ALLREDUCE_GROUP=<stable_group_name>
```

If gradients diverge between replicas:

1. Verify all replicas call `grad.all_reduce()` every step.
2. Verify gradients are averaged after all-reduce sum.
3. Print per-replica max drift after each update.
4. Mixed backend pairs (e.g. CUDA + Vulkan) are supported in the demo, but if
   a specific pair is unstable on your driver stack, retry with
   `CUDA_LAUNCH_BLOCKING=1` and narrow to a minimal reproducer per backend.

---

## 6) Test-suite contamination from backend overrides

### Symptom

Parameterized suites unexpectedly include `cuda_0`, then fail with
`CUDA backend not compiled`.

### Cause

A test temporarily overrides backend registration and does not restore it.

### Fix pattern

- Use RAII-scoped override helpers in tests.
- Always restore the backend factory matching compile configuration.

---

## 7) Fast triage command set

```bash
# Build
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j

# Targeted C++ backend/dispatch tests
./build/debug/munet_tests --gtest_filter='BackendManagerTest.*'

# List currently parameterized devices in AllBackends tests
./build/debug/munet_tests --gtest_filter='AllBackends/*' --gtest_list_tests

# Python binding sanity
PYTHONPATH=build/debug python - <<'PY'
import munet
print("matmul:", hasattr(munet, "matmul"))
PY
```

---

## 8) What to include when filing a bug

Please include:

1. Exact command run.
2. Full traceback/log output.
3. Whether `CUDA_LAUNCH_BLOCKING=1` changes the failing line.
4. Output of fallback telemetry snapshot (if relevant).
5. GPU/driver/runtime details (`nvidia-smi`, CUDA version, visible devices).
