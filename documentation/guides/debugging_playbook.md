# Debugging Playbook

This playbook is a practical checklist for investigating MuNet runtime/demo
issues, especially Vulkan backend and dispatch problems.

---

## 0) Start with a clean, known-good build

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j
```

For Python runs, prefer the just-built module:

```bash
PYTHONPATH=build/debug python -c "import munet_nn as munet; print('ok')"
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

MuNet supports these matmul forms:

- `a @ b`
- `a.matmul(b)`
- `munet.matmul(a, b)`

If you still see this error, your runtime may be importing an older build.
Confirm:

```bash
PYTHONPATH=build/debug python - <<'PY'
import munet_nn as munet
print(hasattr(munet, "matmul"))
x = munet.ones((2,2)); y = munet.ones((2,2))
print((x @ y).shape, x.matmul(y).shape, munet.matmul(x, y).shape)
PY
```

---

## 2) Vulkan backend discovery is too optimistic

### Symptom

- Device appears in probe list, but fails later during backward/optimizer.

### Recommended check

Probe each device with a **real backend op + copy-back** (not allocation-only).
This catches many invalid devices early without over-constraining discovery.

```python
a = munet.ones((1,), device=dev)
b = munet.ones((1,), device=dev)
c = a + b
_ = c.to(munet.Device(munet.DeviceType.VULKAN, 0))
```

If logs show `invalid device ordinal` for higher indices, that is expected when
your probe range exceeds actual device count.

---


### Symptom


### Common root causes and fixes

1. **Asynchronous error reporting** hides the true failing op.
   - Re-run with:
     ```bash
     ```

2. **Cross-device autograd graph edges** from `vulkan_tensor.to(dev)` replicas.
   - Ensure per-device parameters are leaf tensors:
     ```python
     replica = vulkan_param.to(dev).detach()
     replica.requires_grad = True
     ```

3. **One “available” Vulkan is not actually healthy for autograd kernels.**
   - Use the real forward+backward health probe in section 2.

4. **Silent fallback masking backend behavior.**
   - Temporarily enable fail-fast fallback:
     ```bash
     MUNET_FAIL_FAST_VULKAN_UNSUPPORTED=1 python ...
     ```

---

## 4) Unexpected Vulkan backend→Vulkan fallback

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
4. Enable `MUNET_FAIL_FAST_VULKAN_UNSUPPORTED=1` to catch first unexpected fallback with a stack trace

---

## 5) Multi-device all-reduce issues (vulkan fallback mode)

For backend all-reduce vulkan fallback, ensure rendezvous env knobs are set for
the current run:

```bash
MUNET_ALLREDUCE_MODE=vulkan_staging
MUNET_ALLREDUCE_WORLD_SIZE=<num_devices>
MUNET_ALLREDUCE_GROUP=<stable_group_name>
MUNET_ALLREDUCE_TIMEOUT_MS=30000
```

Important: `MUNET_ALLREDUCE_WORLD_SIZE` must match the number of **active
participants** in the current run (not total discovered devices).

If gradients diverge between replicas:

1. Verify all replicas participate in the reduction each step.
2. Verify reduced gradients are averaged (not summed) before optimizer step.
3. Print per-replica max drift after each update.
   a specific pair is unstable on your driver stack, retry with

---

## 6) Test-suite contamination from backend overrides

### Symptom


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
import munet_nn as munet
print("matmul:", hasattr(munet, "matmul"))
PY
```

---

## 8) What to include when filing a bug

Please include:

1. Exact command run.
2. Full traceback/log output.
4. Output of fallback telemetry snapshot (if relevant).

---

## 9) Offload plan troubleshooting (Phase 1)

If using `model.offload(device, layers=[...])`:

1. Verify layer paths exist via `model.named_modules()`.
2. Verify plan with `model.offload_plan()`.
3. If you hit vulkan-layer errors, paths must match module names exactly
   (e.g. `0`, `1`, `encoder.block0`, etc.).
4. For boundary-transfer debugging, run with:
   - `MUNET_DISPATCH_DECISION_DUMP=1`
   - optional `MUNET_FAIL_FAST_VULKAN_UNSUPPORTED=1`

## 10) Offload plan validation and transfer hotspots (Phase 2)

For plan validation and transfer-cost insight:

1. Run plan checks with a representative sample:
   - `report = model.validate_offload_plan(sample_input)`
   - inspect: `report.valid`, `report.errors`, `report.warnings`
2. Inspect boundary estimates:
   - `report.estimated_boundaries`
   - `report.estimated_ping_pong_boundaries`
3. Track runtime transfer telemetry:
   - `model.reset_offload_telemetry()`
   - run one/more forwards
   - `snap = model.offload_telemetry_snapshot()`
   - inspect: `snap.boundary_transfer_count`, `snap.boundary_transfer_bytes`,
     `snap.direction_counts`
4. Tune warnings:
   - `model.set_offload_warnings(True/False)`
   - `model.set_offload_warning_threshold_bytes(...)`
