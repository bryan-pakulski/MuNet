# Coverage feedback loop plan

This document proposes additional coverage reports to complement `dtype_coverage_report.csv` so MuNet has a tight build-measure-prioritize loop.

## 1) Runtime execution coverage report

**Goal:** detect where dispatch says an op is available but runtime execution still fails.

**Report shape:**
- backend, dtype, op, input_shape_case, dispatch_path (backend/fallback), runtime_status (pass/fail), error

**Why next:** catches gaps between capability metadata and real kernels.

## 2) Numerical parity report (reference correctness)

**Goal:** measure error against a reference implementation (CPU float32 baseline or PyTorch) for each op/backend/dtype.

**Report shape:**
- backend, dtype, op, shape_case, max_abs_err, max_rel_err, status (within threshold)

**Why next:** tells us where fallback/native paths are numerically unstable.

## 3) Gradient parity report (autograd)

**Goal:** verify backward correctness for training ops across fallback and native paths.

**Report shape:**
- backend, dtype, op, gradcheck_status, grad_err, error

**Why next:** highlights missing or incorrect backward support quickly.

## 4) Fallback rate report

**Goal:** quantify how often workloads hit CPU fallback instead of native backend kernels.

**Report shape:**
- workload, backend, dtype, op, calls_total, calls_fallback, fallback_pct

**Why next:** converts parity gaps into prioritizable product impact.

## 5) Serialization compatibility report

**Goal:** ensure model artifacts round-trip across dtype/backend combinations.

**Report shape:**
- artifact_version, dtype, backend, load_status, forward_status, mismatch_details

**Why next:** deploy safety for mixed-dtype models.

## 6) Performance coverage report

**Goal:** tie correctness + parity coverage to performance regressions.

**Report shape:**
- backend, dtype, op, shape_case, latency_ms, throughput, vs_baseline_pct

**Why next:** prevents fallback parity wins from silently becoming performance losses.

## 7) Memory/transfer behavior report

**Goal:** track transfer volume and allocator behavior per backend/dtype.

**Report shape:**
- backend, dtype, op, h2d_bytes, d2h_bytes, d2d_bytes, alloc_bytes, peak_bytes

**Why next:** identifies hidden transfer bottlenecks in fallback-heavy paths.

## Recommended implementation order

1. Runtime execution coverage (fastest value)
2. Numerical parity report
3. Gradient parity report
4. Fallback rate report
5. Performance + memory reports
6. Serialization compatibility report

## CI integration recommendation

- Produce all reports as machine-readable CSV/JSON artifacts per build.
- Fail CI on severe regressions (new runtime failures, gradient check fails, large numeric drift).
- Track trends over time (fallback %, latency regressions) in dashboard snapshots.
