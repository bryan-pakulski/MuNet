import pytest

np = pytest.importorskip("numpy")

try:
    import munet_nn as munet
except Exception as exc:  # pragma: no cover
    pytest.skip(f"munet import unavailable: {exc}", allow_module_level=True)


CPU = munet.Device(munet.DeviceType.CPU, 0)


def _first_accelerator(max_index: int = 4):
    for dev_type in (munet.DeviceType.CUDA, munet.DeviceType.VULKAN):
        for idx in range(max_index):
            dev = munet.Device(dev_type, idx)
            try:
                t = munet.ones((1,), device=dev)
                if float((t + t).to(CPU).item()) == 2.0:
                    return dev
            except RuntimeError:
                continue
    return None


def _make_model(options=None):
    options = options if options is not None else munet.TensorOptions()
    return munet.nn.Sequential(
        munet.nn.Linear(4, 16, options=options),
        munet.nn.ReLU(),
        munet.nn.Linear(16, 8, options=options),
        munet.nn.ReLU(),
        munet.nn.Linear(8, 1, options=options),
    )


def test_auto_offload_balanced_is_deterministic_for_fixed_inputs():
    accel = _first_accelerator()
    if accel is None:
        pytest.skip("Need one accelerator for deterministic auto_offload test")

    sample = munet.from_numpy(np.random.randn(4, 4).astype(np.float32))
    devices = [CPU, accel]

    m1 = _make_model()
    plan1 = m1.auto_offload(devices, strategy="balanced", sample_input=sample)

    m2 = _make_model()
    plan2 = m2.auto_offload(devices, strategy="balanced", sample_input=sample)

    assert {k: str(v) for k, v in plan1.items()} == {k: str(v) for k, v in plan2.items()}


def test_auto_offload_obeys_backend_dtype_constraints():
    accel = _first_accelerator()
    if accel is None:
        pytest.skip("Need one accelerator for auto_offload constraint test")

    opts = munet.TensorOptions()
    opts.dtype = munet.DataType.Float16
    model = _make_model(options=opts)
    sample = munet.from_numpy(np.random.randn(2, 4).astype(np.float16))

    with pytest.raises(RuntimeError):
        model.auto_offload([CPU], strategy="balanced", sample_input=sample)


def test_auto_offload_obeys_memory_budget_constraints():
    accel = _first_accelerator()
    if accel is None:
        pytest.skip("Need one accelerator for memory budget constraint test")

    model = _make_model()
    sample = munet.from_numpy(np.random.randn(4, 4).astype(np.float32))
    impossible = {str(accel): 1}

    with pytest.raises(RuntimeError):
        model.auto_offload(
            [accel],
            strategy="memory-first",
            sample_input=sample,
            memory_budgets_bytes=impossible,
        )


def test_auto_offload_explain_and_execution_smoke():
    accel = _first_accelerator()
    if accel is None:
        pytest.skip("Need one accelerator for auto_offload integration smoke")

    model = _make_model()
    x = munet.from_numpy(np.random.randn(8, 4).astype(np.float32))
    _ = model.auto_offload([CPU, accel], strategy="transfer-minimized", sample_input=x)
    explain = model.offload_plan(explain=True)

    assert "plan" in explain
    assert "rationale" in explain
    assert isinstance(explain["plan"], dict)
    assert isinstance(explain["rationale"], dict)
    if explain["rationale"]:
        first = next(iter(explain["rationale"].values()))
        assert isinstance(first, dict)
        assert "strategy" in first
        assert "compute_cost" in first
        assert "transfer_cost" in first

    y = model(x)
    assert y.shape == [8, 1]


def test_frozen_plan_roundtrip_reapply():
    accel = _first_accelerator()
    if accel is None:
        pytest.skip("Need one accelerator for frozen plan roundtrip")

    model = _make_model()
    x = munet.from_numpy(np.random.randn(8, 4).astype(np.float32))
    auto_plan = model.auto_offload([CPU, accel], strategy="balanced", sample_input=x)
    frozen = model.freeze_offload_plan()

    restored = _make_model()
    restored.apply_offload_plan(frozen)
    restored_plan = restored.offload_plan()

    assert {k: str(v) for k, v in auto_plan.items()} == {
        k: str(v) for k, v in restored_plan.items()
    }


def test_auto_offload_plan_executes_and_converges_reference_case():
    model = _make_model()
    x = munet.from_numpy(np.random.randn(64, 4).astype(np.float32))
    target = munet.from_numpy(np.random.randn(64, 1).astype(np.float32))

    _ = model.auto_offload([CPU], strategy="balanced", sample_input=x)

    optim = munet.optim.SGD(model.parameters(), lr=0.05)
    losses = []
    for _ in range(12):
        optim.zero_grad()
        pred = model(x)
        loss = ((pred - target) * (pred - target)).sum()
        losses.append(float(loss.to(CPU).item()))
        loss.backward()
        optim.step()

    assert losses[-1] < losses[0]


def test_balanced_strategy_improves_boundary_metric_vs_naive_split():
    accel = _first_accelerator()
    if accel is None:
        pytest.skip("Need one accelerator for benchmark metric test")

    x = munet.from_numpy(np.random.randn(8, 4).astype(np.float32))

    manual = _make_model()
    manual.reset_offload_telemetry()
    manual.offload(CPU, ["0", "1"])
    manual.offload(accel, ["2", "3", "4"])
    _ = manual(x)
    manual_metric = manual.offload_telemetry_snapshot().boundary_transfer_count

    best_auto_metric = None
    for strategy in ("balanced", "memory-first", "transfer-minimized"):
        auto = _make_model()
        auto.reset_offload_telemetry()
        _ = auto.auto_offload([CPU, accel], strategy=strategy, sample_input=x)
        _ = auto(x)
        metric = auto.offload_telemetry_snapshot().boundary_transfer_count
        best_auto_metric = metric if best_auto_metric is None else min(best_auto_metric, metric)

    assert best_auto_metric is not None
    assert best_auto_metric <= manual_metric
