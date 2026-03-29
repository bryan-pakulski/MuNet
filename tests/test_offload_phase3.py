import pytest

np = pytest.importorskip("numpy")

try:
    import munet
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

    y = model(x)
    assert y.shape == [8, 1]
