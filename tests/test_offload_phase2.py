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
                x = munet.ones((1,), device=dev)
                if float((x + x).to(CPU).item()) == 2.0:
                    return dev
            except RuntimeError:
                continue
    return None


def _make_model(options=None):
    options = options if options is not None else munet.TensorOptions()
    return munet.nn.Sequential(
        munet.nn.Linear(4, 8, options=options),
        munet.nn.ReLU(),
        munet.nn.Linear(8, 1, options=options),
    )


def test_validate_offload_plan_reports_unsupported_dtype_backend_combo():
    opts = munet.TensorOptions()
    opts.dtype = munet.DataType.Float16
    model = _make_model(options=opts)
    model.offload(CPU, layers=["0", "2"])

    sample = munet.from_numpy(np.random.randn(2, 4).astype(np.float16))
    report = model.validate_offload_plan(sample)

    assert not report.valid
    assert any("does not support dtype" in msg for msg in report.errors)


def test_validate_offload_plan_detects_ping_pong_pattern():
    accel = _first_accelerator()
    if accel is None:
        pytest.skip("Need one accelerator to construct ping-pong pattern")

    model = _make_model()
    # 0->CPU, 1->ACCEL, 2->CPU creates alternating boundary pattern.
    model.offload(CPU, layers=["0", "2"])
    model.offload(accel, layers=["1"])
    sample = munet.from_numpy(np.random.randn(2, 4).astype(np.float32))

    report = model.validate_offload_plan(sample)
    assert report.estimated_boundaries >= 2
    assert report.estimated_ping_pong_boundaries >= 1
    assert any("ping-pong" in w for w in report.warnings)


def test_offload_telemetry_reset_snapshot_consistency():
    accel = _first_accelerator()
    if accel is None:
        pytest.skip("Need one accelerator for transfer telemetry smoke")

    model = _make_model()
    model.offload(CPU, layers=["0", "1"])
    model.offload(accel, layers=["2"])
    model.set_offload_warnings(False)
    model.reset_offload_telemetry()

    x = munet.from_numpy(np.random.randn(8, 4).astype(np.float32))
    _ = model(x)
    snap = model.offload_telemetry_snapshot()
    assert snap.boundary_transfer_count >= 1
    assert snap.boundary_transfer_bytes > 0

    model.reset_offload_telemetry()
    after = model.offload_telemetry_snapshot()
    assert after.boundary_transfer_count == 0
    assert after.boundary_transfer_bytes == 0
    assert after.direction_counts == {}
