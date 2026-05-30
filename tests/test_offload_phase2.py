import pytest

np = pytest.importorskip("numpy")

try:
    import munet_nn as munet
except Exception as exc:  # pragma: no cover
    pytest.skip(f"munet import unavailable: {exc}", allow_module_level=True)


Host = munet.Device(munet.DeviceType.VULKAN, 0)


def _first_accelerator(max_index: int = 4):
    for dev_type in (munet.DeviceType.VULKAN,):
        for idx in range(max_index):
            dev = munet.Device(dev_type, idx)
            try:
                x = munet.ones((1,), device=dev)
                if float((x + x).to(Host).item()) == 2.0:
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
    model.offload(Host, layers=["0", "2"])

    sample = munet.from_numpy(np.random.randn(2, 4).astype(np.float16))
    report = model.validate_offload_plan(sample)

    assert not report.valid
    assert any("does not support dtype" in msg for msg in report.errors)
