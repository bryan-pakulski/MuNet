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


def _mini_unet_like():
    model = munet.nn.Sequential()
    model.add(munet.nn.Conv2d(3, 8, 3, padding=1))
    model.add(munet.nn.ReLU())
    model.add(munet.nn.MaxPool2d(2, 2))
    model.add(munet.nn.Conv2d(8, 16, 3, padding=1))
    model.add(munet.nn.ReLU())
    model.add(munet.nn.Upsample(2))
    model.add(munet.nn.Conv2d(16, 8, 3, padding=1))
    model.add(munet.nn.ReLU())
    model.add(munet.nn.Conv2d(8, 2, 1))
    return model


def test_unet_like_auto_offload_explain_schema_and_execution():
    accel = _first_accelerator()
    if accel is None:
        pytest.skip("Need one accelerator for UNet-like offload test")

    model = _mini_unet_like()
    x = munet.from_numpy(np.random.randn(2, 3, 32, 32).astype(np.float32))

    _ = model.auto_offload([CPU, accel], strategy="transfer-minimized", sample_input=x)
    explain = model.offload_plan(explain=True)

    assert "plan" in explain and isinstance(explain["plan"], dict)
    assert "rationale" in explain and isinstance(explain["rationale"], dict)
    assert len(explain["plan"]) > 0
    for _, entry in explain["rationale"].items():
        assert isinstance(entry, dict)
        assert "strategy" in entry
        assert "compute_cost" in entry
        assert "transfer_cost" in entry

    y = model(x)
    assert y.shape == [2, 2, 32, 32]
