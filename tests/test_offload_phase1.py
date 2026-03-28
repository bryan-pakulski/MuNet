import pytest
np = pytest.importorskip("numpy")

try:
    import munet
except Exception as exc:  # pragma: no cover - environment-dependent import
    pytest.skip(f"munet import unavailable: {exc}", allow_module_level=True)


CPU = munet.Device(munet.DeviceType.CPU, 0)


def _detect_accelerators(max_index: int = 4):
    devices = []
    for dev_type in (munet.DeviceType.CUDA, munet.DeviceType.VULKAN):
        for idx in range(max_index):
            dev = munet.Device(dev_type, idx)
            try:
                a = munet.ones((1,), device=dev)
                b = munet.ones((1,), device=dev)
                c = a + b
                if float(c.to(CPU).item()) != 2.0:
                    raise RuntimeError("probe mismatch")
            except RuntimeError:
                continue
            devices.append(dev)
    return devices


def _make_model():
    return munet.nn.Sequential(
        munet.nn.Linear(4, 8),
        munet.nn.ReLU(),
        munet.nn.Linear(8, 1),
    )


def test_offload_plan_crud_and_path_resolution_cpu():
    model = _make_model()
    model.offload(CPU, layers=["0", "1"])
    plan = model.offload_plan()

    assert "0" in plan
    assert "1" in plan
    assert str(plan["0"]) == "cpu:0"
    assert str(plan["1"]) == "cpu:0"

    model.clear_offload()
    assert model.offload_plan() == {}


def test_offload_unknown_layer_raises():
    model = _make_model()
    with pytest.raises(RuntimeError):
        model.offload(CPU, layers=["does_not_exist"])


def test_offload_boundary_transfer_and_backward_mixed_chain():
    accelerators = _detect_accelerators(max_index=4)
    if len(accelerators) < 2:
        pytest.skip("Need >=2 accelerators for boundary transfer mixed-chain test")

    d0, d1 = accelerators[0], accelerators[1]
    model = _make_model()
    model.offload(d0, layers=["0", "1"])
    model.offload(d1, layers=["2"])

    x = munet.from_numpy(np.random.randn(16, 4).astype(np.float32))  # CPU input
    y = munet.from_numpy(np.random.randn(16, 1).astype(np.float32)).to(d1)

    pred = model(x)
    loss = pred.mse_loss(y)
    loss.backward()

    for p in model.parameters():
        assert p.has_grad()
        p.step(0.01)

    assert str(pred.device) == str(d1)


def test_offload_inference_parity_vs_single_device():
    accelerators = _detect_accelerators(max_index=4)
    if len(accelerators) < 2:
        pytest.skip("Need >=2 accelerators for inference parity test")

    d0, d1 = accelerators[0], accelerators[1]
    x = munet.from_numpy(np.random.randn(8, 4).astype(np.float32))

    baseline = _make_model()
    offloaded = _make_model()
    offloaded.offload(d0, layers=["0", "1"])
    offloaded.offload(d1, layers=["2"])

    # Copy weights from baseline -> offloaded for parity check.
    base_params = baseline.named_parameters()
    off_params = offloaded.named_parameters()
    for name, p in base_params.items():
        if name in off_params:
            off_params[name].replace_(p.to(off_params[name].device))

    with munet.no_grad():
        out_base = baseline(x).detach().to(CPU)
        out_off = offloaded(x).detach().to(CPU)

    np.testing.assert_allclose(
        np.array(out_base, copy=False),
        np.array(out_off, copy=False),
        rtol=1e-4,
        atol=1e-5,
    )
