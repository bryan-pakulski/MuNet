import pytest

try:
    import munet_nn as munet
except Exception as exc:  # pragma: no cover
    pytest.skip(f"munet import unavailable: {exc}", allow_module_level=True)


def test_tensor_training_surface_exposed():
    required = {
        "backward",
        "grad",
        "has_grad",
        "zero_grad",
        "register_gradient_hook",
        "all_reduce",
        "step",
        "clone",
        "detach",
        "fill_",
        "uniform_",
        "to",
        "to_options",
        "layer_norm",
        "batch_norm",
        "mse_loss",
        "cross_entropy",
    }
    missing = sorted(name for name in required if not hasattr(munet.Tensor, name))
    assert not missing, f"Tensor binding missing training/inference methods: {missing}"


def test_module_optimizer_inference_surface_exposed():
    assert hasattr(munet.nn.Module, "train")
    assert hasattr(munet.nn.Module, "eval")
    assert hasattr(munet.nn.Module, "parameters")
    assert hasattr(munet.optim, "SGD")
    assert hasattr(munet.optim, "Adam")
    assert hasattr(munet, "inference")
    assert hasattr(munet.inference, "Engine")


def test_backend_probe_apis_exposed_and_vulkan_default():
    assert hasattr(munet, "list_available_backends")
    assert hasattr(munet, "backend_status")

    backends = set(munet.list_available_backends())
    assert "vulkan" in backends or backends == set()

    status = munet.backend_status()
    assert isinstance(status, dict)
    assert "statuses" in status
    assert "summary" in status
    assert "No Vulkan backend is active" in status["summary"] or status["accelerator_loaded"] is True


def test_backend_status_reason_codes_stable_tokens():
    status = munet.backend_status()
    entries = status.get("statuses", [])
    assert any(item.get("name") == "vulkan" for item in entries)

    valid_reason_codes = {"ok", "runtime_dependency_missing"}
    for item in entries:
        assert item.get("reason_code") in valid_reason_codes


def _accelerator_available(backend_name):
    status = munet.backend_status()
    for item in status.get("statuses", []):
        if item.get("name") != backend_name or item.get("source") != "builtin":
            continue
        if item.get("active"):
            return True, "ok"
        reason = item.get("detail") or item.get("reason_code") or "vulkan"
        return False, reason
    return False, "builtin status entry not found"


def _gpu_smoke_matmul(device_type):
    dev = munet.Device(device_type, 0)
    a = munet.ones([2, 2], dev)
    b = munet.ones([2, 2], dev)
    out = munet.matmul(a, b)
    out_host = out.to(munet.Device(munet.DeviceType.VULKAN, 0))
    assert out_host.shape == [2, 2]
    import numpy as np
    np_out = np.array(out_host, copy=False)
    assert (np_out == 2.0).all()


def test_vulkan_gpu_smoke_if_available():
    available, reason = _accelerator_available("vulkan")
    if not available:
        pytest.skip(f"Vulkan accelerator not active in this environment: {reason}")
    _gpu_smoke_matmul(munet.DeviceType.VULKAN)


def test_sequential_accepts_python_list_forms():
    layers = [munet.nn.Linear(2, 3), munet.nn.ReLU(), munet.nn.Linear(3, 1)]

    positional_list = munet.nn.Sequential(layers)
    keyword_list = munet.nn.Sequential(layers=layers)
    positional_args = munet.nn.Sequential(*layers)

    assert len(list(positional_list)) == 3
    assert len(list(keyword_list)) == 3
    assert len(list(positional_args)) == 3
