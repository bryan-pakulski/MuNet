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
