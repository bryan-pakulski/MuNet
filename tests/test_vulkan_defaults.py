"""Default user workflows execute on Vulkan; CPU fallback must be explicit."""
import json
import zipfile

import numpy as np
import pytest
import munet as mu
from munet import _native


def test_default_training_export_python_and_cpp_inference(tmp_path, device, monkeypatch):
    if device == "cpu":
        pytest.skip("default execution requires Vulkan")
    mu.manual_seed(31)
    model = mu.nn.Linear(2, 1)
    x = np.array([[1, 2], [2, 3]], np.float32)
    y = np.array([[1], [2]], np.float32)
    step = mu.train_step(model, mu.optim.SGD(model.parameters(), lr=.01), mu.nn.MSELoss())
    losses = [step(x, y).item() for _ in range(3)]
    assert losses[-1] < losses[0]
    assert step.stats()["runs"] == 3 and step.stats()["submissions"] >= 3

    inference = model.eval().compile()
    expected = inference.predict(x)
    assert inference.stats()["submissions"] > 0
    path = tmp_path / "default.mnet"
    inference.save(path)
    with zipfile.ZipFile(path) as archive:
        assert json.loads(archive.read("manifest.json"))["vulkan"]
    saved = tmp_path / "save.mnet"
    mu.save(inference, saved)
    exported = model.export(tmp_path / "export.mnet", x)

    # Both default save/export produce ready-to-deploy Vulkan artifacts.
    def no_compiler(*args, **kwargs):
        raise AssertionError("deployment must use embedded shaders")
    monkeypatch.setattr(mu.core, "_compile_shaders", no_compiler)
    for artifact in (path, saved, exported):
        python = mu.load(artifact)
        cpp = _native.InferenceModel(str(artifact))
        np.testing.assert_allclose(python.predict(x), expected, rtol=2e-5)
        np.testing.assert_allclose(cpp.run([x])[0], expected, rtol=2e-5)
        assert python.stats()["submissions"] > 0 and cpp.stats()["submissions"] > 0


def test_unavailable_vulkan_reports_failure_and_cpu_fallback_works(tmp_path, monkeypatch):
    import munet.serialization as serialization
    monkeypatch.setattr(_native, "vulkan_built", lambda: False)
    x = np.array([2, 3], np.float32)
    default = mu.compile(lambda x: x * 2)
    with pytest.raises(RuntimeError, match="no Vulkan runtime"):
        default(x)
    assert default.stats() == {"compiled": False}

    def no_compiler(*args, **kwargs):
        raise AssertionError("CPU fallback must not need a shader compiler")
    monkeypatch.setattr(mu.core, "_compile_shaders", no_compiler)
    monkeypatch.setattr(serialization, "_compile_shaders", no_compiler)
    cpu = mu.compile(lambda x: x * 2, device="cpu")
    np.testing.assert_array_equal(cpu.predict(x), x * 2)
    path = tmp_path / "cpu.mnet"
    cpu.save(path)
    with zipfile.ZipFile(path) as archive:
        assert "vulkan" not in json.loads(archive.read("manifest.json"))
    for artifact in (path, mu.export(lambda x: x * 2, tmp_path / "export.mnet", x, include_vulkan=False)):
        np.testing.assert_array_equal(mu.load(artifact, device="cpu").predict(x), x * 2)
        np.testing.assert_array_equal(_native.InferenceModel(str(artifact), "cpu").run([x])[0], x * 2)
        with pytest.raises(RuntimeError, match="no Vulkan runtime"):
            mu.load(artifact)
