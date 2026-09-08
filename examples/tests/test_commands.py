"""Offline CPU checks of the documented train/resume/inference entry points."""
import importlib
import json
import sys
import urllib.request
import numpy as np
from PIL import Image
import pytest
from munet.checkpoint import load_state


def invoke(module, arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module, *map(str, arguments)])
    importlib.import_module(module).main()


@pytest.mark.parametrize("name,arguments", [
    ("mnist", ["--synthetic", "--limit", "16", "--eval-samples", "3", "--width", "2"]),
    ("segmentation", ["--samples", "8", "--eval-samples", "3", "--size", "16", "--width", "2"]),
    ("language_model", ["--synthetic", "--context", "4", "--width", "4", "--heads", "2", "--layers", "1", "--sample-tokens", "4"]),
])
def test_train_resume_and_infer(name, arguments, tmp_path, monkeypatch, capsys):
    def unexpected_download(*args, **kwargs):
        raise AssertionError("offline example attempted a download")
    monkeypatch.setattr(urllib.request, "urlopen", unexpected_download)
    arguments = [*arguments, "--steps", "1", "--batch-size", "2", "--output", tmp_path]
    invoke(f"examples.{name}.train", arguments, monkeypatch)
    checkpoint = tmp_path / "training.mnet"
    invoke(f"examples.{name}.train", [*arguments, "--resume", checkpoint], monkeypatch)
    assert load_state(checkpoint)["steps"] == 2
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["steps"] == 2 and (tmp_path / "inference.mnet").is_file()
    assert all(np.isfinite(value) for value in metrics["final"].values())
    capsys.readouterr()
    if name == "language_model":
        invoke("examples.language_model.generate", ["--checkpoint", checkpoint, "--prompt", "the", "--tokens", "4", "--output", tmp_path / "generated.txt"], monkeypatch)
        assert len((tmp_path / "generated.txt").read_text()) == 7
    elif name == "mnist":
        invoke("examples.mnist.infer", ["--checkpoint", checkpoint, "--image", tmp_path / "sample.png"], monkeypatch)
        result = json.loads(capsys.readouterr().out)
        assert 0 <= result["digit"] <= 9 and len(result["probabilities"]) == 10
        np.testing.assert_allclose(sum(result["probabilities"]), 1., atol=1e-6)
        assert Image.open(tmp_path / "predictions.png").size[0] > 0
    else:
        output = tmp_path / "prediction"
        invoke("examples.segmentation.infer", ["--checkpoint", checkpoint, "--image", tmp_path / "sample.png", "--output", output], monkeypatch)
        mask = np.asarray(Image.open(output / "mask.png"))
        assert mask.shape == (16, 16) and set(np.unique(mask)).issubset({0, 255})
        assert Image.open(tmp_path / "masks.png").size[0] > 0
