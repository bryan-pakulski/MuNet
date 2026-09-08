"""Exercise learning, exact continuation and exported execution on each backend."""
import numpy as np
import pytest
import munet as mu
from examples.common import Trainer, cross_entropy, export
from examples.mnist.model import DigitCNN
from examples.mnist.data import normalize, synthetic
from examples.segmentation.model import TinyUNet, mask_loss
from examples.segmentation.data import shapes
from examples.language_model.model import TinyGPT
from examples.language_model.generate import sample


def case(name):
    if name == "mnist":
        images, labels = synthetic(2, 7)
        return lambda: DigitCNN(width=2), cross_entropy, normalize(images), labels
    if name == "segmentation":
        images, masks = shapes(2, 16, 7)
        return lambda: TinyUNet(width=2), mask_loss, images, masks
    tokens = np.array([[1, 2, 3, 1], [2, 3, 1, 2]], np.float32)
    return lambda: TinyGPT(4, context=4, width=4, heads=2, layers=1), cross_entropy, tokens, (tokens % 3) + 1


def assert_state_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_state_equal(a, b)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    else:
        assert actual == expected


@pytest.mark.parametrize("name", ["mnist", "segmentation", "language_model"])
def test_learn_resume_export(name, device, tmp_path):
    factory, loss, inputs, targets = case(name)
    config = {"case": name}
    model = factory()
    trainer = Trainer(model, loss, example=name, config=config, device=device, lr=.01)
    losses = [trainer.step(inputs, targets) for _ in range(8)]
    assert losses[-1] < losses[0], losses
    # Advance the sampler too: a resumed job must pick the same next batch.
    trainer.rng.integers(100, size=7)
    path = tmp_path / "training.mnet"
    trainer.save(path)
    restored = Trainer(factory(), loss, example=name, config=config, device=device, lr=.01)
    restored.resume(path)
    np.testing.assert_array_equal(trainer.rng.integers(100, size=10), restored.rng.integers(100, size=10))
    np.testing.assert_allclose(trainer.step(inputs, targets), restored.step(inputs, targets), rtol=1e-6)
    assert trainer.steps == restored.steps == 9
    assert_state_equal(restored.model.state_dict(), model.state_dict())
    assert_state_equal(restored.optimizer.state_dict(), trainer.optimizer.state_dict())
    # Checkpoint metadata protects against accidentally resuming a different dataset/model.
    incompatible = Trainer(factory(), loss, example=name, config={"case": "changed"}, device=device)
    with pytest.raises(ValueError, match="same model, dataset"):
        incompatible.resume(path)
    export(model, inputs[:1], tmp_path, device)
    assert (tmp_path / "inference.mnet").is_file()


def test_language_model_cannot_see_future_tokens(device):
    model = TinyGPT(6, context=6, width=8, heads=2, layers=2)
    predict = mu.compile(model.eval(), device=device)
    original = np.array([[1, 2, 3, 1, 2, 3]], np.float32)
    altered = np.array([[1, 2, 3, 5, 4, 5]], np.float32)
    before = predict(original).numpy().copy()
    after = predict(altered).numpy()
    np.testing.assert_allclose(before[:, :3], after[:, :3], rtol=1e-6, atol=1e-6)
    assert np.max(abs(before[:, 3:] - after[:, 3:])) > .01
    # Padded short prompts and sliding windows are both exercised here.
    vocabulary = ["<unk>", "a", "b", "c", "d", "e"]
    a = sample(model, vocabulary, "ab", tokens=8, seed=3, device=device)
    b = sample(model, vocabulary, "ab", tokens=8, seed=3, device=device)
    assert a == b and a.startswith("ab") and len(a) == 10
