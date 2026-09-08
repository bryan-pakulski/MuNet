import numpy as np
import pytest
import munet as mu
from munet.checkpoint import load_state, save_state


def test_simple_training_and_checkpoint_continuation(device, tmp_path):
    mu.manual_seed(12)
    model = mu.nn.Sequential(mu.nn.Linear(3, 8), mu.nn.ReLU(), mu.nn.Linear(8, 2))
    optimizer = mu.optim.AdamW(model.parameters(), lr=.02)
    step = mu.train_step(model, optimizer, mu.nn.CrossEntropyLoss(), device=device, max_grad_norm=1.)
    x = np.array([[1, 0, 1], [-1, 0, -1], [1, 1, 0], [-1, -1, 0]], np.float32)
    y = np.array([0, 1, 0, 1], np.float32)
    initial = {k: v.copy() for k, v in model.state_dict().items()}
    step.prepare(x, y)
    assert step.stats()["runs"] == 0
    for k, v in initial.items(): np.testing.assert_array_equal(model.state_dict()[k], v)
    losses = [step(x, y).item() for _ in range(20)]
    assert losses[-1] < losses[0] * .5
    save_state({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, tmp_path / "state.mnet")
    expected = step(x, y).item()
    expected_weights = model.state_dict()
    state = load_state(tmp_path / "state.mnet")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    assert step(x, y).item() == pytest.approx(expected, rel=1e-6)
    for k, v in expected_weights.items(): np.testing.assert_allclose(model.state_dict()[k], v, atol=1e-6)
    assert "Linear(in_features=3, out_features=8)" in repr(model)


def test_export_modes_names_and_owned_predictions(device, tmp_path):
    model = mu.nn.Sequential(mu.nn.BatchNorm2d(2), mu.nn.Flatten(), mu.nn.Linear(8, 3))
    model.train(); model[0].eval()  # Preserve intentionally mixed modes on export.
    x = np.ones((1, 2, 2, 2), np.float32)
    state = {k: v.copy() for k, v in model.state_dict().items()}
    path = model.export(tmp_path / "model.mnet", x, input_names=["image"], output_names=["logits"])
    assert model.training and not model[0].training
    for k, v in state.items(): np.testing.assert_array_equal(model.state_dict()[k], v)
    restored = mu.load(path, device=device)
    assert restored.inputs == (mu.TensorSpec("image", (1, 2, 2, 2)),)
    assert restored.outputs == (mu.TensorSpec("logits", (1, 3)),)
    expected = model.eval().compile(device=device).predict(x)
    prediction = restored.predict(x)
    restored.predict(x * 2)
    np.testing.assert_allclose(prediction, expected, rtol=2e-5, atol=2e-6)
    borrowed = restored(x)
    np.testing.assert_allclose(np.asarray(borrowed), expected, rtol=2e-5, atol=2e-6)
    restored(x)
    with pytest.raises(RuntimeError, match="storage was reused"): borrowed.numpy()


def test_nested_predictions_prepare_save_and_helpful_errors(tmp_path):
    x = np.ones((2, 3), np.float32)
    f = mu.compile(lambda x: {"a": x + 1, "b": (x * 2, "metadata", None)}, device="cpu")
    with pytest.raises(RuntimeError, match="prepare"): _ = f.inputs
    f.prepare(x).save(tmp_path / "nested.mnet")
    g = mu.load(tmp_path / "nested.mnet", device="cpu")
    result = g.predict(x)
    assert isinstance(result["b"], tuple) and result["b"][1:] == ("metadata", None)
    np.testing.assert_array_equal(result["a"], x + 1)
    with pytest.raises(TypeError, match="input 0.*np.asarray"): g(x.astype(np.float64))
    with pytest.raises(ValueError, match="shape guard"): g(x[:1])
    with pytest.raises(RuntimeError, match="model.eval.*compile"): mu.nn.Linear(3, 2)(x)
    with pytest.raises(ValueError, match="device must"): mu.compile(lambda x: x, device="vulkan:-1")
    with pytest.raises(ValueError, match="unique nonempty"): mu.export(lambda x: x, tmp_path / "bad.mnet", x, input_names=[""])


@pytest.mark.parametrize("dim,shape", [(1, (2, 3, 4)), (-1, (2, 4, 3))])
def test_cross_entropy_values_and_gradients(dim, shape, device):
    logits = np.random.default_rng(2).normal(size=shape).astype(np.float32)
    targets = np.array([[0, 1, 2, 0], [1, 2, 0, 1]], np.float32)
    def forward(x, labels):
        loss = mu.nn.functional.cross_entropy(x, labels, dim=dim)
        return loss, mu.grad(loss, [x])[0]
    value, derivative = mu.compile(forward, device=device)(logits, targets)
    axis = dim % len(shape)
    shifted = logits - logits.max(axis, keepdims=True)
    probability = np.exp(shifted) / np.exp(shifted).sum(axis, keepdims=True)
    expected = -np.take_along_axis(np.log(probability), np.expand_dims(targets.astype(int), axis), axis).mean()
    onehot = np.moveaxis(np.eye(3, dtype=np.float32)[targets.astype(int)], -1, axis)
    np.testing.assert_allclose(value.item(), expected, rtol=1e-5)
    np.testing.assert_allclose(derivative.numpy(), (probability - onehot) / targets.size, rtol=3e-5, atol=1e-6)


def test_export_rejects_updates_and_alias_exports(tmp_path):
    import munet_nn as legacy
    assert legacy.export is mu.export and legacy.train_step is mu.train_step
    assert legacy.cat is mu.cat and legacy.Buffer is mu.Buffer
    mu.manual_seed(7); a = mu.nn.Linear(2, 1)
    mu.manual_seed(7); b = mu.nn.Linear(2, 1)
    np.testing.assert_array_equal(a.weight.numpy(), b.weight.numpy())
    step = mu.train_step(a, mu.optim.SGD(a.parameters()), mu.nn.MSELoss(), device="cpu")
    with pytest.raises(ValueError, match="inference export cannot contain"):
        mu.export(step.fn, tmp_path / "train.mnet", (np.ones((1, 2), np.float32), np.zeros((1, 1), np.float32)))


def test_prepare_recapture_invalidates_data_but_preserves_result_shape():
    class DifferentOutput(mu.nn.Module):
        def forward(self, x): return x if self.training else x.mean()
    model = DifferentOutput()
    f = model.compile()
    x = np.ones((2, 3), np.float32)
    result = f(x)
    model.eval()
    f.prepare(x)
    assert f.outputs[0].shape == () and result.shape == (2, 3)
    with pytest.raises(RuntimeError, match="storage was reused"): result.numpy()
