import numpy as np
import pytest
import munet as mu
from munet import _native


def test_broadcast_gradients_against_finite_differences(device):
    x = np.asarray([[0.4, -0.7, 1.2], [0.8, 0.3, -0.2]], np.float32)
    bias = np.asarray([0.2, -0.1, 0.3], np.float32)
    def function(a, b):
        u = ((a + b).sigmoid() * (a - b)).square().mean()
        return u, *mu.grad(u, [a, b])
    compiled = mu.compile(function, device=device)
    loss, dx, db = [r.numpy() for r in compiled(x, bias)]
    def reference(a, b): return np.mean(((1 / (1 + np.exp(-(a + b)))) * (a - b)) ** 2)
    np.testing.assert_allclose(loss, reference(x, bias), atol=2e-6)
    for arr, actual, other, first in [(x, dx, bias, True), (bias, db, x, False)]:
        numerical = np.empty_like(arr)
        for index in np.ndindex(arr.shape):
            plus, minus = arr.copy(), arr.copy()
            plus[index] += 0.001; minus[index] -= 0.001
            numerical[index] = ((reference(plus, other) - reference(minus, other)) if first else
                                (reference(other, plus) - reference(other, minus))) / 0.002
        np.testing.assert_allclose(actual, numerical, rtol=2e-3, atol=3e-5)


def test_all_first_order_rules_against_torch(device):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    a = rng.normal(size=(3, 4)).astype(np.float32)
    b = rng.normal(size=(4, 2)).astype(np.float32)
    def function(x, w):
        z = (x @ w).relu() + 1.5
        loss = (z.sqrt().log().exp() / z.sigmoid()).sum(0).mean()
        return loss, *mu.grad(loss, [x, w])
    results = [r.numpy() for r in mu.compile(function, device=device)(a, b)]
    x, w = torch.tensor(a, requires_grad=True), torch.tensor(b, requires_grad=True)
    z = (x @ w).relu() + 1.5
    expected = (z.sqrt().log().exp() / z.sigmoid()).sum(0).mean()
    expected.backward()
    for actual, reference in zip(results, [expected, x.grad, w.grad]):
        np.testing.assert_allclose(actual, reference.detach().numpy(), atol=2e-5, rtol=2e-5)


def test_fusion_and_reused_buffers_preserve_branches(device):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(8, 16)).astype(np.float32)
    def function(a):
        branch = (a + 2).square()
        y = ((a.sigmoid() + 1) * 3 - 2) / 4
        return (y + branch).mean(0), (branch - y).sum(1)
    fused = mu.compile(function, device=device)
    plain = mu.compile(function, device=device, fuse=False)
    for source in [x, x * 2, x - 1]:
        actual, expected = fused(source), plain(source)
        for a, b in zip(actual, expected): np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=2e-5, atol=2e-5)
    assert fused.stats()["kernels"] < plain.stats()["kernels"]


def test_training_replays_keep_parameter_updates_and_no_hidden_readback(device):
    rng = np.random.default_rng(17)
    model = mu.nn.Linear(3, 2, rng=rng)
    optimizer = mu.optim.SGD(model.parameters(), lr=0.08)
    @mu.compile(device=device)
    def step(x, y):
        optimizer.zero_grad()
        loss = (model(x) - y).square().mean()
        loss.backward()
        optimizer.step()
        return loss
    x = rng.normal(size=(24, 3)).astype(np.float32)
    y = (x @ np.asarray([[1, 2], [-2, 1], [0.5, -1]], np.float32) + 0.4).astype(np.float32)
    first = step(x, y).item()
    before = step.stats()
    for _ in range(80): result = step(x, y)
    after = step.stats()
    assert result.item() < first * 0.001
    if device == "vulkan":
        assert after["download_bytes"] == before["download_bytes"]
        assert after["upload_bytes"] - before["upload_bytes"] == 80 * (x.nbytes + y.nbytes)
        assert after["submissions"] - before["submissions"] == 80
    np.testing.assert_allclose(model.weight.numpy(), np.asarray([[1, -2, 0.5], [2, 1, -1]], np.float32), atol=0.04)


def test_training_one_step_matches_torch(device):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(21)
    model = mu.nn.Sequential(mu.nn.Linear(3, 5, rng=rng), mu.nn.ReLU(), mu.nn.Linear(5, 2, rng=rng))
    reference = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))
    with torch.no_grad():
        for target, source in zip(reference.parameters(), model.parameters()): target.copy_(torch.from_numpy(source.numpy()))
    optimizer = mu.optim.SGD(model.parameters(), lr=0.05)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    @mu.compile(device=device)
    def step(x, y):
        optimizer.zero_grad(); loss = (model(x) - y).square().mean()
        loss.backward(); optimizer.step(); return loss
    for _ in range(3):
        x = rng.normal(size=(7, 3)).astype(np.float32); y = rng.normal(size=(7, 2)).astype(np.float32)
        actual = step(x, y).item()
        ref_optimizer.zero_grad(); expected = (reference(torch.from_numpy(x)) - torch.from_numpy(y)).square().mean()
        expected.backward(); ref_optimizer.step()
        np.testing.assert_allclose(actual, expected.detach().numpy(), rtol=2e-5, atol=2e-5)
        for target, source in zip(reference.parameters(), model.parameters()):
            np.testing.assert_allclose(source.numpy(), target.detach().numpy(), rtol=2e-5, atol=2e-5)


def test_guards_and_result_lifetime(device):
    f = mu.compile(lambda x: (x + 1).square(), device=device)
    x = np.ones((2, 3), np.float32)
    previous = f(x)
    f(x * 2)
    with pytest.raises(RuntimeError, match="reused"): previous.numpy()
    with pytest.raises(ValueError, match="shape guard"): f(np.ones((3, 3), np.float32))
    with pytest.raises(TypeError, match="float32"): f(x.astype(np.float64))
    with pytest.raises(TypeError, match="control flow"):
        mu.compile(lambda a: a if a else -a, device=device)(x)


def test_scalar_input_keeps_rank_zero(device):
    def function(x):
        y = x.square().mean()
        return y, mu.grad(y, [x])[0]
    result, derivative = mu.compile(function, device=device)(np.asarray(3, np.float32))
    assert result.shape == derivative.shape == ()
    np.testing.assert_allclose(result.numpy(), 9)
    np.testing.assert_allclose(derivative.numpy(), 6)


def test_shared_parameter_and_cross_program_updates(device):
    p = mu.Parameter(np.asarray([1, 2, 3], np.float32))
    optimizer = mu.optim.SGD([p, p], lr=0.1)
    @mu.compile(device=device)
    def train(x):
        optimizer.zero_grad(); loss = (p * x + p).square().mean()
        loss.backward(); optimizer.step(); return loss
    inference = mu.compile(lambda x: x * p, device=device)
    x = np.asarray([0.5, 1.0, 2.0], np.float32)
    inference(x)
    train(x)
    expected = p.numpy() * x
    np.testing.assert_allclose(inference(x).numpy(), expected, rtol=2e-5)
    p.assign(np.ones(3, np.float32))
    np.testing.assert_allclose(inference(x).numpy(), x)


def test_native_rejects_malformed_graph_and_snapshots_updates(device):
    graph = _native.Graph()
    a = graph.leaf("parameter", "a", [2], [1, 2])
    b = graph.leaf("parameter", "b", [2], [3, 4])
    with pytest.raises(ValueError): graph.op("add", [a, 999])
    with pytest.raises(ValueError): graph.leaf("input", "bad", [-2])
    with pytest.raises(ValueError): graph.op("sum", [a], [0, 0])
    with pytest.raises(ValueError): graph.op("invented", [a])
    from munet.serialization import from_graph
    program = from_graph(graph, [a, b], [(a, b), (b, a)], [], [], False, device=device)
    got = [r.numpy() for r in program()]
    np.testing.assert_array_equal(got[0], [3, 4]); np.testing.assert_array_equal(got[1], [1, 2])
