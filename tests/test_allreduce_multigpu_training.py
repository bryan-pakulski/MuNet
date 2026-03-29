import os
import threading

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
                a = munet.ones((1,), device=dev, dtype=munet.DataType.Float32)
                b = munet.ones((1,), device=dev, dtype=munet.DataType.Float32)
                c = a + b
                if float(c.to(CPU).item()) != 2.0:
                    raise RuntimeError("accelerator probe produced unexpected value")
            except RuntimeError:
                continue
            devices.append(dev)
    return devices


def _allreduce_grads_via_runtime(param_replicas):
    errors = []

    def _target(param):
        try:
            param.grad.all_reduce()
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(exc)

    threads = [threading.Thread(target=_target, args=(p,)) for p in param_replicas]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError(f"all_reduce failed: {errors[0]}")

    # Runtime all_reduce computes SUM; convert to mean for SGD parity.
    scale = 1.0 / float(len(param_replicas))
    for p in param_replicas:
        p.grad.replace_(p.grad * scale)


def _require_devices(device_specs):
    available = _detect_accelerators(max_index=4)
    available_set = {(d.type, d.index) for d in available}
    wanted = []
    for dev_type, index in device_specs:
        if (dev_type, index) not in available_set:
            pytest.skip(
                f"Required device {dev_type.name}:{index} unavailable for scenario"
            )
        wanted.append(munet.Device(dev_type, index))
    return wanted


def _build_replicated_mlp(accelerators):
    rng = np.random.default_rng(7)
    w1_np = rng.normal(0.0, 0.15, size=(4, 8)).astype(np.float32)
    b1_np = np.zeros((8,), dtype=np.float32)
    w2_np = rng.normal(0.0, 0.15, size=(8, 1)).astype(np.float32)
    b2_np = np.zeros((1,), dtype=np.float32)

    def _replica(arr, dev):
        t = munet.from_numpy(arr).to(dev).detach()
        t.requires_grad = True
        return t

    w1 = [_replica(w1_np, dev) for dev in accelerators]
    b1 = [_replica(b1_np, dev) for dev in accelerators]
    w2 = [_replica(w2_np, dev) for dev in accelerators]
    b2 = [_replica(b2_np, dev) for dev in accelerators]
    return w1, b1, w2, b2


def _forward_mlp(x, w1, b1, w2, b2):
    h = (x.matmul(w1) + b1).relu()
    return h.matmul(w2) + b2


def _assert_tensors_synced_across_replicas(tensors, rtol=1e-4, atol=1e-5):
    base = np.array(tensors[0].detach().to(CPU), copy=False)
    for t in tensors[1:]:
        other = np.array(t.detach().to(CPU), copy=False)
        np.testing.assert_allclose(base, other, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "scenario_name,device_specs",
    [
        pytest.param(
            "cuda0_vulkan1",
            [(munet.DeviceType.CUDA, 0), (munet.DeviceType.VULKAN, 1)],
            id="cuda0_vulkan1",
        ),
        pytest.param(
            "vulkan0_vulkan1",
            [(munet.DeviceType.VULKAN, 0), (munet.DeviceType.VULKAN, 1)],
            id="vulkan0_vulkan1",
        ),
    ],
)
def test_multigpu_e2e_complex_training_scenarios(scenario_name, device_specs):
    accelerators = _require_devices(device_specs)
    assert len(accelerators) == 2

    # Set all-reduce env knobs to match active participants in this scenario.
    os.environ["MUNET_ALLREDUCE_WORLD_SIZE"] = str(len(accelerators))
    os.environ["MUNET_ALLREDUCE_MODE"] = "host_fallback"
    os.environ["MUNET_ALLREDUCE_GROUP"] = f"pytest_multigpu_{scenario_name}"
    os.environ["MUNET_ALLREDUCE_TIMEOUT_MS"] = "30000"

    w1, b1, w2, b2 = _build_replicated_mlp(accelerators)

    rng = np.random.default_rng(42)
    x_np = rng.normal(size=(64, 4)).astype(np.float32)
    true_w = np.array([[0.4], [-0.3], [0.2], [0.6]], dtype=np.float32)
    true_b = np.array([0.15], dtype=np.float32)
    y_np = x_np @ true_w + true_b

    lr = 0.03
    steps = 4
    first_mean_loss = None
    last_mean_loss = None
    shard = len(x_np) // len(accelerators)

    for _ in range(steps):
        losses = []
        for rank, dev in enumerate(accelerators):
            xs = munet.from_numpy(x_np[rank * shard:(rank + 1) * shard]).to(dev)
            ys = munet.from_numpy(y_np[rank * shard:(rank + 1) * shard]).to(dev)
            pred = _forward_mlp(xs, w1[rank], b1[rank], w2[rank], b2[rank])
            loss = pred.mse_loss(ys)
            loss.backward()
            losses.append(float(loss.detach().to(CPU).item()))

        _allreduce_grads_via_runtime(w1)
        _allreduce_grads_via_runtime(b1)
        _allreduce_grads_via_runtime(w2)
        _allreduce_grads_via_runtime(b2)

        # Brutal invariants: gradients and params should remain synchronized.
        _assert_tensors_synced_across_replicas([t.grad for t in w1])
        _assert_tensors_synced_across_replicas([t.grad for t in b1])
        _assert_tensors_synced_across_replicas([t.grad for t in w2])
        _assert_tensors_synced_across_replicas([t.grad for t in b2])

        for group in (w1, b1, w2, b2):
            for t in group:
                t.step(lr)

        _assert_tensors_synced_across_replicas(w1)
        _assert_tensors_synced_across_replicas(b1)
        _assert_tensors_synced_across_replicas(w2)
        _assert_tensors_synced_across_replicas(b2)

        mean_loss = float(np.mean(losses))
        if first_mean_loss is None:
            first_mean_loss = mean_loss
        last_mean_loss = mean_loss

    assert first_mean_loss is not None and last_mean_loss is not None
    assert last_mean_loss <= first_mean_loss * 1.25


def test_python_all_reduce_binding_cpu_smoke():
    os.environ["MUNET_ALLREDUCE_WORLD_SIZE"] = "2"
    os.environ["MUNET_ALLREDUCE_MODE"] = "host_fallback"
    os.environ["MUNET_ALLREDUCE_GROUP"] = "pytest_python_binding_cpu"

    a = munet.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    b = munet.from_numpy(np.array([3.0, 4.0], dtype=np.float32))

    t0 = threading.Thread(target=lambda: a.all_reduce())
    t1 = threading.Thread(target=lambda: b.all_reduce())
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    np.testing.assert_allclose(np.array(a, copy=False), np.array([4.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(np.array(b, copy=False), np.array([4.0, 6.0], dtype=np.float32))
