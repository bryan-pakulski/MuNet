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
                _ = munet.ones((1,), device=dev)
            except RuntimeError:
                continue
            devices.append(dev)
    return devices


def _allreduce_grads_via_runtime(param_replicas):
    threads = [threading.Thread(target=lambda p=p: p.grad.all_reduce()) for p in param_replicas]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Runtime all_reduce computes SUM; convert to mean for SGD parity.
    scale = 1.0 / float(len(param_replicas))
    for p in param_replicas:
        p.grad.replace_(p.grad * scale)


def _build_replicated_linear(accelerators):
    w_cpu = munet.Tensor((4, 1), device=CPU, dtype=munet.DataType.Float32, requires_grad=True)
    w_cpu.uniform_(-0.2, 0.2)
    b_cpu = munet.Tensor((1,), device=CPU, dtype=munet.DataType.Float32, requires_grad=True)
    b_cpu.fill_(0.0)

    # Create one model replica per accelerator.
    w_replicas = [w_cpu.to(dev) for dev in accelerators]
    b_replicas = [b_cpu.to(dev) for dev in accelerators]

    # Ensure autograd tracking on replicas.
    for w in w_replicas:
        w.requires_grad = True
    for b in b_replicas:
        b.requires_grad = True

    return w_replicas, b_replicas


def test_multigpu_data_parallel_gradient_allreduce_like_training_step():
    accelerators = _detect_accelerators()
    if len(accelerators) < 2:
        pytest.skip("Need at least two accelerator devices (CUDA/Vulkan) for multi-device all-reduce training test")

    # Set all-reduce env knobs for parity with runtime expectations used by C++ paths.
    os.environ["MUNET_ALLREDUCE_WORLD_SIZE"] = str(len(accelerators))
    os.environ["MUNET_ALLREDUCE_MODE"] = "host_fallback"
    os.environ["MUNET_ALLREDUCE_GROUP"] = "pytest_multigpu_demo"

    w_replicas, b_replicas = _build_replicated_linear(accelerators)

    # Same batch split over replicas.
    x_np = np.array(
        [[1.0, 0.5, -0.2, 0.3],
         [0.2, -1.0, 0.5, 0.7],
         [0.3, 0.8, -0.5, 0.4],
         [0.9, -0.1, 0.2, -0.6]],
        dtype=np.float32,
    )
    y_np = np.array([[0.5], [0.1], [0.7], [0.3]], dtype=np.float32)

    losses = []
    for rank, dev in enumerate(accelerators[:2]):
        xs = munet.from_numpy(x_np[rank * 2 : (rank + 1) * 2]).to(dev)
        ys = munet.from_numpy(y_np[rank * 2 : (rank + 1) * 2]).to(dev)
        preds = xs.matmul(w_replicas[rank]) + b_replicas[rank]
        loss = munet.mse_loss(preds, ys)
        loss.backward()
        losses.append(loss)

    _allreduce_grads_via_runtime([w_replicas[0], w_replicas[1]])
    _allreduce_grads_via_runtime([b_replicas[0], b_replicas[1]])

    # After all-reduce average, gradients should be equal across replicas.
    w0g = np.array(w_replicas[0].grad.to(CPU), copy=False)
    w1g = np.array(w_replicas[1].grad.to(CPU), copy=False)
    b0g = np.array(b_replicas[0].grad.to(CPU), copy=False)
    b1g = np.array(b_replicas[1].grad.to(CPU), copy=False)

    np.testing.assert_allclose(w0g, w1g, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(b0g, b1g, rtol=1e-4, atol=1e-5)

    # Apply one synchronized step and confirm params stay in sync.
    lr = 0.05
    for w in w_replicas[:2]:
        w.step(lr)
    for b in b_replicas[:2]:
        b.step(lr)

    w0 = np.array(w_replicas[0].to(CPU), copy=False)
    w1 = np.array(w_replicas[1].to(CPU), copy=False)
    b0 = np.array(b_replicas[0].to(CPU), copy=False)
    b1 = np.array(b_replicas[1].to(CPU), copy=False)

    np.testing.assert_allclose(w0, w1, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(b0, b1, rtol=1e-4, atol=1e-5)


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
