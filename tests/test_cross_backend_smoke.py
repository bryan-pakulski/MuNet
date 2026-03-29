import pytest

np = pytest.importorskip("numpy")

try:
    import munet
except Exception as exc:  # pragma: no cover - environment-dependent import
    pytest.skip(f"munet import unavailable: {exc}", allow_module_level=True)


CPU = munet.Device(munet.DeviceType.CPU, 0)


def _healthy_device(dev: "munet.Device") -> bool:
    try:
        a = munet.ones((4,), device=dev)
        b = (a + a).to(CPU)
        return np.allclose(np.array(b, copy=False), np.array([2, 2, 2, 2], dtype=np.float32))
    except RuntimeError:
        return False


def _pick_cuda_vulkan_pair(max_index: int = 4):
    cuda = None
    vulkan = None

    for idx in range(max_index):
        dev = munet.Device(munet.DeviceType.CUDA, idx)
        if _healthy_device(dev):
            cuda = dev
            break

    for idx in range(max_index):
        dev = munet.Device(munet.DeviceType.VULKAN, idx)
        if _healthy_device(dev):
            vulkan = dev
            break

    if cuda is None or vulkan is None:
        return None
    return cuda, vulkan


def test_cuda_to_vulkan_transfer_smoke():
    pair = _pick_cuda_vulkan_pair(max_index=4)
    if pair is None:
        pytest.skip("Need at least one healthy CUDA and one healthy Vulkan device")

    cuda, vulkan = pair
    x_np = np.random.randn(8, 8).astype(np.float32)
    x_cuda = munet.from_numpy(x_np).to(cuda)
    x_vk = x_cuda.to(vulkan)
    x_back = x_vk.to(CPU)

    np.testing.assert_allclose(np.array(x_back, copy=False), x_np, rtol=1e-4, atol=1e-5)


def test_cross_backend_offload_forward_backward_smoke():
    pair = _pick_cuda_vulkan_pair(max_index=4)
    if pair is None:
        pytest.skip("Need at least one healthy CUDA and one healthy Vulkan device")

    cuda, vulkan = pair
    model = munet.nn.Sequential(
        munet.nn.Linear(4, 8),
        munet.nn.ReLU(),
        munet.nn.Linear(8, 1),
    )
    model.offload(cuda, layers=["0", "1"])
    model.offload(vulkan, layers=["2"])

    x = munet.from_numpy(np.random.randn(16, 4).astype(np.float32))
    y = munet.from_numpy(np.random.randn(16, 1).astype(np.float32)).to(vulkan)

    pred = model(x)
    loss = pred.mse_loss(y)
    loss.backward()

    for p in model.parameters():
        assert p.has_grad()
