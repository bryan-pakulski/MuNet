import sys
import os
import json
import unittest
import tempfile
import subprocess

try:
    import numpy as np
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy"], check=True)
    import numpy as np

# Dynamically add build output directories to sys.path so Python can find
# munet_nn*.so (CMake places it in build/debug by default).
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
candidate_build_dirs = [
    os.path.join(repo_root, "build", "debug"),
    os.path.join(repo_root, "build", "release"),
    os.path.join(repo_root, "build"),
]
for candidate in candidate_build_dirs:
    if os.path.isdir(candidate):
        sys.path.insert(0, candidate)

try:
    import munet_nn as munet
except ImportError as e:
    print("\n[ERROR] munet import failed; attempting run local build for tests...\n")
    sys.exit(1)

def _sequential(layers):
    return munet.nn.Sequential(*layers)


class TestBindings(unittest.TestCase):
    def _available_non_host_devices(self):
        devices = []
        for device_type in (munet.DeviceType.VULKAN,):
            dev = munet.Device(device_type, 0)
            try:
                probe = munet.ones([1], device=dev)
                devices.append(probe.device)
            except RuntimeError:
                continue
        return devices

    def test_mse_loss(self):
        pred_np = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        target_np = np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32)

        # Load into MuNet tensors
        pred = munet.Tensor(pred_np.shape)
        target = munet.Tensor(target_np.shape)

        # We can use the buffer protocol via np.array to write directly to C++ memory
        np.array(pred, copy=False)[:] = pred_np
        np.array(target, copy=False)[:] = target_np

        pred.requires_grad = True

        loss = pred.mse_loss(target)
        loss_val = loss.item()

        # MSE Forward Check
        self.assertTrue(np.isclose(loss_val, 0.5))

        loss.backward()
        grad = pred.grad.detach().numpy()

        # MSE Backward Check
        expected_grad = np.array([0.0, 0.5, 0.0, 0.5], dtype=np.float32)
        self.assertTrue(np.allclose(grad, expected_grad, atol=1e-6))

    def test_available_accelerators_and_devices_introspection(self):
        accelerators = munet.available_accelerators()
        self.assertIsInstance(accelerators, list)
        self.assertGreaterEqual(len(accelerators), 1)

        by_name = {entry["name"]: entry for entry in accelerators}
        self.assertIn("vulkan", by_name)

        vulkan_entry = by_name["vulkan"]
        self.assertGreaterEqual(len(vulkan_entry["devices"]), 0)

        devices = munet.available_devices()
        self.assertIsInstance(devices, list)
        self.assertGreaterEqual(len(devices), 1)
        self.assertEqual(devices[0].type, munet.DeviceType.VULKAN)

    def test_cross_entropy_loss(self):
        logits_np = np.array([[2.0, 1.0, 0.1], [0.1, 1.0, 2.0]], dtype=np.float32)

        targets_np = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        with munet.no_grad():
            logits = munet.Tensor(list(logits_np.shape))
            targets = munet.Tensor(list(targets_np.shape))

        np.array(logits, copy=False)[:] = logits_np
        np.array(targets, copy=False)[:] = targets_np

        logits.requires_grad = True

        loss = logits.cross_entropy(targets)
        loss_val = loss.item()

        # CE Forward Check (Matches PyTorch F.cross_entropy)
        self.assertTrue(np.isclose(loss_val, 0.417022, atol=1e-4))

        loss.backward()
        grad = np.array(logits.grad, copy=False)

        # CE Backward Check
        expected_grad = np.array(
            [[-0.17050, 0.12122, 0.04928], [0.04928, 0.12122, -0.17050]],
            dtype=np.float32,
        )

        self.assertTrue(np.allclose(grad, expected_grad, atol=1e-4))

    def test_unary_math_ops(self):
        x = munet.Tensor([3])
        np.array(x, copy=False)[:] = np.array([1.0, 4.0, 9.0], dtype=np.float32)

        sqrt_out = np.array(x.sqrt().detach(), copy=False)
        self.assertTrue(np.allclose(sqrt_out, np.array([1.0, 2.0, 3.0], dtype=np.float32)))

        roundtrip = np.array(x.log().exp().detach(), copy=False)
        self.assertTrue(np.allclose(roundtrip, np.array([1.0, 4.0, 9.0], dtype=np.float32), atol=1e-5))

        trig = munet.Tensor([2])
        np.array(trig, copy=False)[:] = np.array([0.0, np.pi / 2.0], dtype=np.float32)
        self.assertTrue(np.allclose(np.array(trig.sin().detach(), copy=False), np.array([0.0, 1.0], dtype=np.float32), atol=1e-5))
        self.assertTrue(np.allclose(np.array(trig.cos().detach(), copy=False), np.array([1.0, 0.0], dtype=np.float32), atol=1e-5))
        self.assertTrue(np.allclose(np.array(x.rsqrt().detach(), copy=False), np.array([1.0, 0.5, 1.0 / 3.0], dtype=np.float32), atol=1e-5))

    def test_unary_math_ops_backward(self):
        x = munet.Tensor([3], requires_grad=False)
        np.array(x, copy=False)[:] = np.array([1.0, 4.0, 9.0], dtype=np.float32)
        x.requires_grad = True

        loss = x.log().sum()
        loss.backward()
        grad = np.array(x.grad.detach(), copy=False)
        expected = np.array([1.0, 0.25, 1.0 / 9.0], dtype=np.float32)
        self.assertTrue(np.allclose(grad, expected, atol=1e-6))

    def test_mean_narrow_and_rmsnorm(self):
        x = munet.Tensor([2, 4], requires_grad=False)
        np.array(x, copy=False)[:] = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 6.0, 8.0, 10.0]], dtype=np.float32)
        x.requires_grad = True

        mean = x.mean(-1)
        self.assertTrue(np.allclose(np.array(mean.detach(), copy=False), np.array([2.5, 7.0], dtype=np.float32)))

        narrowed = x.detach().narrow(1, 1, 2)
        self.assertEqual(narrowed.shape, [2, 2])
        self.assertEqual(narrowed.storage_offset, 1)
        self.assertTrue(
            np.allclose(
                np.array(narrowed, copy=False),
                np.array([[3.0, 4.0], [8.0, 10.0]], dtype=np.float32),
            )
        )

        rms = munet.nn.RMSNorm(4)
        out = rms(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(rms.weight.grad)

    def test_tensor_creation(self):
        """Test basic tensor creation and properties mapping."""
        t = munet.Tensor([2, 3], requires_grad=True)
        self.assertEqual(t.shape, [2, 3])
        self.assertTrue(t.requires_grad)

        # Check Enums mapped properly
        self.assertEqual(t.device.type, munet.DeviceType.VULKAN)
        self.assertEqual(t.dtype, munet.DataType.Float32)

    def test_backend_supports_api(self):
        host = munet.Device(munet.DeviceType.VULKAN, 0)
        self.assertTrue(munet.supports(host, munet.BackendFeature.Matmul, munet.DataType.Float32))
        self.assertFalse(munet.supports(host, munet.BackendFeature.Matmul, munet.DataType.Float16))
        self.assertTrue(munet.supports(host, munet.BackendFeature.RandomFill, munet.DataType.Float16))
        self.assertFalse(munet.supports(host, munet.BackendFeature.RandomFill, munet.DataType.Int32))

    def test_numpy_buffer_protocol(self):
        """Test zero-copy memory sharing between C++ and NumPy."""
        t = munet.Tensor([2, 2])

        # Create a NumPy array that points directly to the C++ Tensor's memory
        arr = np.array(t, copy=False)

        # Modify memory via Python/NumPy
        arr[:] = 5.0

        # Verify the underlying data was actually changed
        self.assertEqual(arr[0, 0], 5.0)
        self.assertEqual(arr[1, 1], 5.0)

        arr[0, 1] = 42.0
        self.assertEqual(arr[0, 1], 42.0)

    def test_autograd_add(self):
        """Test math operations and the backward pass DAG."""
        # 1. Create leaves, disable grad as we don't allow modification
        # of tensors with grad enabled for safety.
        a = munet.Tensor([1], requires_grad=False)
        b = munet.Tensor([1], requires_grad=False)

        # Set values natively using NumPy
        np.array(a, copy=False)[:] = 10.0
        np.array(b, copy=False)[:] = 20.0

        a.requires_grad = True
        b.requires_grad = True

        # 2. Forward pass (C++ operator overloaded)
        c = a + b

        # 3. Verify forward result
        c_arr = c.detach().numpy()
        self.assertEqual(c_arr[0], 30.0)

        # 4. Backward pass
        c.backward()

        # 5. Verify gradients (dz/da = 1, dz/db = 1)
        a_grad = a.grad.detach().numpy()
        b_grad = b.grad.detach().numpy()

        self.assertEqual(a_grad[0], 1.0)
        self.assertEqual(b_grad[0], 1.0)

    def test_vulkan_backend(self):
        """Test moving tensors to Vulkan backend."""
        try:
            munet.Device(munet.DeviceType.VULKAN, 0)
        except RuntimeError:
            print("\nSkipping Vulkan tests (not compiled or no GPU).")
            return

        a_host = munet.Tensor([2], requires_grad=False)
        np.array(a_host, copy=False)[:] = [9.0, 10.0]

        vk_dev = munet.Device(munet.DeviceType.VULKAN, 0)

        try:
            a_vk = a_host.to(vk_dev)
        except RuntimeError as e:
            print(f"\nSkipping Vulkan test due to: {e}")
            return

        self.assertEqual(a_vk.device.type, munet.DeviceType.VULKAN)

        # Test moving it back to Host to read the memory
        a_back = a_vk.to(munet.Device(munet.DeviceType.VULKAN, 0))
        result = a_back.detach().numpy()
        self.assertEqual(result[0], 9.0)
        self.assertEqual(result[1], 10.0)

    def test_neural_network_forward_backward(self):
        """Test a mini neural network forward and backward pass!"""

        # We will create a mini computational graph:
        # y = relu(X @ W1) @ W2
        # Then calculate gradients for W1 and W2.

        with munet.no_grad():
            X = munet.Tensor([1, 3])
            W1 = munet.Tensor([3, 4])
            W2 = munet.Tensor([4, 1])

        np.array(X, copy=False)[:] = [[1.0, 2.0, -1.0]]

        # W1
        np.array(W1, copy=False)[:] = [
            [1.0, 0.5, -1.0, 2.0],
            [-2.0, 1.0, 0.5, -0.5],
            [0.0, -1.0, 1.0, 1.0],
        ]

        # W2
        np.array(W2, copy=False)[:] = [[1.0], [-1.0], [2.0], [0.5]]

        W1.requires_grad = True
        W2.requires_grad = True

        # --- Forward Pass ---
        hidden = X @ W1
        activation = hidden.relu()
        output = activation @ W2

        # Verify Forward Pass Math
        # X @ W1 = [[-3.0, 3.5, -1.0, 0.0]]
        # relu(X @ W1) = [[0.0, 3.5, 0.0, 0.0]]
        # relu @ W2 = 0*1 + 3.5*-1 + 0*2 + 0*0.5 = -3.5

        fp_result = output.detach().numpy()
        self.assertEqual(fp_result[0], -3.5)

        # --- Backward Pass ---
        output.backward()

        # Verify Backward Pass
        # dOutput = 1.0
        # dW2 = activation.T @ dOutput = [[0.0], [3.5], [0.0], [0.0]]
        dw2_result = np.array(W2.grad, copy=False)
        self.assertEqual(dw2_result[1][0], 3.5)
        self.assertEqual(dw2_result[0][0], 0.0)

        # dActivation = dOutput @ W2.T = [[1.0, -1.0, 2.0, 0.5]]
        # dHidden = dActivation * (hidden > 0) -> only the 2nd element was > 0.
        # dHidden = [[0.0, -1.0, 0.0, 0.0]]
        # dW1 = X.T @ dHidden = [[1], [2], [-1]] @ [[0.0, -1.0, 0.0, 0.0]]

        dw1_result = np.array(W1.grad, copy=False)
        self.assertEqual(dw1_result[0][1], -1.0)  # 1.0 * -1.0
        self.assertEqual(dw1_result[1][1], -2.0)  # 2.0 * -1.0
        self.assertEqual(dw1_result[2][1], 1.0)  # -1.0 * -1.0

    def test_full_training_loop(self):
        """Train a 2-layer MLP to overfit on dummy data."""
        # GPU or Host
        dev = munet.Device(munet.DeviceType.VULKAN, 0)

        # Data: Predict sum of features
        x = munet.Tensor([2, 2], device=dev, requires_grad=False)
        y = munet.Tensor([2, 1], device=dev, requires_grad=False)

        np.array(x, copy=False)[:] = [[1.0, 1.0], [1.0, 0.0]]
        np.array(y, copy=False)[:] = [[0.0], [1.0]]  # Targets

        # Weights
        w1 = munet.Tensor([2, 4], device=dev, requires_grad=True)
        w2 = munet.Tensor([4, 1], device=dev, requires_grad=True)
        w1.uniform_(-1.0, 1.0)
        w2.uniform_(-1.0, 1.0)

        optimizer = munet.optim.SGD([w1, w2], lr=0.01)

        for epoch in range(50):
            optimizer.zero_grad()

            # Forward
            h = x @ w1
            pred = h @ w2

            # MSE Loss
            diff = pred - y
            loss = (diff * diff).sum()

            # Backward
            loss.backward()

            # Optimize
            optimizer.step()

            if epoch == 0:
                loss_start = loss.item()
            if epoch == 49:
                loss_end = loss.item()

        # Training API smoke test: ensure loop runs and loss stays finite.
        print(
            f"\nTraining Loop -> Start Loss: {loss_start:.4f} | End Loss: {loss_end:.4f}"
        )
        self.assertTrue(np.isfinite(loss_start))
        self.assertTrue(np.isfinite(loss_end))

    def test_multi_device_model_parallelism(self):
        """Test seamless autograd across Host and GPU boundaries."""
        # Find an available GPU
        gpu_dev = None
        for dev_type in (munet.DeviceType.VULKAN,):
            try:
                candidate = munet.Device(dev_type, 0)
                _ = munet.ones([1], device=candidate)
                gpu_dev = candidate
                break
            except RuntimeError:
                continue

        if gpu_dev is None:
            print("\nSkipping multi-device test (No GPU available).")
            return
        host_dev = munet.Device(munet.DeviceType.VULKAN, 0)
        # --- Layer 1 on Host ---
        x_host = munet.Tensor([1, 3], device=host_dev)
        w1_host = munet.Tensor([3, 4], device=host_dev, requires_grad=True)
        np.array(x_host, copy=False)[:] = [[1.0, 2.0, 3.0]]
        w1_host.uniform_(-0.5, 0.5)
        # --- Layer 2 on GPU ---
        w2_gpu = munet.Tensor([4, 2], device=gpu_dev, requires_grad=True)
        w2_gpu.uniform_(-0.5, 0.5)
        # --- Forward Pass (Cross-device) ---
        h_host = x_host @ w1_host
        h_gpu = h_host.to(gpu_dev)  # Autograd boundary crossing!
        pred_gpu = h_gpu @ w2_gpu

        target_gpu = munet.Tensor([1, 2], device=gpu_dev)
        target_gpu.uniform_(-1.0, 1.0)

        diff = pred_gpu - target_gpu
        loss = (diff * diff).sum()
        # --- Backward Pass ---
        loss.backward()
        # --- Verify Gradients Synchronized Correctly ---
        self.assertTrue(w1_host.grad is not None)
        self.assertTrue(w2_gpu.grad is not None)
        self.assertEqual(w1_host.grad.device.type, munet.DeviceType.VULKAN)
        self.assertEqual(w2_gpu.grad.device.type, gpu_dev.type)
        print(f"\nSuccessfully backpropagated from {gpu_dev.type} back to Host!")

    def test_default_grad_mode(self):
        # By default, operations on requires_grad=True tensors build a graph
        x = munet.ones([2, 2], requires_grad=True)
        y = x + x
        self.assertTrue(y.requires_grad)

    def test_no_grad_context(self):
        x = munet.ones([2, 2], requires_grad=True)

        with munet.no_grad():
            y = x + x
            # Inside no_grad, graph building is disabled
            self.assertFalse(y.requires_grad)

        # Context exit restores the previous state
        z = x + x
        self.assertTrue(z.requires_grad)

    def test_nested_enable_grad(self):
        x = munet.ones([2, 2], requires_grad=True)

        with munet.no_grad():
            y = x * 2
            self.assertFalse(y.requires_grad)

            # Nested context to temporarily re-enable gradients
            with munet.enable_grad():
                z = x * 3
                self.assertTrue(z.requires_grad)

            # Exiting enable_grad restores the no_grad state
            w = x * 4
            self.assertFalse(w.requires_grad)

    def test_backward_with_no_grad(self):
        x = munet.ones([2], requires_grad=True)

        # y is detached from the graph
        with munet.no_grad():
            y = x * 2

        # z connects only to x directly, y is treated as a constant
        z = x * y
        out = z.sum()
        out.backward()

        # Forward was: x * (x * 2), where y=(x*2) is constant 2
        # So z = x * 2. dz/dx = 2.
        # Since x is ones([2]), grad should be [2.0, 2.0]
        # (If graph building wasn't disabled for y, grad would be 4.0)
        grad_np = x.grad.detach().numpy()
        self.assertEqual(grad_np[0], 2.0)
        self.assertEqual(grad_np[1], 2.0)

    def test_detach(self):
        """Test that detached tensors do not propagate gradients."""
        x = munet.Tensor([1])
        np.array(x, copy=False)[:] = 2.0
        x.requires_grad = True

        y = x * x
        z = y.detach()

        out = z * x
        out.backward()

        # out = (x^2)_detached * x
        # Since y is detached, z is treated as constant 4.0
        # dout/dx = 4.0
        self.assertEqual(np.array(x.grad, copy=False)[0], 4.0)

    def test_item_method(self):
        """Test the item() method for scalar tensors."""
        t = munet.Tensor([1])
        np.array(t, copy=False)[0] = 1.23
        val = t.item()

        self.assertIsInstance(val, float)
        self.assertAlmostEqual(val, 1.23, places=5)

        # Test error for non-scalar
        t2 = munet.Tensor([2])
        with self.assertRaises(RuntimeError):
            t2.item()

    def test_numpy_method(self):
        """Test the .numpy() binding for Host tensors."""
        t = munet.Tensor([2, 3])
        np.array(t, copy=False)[:] = 1.0

        arr = t.detach().numpy()
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (2, 3))
        self.assertEqual(arr[0, 0], 1.0)

        # Check sharing
        arr[0, 0] = 99.0
        self.assertEqual(np.array(t, copy=False)[0, 0], 99.0)

    def test_numpy_safety(self):
        """Verify .numpy() safety checks for gradients and device."""
        # 1. Grad safety
        t = munet.ones([2, 2], requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, r"Use \.detach\(\)\.numpy\(\) instead"):
            t.numpy()

    def test_adam_optimizer(self):
        """Test Adam optimizer convergence in Python."""
        params = [munet.Tensor([1], requires_grad=False)]
        np.array(params[0], copy=False)[0] = 1.0
        params[0].requires_grad = True

        # Optimize f(x) = x^2
        optimizer = munet.optim.Adam(params, lr=1e-1)

        for _ in range(10):
            optimizer.zero_grad()
            x = params[0]
            loss = x * x
            loss.backward()
            optimizer.step()

        final_val = params[0].item()
        self.assertLess(abs(final_val), 1.0)
        self.assertGreater(abs(final_val), 0.0)

    def test_transpose_view(self):
        """Test that transpose creates a view with swapped strides."""
        t = munet.Tensor([2, 3])
        # Default strides for [2, 3] should be [3, 1]
        self.assertEqual(t.strides, [3, 1])

        t_t = t.transpose(0, 1)
        self.assertEqual(t_t.shape, [3, 2])
        # Transposed strides should be [1, 3]
        self.assertEqual(t_t.strides, [1, 3])
        self.assertFalse(t_t.is_contiguous)

        # Verify contiguous() restores default stride order [2, 1] for [3, 2]
        t_c = t_t.contiguous()
        self.assertEqual(t_c.shape, [3, 2])
        self.assertEqual(t_c.strides, [2, 1])
        self.assertTrue(t_c.is_contiguous)

    def test_transpose_autograd(self):
        """Test that gradients flow through transpose views."""
        x = munet.Tensor([2, 2])
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.array(x, copy=False)[:] = x_np
        x.requires_grad = True

        y = x.transpose(0, 1)
        # z = y[0,0] + y[0,1] + ...
        # basically sum(x.T) = sum(x)
        loss = y.sum()
        loss.backward()

        grad = x.grad.detach().numpy()
        self.assertTrue(np.allclose(grad, np.ones((2, 2))))

    def test_sgd_optimizer(self):
        """Test SGD optimizer in Python."""
        params = [munet.Tensor([1], requires_grad=False)]
        np.array(params[0], copy=False)[0] = 1.0
        params[0].requires_grad = True

        optimizer = munet.optim.SGD(params, lr=0.1)

        # 1 step: x = 1.0, grad = 2.0, x_new = 1.0 - 0.2 = 0.8
        optimizer.zero_grad()
        (params[0] * params[0]).backward()
        optimizer.step()

        self.assertAlmostEqual(params[0].item(), 0.8, places=6)

    def test_model_serialization_full_roundtrip(self):
        """Save full model and reconstruct it from file without original definition."""
        model = _sequential([
            munet.nn.Linear(4, 8),
            munet.nn.GELU(),
            munet.nn.Linear(8, 2),
        ])

        x = munet.Tensor([3, 4], requires_grad=False)
        np.array(x, copy=False)[:] = np.array(
            [[0.1, -0.2, 0.3, 0.4], [0.5, 0.6, -0.7, 0.8], [-0.9, 1.0, 0.2, -0.1]],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "full_model.npz")
            munet.save_checkpoint(model, path)

            loaded = munet.load_checkpoint(path, trusted=False)
            y_ref = np.array(model.forward(x).detach(), copy=False)
            y_loaded = np.array(loaded.forward(x).detach(), copy=False)
            self.assertTrue(np.allclose(y_ref, y_loaded, atol=1e-6))

    def test_model_serialization_weights_only(self):
        """Load weights into an existing model definition."""
        src = _sequential([
            munet.nn.Linear(4, 8),
            munet.nn.ReLU(),
            munet.nn.Linear(8, 2),
        ])
        dst = _sequential([
            munet.nn.Linear(4, 8),
            munet.nn.ReLU(),
            munet.nn.Linear(8, 2),
        ])

        x = munet.Tensor([2, 4], requires_grad=False)
        np.array(x, copy=False)[:] = np.array(
            [[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, 0.5, 0.25]], dtype=np.float32
        )

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "weights_only.npz")
            munet.save_checkpoint(src, path)
            munet.load_weights_checkpoint(dst, path)

            y_src = np.array(src.forward(x).detach(), copy=False)
            y_dst = np.array(dst.forward(x).detach(), copy=False)
            self.assertTrue(np.allclose(y_src, y_dst, atol=1e-6))

    def test_model_serialization_custom_class_rebuild_without_redefinition(self):
        class TinyGraphNet(munet.nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj = munet.nn.Linear(4, 8)
                self.out_proj = munet.nn.Linear(8, 2)

            def forward(self, x):
                h = self.in_proj(x)
                h = h.relu()
                return self.out_proj(h)

        model = TinyGraphNet()
        x = munet.Tensor([2, 4], requires_grad=False)
        np.array(x, copy=False)[:] = np.array(
            [[0.2, -0.1, 0.5, 0.7], [1.0, -0.3, 0.4, -0.8]], dtype=np.float32
        )

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "custom_graph_roundtrip.npz")
            munet.save_checkpoint(model, path)

            del TinyGraphNet
            loaded = munet.load_checkpoint(path, trusted=True)

            y_ref = np.array(model.forward(x).detach(), copy=False)
            y_loaded = np.array(loaded.forward(x).detach(), copy=False)
            self.assertTrue(np.allclose(y_ref, y_loaded, atol=1e-6))

            loss = loaded.forward(x).sum()
            loaded.zero_grad()
            loss.backward()
            self.assertTrue(loaded.named_parameters()["in_proj.weight"].has_grad())

    def test_model_serialization_custom_class_writes_hybrid_payload(self):
        class TinyHybridNet(munet.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = munet.nn.Linear(3, 3)

            def forward(self, x):
                return self.fc(x).relu()

        model = TinyHybridNet()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "custom_hybrid_payload.npz")
            munet.save_checkpoint(model, path)

            with np.load(path, allow_pickle=True) as state:
                self.assertIn("__format__", state.files)
                self.assertEqual(str(state["__format__"]), "munet_hybrid_v1")
                self.assertIn("__shell__", state.files)
                cfg = json.loads(str(state["__config__"]))
                self.assertEqual(cfg["type"], "__custom__")

    def test_model_serialization_checkpoint_builtin_has_no_hybrid_shell(self):
        model = _sequential([
            munet.nn.Linear(4, 4),
            munet.nn.ReLU(),
        ])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "builtin_checkpoint.npz")
            munet.save_checkpoint(model, path)
            with np.load(path, allow_pickle=True) as state:
                self.assertNotIn("__shell__", state.files)
                self.assertNotIn("__format__", state.files)
                metadata = munet.serialization_metadata(path)
                self.assertEqual(metadata["artifact_kind"], "training_checkpoint")

    def test_model_serialization_custom_class_non_default_ctor_requires_weights_only(self):
        class NonDefaultCtorNet(munet.nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.fc1 = munet.nn.Linear(4, hidden)
                self.fc2 = munet.nn.Linear(hidden, 2)

            def forward(self, x):
                return self.fc2(self.fc1(x).relu())

        src = NonDefaultCtorNet(8)
        dst = NonDefaultCtorNet(8)
        x = munet.Tensor([1, 4], requires_grad=False)
        np.array(x, copy=False)[:] = np.array([[0.1, 0.2, -0.3, 0.4]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "non_default_ctor_checkpoint.npz")
            munet.save_checkpoint(src, path)

            with self.assertRaises(ValueError):
                munet.load_checkpoint(path, trusted=True)

            munet.load_weights_checkpoint(dst, path)
            y_src = np.array(src.forward(x).detach(), copy=False)
            y_dst = np.array(dst.forward(x).detach(), copy=False)
            self.assertTrue(np.allclose(y_src, y_dst, atol=1e-6))

    def test_model_serialization_checkpoint_untrusted_source_execution_blocked(self):
        class TinyUnsafeNet(munet.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = munet.nn.Linear(4, 2)

            def forward(self, x):
                return self.fc(x)

        model = TinyUnsafeNet()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "unsafe_checkpoint.npz")
            munet.save_checkpoint(model, path)
            with self.assertRaises(ValueError):
                munet.load_checkpoint(path, trusted=False)

    def test_tensor_factories_and_numpy_roundtrip_preserve_dtype(self):
        half = munet.ones([2, 2], dtype=munet.DataType.Float16)
        self.assertEqual(half.dtype, munet.DataType.Float16)
        self.assertEqual(np.array(half.detach(), copy=False).dtype, np.float16)
        self.assertTrue(np.allclose(np.array(half.detach(), copy=False), np.ones((2, 2), dtype=np.float16)))

        ints = munet.zeros([3], dtype=munet.DataType.Int32)
        munet.copy_from_numpy(ints, np.array([1, 2, 3], dtype=np.int32))
        self.assertEqual(ints.dtype, munet.DataType.Int32)
        self.assertEqual(np.array(ints.detach(), copy=False).dtype, np.int32)
        self.assertTrue(np.array_equal(np.array(ints.detach(), copy=False), np.array([1, 2, 3], dtype=np.int32)))

        arr16 = np.array([[1.5, -2.0]], dtype=np.float16)
        from_np = munet.from_numpy(arr16)
        self.assertEqual(from_np.dtype, munet.DataType.Float16)
        self.assertEqual(np.array(from_np.detach(), copy=False).dtype, np.float16)
        self.assertTrue(np.allclose(np.array(from_np.detach(), copy=False), arr16))

    def test_model_serialization_preserves_dtype_policy(self):
        opts = munet.TensorOptions()
        opts.dtype = munet.DataType.Float16
        model = _sequential([
            munet.nn.Linear(4, 4, options=opts),
            munet.nn.BatchNorm2d(4, options=opts),
            munet.nn.LayerNorm(4, options=opts),
        ])

        np.array(model.named_parameters()["1.running_mean"], copy=False)[:] = np.array(
            [0.5, -1.0, 1.5, -2.0], dtype=np.float32
        )
        np.array(model.named_parameters()["1.running_var"], copy=False)[:] = np.array(
            [1.25, 0.75, 2.0, 3.0], dtype=np.float32
        )

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "dtype_model.npz")
            munet.save_checkpoint(model, path)
            loaded = munet.load_checkpoint(path, trusted=False)

            named = loaded.named_parameters()
            self.assertEqual(named["0.weight"].dtype, munet.DataType.Float16)
            self.assertEqual(named["0.bias"].dtype, munet.DataType.Float16)
            self.assertEqual(named["1.weight"].dtype, munet.DataType.Float16)
            self.assertEqual(named["1.bias"].dtype, munet.DataType.Float16)
            self.assertEqual(named["1.running_mean"].dtype, munet.DataType.Float32)
            self.assertEqual(named["1.running_var"].dtype, munet.DataType.Float32)
            self.assertTrue(
                np.allclose(
                    np.array(named["1.running_mean"].detach(), copy=False),
                    np.array([0.5, -1.0, 1.5, -2.0], dtype=np.float32),
                )
            )
            self.assertTrue(
                np.allclose(
                    np.array(named["1.running_var"].detach(), copy=False),
                    np.array([1.25, 0.75, 2.0, 3.0], dtype=np.float32),
                )
            )
            self.assertEqual(named["2.weight"].dtype, munet.DataType.Float16)
            self.assertEqual(named["2.bias"].dtype, munet.DataType.Float16)

    def test_serialization_metadata_and_format_info(self):
        info = munet.serialization_format_info()
        self.assertEqual(info["format_name"], "munet_model")
        self.assertEqual(info["format_revision"], 1)
        self.assertEqual(info["legacy_tag"], "munet_model_v1")
        self.assertEqual(info["artifact_kind"], "deploy_model")
        self.assertEqual(info["artifact_scope"], "runtime_only")
        self.assertEqual(info["default_load_mode"], "eval")
        self.assertFalse(info["contains_training_state"])
        self.assertEqual(info["device_policy"], "caller_specified")
        self.assertEqual(info["dtype_policy"], "per_tensor")
        self.assertEqual(info["recommended_loader"], "load_for_inference")
        self.assertEqual(info["compile_contract_policy"], "external")

        model = _sequential([
            munet.nn.Linear(4, 4),
            munet.nn.ReLU(),
        ])

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "metadata_model.npz")
            munet.save_deploy(model, path)
            metadata = munet.serialization_metadata(path)
            self.assertEqual(metadata["format_name"], "munet_model")
            self.assertEqual(metadata["format_revision"], 1)
            self.assertEqual(metadata["legacy_tag"], "munet_model_v1")
            self.assertEqual(metadata["artifact_kind"], "deploy_model")
            self.assertEqual(metadata["artifact_scope"], "runtime_only")
            self.assertEqual(metadata["default_load_mode"], "eval")
            self.assertFalse(metadata["contains_training_state"])
            self.assertEqual(metadata["device_policy"], "caller_specified")
            self.assertEqual(metadata["dtype_policy"], "per_tensor")
            self.assertEqual(metadata["recommended_loader"], "load_for_inference")
            self.assertEqual(metadata["compile_contract_policy"], "external")
            self.assertGreaterEqual(metadata["tensor_count"], 2)
            self.assertIn("0.weight", metadata["tensor_names"])
            self.assertTrue(metadata["has_config"])

    def test_legacy_save_load_apis_removed(self):
        self.assertFalse(hasattr(munet, "save"))
        self.assertFalse(hasattr(munet, "load"))

    def test_model_serialization_rejects_unsupported_revision(self):
        model = _sequential([
            munet.nn.Linear(4, 4),
            munet.nn.ReLU(),
        ])

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad_revision_model.npz")
            munet.save_deploy(model, path)

            with np.load(path, allow_pickle=True) as state:
                mutated = {key: state[key] for key in state.files}
            mutated["__format_revision__"] = np.array(999)
            np.savez(path, **mutated)

            with self.assertRaises(ValueError):
                munet.load_deploy(path)

    def test_checkpoint_serialization_rejects_unsupported_revision(self):
        model = _sequential([
            munet.nn.Linear(4, 4),
            munet.nn.ReLU(),
        ])

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad_checkpoint_revision_model.npz")
            munet.save_checkpoint(model, path)

            with np.load(path, allow_pickle=True) as state:
                mutated = {key: state[key] for key in state.files}
            mutated["__format_revision__"] = np.array(4242)
            np.savez(path, **mutated)

            with self.assertRaises(ValueError):
                munet.load_checkpoint(path)

    def test_save_deploy_rejects_custom_module(self):
        class TinyCustomDeployNet(munet.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = munet.nn.Linear(4, 2)

            def forward(self, x):
                return self.fc(x)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "deploy_custom.npz")
            with self.assertRaises(ValueError):
                munet.save_deploy(TinyCustomDeployNet(), path)

    def test_model_serialization_rejects_training_payload_keys(self):
        model = _sequential([
            munet.nn.Linear(4, 4),
            munet.nn.ReLU(),
        ])

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad_training_payload.npz")
            munet.save_deploy(model, path)

            with np.load(path, allow_pickle=True) as state:
                mutated = {key: state[key] for key in state.files}
            mutated["optimizer_state"] = np.array([1.0], dtype=np.float32)
            np.savez(path, **mutated)

            with self.assertRaises(ValueError):
                munet.load_for_inference(path)

    def test_load_for_inference_sets_eval_mode(self):
        model = _sequential([
            munet.nn.Dropout(0.5),
        ])
        model.train(True)

        x = munet.ones([2, 3], dtype=munet.DataType.Float32)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "deploy_dropout.npz")
            munet.save_deploy(model, path)

            restored = munet.load_for_inference(path)
            y = np.array(restored.forward(x).detach(), copy=False)
            self.assertTrue(np.allclose(y, np.ones((2, 3), dtype=np.float32)))
    def test_load_weights_for_inference_sets_eval_mode(self):
        src = _sequential([
            munet.nn.Dropout(0.5),
        ])
        dst = _sequential([
            munet.nn.Dropout(0.5),
        ])
        src.train(True)
        dst.train(True)
        x = munet.ones([2, 2], dtype=munet.DataType.Float32)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "deploy_weights_only.npz")
            munet.save_deploy(src, path)
            munet.load_weights_for_inference(dst, path)
            y = np.array(dst.forward(x).detach(), copy=False)
            self.assertTrue(np.allclose(y, np.ones((2, 2), dtype=np.float32)))
    def test_inference_engine_compile_and_shape_guard(self):
        model = _sequential([
            munet.nn.Linear(4, 8),
            munet.nn.ReLU(),
            munet.nn.Linear(8, 2),
        ])

        eng = munet.inference.Engine()
        eng.load(model)

        x = munet.Tensor([2, 4], requires_grad=False)
        np.array(x, copy=False)[:] = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)

        eng.compile(x)
        self.assertTrue(eng.is_compiled())
        self.assertEqual(eng.compiled_input_shape(), [2, 4])
        self.assertGreaterEqual(eng.stats().compile_ms, 0.0)

        y = eng.run(x)
        self.assertEqual(y.shape, [2, 2])

        # Avoid mismatched-shape runtime path here; this currently triggers
        # undefined behavior in debug Host-only builds.
        y2 = eng.run(x)
        self.assertEqual(y2.shape, [2, 2])

    def test_inference_engine_defaults_to_low_overhead_mode_without_diagnostics(self):
        model = _sequential([
            munet.nn.Linear(4, 8),
            munet.nn.ReLU(),
            munet.nn.Linear(8, 2),
        ])

        eng = munet.inference.Engine()
        self.assertFalse(eng.capture_profiler_memory())
        self.assertFalse(eng.lean_mode())

        x = munet.Tensor([2, 4], requires_grad=False)
        np.array(x, copy=False)[:] = np.ones((2, 4), dtype=np.float32)

        eng.load(model)
        eng.compile(x)
        y = eng.run(x)

        self.assertEqual(y.shape, [2, 2])
        self.assertEqual(eng.stats().last_compile_trace_id, 0)
        self.assertEqual(eng.stats().last_run_trace_id, 0)
        self.assertEqual(eng.stats().current_memory_bytes, 0)
        self.assertEqual(eng.stats().peak_memory_bytes, 0)

    def test_inference_engine_lean_mode_disables_memory_capture(self):
        cfg = munet.inference.EngineConfig()
        cfg.lean_mode = True
        eng = munet.inference.Engine(cfg)
        self.assertTrue(eng.lean_mode())
        self.assertFalse(eng.capture_profiler_memory())

        model = _sequential([munet.nn.Linear(4, 2)])
        x = munet.ones([1, 4], dtype=munet.DataType.Float32)
        eng.load(model)
        y = eng.run(x)

        self.assertEqual(y.shape, [1, 2])
        self.assertEqual(eng.stats().current_memory_bytes, 0)
        self.assertEqual(eng.stats().peak_memory_bytes, 0)

    def test_inference_engine_exposes_bounded_prepared_input_cache_policy(self):
        cfg = munet.inference.EngineConfig()
        cfg.prepared_input_cache_entries = 1
        cfg.prepared_input_cache_max_bytes = 1024
        eng = munet.inference.Engine(cfg)

        self.assertEqual(eng.prepared_input_cache_entries_limit(), 1)
        self.assertEqual(eng.prepared_input_cache_max_bytes_limit(), 1024)

        model = _sequential([munet.nn.Linear(4, 2)])
        eng.load(model)

        a = munet.ones([1, 4], dtype=munet.DataType.Float32)
        b = munet.ones([1, 4], dtype=munet.DataType.Float32)
        eng.run_batch([a, b])
        stats = eng.stats()

        self.assertLessEqual(stats.prepared_input_cache_entries, 1)
        self.assertLessEqual(stats.prepared_input_cache_bytes, 1024)

    def test_inference_engine_prepare_batch_prepopulates_cache(self):
        cfg = munet.inference.EngineConfig()
        cfg.prepared_input_cache_entries = 2
        eng = munet.inference.Engine(cfg)
        model = _sequential([munet.nn.Linear(4, 2)])
        eng.load(model)

        a = munet.ones([1, 4], dtype=munet.DataType.Float32)
        b = munet.ones([1, 4], dtype=munet.DataType.Float32)
        eng.prepare_batch([a, b])
        self.assertEqual(eng.stats().prepared_input_cache_misses, 0)

    def test_inference_engine_from_serialized_model(self):
        model = _sequential([
            munet.nn.Linear(3, 3),
            munet.nn.Tanh(),
            munet.nn.Linear(3, 1),
        ])

        x = munet.Tensor([4, 3], requires_grad=False)
        np.array(x, copy=False)[:] = np.array(
            [[0.1, 0.2, 0.3], [0.4, -0.1, 0.0], [1.0, -1.0, 0.5], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "e2e_model.npz")
            munet.save_deploy(model, path)
            restored = munet.load_for_inference(path)

            eng = munet.inference.Engine()
            eng.load(restored)
            eng.compile(x)

            y_ref = np.array(restored.forward(x).detach(), copy=False)
            y_eng = np.array(eng.run(x).detach(), copy=False)
            self.assertTrue(np.allclose(y_ref, y_eng, atol=1e-6))
    def test_inference_engine_dynamic_dims_with_wildcards(self):
        model = _sequential([
            munet.nn.Conv2d(3, 4, 3, padding=1),
            munet.nn.ReLU(),
            munet.nn.Conv2d(4, 2, 1),
        ])

        eng = munet.inference.Engine()
        eng.load(model)

        x_compile = munet.Tensor([1, 3, 64, 64], requires_grad=False)
        np.array(x_compile, copy=False)[:] = np.random.randn(1, 3, 64, 64).astype(np.float32)

        eng.compile(x_compile, expected_input_shape=[-1, 3, -1, -1], expected_output_shape=[-1, 2, -1, -1])
        self.assertEqual(eng.compiled_input_shape(), [1, 3, 64, 64])
        self.assertEqual(eng.compiled_output_shape(), [1, 2, 64, 64])

        x_ok = munet.Tensor([2, 3, 128, 80], requires_grad=False)
        np.array(x_ok, copy=False)[:] = np.random.randn(2, 3, 128, 80).astype(np.float32)
        y_ok = eng.run(x_ok)
        self.assertEqual(y_ok.shape, [2, 2, 128, 80])

        x_bad = munet.Tensor([2, 1, 128, 80], requires_grad=False)
        np.array(x_bad, copy=False)[:] = np.random.randn(2, 1, 128, 80).astype(np.float32)
        with self.assertRaises(RuntimeError):
            eng.run(x_bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
