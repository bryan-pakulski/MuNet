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
    def _available_non_cpu_devices(self):
        devices = []
        for device_type in (munet.DeviceType.CUDA, munet.DeviceType.VULKAN):
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
        self.assertGreaterEqual(len(accelerators), 3)

        by_name = {entry["name"]: entry for entry in accelerators}
        self.assertIn("cpu", by_name)
        self.assertIn("cuda", by_name)
        self.assertIn("vulkan", by_name)

        cpu_entry = by_name["cpu"]
        self.assertTrue(cpu_entry["available"])
        self.assertGreaterEqual(len(cpu_entry["devices"]), 1)
        self.assertEqual(cpu_entry["devices"][0].type, munet.DeviceType.CPU)

        devices = munet.available_devices()
        self.assertIsInstance(devices, list)
        self.assertGreaterEqual(len(devices), 1)
        self.assertEqual(devices[0].type, munet.DeviceType.CPU)

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
        self.assertEqual(t.device.type, munet.DeviceType.CPU)
        self.assertEqual(t.dtype, munet.DataType.Float32)

    def test_backend_supports_api(self):
        cpu = munet.Device(munet.DeviceType.CPU, 0)
        self.assertTrue(munet.supports(cpu, munet.BackendFeature.Matmul, munet.DataType.Float32))
        self.assertFalse(munet.supports(cpu, munet.BackendFeature.Matmul, munet.DataType.Float16))
        self.assertTrue(munet.supports(cpu, munet.BackendFeature.RandomFill, munet.DataType.Float16))
        self.assertFalse(munet.supports(cpu, munet.BackendFeature.RandomFill, munet.DataType.Int32))

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

    def test_cuda_backend(self):
        """Test moving tensors to GPU and doing math."""
        # Check if CUDA was compiled in
        try:
            munet.Device(munet.DeviceType.CUDA, 0)
        except RuntimeError:
            print("\nSkipping CUDA tests (not compiled or no GPU).")
            return

        # 1. Create on CPU
        with munet.no_grad():
            a_cpu = munet.Tensor([2])
            b_cpu = munet.Tensor([2])

        np.array(a_cpu, copy=False)[:] = [1.0, 2.0]
        np.array(b_cpu, copy=False)[:] = [3.0, 4.0]

        a_cpu.requires_grad = True
        b_cpu.requires_grad = True

        # 2. Move to GPU!
        cuda_dev = munet.Device(munet.DeviceType.CUDA, 0)

        try:
            a_gpu = a_cpu.to(cuda_dev)
            b_gpu = b_cpu.to(cuda_dev)
        except RuntimeError as e:
            # Catch backend not implemented (if CMake didn't find CUDA)
            print(f"\nSkipping CUDA test due to: {e}")
            return

        # Ensure devices changed
        self.assertEqual(a_gpu.device.type, munet.DeviceType.CUDA)

        # 3. Add on GPU (invokes add_kernel on the device)
        c_gpu = a_gpu + b_gpu

        # 4. Bring result back to CPU to verify
        c_cpu = c_gpu.to(munet.Device(munet.DeviceType.CPU, 0))

        result = c_cpu.detach().numpy()
        self.assertEqual(result[0], 4.0)
        self.assertEqual(result[1], 6.0)

    def test_vulkan_backend(self):
        """Test moving tensors to Vulkan backend."""
        try:
            munet.Device(munet.DeviceType.VULKAN, 0)
        except RuntimeError:
            print("\nSkipping Vulkan tests (not compiled or no GPU).")
            return

        a_cpu = munet.Tensor([2], requires_grad=False)
        np.array(a_cpu, copy=False)[:] = [9.0, 10.0]

        vk_dev = munet.Device(munet.DeviceType.VULKAN, 0)

        try:
            a_vk = a_cpu.to(vk_dev)
        except RuntimeError as e:
            print(f"\nSkipping Vulkan test due to: {e}")
            return

        self.assertEqual(a_vk.device.type, munet.DeviceType.VULKAN)

        # Test moving it back to CPU to read the memory
        a_back = a_vk.to(munet.Device(munet.DeviceType.CPU, 0))
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
        # GPU or CPU
        dev = munet.Device(munet.DeviceType.CPU, 0)

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
        """Test seamless autograd across CPU and GPU boundaries."""
        # Find an available GPU
        gpu_dev = None
        for dev_type in (munet.DeviceType.CUDA, munet.DeviceType.VULKAN):
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
        cpu_dev = munet.Device(munet.DeviceType.CPU, 0)
        # --- Layer 1 on CPU ---
        x_cpu = munet.Tensor([1, 3], device=cpu_dev)
        w1_cpu = munet.Tensor([3, 4], device=cpu_dev, requires_grad=True)
        np.array(x_cpu, copy=False)[:] = [[1.0, 2.0, 3.0]]
        w1_cpu.uniform_(-0.5, 0.5)
        # --- Layer 2 on GPU ---
        w2_gpu = munet.Tensor([4, 2], device=gpu_dev, requires_grad=True)
        w2_gpu.uniform_(-0.5, 0.5)
        # --- Forward Pass (Cross-device) ---
        h_cpu = x_cpu @ w1_cpu
        h_gpu = h_cpu.to(gpu_dev)  # Autograd boundary crossing!
        pred_gpu = h_gpu @ w2_gpu

        target_gpu = munet.Tensor([1, 2], device=gpu_dev)
        target_gpu.uniform_(-1.0, 1.0)

        diff = pred_gpu - target_gpu
        loss = (diff * diff).sum()
        # --- Backward Pass ---
        loss.backward()
        # --- Verify Gradients Synchronized Correctly ---
        self.assertTrue(w1_cpu.grad is not None)
        self.assertTrue(w2_gpu.grad is not None)
        self.assertEqual(w1_cpu.grad.device.type, munet.DeviceType.CPU)
        self.assertEqual(w2_gpu.grad.device.type, gpu_dev.type)
        print(f"\nSuccessfully backpropagated from {gpu_dev.type} back to CPU!")

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
        """Test the .numpy() binding for CPU tensors."""
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
        with self.assertRaisesRegex(RuntimeError, "Use .detach\(\).numpy\(\) instead"):
            t.numpy()

        # 2. Device safety
        gpu_dev = None
        for dev_type in (munet.DeviceType.CUDA, munet.DeviceType.VULKAN):
            try:
                gpu_dev = munet.Device(dev_type, 0)
                # verify it's actually usable
                munet.ones([1], device=gpu_dev)
                break
            except:
                continue

        if gpu_dev:
            try:
                t_gpu = munet.ones([2], device=gpu_dev)
            except RuntimeError:
                return
            with self.assertRaisesRegex(
                RuntimeError,
                "Cannot convert GPU tensor to NumPy array directly. Call `.to\(Device\(DeviceType.CPU\)\)` first.",
            ):
                t_gpu.numpy()

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

    def test_inference_load_serialized_alias_and_device(self):
        model = _sequential([
            munet.nn.Dropout(0.5),
        ])
        model.train(True)

        x = munet.ones([1, 3], dtype=munet.DataType.Float32)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "deploy_alias.npz")
            munet.save_deploy(model, path)

            restored = munet.inference.load_serialized(
                path,
                device=munet.Device(munet.DeviceType.CPU, 0),
            )
            y = np.array(restored.forward(x).detach(), copy=False)
            self.assertTrue(np.allclose(y, np.ones((1, 3), dtype=np.float32)))

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

    def test_model_serialization_from_non_cpu_device_preserves_dtype(self):
        devices = self._available_non_cpu_devices()
        if not devices:
            print("\nSkipping non-CPU serialization test (no CUDA/Vulkan device available).")
            return

        opts = munet.TensorOptions()
        opts.dtype = munet.DataType.Float16
        model = _sequential([
            munet.nn.Linear(4, 4, options=opts),
            munet.nn.LayerNorm(4, options=opts),
        ])
        model.to(devices[0])

        x = munet.ones([2, 4], dtype=munet.DataType.Float16).to(devices[0])

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "non_cpu_dtype_model.npz")
            munet.save_checkpoint(model, path)
            loaded = munet.load_checkpoint(path, trusted=False)

            y_ref = np.array(model.forward(x).detach().to(munet.Device(munet.DeviceType.CPU, 0)), copy=False)
            y_loaded = np.array(
                loaded.forward(x.to(munet.Device(munet.DeviceType.CPU, 0))).detach(),
                copy=False,
            )
            self.assertEqual(loaded.named_parameters()["0.weight"].dtype, munet.DataType.Float16)
            self.assertEqual(loaded.named_parameters()["1.weight"].dtype, munet.DataType.Float16)
            self.assertTrue(np.allclose(y_ref.astype(np.float32), y_loaded.astype(np.float32), atol=5e-2))


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
        # undefined behavior in debug CPU-only builds.
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

    def test_inference_engine_preserves_float16_across_available_devices(self):
        opts = munet.TensorOptions()
        opts.dtype = munet.DataType.Float16
        model = _sequential([
            munet.nn.Linear(2, 2, options=opts),
            munet.nn.ReLU(),
        ])

        x = munet.ones([1, 2], dtype=munet.DataType.Float16)
        expected = np.array(
            model.forward(x).detach().to(munet.Device(munet.DeviceType.CPU, 0)),
            copy=False,
        )

        for dev in [munet.Device(munet.DeviceType.CPU, 0)] + self._available_non_cpu_devices():
            if not munet.supports(dev, munet.BackendFeature.Matmul, munet.DataType.Float16):
                continue
            eng = munet.inference.Engine()
            eng.set_device(dev)
            eng.load(model)

            y = eng.run(x.to(dev)).detach().to(munet.Device(munet.DeviceType.CPU, 0))
            self.assertEqual(y.dtype, munet.DataType.Float16)
            self.assertTrue(
                np.allclose(
                    np.array(y, copy=False).astype(np.float32),
                    expected.astype(np.float32),
                    atol=5e-2,
                )
            )




    def test_compile_onnx_to_munet_module(self):
        try:
            import onnx
            from onnx import TensorProto, helper
        except Exception:
            print("\nSkipping ONNX compile test (onnx not installed).")
            return

        with tempfile.TemporaryDirectory() as d:
            onnx_path = os.path.join(d, "linear_relu.onnx")

            x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])
            y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 2])

            W = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float32)
            B = np.array([0.5, -1.0], dtype=np.float32)

            w_init = helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten().tolist())
            b_init = helper.make_tensor("B", TensorProto.FLOAT, B.shape, B.flatten().tolist())

            gemm = helper.make_node("Gemm", ["x", "W", "B"], ["z"], transB=0)
            relu = helper.make_node("Relu", ["z"], ["y"])

            graph = helper.make_graph([gemm, relu], "linear_relu_graph", [x_info], [y_info], [w_init, b_init])
            model = helper.make_model(graph, producer_name="munet_compile_test", opset_imports=[helper.make_opsetid("", 11)])
            model.ir_version = 7
            onnx.save(model, onnx_path)

            module = munet.inference.compile_onnx(onnx_path)

            x = munet.from_numpy(np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], dtype=np.float32))
            y = module.forward(x)
            y_np = np.array(y.detach(), copy=False)

            expected = np.maximum(x.numpy() @ W + B, 0.0)
            self.assertTrue(np.allclose(y_np, expected, atol=1e-5))

    def test_compile_onnx_report_unsupported_ops(self):
        try:
            import onnx
            from onnx import TensorProto, helper
        except Exception:
            print("\nSkipping ONNX unsupported-op report test (onnx not installed).")
            return

        with tempfile.TemporaryDirectory() as d:
            onnx_path = os.path.join(d, "unsupported.onnx")
            x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])
            y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 3])
            node = helper.make_node("Erf", ["x"], ["y"])
            graph = helper.make_graph([node], "unsupported_graph", [x_info], [y_info])
            model = helper.make_model(graph, producer_name="munet_compile_report_test", opset_imports=[helper.make_opsetid("", 11)])
            model.ir_version = 7
            onnx.save(model, onnx_path)

            missing = munet.inference.report_onnx_unsupported_ops(onnx_path)
            self.assertIn("Erf", missing)

            with self.assertRaises(ValueError) as ctx:
                munet.inference.compile_onnx(onnx_path)

            msg = str(ctx.exception)
            self.assertIn("unsupported_unique", msg)
            self.assertIn("unsupported_total", msg)
            self.assertIn("Erf", msg)

    def test_onnx_native_conversion_map_api(self):
        mp = munet.inference.onnx_native_conversion_map()
        self.assertIn("Gemm", mp)
        self.assertEqual(mp["Gemm"]["status"], "lowered")
        self.assertEqual(mp["LeakyRelu"]["status"], "lowered")
        self.assertEqual(mp["Gelu"]["status"], "lowered")
        self.assertEqual(mp["GlobalAveragePool"]["status"], "lowered")
        self.assertEqual(mp["Add"]["status"], "lowered")
        self.assertEqual(mp["Sub"]["status"], "lowered")
        self.assertEqual(mp["Mul"]["status"], "lowered")
        self.assertEqual(mp["Div"]["status"], "lowered")
        self.assertEqual(mp["Reshape"]["status"], "lowered")
        self.assertEqual(mp["Transpose"]["status"], "lowered")
        self.assertEqual(mp["Concat"]["status"], "lowered")
        self.assertEqual(mp["Squeeze"]["status"], "lowered")
        self.assertEqual(mp["Expand"]["status"], "lowered")
        self.assertEqual(mp["Tile"]["status"], "lowered")
        self.assertEqual(mp["ConstantOfShape"]["status"], "lowered")
        self.assertEqual(mp["Gather"]["status"], "lowered")

    def test_compile_onnx_lower_leakyrelu_and_globalavgpool(self):
        try:
            import onnx
            from onnx import TensorProto, helper
        except Exception:
            print("\nSkipping ONNX LeakyRelu/GlobalAveragePool lowering test (onnx not installed).")
            return

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "conv_lrelu_gap.onnx")

            x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
            y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 1])

            W = np.array([[[[1.0]]]], dtype=np.float32)
            B = np.array([0.0], dtype=np.float32)
            w_init = helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten().tolist())
            b_init = helper.make_tensor("B", TensorProto.FLOAT, B.shape, B.flatten().tolist())

            conv = helper.make_node("Conv", ["x", "W", "B"], ["z"])
            lrelu = helper.make_node("LeakyRelu", ["z"], ["a"], alpha=0.1)
            gap = helper.make_node("GlobalAveragePool", ["a"], ["y"])

            graph = helper.make_graph([conv, lrelu, gap], "conv_lrelu_gap", [x_info], [y_info], [w_init, b_init])
            model = helper.make_model(graph, producer_name="munet_lrelu_gap_test", opset_imports=[helper.make_opsetid("", 11)])
            model.ir_version = 7
            onnx.save(model, path)

            module = munet.inference.compile_onnx(path)
            x = munet.from_numpy(np.array([[[[-1.0, 3.0], [2.0, -4.0]]]], dtype=np.float32))
            out = np.array(module.forward(x).detach(), copy=False)

            expected = np.array([[[[( -0.1 + 3.0 + 2.0 - 0.4 ) / 4.0]]]], dtype=np.float32)
            self.assertTrue(np.allclose(out, expected, atol=1e-5))

    def test_compile_onnx_preserves_float16_dtype_through_graph_runtime(self):
        try:
            import onnx
            from onnx import TensorProto, helper, numpy_helper
        except Exception:
            print("\nSkipping ONNX float16 dtype test (onnx not installed).")
            return

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "float16_add.onnx")
            x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 2])
            y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 2])
            bias = numpy_helper.from_array(np.array([[0.5, -1.0]], dtype=np.float16), name="bias")

            node = helper.make_node("Add", ["x", "bias"], ["y"])
            graph = helper.make_graph([node], "float16_add_graph", [x_info], [y_info], [bias])
            model = helper.make_model(graph, producer_name="munet_float16_add_test", opset_imports=[helper.make_opsetid("", 11)])
            model.ir_version = 7
            onnx.save(model, path)

            module = munet.inference.compile_onnx(path)
            x = munet.from_numpy(np.array([[1.5, 2.0]], dtype=np.float16))
            y = module.forward(x)

            self.assertEqual(y.dtype, munet.DataType.Float16)
            self.assertEqual(np.array(y.detach(), copy=False).dtype, np.float16)
            self.assertTrue(
                np.allclose(
                    np.array(y.detach(), copy=False),
                    np.array([[2.0, 1.0]], dtype=np.float16),
                    atol=1e-3,
                )
            )

    def test_compile_onnx_cast_preserves_requested_dtype(self):
        try:
            import onnx
            from onnx import TensorProto, helper
        except Exception:
            print("\nSkipping ONNX cast dtype test (onnx not installed).")
            return

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cast_to_float16.onnx")
            x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
            y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 2])
            cast = helper.make_node("Cast", ["x"], ["y"], to=TensorProto.FLOAT16)
            graph = helper.make_graph([cast], "cast_to_float16_graph", [x_info], [y_info])
            model = helper.make_model(graph, producer_name="munet_cast_test", opset_imports=[helper.make_opsetid("", 11)])
            model.ir_version = 7
            onnx.save(model, path)

            module = munet.inference.compile_onnx(path)
            x = munet.from_numpy(np.array([[1.5, -2.25]], dtype=np.float32))
            y = module.forward(x)

            self.assertEqual(y.dtype, munet.DataType.Float16)
            self.assertEqual(np.array(y.detach(), copy=False).dtype, np.float16)
            self.assertTrue(
                np.allclose(
                    np.array(y.detach(), copy=False).astype(np.float32),
                    np.array([[1.5, -2.25]], dtype=np.float32),
                    atol=1e-3,
                )
            )

    def test_compile_onnx_binary_ops_add_sub_mul_div(self):
        try:
            import onnx
            from onnx import TensorProto, helper
        except Exception:
            print("\nSkipping ONNX binary ops lowering test (onnx not installed).")
            return

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "binary_ops.onnx")
            x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
            y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])

            c_add = helper.make_tensor("c_add", TensorProto.FLOAT, [1, 3], [1.0, 2.0, 3.0])
            c_sub = helper.make_tensor("c_sub", TensorProto.FLOAT, [1, 3], [0.5, 1.0, 1.5])
            c_mul = helper.make_tensor("c_mul", TensorProto.FLOAT, [1, 3], [2.0, 2.0, 2.0])
            c_div = helper.make_tensor("c_div", TensorProto.FLOAT, [1, 3], [2.0, 4.0, 8.0])

            n1 = helper.make_node("Add", ["x", "c_add"], ["a"])
            n2 = helper.make_node("Sub", ["a", "c_sub"], ["b"])
            n3 = helper.make_node("Mul", ["b", "c_mul"], ["c"])
            n4 = helper.make_node("Div", ["c", "c_div"], ["y"])

            graph = helper.make_graph([n1, n2, n3, n4], "binary_ops_graph", [x_info], [y_info], [c_add, c_sub, c_mul, c_div])
            model = helper.make_model(graph, producer_name="munet_binary_ops_test", opset_imports=[helper.make_opsetid("", 11)])
            model.ir_version = 7
            onnx.save(model, path)

            module = munet.inference.compile_onnx(path)
            x = np.array([[2.0, 4.0, 8.0]], dtype=np.float32)
            y = np.array(module.forward(munet.from_numpy(x)).detach(), copy=False)
            expected = (((x + np.array([[1.0, 2.0, 3.0]], dtype=np.float32)) - np.array([[0.5, 1.0, 1.5]], dtype=np.float32)) * 2.0) / np.array([[2.0, 4.0, 8.0]], dtype=np.float32)
            self.assertTrue(np.allclose(y, expected, atol=1e-6))

    def test_compile_onnx_phase1_shape_index_ops(self):
        try:
            import onnx
            from onnx import TensorProto, helper
        except Exception:
            print("\nSkipping ONNX phase1 shape/index ops test (onnx not installed).")
            return

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "phase1_shape_index.onnx")
            x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 1, 2])
            y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2, 2])

            # ConstantOfShape input shape
            shp = helper.make_tensor("shape_vec", TensorProto.INT64, [4], [2, 1, 2, 2])
            cshape = helper.make_node("ConstantOfShape", ["shape_vec"], ["base"])
            add = helper.make_node("Add", ["base", "x"], ["a"])

            # Expand to same shape (exercise operator semantics)
            exp_shape = helper.make_tensor("exp_shape", TensorProto.INT64, [4], [2, 1, 2, 2])
            expand = helper.make_node("Expand", ["a", "exp_shape"], ["e"])

            # Tile channel dim then gather first channel back
            reps = helper.make_tensor("reps", TensorProto.INT64, [4], [1, 2, 1, 1])
            tile = helper.make_node("Tile", ["e", "reps"], ["t"])
            idx = helper.make_tensor("idx", TensorProto.INT64, [1], [0])
            gather = helper.make_node("Gather", ["t", "idx"], ["g"], axis=1)
            sq_axes = helper.make_tensor("sq_axes", TensorProto.INT64, [1], [1])
            squeeze = helper.make_node("Squeeze", ["g", "sq_axes"], ["y"])

            graph = helper.make_graph(
                [cshape, add, expand, tile, gather, squeeze],
                "phase1_shape_index_graph",
                [x_info],
                [y_info],
                [shp, exp_shape, reps, idx, sq_axes],
            )
            model = helper.make_model(graph, producer_name="munet_phase1_test", opset_imports=[helper.make_opsetid("", 13)])
            model.ir_version = 7
            onnx.save(model, path)

            module = munet.inference.compile_onnx(path)
            x = munet.from_numpy(np.array([[[[2.0, 4.0]]]], dtype=np.float32))
            out = np.array(module.forward(x).detach(), copy=False)
            expected = np.array([[[2.0, 4.0], [2.0, 4.0]], [[2.0, 4.0], [2.0, 4.0]]], dtype=np.float32)
            self.assertTrue(np.allclose(out, expected, atol=1e-6))

    def test_compile_onnx_strict_failure_reports_counts(self):
        try:
            import onnx
            from onnx import TensorProto, helper
        except Exception:
            print("\nSkipping ONNX strict-failure report test (onnx not installed).")
            return

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "strict_fail.onnx")
            x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])
            y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 3])
            soft = helper.make_node("Softmax", ["x"], ["z1"])
            erf = helper.make_node("Erf", ["z1"], ["y"])
            graph = helper.make_graph([soft, erf], "strict_fail_graph", [x_info], [y_info])
            model = helper.make_model(graph, producer_name="munet_strict_fail_test", opset_imports=[helper.make_opsetid("", 11)])
            model.ir_version = 7
            onnx.save(model, path)

            with self.assertRaises(ValueError) as ctx:
                munet.inference.compile_onnx(path)

            msg = str(ctx.exception)
            self.assertIn("unsupported_total=1", msg)
            self.assertIn("Erf", msg)
            self.assertIn("op_counts=", msg)

    def test_onnx_conversion_coverage_report_generated_graph(self):
        try:
            import onnx
            from onnx import TensorProto, helper
        except Exception:
            print("\nSkipping ONNX coverage-report test (onnx not installed).")
            return

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "coverage_graph.onnx")
            x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])
            y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 2])

            W = np.array([[1.0, 0.5], [0.0, -1.0], [2.0, 1.0]], dtype=np.float32)
            B = np.array([0.25, -0.75], dtype=np.float32)
            w_init = helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten().tolist())
            b_init = helper.make_tensor("B", TensorProto.FLOAT, B.shape, B.flatten().tolist())

            nodes = [
                helper.make_node("Gemm", ["x", "W", "B"], ["z"], transB=0),
                helper.make_node("Relu", ["z"], ["a"]),
                helper.make_node("Sigmoid", ["a"], ["y"]),
            ]
            graph = helper.make_graph(nodes, "coverage_graph", [x_info], [y_info], [w_init, b_init])
            model = helper.make_model(graph, producer_name="munet_coverage_test", opset_imports=[helper.make_opsetid("", 11)])
            model.ir_version = 7
            onnx.save(model, path)

            report = munet.inference.onnx_conversion_coverage_report(path)
            self.assertEqual(report["total_nodes"], 3)
            self.assertTrue(report["fully_lowerable"])
            self.assertTrue(report["native_deployable"])
            self.assertEqual(report["runtime_role"], "deploy_runtime")
            self.assertEqual(report["device_policy"], "caller_specified")
            self.assertEqual(report["dtype_policy"], "preserve_onnx_io_types")
            self.assertEqual(report["shape_contract_policy"], "caller_declared_at_engine_compile")
            self.assertEqual(report["warm_state"], "not_embedded")
            self.assertEqual(report["coverage"]["unsupported"], [])
            self.assertEqual(report["coverage"]["unmapped"], [])

    def test_compile_onnx_native_module_can_save_deploy_artifact(self):
        try:
            import onnx
            from onnx import TensorProto, helper
        except Exception:
            print("\nSkipping ONNX native-output save test (onnx not installed).")
            return

        with tempfile.TemporaryDirectory() as d:
            onnx_path = os.path.join(d, "linear_relu_native.onnx")
            native_path = os.path.join(d, "linear_relu_native.npz")

            x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])
            y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 2])

            W = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float32)
            B = np.array([0.5, -1.0], dtype=np.float32)

            w_init = helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten().tolist())
            b_init = helper.make_tensor("B", TensorProto.FLOAT, B.shape, B.flatten().tolist())

            gemm = helper.make_node("Gemm", ["x", "W", "B"], ["z"], transB=0)
            relu = helper.make_node("Relu", ["z"], ["y"])

            graph = helper.make_graph([gemm, relu], "linear_relu_native_graph", [x_info], [y_info], [w_init, b_init])
            model = helper.make_model(graph, producer_name="munet_native_save_test", opset_imports=[helper.make_opsetid("", 11)])
            model.ir_version = 7
            onnx.save(model, onnx_path)

            from munet_nn import inference

            module = inference.compile_onnx(onnx_path, output_path=native_path)
            restored = munet.load_for_inference(native_path)

            x_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
            expected = np.maximum(x_np @ W + B, 0.0)

            y0 = np.array(module.forward(munet.from_numpy(x_np)).detach(), copy=False)
            y1 = np.array(restored.forward(munet.from_numpy(x_np)).detach(), copy=False)
            self.assertTrue(np.allclose(y0, expected, atol=1e-5))
            self.assertTrue(np.allclose(y1, expected, atol=1e-5))

            metadata = munet.serialization_metadata(native_path)
            self.assertEqual(metadata["artifact_kind"], "deploy_model")

    def test_yolov5n_onnx_conversion_coverage_report(self):
        try:
            import onnx  # noqa: F401
        except Exception:
            print("\nSkipping yolov5n coverage test (onnx not installed).")
            return

        import urllib.error

        with tempfile.TemporaryDirectory() as d:
            yolopath = os.path.join(d, "yolov5n.onnx")
            try:
                munet.inference.download_yolov5n_onnx(yolopath)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                print(f"\nSkipping yolov5n coverage test (download unavailable): {e}")
                return

            report = munet.inference.onnx_conversion_coverage_report(yolopath)
            self.assertGreater(report["total_nodes"], 0)
            self.assertIn("Conv", report["unique_ops"])
            self.assertEqual(report["coverage"]["unsupported"], [])
            self.assertEqual(report["coverage"]["unmapped"], [])
            self.assertTrue(report["fully_lowerable"])

    def test_compile_onnx_graph_runtime_branching_ops(self):
        try:
            import onnx
            from onnx import TensorProto, helper
        except Exception:
            print("\nSkipping ONNX graph-runtime branching test (onnx not installed).")
            return

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "branching_ops.onnx")
            x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2, 2])
            y_info = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1, 2, 2, 2])

            split = helper.make_node("Split", ["x"], ["a", "b"], axis=1, split=[1, 1])
            idn = helper.make_node("Identity", ["b"], ["b2"])
            cat = helper.make_node("Concat", ["a", "b2"], ["c"], axis=1)
            shape_const = helper.make_tensor("shape", TensorProto.INT64, [4], [1, 2, 2, 2])
            reshape = helper.make_node("Reshape", ["c", "shape"], ["r"])
            trans = helper.make_node("Transpose", ["r"], ["t"], perm=[0, 1, 2, 3])
            unsq_axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
            unsq = helper.make_node("Unsqueeze", ["t", "axes"], ["u"])
            shape = helper.make_node("Shape", ["u"], ["sh"])
            st = helper.make_tensor("st", TensorProto.INT64, [1], [0])
            en = helper.make_tensor("en", TensorProto.INT64, [1], [1])
            ax = helper.make_tensor("ax", TensorProto.INT64, [1], [0])
            sl = helper.make_node("Slice", ["sh", "st", "en", "ax"], ["slv"])
            pw = helper.make_tensor("pw", TensorProto.FLOAT, [1], [2.0])
            pwn = helper.make_node("Pow", ["slv", "pw"], ["p2"])
            fl = helper.make_node("Floor", ["p2"], ["f"])
            cast = helper.make_node("Cast", ["f"], ["f32"], to=TensorProto.FLOAT)
            add = helper.make_node("Add", ["u", "f32"], ["out"])

            graph = helper.make_graph(
                [split, idn, cat, reshape, trans, unsq, shape, sl, pwn, fl, cast, add],
                "branching_ops",
                [x_info],
                [y_info],
                [shape_const, unsq_axes, st, en, ax, pw],
            )
            model = helper.make_model(graph, producer_name="munet_graph_runtime_test", opset_imports=[helper.make_opsetid("", 11)])
            model.ir_version = 7
            onnx.save(model, path)

            module = munet.inference.compile_onnx(path)
            x = munet.from_numpy(np.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32))
            y = module.forward(x)
            y_np = np.array(y.detach(), copy=False)
            self.assertEqual(list(y_np.shape), [1, 1, 2, 2, 2])

            report = munet.inference.onnx_conversion_coverage_report(path)
            self.assertTrue(report["fully_lowerable"])
            self.assertFalse(report["native_deployable"])
            self.assertEqual(report["runtime_role"], "development_tooling")

    def test_onnx_runtime_package_boundary_api(self):
        boundary = munet.inference.onnx_runtime_package_boundary()
        self.assertIn("deploy_runtime", boundary)
        self.assertIn("development_tooling", boundary)
        self.assertIn("compile_onnx(model_path)", boundary["deploy_runtime"][0])
        self.assertIn("packaging_policy", boundary)

    def test_onnx_inference_wrapper(self):
        try:
            import onnx
            from onnx import TensorProto, helper
        except Exception:
            print("\nSkipping ONNX load_onnx deprecation test (onnx not installed).")
            return

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "add_bias.onnx")

            x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3])
            y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 3])
            b_init = helper.make_tensor(
                "b", TensorProto.FLOAT, [1, 3], np.array([[1.0, 2.0, 3.0]], dtype=np.float32).flatten().tolist()
            )
            add_node = helper.make_node("Add", ["x", "b"], ["y"])

            graph = helper.make_graph([add_node], "add_graph", [x_info], [y_info], [b_init])
            model = helper.make_model(graph, producer_name="munet_test", opset_imports=[helper.make_opsetid("", 11)])
            model.ir_version = 7
            onnx.save(model, path)

            with self.assertRaises(RuntimeError):
                munet.inference.load_onnx(path)

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
