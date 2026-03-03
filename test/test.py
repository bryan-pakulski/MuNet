import unittest
import munet
import numpy as np

class TestModel(unittest.TestCase):

    def test_gpu_stability(self):
        model = munet.Model()
        model.add(munet.Linear(128, 64))
        model.add(munet.ReLU())
        model.add(munet.Linear(64, 10))

        # Move to GPU
        for p in model.parameters():
            p.to_gpu()

        optimizer = munet.SGD(model.parameters(), lr=0.01)

        # Synthetic Data
        BATCH_SIZE = 64
        input_data = np.random.randn(BATCH_SIZE, 128).astype(np.float32)
        target_data = np.zeros((BATCH_SIZE, 10), dtype=np.float32)
        target_data[:, 0] = 1.0  # Dummy targets

        # Reusable Tensors
        x_gpu = munet.Tensor([BATCH_SIZE, 128], munet.Device.CUDA)
        y_gpu = munet.Tensor([BATCH_SIZE, 10], munet.Device.CUDA)

        print("Starting Training Loop (200 batches)...")

        for i in range(200):
            # 1. Copy Data
            x_gpu.copy_from_numpy(input_data)
            y_gpu.copy_from_numpy(target_data)

            # 2. Forward
            optimizer.zero_grad()
            logits = model.forward(x_gpu)

            # 3. Loss
            loss, grad = munet.cross_entropy_loss(logits, y_gpu)

            # Check for NaN
            if np.isnan(loss):
                self.fail(f"FAILED: Loss is NaN at batch {i}")

            # 4. Backward
            model.backward(grad)
            optimizer.step()

            if i % 50 == 0:
                print(f"Batch {i}: {loss:.4f}")


class TestBatchNorm2D(unittest.TestCase):

    def test_batchnorm2d_forward_eval_mode(self):
        # Create BatchNorm2D with 2 features
        bn = munet.BatchNorm2D(2, eps=1e-5, momentum=0.1)

        # Test input: [N=1, C=2, H=2, W=2]
        x_np = np.array(
            [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32
        )
        x = munet.Tensor.from_numpy(x_np)

        # In eval mode, output should just use running_mean (0) and running_var (1)
        bn.eval()
        out = bn.forward(x)
        out_np = out.numpy()

        # Expected: x * (1 / sqrt(1 + 1e-5))
        scale = 1.0 / np.sqrt(1.0 + 1e-5)
        expected = x_np * scale

        np.testing.assert_allclose(out_np, expected, rtol=1e-5, atol=1e-5)

    def test_batchnorm2d_forward_train_mode(self):
        bn = munet.BatchNorm2D(2, eps=1e-5, momentum=0.1)

        x_np = np.array(
            [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32
        )
        x = munet.Tensor.from_numpy(x_np)

        # In train mode, it calculates mean and var dynamically
        bn.train()
        out = bn.forward(x)
        out_np = out.numpy()

        # Channel 0: [1,2,3,4], Mean=2.5, Var=1.25
        mean0 = 2.5
        var0 = 1.25
        inv_std0 = 1.0 / np.sqrt(var0 + 1e-5)
        exp_c0 = (np.array([[1.0, 2.0], [3.0, 4.0]]) - mean0) * inv_std0

        # Channel 1: [5,6,7,8], Mean=6.5, Var=1.25
        mean1 = 6.5
        var1 = 1.25
        inv_std1 = 1.0 / np.sqrt(var1 + 1e-5)
        exp_c1 = (np.array([[5.0, 6.0], [7.0, 8.0]]) - mean1) * inv_std1

        expected = np.array([[exp_c0, exp_c1]], dtype=np.float32)

        np.testing.assert_allclose(out_np, expected, rtol=1e-5, atol=1e-5)

    def test_batchnorm2d_running_stats_update(self):
        bn = munet.BatchNorm2D(1, eps=1e-5, momentum=0.1)
        bn.train()

        # N=1, C=1, H=2, W=2
        x_np = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        x = munet.Tensor.from_numpy(x_np)

        bn.forward(x)

        # Fetch internal parameters (Need to add a python-binding to fetch running stats or just get them from parameters())
        params = bn.parameters()

        rm_tensor = None
        rv_tensor = None
        for name, tensor in params.items():
            if name == "running_mean":
                rm_tensor = tensor
            elif name == "running_var":
                rv_tensor = tensor

        rm_np = rm_tensor.numpy()
        rv_np = rv_tensor.numpy()

        # Initial RM=0, RV=1. Mean=2.5, Var=1.25.
        # Unbiased var for running_var (M=4): 1.25 * (4/3) = 1.66667
        # RM = 0.9 * 0 + 0.1 * 2.5 = 0.25
        # RV = 0.9 * 1.0 + 0.1 * 1.66667 = 1.066667

        np.testing.assert_allclose(rm_np, np.array([0.25]), rtol=1e-4)
        np.testing.assert_allclose(rv_np, np.array([1.066667]), rtol=1e-4)

    def test_batchnorm2d_backward(self):
        bn = munet.BatchNorm2D(2)
        x_np = np.array(
            [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32
        )
        x = munet.Tensor.from_numpy(x_np)

        # Forward pass to populate cache
        bn.train()
        bn.forward(x)

        # Simulate an incoming gradient (all 1s)
        grad_np = np.ones((1, 2, 2, 2), dtype=np.float32)
        grad = munet.Tensor.from_numpy(grad_np)

        grad_in = bn.backward(grad)

        # Shapes should match
        self.assertEqual(grad_in.shape, [1, 2, 2, 2])

        params = bn.parameters()
        # Since grad output is entirely 1s, and input xhat sums to 0,
        # bias gradient should just be sum(go) = 4 per channel.
        # weight gradient should be sum(go * xhat) = 0 per channel.
        gw = params["weight"].grad().numpy()
        gb = params["bias"].grad().numpy()

        np.testing.assert_allclose(gb, np.array([4.0, 4.0]), rtol=1e-5)
        np.testing.assert_allclose(gw, np.array([0.0, 0.0]), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
