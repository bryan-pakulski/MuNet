import sys
import os
import unittest
import numpy as np

# Dynamically add the 'build' directory to sys.path so Python can find 'munet.so'
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
sys.path.insert(0, build_dir)

try:
    import munet
except ImportError as e:
    print(
        f"\n[ERROR] Failed to import munet.\nMake sure you ran 'make' and munet*.so exists in: {build_dir}\n"
    )
    raise e


class TestLoss(unittest.TestCase):
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
        loss_val = np.array(loss, copy=False)[0]                                                                                                                                             
                                                                                                                                                                                      
        # MSE Forward Check                                                                                                                                                                  
        self.assertTrue(np.isclose(loss_val, 0.5))                                                                                                                                           
                                                                                                                                                                                      
        loss.backward()                                                                                                                                                                      
        grad = np.array(pred.grad, copy=False)                                                                                                                                               
                                                                                                                                                                                      
        # MSE Backward Check                                                                                                                                                                 
        expected_grad = np.array([0.0, 0.5, 0.0, 0.5], dtype=np.float32)                                                                                                                     
        self.assertTrue(np.allclose(grad, expected_grad, atol=1e-6))                                                                                                                        
                                                                                                                                                                                          
    def test_cross_entropy_loss(self):                                                                                                                                                           
        logits_np = np.array([                                                                                                                                                               
         [2.0, 1.0, 0.1],                                                                                                                                                                 
         [0.1, 1.0, 2.0]                                                                                                                                                                  
        ], dtype=np.float32)                                                                                                                                                                 
                                                                                                                                                                                          
        targets_np = np.array([                                                                                                                                                              
         [1.0, 0.0, 0.0],                                                                                                                                                                 
         [0.0, 0.0, 1.0]                                                                                                                                                                  
        ], dtype=np.float32)                                                                                                                                                                 
                                                                                                                                                                                          
        logits = munet.Tensor(list(logits_np.shape))                                                                                                                                         
        targets = munet.Tensor(list(targets_np.shape))                                                                                                                                       
                                                                                                                                                                                          
        np.array(logits, copy=False)[:] = logits_np.flatten()                                                                                                                                
        np.array(targets, copy=False)[:] = targets_np.flatten()                                                                                                                              
                                                                                                                                                                                          
        logits.requires_grad = True                                                                                                                                                          
                                                                                                                                                                                          
        loss = logits.cross_entropy(targets)                                                                                                                                                 
        loss_val = np.array(loss, copy=False)[0]                                                                                                                                             
                                                                                                                                                                                          
        # CE Forward Check (Matches PyTorch F.cross_entropy)                                                                                                                                 
        self.assertTrue(np.isclose(loss_val, 0.417022, atol=1e-4))                                                                                                                                     
                                                                                                                                                                                          
        loss.backward()                                                                                                                                                                      
        grad = np.array(logits.grad, copy=False)                                                                                                                                             
                                                                                                                                                                                          
        # CE Backward Check                                                                                                                                                                  
        expected_grad = np.array([                                                                                                                                                           
         [-0.17049,  0.12101,  0.04948],                                                                                                                                                  
         [ 0.04948,  0.12101, -0.17049]                                                                                                                                                   
        ], dtype=np.float32).flatten()                                                                                                                                                       
                                                                                                                                                                                          
        self.assertTrue(np.allclose(grad, expected_grad, atol=1e-4))        

class TestMuNetPythonBindings(unittest.TestCase):

    def test_tensor_creation(self):
        """Test basic tensor creation and properties mapping."""
        t = munet.Tensor([2, 3], requires_grad=True)
        self.assertEqual(t.shape, [2, 3])
        self.assertTrue(t.requires_grad)

        # Check Enums mapped properly
        self.assertEqual(t.device.type, munet.DeviceType.CPU)
        self.assertEqual(t.dtype, munet.DataType.Float32)

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
        # 1. Create leaves
        a = munet.Tensor([1], requires_grad=True)
        b = munet.Tensor([1], requires_grad=True)

        # Set values natively using NumPy
        np.array(a, copy=False)[:] = 10.0
        np.array(b, copy=False)[:] = 20.0

        # 2. Forward pass (C++ operator overloaded)
        c = a + b

        # 3. Verify forward result
        c_arr = np.array(c, copy=False)
        self.assertEqual(c_arr[0], 30.0)

        # 4. Backward pass
        c.backward()

        # 5. Verify gradients (dz/da = 1, dz/db = 1)
        a_grad = np.array(a.grad, copy=False)
        b_grad = np.array(b.grad, copy=False)

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
        a_cpu = munet.Tensor([2], requires_grad=True)
        b_cpu = munet.Tensor([2], requires_grad=True)

        np.array(a_cpu, copy=False)[:] = [1.0, 2.0]
        np.array(b_cpu, copy=False)[:] = [3.0, 4.0]

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

        result = np.array(c_cpu, copy=False)
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
        result = np.array(a_back, copy=False)
        self.assertEqual(result[0], 9.0)
        self.assertEqual(result[1], 10.0)

    def test_neural_network_forward_backward(self):                            
        """Test a mini neural network forward and backward pass!"""            
                                                                               
        # We will create a mini computational graph:                           
        # y = relu(X @ W1) @ W2                                                
        # Then calculate gradients for W1 and W2.                              
                                                                               
        X = munet.Tensor([1, 3], requires_grad=False)                          
        W1 = munet.Tensor([3, 4], requires_grad=True)                          
        W2 = munet.Tensor([4, 1], requires_grad=True)                          
                                                                               
        np.array(X, copy=False)[:] = [[1.0, 2.0, -1.0]]                        
                                                                               
        # W1                                                                   
        np.array(W1, copy=False)[:] = [                                        
            [ 1.0,  0.5, -1.0,  2.0],                                          
            [-2.0,  1.0,  0.5, -0.5],                                          
            [ 0.0, -1.0,  1.0,  1.0]                                           
        ]                                                                      
                                                                               
        # W2                                                                   
        np.array(W2, copy=False)[:] = [[1.0], [-1.0], [2.0], [0.5]]            
                                                                               
        # --- Forward Pass ---                                                 
        hidden = X @ W1                                                        
        activation = hidden.relu()                                             
        output = activation @ W2                                               
                                                                               
        # Verify Forward Pass Math                                             
        # X @ W1 = [[-3.0, 3.5, -1.0, 0.0]]                                    
        # relu(X @ W1) = [[0.0, 3.5, 0.0, 0.0]]                                
        # relu @ W2 = 0*1 + 3.5*-1 + 0*2 + 0*0.5 = -3.5                        
                                                                               
        self.assertEqual(np.array(output, copy=False)[0][0], -3.5)             
                                                                               
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
        self.assertEqual(dw1_result[0][1], -1.0) # 1.0 * -1.0                  
        self.assertEqual(dw1_result[1][1], -2.0) # 2.0 * -1.0                  
        self.assertEqual(dw1_result[2][1],  1.0) # -1.0 * -1.0                 

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
        
        lr = 0.01
        
        for epoch in range(50):
            w1.zero_grad()
            w2.zero_grad()
            
            # Forward
            h = (x @ w1).relu()
            pred = h @ w2
            
            # MSE Loss
            diff = pred - y
            loss = (diff * diff).sum()
            
            # Backward
            loss.backward()
            
            # Optimize
            w1.step(lr)
            w2.step(lr)
            
            if epoch == 0:
                loss_start = np.array(loss.to(munet.Device(munet.DeviceType.CPU, 0)), copy=False)[0]
            if epoch == 49:
                loss_end = np.array(loss.to(munet.Device(munet.DeviceType.CPU, 0)), copy=False)[0]
                
        # The network should have learned, meaning loss went down significantly!
        print(f"\nTraining Loop -> Start Loss: {loss_start:.4f} | End Loss: {loss_end:.4f}")
        self.assertLess(loss_end, loss_start)

    def test_multi_device_model_parallelism(self):
        """Test seamless autograd across CPU and GPU boundaries."""
        # Find an available GPU
        gpu_dev = None
        for dev_type in (munet.DeviceType.CUDA, munet.DeviceType.VULKAN):
            try:
                gpu_dev = munet.Device(dev_type, 0)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
