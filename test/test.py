import munet
import numpy as np
import time

def test_gpu_stability():
    print("Initializing Model...")
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
    target_data[:, 0] = 1.0 # Dummy targets
    
    # Reusable Tensors
    x_gpu = munet.Tensor([BATCH_SIZE, 128], munet.Device.CUDA)
    y_gpu = munet.Tensor([BATCH_SIZE, 10], munet.Device.CUDA)
    
    print("Starting Training Loop (200 batches)...")
    start = time.time()
    
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
            print(f"FAILED: Loss is NaN at batch {i}")
            exit(1)
            
        # 4. Backward
        model.backward(grad)
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Batch {i}: {loss:.4f}")

    print(f"Success! Finished in {time.time() - start:.2f}s")

if __name__ == "__main__":
    test_gpu_stability()
