import sys
import os
import numpy as np
from graphviz import Source

sys.path.append(os.path.join(os.path.dirname(__file__), "../../build"))
import munet


class UNet(munet.nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = munet.nn.Conv2d(1, 16, 3, padding=1)
        self.enc2 = munet.nn.Conv2d(16, 32, 3, padding=1)
        self.pool = munet.nn.MaxPool2d(2, 2)

        # Decoder
        self.up = munet.nn.Upsample(2)
        self.dec2 = munet.nn.Conv2d(32 + 16, 16, 3, padding=1)
        self.final = munet.nn.Conv2d(16, 1, 1)

    def forward(self, x):
        # Down
        e1 = self.enc1(x).relu()
        p1 = self.pool(e1)
        e2 = self.enc2(p1).relu()

        # Up with Skip Connection
        up2 = self.up(e2)
        merge2 = munet.cat([up2, e1], dim=1)  # Concatenation Skip

        d2 = self.dec2(merge2).relu()
        return self.final(d2)


# --- Data Gen ---
def generate_shapes(num_samples=100, size=32):
    # Ensure float32! C++ expects 4 bytes per float.
    images = np.zeros((num_samples, 1, size, size), dtype=np.float32)
    masks = np.zeros((num_samples, 1, size, size), dtype=np.float32)

    # Simple background class 0, shape class 1
    masks[:, 0, :, :] = 1.0

    for i in range(num_samples):
        # Create a simple box in channel 0
        # Randomize position slightly to make it a real learning task
        r, c = np.random.randint(5, size - 15, size=2)
        images[i, 0, r : r + 10, c : c + 10] = 1.0
        masks[i, 0, r : r + 10, c : c + 10] = 1.0
    return images, masks


# --- Training ---
if __name__ == "__main__":
    model = UNet()

    x_train, y_train = generate_shapes(1000)

    device = munet.Device(munet.DeviceType.CUDA, 0)

    print(f"Using {device.type.name}")

    model.to(device)

    # Lower LR slightly for stability with MSE
    optimizer = munet.optim.SGD(model.parameters(), lr=0.1)

    print("Training...")
    BATCH_SIZE = 16

    # Fresh state
    munet.reset_profiler()
    for epoch in range(10):
        epoch_loss = 0.0
        batches = 0

        for i in range(0, len(x_train), BATCH_SIZE):
            # 1. Get Numpy Batch
            bx_np = x_train[i : i + BATCH_SIZE]
            by_np = y_train[i : i + BATCH_SIZE]

            # 2. Create CPU Tensor containing data, then move to Device
            # This is the critical fix!
            bx = munet.from_numpy(bx_np).to(device)
            by = munet.from_numpy(by_np).to(device)

            optimizer.zero_grad()

            preds = model.forward(bx)

            # Simple MSE Loss on raw logits
            loss = preds.mse_loss(by)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        print(f"Epoch {epoch}: Avg Loss {epoch_loss / batches:.6f}")

    munet.print_profiler_stats()

    # TODO: this is broken
    # munet.save(model, "unet")
