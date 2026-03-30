import numpy as np

import munet_nn as munet


# --- Data Gen ---
def generate_shapes(num_samples=100, size=32):
    images = np.zeros((num_samples, 3, size, size), dtype=np.float32)
    # Target: Single channel (1 = shape, 0 = background)
    masks = np.zeros((num_samples, 1, size, size), dtype=np.float32)

    for i in range(num_samples):
        # Create a simple box
        images[i, :, 10:20, 10:20] = 1.0
        masks[i, 0, 10:20, 10:20] = 1.0
    return images, masks


def create_model():
    return munet.nn.Sequential(
        [
            # Encoder
            munet.nn.Conv2d(3, 16, 3, padding=1),
            munet.nn.BatchNorm2d(16),
            munet.nn.ReLU(),
            munet.nn.MaxPool2d(2, 2),
            munet.nn.Conv2d(16, 32, 3, padding=1),
            munet.nn.BatchNorm2d(32),
            munet.nn.ReLU(),
            munet.nn.MaxPool2d(2, 2),
            # Bottleneck
            munet.nn.Conv2d(32, 64, 3, padding=1),
            munet.nn.BatchNorm2d(64),
            munet.nn.ReLU(),
            # Decoder
            munet.nn.Upsample(2),
            munet.nn.Conv2d(64, 32, 3, padding=1),
            munet.nn.BatchNorm2d(32),
            munet.nn.ReLU(),
            munet.nn.Upsample(2),
            munet.nn.Conv2d(32, 16, 3, padding=1),
            munet.nn.BatchNorm2d(16),
            munet.nn.ReLU(),
            # Output: 1 Channel + Sigmoid
            munet.nn.Conv2d(16, 1, 1),
            munet.nn.Sigmoid(),
        ]
    )


if __name__ == "__main__":
    device = munet.Device(munet.DeviceType.CPU, 0)
    print("Using CPU")

    x_train, y_train = generate_shapes(200)

    model = create_model()
    model.to(device)

    # SGD with momentum helps convergence
    optimizer = munet.optim.SGD(model.parameters(), lr=0.1)

    print("Training...")
    BATCH_SIZE = 16

    for epoch in range(15):
        epoch_loss = 0.0
        batches = 0

        for i in range(0, len(x_train), BATCH_SIZE):
            if i + BATCH_SIZE > len(x_train):
                break

            bx_np = x_train[i : i + BATCH_SIZE]
            by_np = y_train[i : i + BATCH_SIZE]

            bx = munet.from_numpy(bx_np).to(device)
            by = munet.from_numpy(by_np).to(device)

            optimizer.zero_grad()

            # Forward
            preds = model.forward(bx)

            # Loss (MSE works well with Sigmoid output 0..1)
            loss = preds.mse_loss(by)

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            batches += 1

        print(f"Epoch {epoch}: Avg Loss {epoch_loss / batches:.6f}")
