import os

import munet_nn as munet
import numpy as np
import matplotlib.pyplot as plt
import argparse

argparser = argparse.ArgumentParser(description="Train a simple Unet")
argparser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to use",
    choices=["cpu", "cuda", "vulkan"],
)
args = argparser.parse_args()


# Helper to find the best available device
def get_device():
    if args.device == "cpu":
        return munet.Device(munet.DeviceType.CPU, 0)
    elif args.device == "cuda":
        return munet.Device(munet.DeviceType.CUDA, 0)
    elif args.device == "vulkan":
        return munet.Device(munet.DeviceType.VULKAN, 0)


def generate_shapes(num_samples=100, size=32):
    images = np.zeros((num_samples, 3, size, size), dtype=np.float32)
    masks = np.zeros((num_samples, 2, size, size), dtype=np.float32)
    masks[:, 0, :, :] = 1.0  # Background
    y_grid, x_grid = np.indices((size, size))

    for i in range(num_samples):
        num_objs = np.random.randint(0, 4)
        for _ in range(num_objs):
            cx, cy = np.random.randint(5, size - 5, size=2)
            r = np.random.randint(4, 9)
            colors = [
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 1.0),
                (1.0, 0.0, 1.0),
            ]
            color = colors[np.random.randint(0, len(colors))]
            shape_type = np.random.randint(0, 3)
            mask = np.zeros((size, size), dtype=bool)

            if shape_type == 0:  # Triangle
                top_y, height = cy - r, 2 * r
                dy, dx = y_grid - top_y, np.abs(x_grid - cx)
                mask = (dy >= 0) & (dy <= height) & (dx <= (dy * 0.6))
            elif shape_type == 1:  # Circle
                mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) < r**2
            else:  # Square
                mask = (
                    (x_grid >= cx - r)
                    & (x_grid <= cx + r)
                    & (y_grid >= cy - r)
                    & (y_grid <= cy + r)
                )

            if mask.any():
                for c in range(3):
                    images[i, c, mask] = color[c]
                if shape_type == 0:
                    masks[i, 1, mask], masks[i, 0, mask] = 1.0, 0.0
                else:
                    masks[i, 1, mask], masks[i, 0, mask] = 0.0, 1.0
    return images, masks


def create_segmentation_model():
    model = munet.nn.Sequential()
    # Encoder 1
    model.add(munet.nn.Conv2d(3, 16, 3, padding=1))
    model.add(munet.nn.BatchNorm2d(16))
    model.add(munet.nn.ReLU())
    model.add(munet.nn.MaxPool2d(2, 2))
    # Encoder 2
    model.add(munet.nn.Conv2d(16, 32, 3, padding=1))
    model.add(munet.nn.BatchNorm2d(32))
    model.add(munet.nn.ReLU())
    model.add(munet.nn.MaxPool2d(2, 2))
    # Bottleneck
    model.add(munet.nn.Conv2d(32, 64, 3, padding=1))
    model.add(munet.nn.BatchNorm2d(64))
    model.add(munet.nn.ReLU())
    # Decoder 1
    model.add(munet.nn.Upsample(2))
    model.add(munet.nn.Conv2d(64, 32, 3, padding=1))
    model.add(munet.nn.BatchNorm2d(32))
    model.add(munet.nn.ReLU())
    # Decoder 2
    model.add(munet.nn.Upsample(2))
    model.add(munet.nn.Conv2d(32, 16, 3, padding=1))
    model.add(munet.nn.BatchNorm2d(16))
    model.add(munet.nn.ReLU())
    # Head
    model.add(munet.nn.Conv2d(16, 2, 1))
    return model


def visualize_results(model, x_test, y_test, epoch, device, output_dir="vis_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.eval()
    with munet.no_grad():
        tx = munet.from_numpy(x_test).to(device)
        logits = model(tx)
        preds = np.array(
            logits.to(munet.Device(munet.DeviceType.CPU, 0)).detach(), copy=False
        )

    pred_mask = np.argmax(preds, axis=1)
    true_mask = np.argmax(y_test, axis=1)
    num_samples = len(x_test)
    fig, axes = plt.subplots(num_samples, 3, figsize=(6, 2 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_samples):
        img_rgb = np.transpose(x_test[i], (1, 2, 0))
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(true_mask[i], vmin=0, vmax=1)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(pred_mask[i], vmin=0, vmax=1)
        axes[i, 2].axis("off")
        if i == 0:
            axes[i, 0].set_title("Input")
            axes[i, 1].set_title("Truth")
            axes[i, 2].set_title(f"Pred (Ep {epoch})")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/epoch_{epoch:03d}.png")
    plt.close()
    model.train()


# --- Main Execution ---
device = get_device()
print(f"Using device: {device}")

x_train, y_train = generate_shapes(2000)
model = create_segmentation_model()
model.to(device)

optimizer = munet.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    indices = np.random.permutation(len(x_train))
    x_train, y_train = x_train[indices], y_train[indices]
    epoch_loss, batches = 0, 0
    model.train()

    for i in range(0, len(x_train), 64):
        # Optimization: use no_grad context for data transfer/setup
        with munet.no_grad():
            batch_x = munet.from_numpy(x_train[i : i + 64]).to(device)
            batch_y = munet.from_numpy(y_train[i : i + 64]).to(device)

        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = predictions.cross_entropy(batch_y)

        loss_val = loss.item()
        if np.isnan(loss_val):
            print("NaN detected in loss!")
            break

        loss.backward()
        optimizer.step()
        epoch_loss += loss_val
        batches += 1

    print(f"Epoch {epoch+1} | Avg Loss: {epoch_loss/batches:.4f}")
    if (epoch + 1) % 5 == 0:
        x_test, y_test = generate_shapes(5)
        visualize_results(model, x_test, y_test, epoch + 1, device)

munet.save_checkpoint(model, "unet_complex.npz")
print("Model saved.")
