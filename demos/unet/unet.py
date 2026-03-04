import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import munet from build directory
sys.path.append(os.path.join(os.path.dirname(__file__), "../../build"))
import munet


# --- Core Framework Wrapper ---
class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        params = []
        for attr in dir(self):
            if attr.startswith("__"):
                continue
            try:
                val = getattr(self, attr)
            except:
                continue

            if isinstance(val, munet.Tensor) and val.requires_grad:
                params.append(val)
            elif isinstance(val, Module):
                params.extend(val.parameters())
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def train(self):
        self.training = True
        for attr in dir(self):
            try:
                val = getattr(self, attr)
            except:
                continue
            if isinstance(val, Module):
                val.train()

    def eval(self):
        self.training = False
        for attr in dir(self):
            try:
                val = getattr(self, attr)
            except:
                continue
            if isinstance(val, Module):
                val.eval()

    def to(self, device):
        for attr in dir(self):
            if attr.startswith("__"):
                continue
            try:
                val = getattr(self, attr)
            except:
                continue

            if isinstance(val, munet.Tensor):
                new_t = val.to(device)
                if val.requires_grad:
                    new_t.requires_grad = True
                    new_t.name = val.name
                setattr(self, attr, new_t)
            elif isinstance(val, Module):
                val.to(device)
        return self

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def train(self):
        super().train()
        for l in self.layers:
            l.train()

    def eval(self):
        super().eval()
        for l in self.layers:
            l.eval()

    def to(self, device):
        for l in self.layers:
            l.to(device)
        return self


class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.step(self.lr)


class MeanMSELoss(Module):
    def forward(self, pred, target):
        diff = pred - target
        s = (diff * diff).sum()

        # Normalize sum to mean to prevent gradient explosion
        # Calculate N (num elements)
        N = 1.0
        for dim in pred.shape:
            N *= dim

        # Create scalar tensor for division on the correct device
        scale_cpu = munet.Tensor([1])
        np.array(scale_cpu, copy=False)[0] = 1.0 / N
        scale = scale_cpu.to(pred.device)

        return s * scale


# --- Layers ---
class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.weight = munet.Tensor(
            [out_channels, in_channels, kernel_size, kernel_size], requires_grad=True
        )
        self.bias = munet.Tensor([out_channels], requires_grad=True)
        self.stride = stride
        self.padding = padding
        # Kaiming Init
        limit = np.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels))
        # Initializing on CPU via numpy for safety before moving
        np_w = np.array(self.weight, copy=False)
        np_w[:] = np.random.uniform(-limit, limit, np_w.shape)
        np_b = np.array(self.bias, copy=False)
        np_b[:] = np.random.uniform(-limit, limit, np_b.shape)

    def forward(self, x):
        return x.conv2d(self.weight, self.bias, self.stride, self.padding)


class MaxPool2D(Module):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return x.max_pool2d(self.kernel_size, self.stride, 0)


class Upsample2D(Module):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def forward(self, x):
        return x.upsample2d(self.scale_factor)


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.weight = munet.Tensor([num_features], requires_grad=True)
        self.bias = munet.Tensor([num_features], requires_grad=True)

        # Safe Init via Numpy
        np.array(self.weight, copy=False)[:] = 1.0
        np.array(self.bias, copy=False)[:] = 0.0

        self.running_mean = munet.Tensor([num_features], requires_grad=False)
        self.running_var = munet.Tensor([num_features], requires_grad=False)
        np.array(self.running_mean, copy=False)[:] = 0.0
        np.array(self.running_var, copy=False)[:] = 1.0

    def forward(self, x):
        return x.batch_norm(
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )


# --- Data Gen ---
def generate_shapes(num_samples=100, size=32):
    images = np.zeros((num_samples, 3, size, size), dtype=np.float32)
    masks = np.zeros((num_samples, 2, size, size), dtype=np.float32)
    masks[:, 0, :, :] = 1.0
    y_grid, x_grid = np.indices((size, size))
    for i in range(num_samples):
        num_objs = np.random.randint(0, 4)
        for _ in range(num_objs):
            cx, cy = np.random.randint(5, size - 5, size=2)
            r = np.random.randint(4, 9)
            colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            color = colors[np.random.randint(0, len(colors))]
            shape_type = np.random.randint(0, 3)
            mask = np.zeros((size, size), dtype=bool)
            if shape_type == 0:
                top_y = cy - r
                height = 2 * r
                dy = y_grid - top_y
                dx = np.abs(x_grid - cx)
                mask = (dy >= 0) & (dy <= height) & (dx <= (dy * 0.6))
            elif shape_type == 1:
                dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
                mask = dist_sq < r**2
            else:
                mask = (
                    (x_grid >= cx - r)
                    & (x_grid <= cx + r)
                    & (y_grid >= cy - r)
                    & (y_grid <= cy + r)
                )
            if mask.any():
                for c in range(3):
                    images[i, c, mask] = color[c]
                masks[i, 1, mask] = 1.0
                masks[i, 0, mask] = 0.0
    return images, masks


def create_segmentation_model():
    return Sequential(
        Conv2D(3, 16, 3, padding=1),
        BatchNorm2d(16),
        ReLU(),
        MaxPool2D(2, 2),
        Conv2D(16, 32, 3, padding=1),
        BatchNorm2d(32),
        ReLU(),
        MaxPool2D(2, 2),
        Conv2D(32, 64, 3, padding=1),
        BatchNorm2d(64),
        ReLU(),
        Conv2D(64, 64, 3, padding=1),
        BatchNorm2d(64),
        ReLU(),
        Upsample2D(2),
        Conv2D(64, 32, 3, padding=1),
        BatchNorm2d(32),
        ReLU(),
        Upsample2D(2),
        Conv2D(32, 16, 3, padding=1),
        BatchNorm2d(16),
        ReLU(),
        Conv2D(16, 2, 1),
    )


def visualize_results(model, x_test, y_test, epoch, device, output_dir="vis_results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.eval()

    # Input
    tx = munet.Tensor(x_test.shape).to(device)
    # Write to CPU temp then move
    tmp_cpu = munet.Tensor(x_test.shape)
    np.array(tmp_cpu, copy=False)[:] = x_test
    tx = tmp_cpu.to(device)

    logits = model(tx)
    preds = np.array(logits.to(munet.Device(munet.DeviceType.CPU, 0)), copy=False)

    pred_mask = np.argmax(preds, axis=1)
    true_mask = np.argmax(y_test, axis=1)

    fig, axes = plt.subplots(len(x_test), 3, figsize=(6, 2 * len(x_test)))
    for i in range(len(x_test)):
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
            axes[i, 2].set_title(f"Pred Ep{epoch}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/epoch_{epoch:03d}.png")
    plt.close()
    model.train()


if __name__ == "__main__":
    try:
        device = munet.Device(munet.DeviceType.CPU, 0)
        print("Using Device: CUDA")
    except:
        device = munet.Device(munet.DeviceType.CPU, 0)
        print("Using Device: CPU")

    print("Generating Dataset...")
    x_train, y_train = generate_shapes(500)
    x_test, y_test = generate_shapes(5)

    model = create_segmentation_model()
    model.to(device)

    # Increased LR slightly since we are now using MeanMSELoss
    optimizer = SGD(model.parameters(), lr=1.5)
    criterion = MeanMSELoss()

    BATCH_SIZE = 16
    print("Starting training...")
    for epoch in range(20):
        epoch_loss = 0
        batches = 0

        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]

        for i in range(0, len(x_train), BATCH_SIZE):
            batch_x_np = x_train[i : i + BATCH_SIZE]
            batch_y_np = y_train[i : i + BATCH_SIZE]

            bx_cpu = munet.Tensor(batch_x_np.shape)
            by_cpu = munet.Tensor(batch_y_np.shape)
            np.array(bx_cpu, copy=False)[:] = batch_x_np
            np.array(by_cpu, copy=False)[:] = batch_y_np

            bx = bx_cpu.to(device)
            by = by_cpu.to(device)

            model.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            scalar_loss = np.array(
                loss.to(munet.Device(munet.DeviceType.CPU, 0)), copy=False
            )[0]
            if np.isnan(scalar_loss) or np.isinf(scalar_loss):
                print("LOSS EXPLODED!")
                sys.exit(1)

            epoch_loss += scalar_loss
            batches += 1

        avg_loss = epoch_loss / batches
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.6f}")
        visualize_results(model, x_test, y_test, epoch + 1, device)
