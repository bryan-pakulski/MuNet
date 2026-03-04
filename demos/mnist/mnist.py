import sys
import os
import gzip
import urllib.request
import numpy as np

# Ensure we can import munet from build directory
sys.path.append(os.path.join(os.path.dirname(__file__), "../../build"))
import munet
import time

# --- Mini Framework ---


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        params = []
        # Filter attributes to find Tensors and Sub-Modules
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
            elif isinstance(val, (list, tuple)):  # FIX: Support tuples (Sequential)
                for item in val:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weight = munet.Tensor([in_features, out_features], requires_grad=True)
        self.bias = munet.Tensor([1, out_features], requires_grad=True)

        # Xavier Init
        limit = np.sqrt(6 / (in_features + out_features))
        self.weight.uniform_(-limit, limit)
        self.bias.uniform_(-limit, limit)

    def forward(self, x):
        out = x @ self.weight
        # Bias broadcast hack: (B, 1) @ (1, Out) -> (B, Out)
        # We need to ensure 'ones' is on the same device as x
        if x.shape[0] > 1:
            ones = munet.Tensor([x.shape[0], 1], device=x.device, requires_grad=False)
            ones.uniform_(1.0, 1.0)
            return out + (ones @ self.bias)
        return out + self.bias


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class MSELoss(Module):
    def forward(self, pred, target):
        diff = pred - target
        return (diff * diff).sum()


class SGD:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.step(self.lr)


# --- Data Loading ---


def load_mnist():
    # Google CVDF Mirror
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    if not os.path.exists("data"):
        os.mkdir("data")

    for f in files:
        path = f"data/{f}"
        if not os.path.exists(path):
            print(f"Downloading {f}...")
            try:
                urllib.request.urlretrieve(base_url + f, path)
            except Exception as e:
                print(f"Download failed: {e}")
                sys.exit(1)

    def parse_images(path):
        with gzip.open(path, "rb") as f:
            f.read(16)
            buf = f.read()
            return (
                np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(-1, 784)
                / 255.0
            )

    def parse_labels(path):
        with gzip.open(path, "rb") as f:
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)

    return (
        parse_images("data/train-images-idx3-ubyte.gz"),
        parse_labels("data/train-labels-idx1-ubyte.gz"),
        parse_images("data/t10k-images-idx3-ubyte.gz"),
        parse_labels("data/t10k-labels-idx1-ubyte.gz"),
    )


def one_hot(labels, num_classes=10):
    res = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    res[np.arange(labels.shape[0]), labels] = 1.0
    return res


# --- Training Loop ---


def train():
    print("Loading Data...")
    X_train, y_train, X_test, y_test = load_mnist()

    # Check for CUDA
    try:
        device = munet.Device(munet.DeviceType.VULKAN, 0)
        # Allocate small tensor to test if VULKAN works
        munet.Tensor([1], device=device)
        print("Using VULKAN GPU")
    except:
        print("CUDA not available, falling back to CPU")
        device = munet.Device(munet.DeviceType.CPU, 0)

    BATCH_SIZE = 128
    LR = 0.0003
    EPOCHS = 50

    model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10))
    optimizer = SGD(model.parameters(), lr=LR)
    criterion = MSELoss()

    # Move params to device
    print(f"Moving {len(model.parameters())} tensors to {device}...")
    for p in model.parameters():
        p_dev = p.to(device)
        p.replace_(p_dev)

    start = time.time()

    num_batches = X_train.shape[0] // BATCH_SIZE
    print(f"Starting training on {num_batches} batches per epoch...")

    for epoch in range(EPOCHS):
        # Shuffle
        perm = np.random.permutation(X_train.shape[0])
        X_train = X_train[perm]
        y_train = y_train[perm]

        loss = None
        logits = None
        s = 0
        e = 0
        i = 0

        for i in range(num_batches):
            s = i * BATCH_SIZE
            e = s + BATCH_SIZE

            # Prepare batch on CPU
            x_cpu = munet.Tensor([BATCH_SIZE, 784])
            y_cpu = munet.Tensor([BATCH_SIZE, 10])
            np.array(x_cpu, copy=False)[:] = X_train[s:e]
            np.array(y_cpu, copy=False)[:] = one_hot(y_train[s:e])

            # Move to Device
            x_batch = x_cpu.to(device)
            y_batch = y_cpu.to(device)

            # Step
            model.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        # Metrics (CPU sync)
        if epoch % 5 == 0:
            loss_val = np.array(
                loss.to(munet.Device(munet.DeviceType.CPU, 0)), copy=False
            )[0]
            # Accuracy check
            logits_np = np.array(
                logits.to(munet.Device(munet.DeviceType.CPU, 0)), copy=False
            )
            preds = np.argmax(logits_np, axis=1)
            acc = (preds == y_train[s:e]).sum() / BATCH_SIZE
            print(
                f"Epoch {epoch+1} [{i}/{num_batches}] Loss: {loss_val:.4f} Acc: {acc*100:.1f}%"
            )

    end = time.time()
    print(f"Training took {end-start} seconds")


if __name__ == "__main__":
    train()
