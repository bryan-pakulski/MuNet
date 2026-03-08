import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../build"))
import munet
import numpy as np
import time


# --- 1. Define Architecture ---
# Hybrid: Conv -> BN -> Pool -> Upsample -> Cat(Skip) -> Flatten -> Linear
class WeirdModelMuNet(munet.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = munet.nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn = munet.nn.BatchNorm2d(4)
        self.pool = munet.nn.MaxPool2d(2, 2)
        self.up = munet.nn.Upsample(2)
        self.flatten = munet.nn.Flatten()
        self.fc = munet.nn.Linear(4 * 32 * 32 + 1 * 32 * 32, 1)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = out.relu()
        out = self.pool(out)
        out = self.up(out)
        # Weird skip connection: concat original input with upsampled feature map
        out = munet.cat([out, identity], dim=1)
        out = self.flatten(out)
        return self.fc(out)


class WeirdModelTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.fc = nn.Linear(4 * 32 * 32 + 1 * 32 * 32, 1)

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn(self.conv(x)))
        out = self.up(self.pool(out))
        out = torch.cat([out, identity], dim=1)
        out = torch.flatten(out, 1)
        return self.fc(out)


# --- 2. Setup ---
device_type = munet.DeviceType.CUDA  # Change to CUDA if available
m_dev = munet.Device(device_type, 0)
t_dev = "cuda"

m_model = WeirdModelMuNet()
t_model = WeirdModelTorch()

# Synchronize weights
with torch.no_grad():
    # Linear: MuNet [In, Out], Torch [Out, In]
    m_model.fc.weight.copy_from_numpy(t_model.fc.weight.detach().numpy().T)
    m_model.fc.bias.copy_from_numpy(t_model.fc.bias.detach().numpy())
    # Conv: Both [Out, In, K, K]
    m_model.conv.weight.copy_from_numpy(t_model.conv.weight.detach().numpy())
    m_model.conv.bias.copy_from_numpy(t_model.conv.bias.detach().numpy())

m_model.to(m_dev)
t_model.to(t_dev)

m_opt = munet.optim.SGD(m_model.parameters(), lr=0.0001)
t_opt = torch.optim.SGD(t_model.parameters(), lr=0.0001)

# --- 3. Training Loop ---
iters = 50
batch_size = 4
x_np = np.random.randn(batch_size, 1, 32, 32).astype(np.float32)
y_np = np.random.randn(batch_size, 1).astype(np.float32)

# Keep device tensors persistent to benchmark kernel/optimizer speed rather than
# host->device transfer overhead each iteration.
m_tx = munet.from_numpy(x_np).to(m_dev)
m_ty = munet.from_numpy(y_np).to(m_dev)
t_tx = torch.from_numpy(x_np).to(t_dev)
t_ty = torch.from_numpy(y_np).to(t_dev)

print(f"{'Iter':<6} | {'MuNet Loss':<12} | {'Torch Loss':<12} | {'Diff':<12}")
print("-" * 50)


def benchmark(name, fn):
    start = time.perf_counter()
    res = fn()
    return res, (time.perf_counter() - start) * 1000


m_times, t_times = [], []

for i in range(iters):
    # MuNet Step
    def munet_step():
        m_opt.zero_grad()
        pred = m_model.forward(m_tx)
        loss = pred.mse_loss(m_ty)
        loss.backward()
        m_opt.step()
        return loss.item()

    # Torch Step
    def torch_step():
        t_opt.zero_grad()
        pred = t_model(t_tx)
        loss = nn.functional.mse_loss(pred, t_ty)
        loss.backward()
        t_opt.step()
        return loss.item()

    m_loss, m_t = benchmark("MuNet", munet_step)
    t_loss, t_t = benchmark("Torch", torch_step)

    if i > 5:  # Warmup
        m_times.append(m_t)
        t_times.append(t_t)

    if i % 10 == 0:
        print(
            f"{i:<6} | {m_loss:<12.6f} | {t_loss:<12.6f} | {abs(m_loss-t_loss):<12.6e}"
        )

print("-" * 50)
print(f"Avg Step Time (MuNet): {np.mean(m_times):.3f} ms")
print(f"Avg Step Time (Torch): {np.mean(t_times):.3f} ms")
