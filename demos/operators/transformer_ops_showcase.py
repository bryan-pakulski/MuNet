import numpy as np

import munet


def main():
    dev = munet.Device(munet.DeviceType.CPU, 0)
    B, T, E, H = 2, 4, 8, 2

    x = munet.rand([B, T, E], dev, requires_grad=False)
    mha = munet.nn.MultiHeadAttention(E, H, causal=True)
    ln = munet.nn.LayerNorm(E)
    drop = munet.nn.Dropout(0.1)
    gelu = munet.nn.GELU()

    mha.eval()
    ln.eval()
    drop.eval()

    with munet.no_grad():
        y = mha.forward(x)
        y = ln.forward(y)
        y = gelu.forward(y)
        y = drop.forward(y)

    y_cpu = y.to(munet.Device(munet.DeviceType.CPU, 0)).detach()
    arr = np.array(y_cpu, copy=False)
    print("Transformer ops showcase complete")
    print(f"output shape={arr.shape}, mean={arr.mean():.5f}, std={arr.std():.5f}")


if __name__ == "__main__":
    main()
