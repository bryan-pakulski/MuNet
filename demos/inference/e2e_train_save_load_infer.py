import os
import sys
import tempfile
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../build"))
import munet


def make_model():
    return munet.nn.Sequential([
        munet.nn.Linear(4, 16),
        munet.nn.GELU(),
        munet.nn.Linear(16, 2),
    ])


def make_dataset(n=128):
    x = np.random.uniform(-1.0, 1.0, size=(n, 4)).astype(np.float32)
    y = np.zeros((n, 2), dtype=np.float32)
    # simple synthetic target
    y[:, 0] = (x[:, 0] + x[:, 1] - x[:, 2]).astype(np.float32)
    y[:, 1] = (x[:, 3] * x[:, 3]).astype(np.float32)
    return x, y


def train_model(model, x_np, y_np, steps=200, bs=32):
    opt = munet.optim.Adam(model.parameters(), lr=5e-3)

    for step in range(steps):
        idx = np.random.randint(0, x_np.shape[0], size=bs)
        xb = munet.from_numpy(x_np[idx])
        yb = munet.from_numpy(y_np[idx])

        opt.zero_grad()
        pred = model.forward(xb)
        loss = pred.mse_loss(yb)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"step {step:3d} | loss={loss.item():.6f}")


def main():
    x_np, y_np = make_dataset(256)
    model = make_model()
    train_model(model, x_np, y_np)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trained_model.npz")
        munet.save(model, path)

        restored = munet.load(path)

        engine = munet.inference.Engine()
        engine.load(restored)

        sample = munet.from_numpy(x_np[:8])
        engine.compile(sample)
        out = engine.run(sample)

        out_np = np.array(out.detach(), copy=False)
        ref_np = np.array(restored.forward(sample).detach(), copy=False)

        print("\n=== E2E inference ===")
        print("compiled shape:", engine.compiled_input_shape())
        print("engine runs:", engine.stats().runs)
        print("compile ms:", engine.stats().compile_ms)
        print("max abs diff vs direct forward:", float(np.max(np.abs(out_np - ref_np))))


if __name__ == "__main__":
    main()
