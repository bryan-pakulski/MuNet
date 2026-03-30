import os
import sys
import tempfile
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../build"))
import munet


def make_model():
    return munet.nn.Sequential(
        [
            munet.nn.Linear(4, 16),
            munet.nn.GELU(),
            munet.nn.Linear(16, 2),
        ]
    )


def make_dataset(n=128):
    x = np.random.uniform(-1.0, 1.0, size=(n, 4)).astype(np.float32)
    y = np.zeros((n, 2), dtype=np.float32)
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
        munet.save_deploy(model, path)
        restored = munet.load_for_inference(path)

        engine = munet.inference.Engine()
        engine.load(restored)

        sample = munet.from_numpy(x_np[:8])
        engine.compile(
            sample, expected_input_shape=[-1, 4], expected_output_shape=[-1, 2]
        )
        out = engine.run(sample)

        # Dynamic batch size
        sample_big = munet.from_numpy(x_np[:32])
        out_big = engine.run(sample_big)

        out_np = np.array(out.detach(), copy=False)
        ref_np = np.array(restored.forward(sample).detach(), copy=False)

        print("\n=== E2E inference ===")
        print("compiled input shape:", engine.compiled_input_shape())
        print("compiled output shape:", engine.compiled_output_shape())
        print("engine runs:", engine.stats().runs)
        print("compile ms:", engine.stats().compile_ms)
        print("dynamic batch output shape:", out_big.shape)
        print("max abs diff vs direct forward:", float(np.max(np.abs(out_np - ref_np))))

    # Dynamic resolution showcase with conv model
    conv = munet.nn.Sequential(
        [
            munet.nn.Conv2d(3, 4, 3, padding=1),
            munet.nn.ReLU(),
            munet.nn.Conv2d(4, 2, 1),
        ]
    )
    conv_engine = munet.inference.Engine()
    conv_engine.load(conv)

    img_64 = munet.from_numpy(np.random.randn(1, 3, 64, 64).astype(np.float32))
    conv_engine.compile(
        img_64,
        expected_input_shape=[-1, 3, -1, -1],
        expected_output_shape=[-1, 2, -1, -1],
    )

    img_128 = munet.from_numpy(np.random.randn(2, 3, 128, 128).astype(np.float32))
    y_128 = conv_engine.run(img_128)
    print("dynamic resolution output shape:", y_128.shape)


if __name__ == "__main__":
    main()
