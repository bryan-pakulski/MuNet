import argparse
import json
from pathlib import Path
import numpy as np
import munet as mu
from munet.interop import to_onnx

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="vulkan")
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--output", type=Path, default=Path("artifacts"))
args = parser.parse_args()
if args.steps < 1: parser.error("--steps must be positive")
rng = np.random.default_rng(7)
x = rng.normal(size=(32, 4)).astype(np.float32)
y = (x @ rng.normal(size=(4, 2)) + 0.3).astype(np.float32)
model = mu.nn.Sequential(mu.nn.Linear(4, 16, rng=rng), mu.nn.ReLU(), mu.nn.Linear(16, 2, rng=rng))
optimizer = mu.optim.SGD(model.parameters(), lr=0.03)

@mu.compile(device=args.device)
def train_step(inputs, targets):
    optimizer.zero_grad()
    loss = mu.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return loss

first = train_step(x, y).item()
for i in range(1, args.steps):
    loss = train_step(x, y)
last = loss.item() if args.steps > 1 else first
inference = mu.compile(model, device=args.device)
expected = inference(x).numpy()
args.output.mkdir(parents=True, exist_ok=True)
mu.save(inference, args.output / "mlp.mnet")
mu.save(train_step, args.output / "training_step.mnet")
to_onnx(inference, args.output / "mlp.onnx")
restored = mu.load(args.output / "mlp.mnet", device=args.device)
np.testing.assert_allclose(restored(x).numpy(), expected, rtol=2e-5, atol=2e-5)
print(json.dumps({"initial_loss": first, "final_loss": last, "training": train_step.stats(),
                  "inference": inference.stats()}, indent=2))
