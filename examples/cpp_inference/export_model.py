"""Train a tiny regressor in Python, then export for the standalone C++ application."""
import argparse
from pathlib import Path
import numpy as np
import munet as mu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cpp-inference/model.mnet"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-vulkan", action="store_true", help="bundle deployment shaders; needs glslangValidator here")
    args = parser.parse_args()
    mu.manual_seed(7)
    model = mu.nn.Linear(4, 2)
    optimizer = mu.optim.SGD(model.parameters(), lr=.05)
    step = mu.train_step(model, optimizer, mu.nn.MSELoss(), device=args.device)
    rng = np.random.default_rng(11)
    weights = np.array([[1, 2], [-1, 1], [.5, -1], [2, .5]], np.float32)
    for _ in range(200):
        x = rng.normal(size=(16, 4)).astype(np.float32)
        step(x, x @ weights + np.float32(.25))
    features = np.array([[1, 2, 3, 4]], np.float32)
    model.export(args.output, features, input_names=["features"], output_names=["prediction"],
                 include_vulkan=args.include_vulkan)
    expected = model.eval().compile(device=args.device).predict(features)
    np.savetxt(args.output.with_suffix(".expected.txt"), expected)
    print(f"Exported {args.output}; Python prediction: {expected.tolist()}")


if __name__ == "__main__": main()
