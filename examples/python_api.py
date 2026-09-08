"""Small, runnable Python API tour: train, checkpoint, export, reload, infer."""
import argparse
from pathlib import Path
import numpy as np
import munet as mu
from munet.checkpoint import save_state, load_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/python-api"))
    args = parser.parse_args()
    mu.manual_seed(7)
    model = mu.nn.Sequential(mu.nn.Linear(2, 16), mu.nn.ReLU(), mu.nn.Linear(16, 2))
    optimizer = mu.optim.AdamW(model.parameters(), lr=.01)
    step = mu.train_step(model, optimizer, mu.nn.CrossEntropyLoss(), device=args.device)
    rng = np.random.default_rng(7)
    for _ in range(100):
        features = rng.normal(size=(32, 2)).astype(np.float32)
        labels = (features[:, 0] + features[:, 1] > 0).astype(np.float32)
        loss = step(features, labels).item()
    args.output.mkdir(parents=True, exist_ok=True)
    save_state({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": rng.bit_generator.state}, args.output / "training.mnet")
    state = load_state(args.output / "training.mnet")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    rng.bit_generator.state = state["rng"]
    features = np.array([[1., 1.], [-1., -1.]], np.float32)
    model.export(args.output / "classifier.mnet", features,
                 input_names=["features"], output_names=["logits"])
    inference = mu.load(args.output / "classifier.mnet", device=args.device)
    print(model)
    print("Input signature:", inference.inputs)
    prediction = inference.predict(features).argmax(-1)
    np.testing.assert_array_equal(prediction, [1, 0])
    print(f"Final training loss: {loss:.4f}; predictions: {prediction.tolist()}")


if __name__ == "__main__": main()
