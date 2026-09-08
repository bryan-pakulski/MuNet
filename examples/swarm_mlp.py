"""Prepare a training swarm, then export its final model after stopping the owner."""
import argparse
from pathlib import Path
import numpy as np
import munet as mu
from munet.swarm import create_job, Owner


def fixture():
    rng = np.random.default_rng(17)
    model = mu.nn.Sequential(mu.nn.Linear(4, 16, rng=rng), mu.nn.ReLU(),
                             mu.nn.Linear(16, 2, rng=rng))
    x = rng.normal(size=(129, 4)).astype(np.float32)
    y = (x @ rng.normal(size=(4, 2))).astype(np.float32)
    return model, x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "export"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cpu-only", action="store_true", help="omit SPIR-V from a new job")
    parser.add_argument("--device", default="vulkan", help="export inference device: vulkan, vulkan:N or cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("swarm-trained.mnet"))
    parser.add_argument("--onnx", action="store_true", help="also export the trained inference graph as ONNX")
    args = parser.parse_args()
    model, x, y = fixture()
    if args.action == "prepare":
        create_job(model, x, y, args.directory, global_batch_size=32, micro_batch_size=4,
                   epochs=args.epochs, lr=0.03, seed=42, include_vulkan=not args.cpu_only)
        print(f"Prepared {args.directory}; start it with: python -m munet.swarm {args.directory}")
        return
    owner = Owner(args.directory)
    try:
        owner.load_into(model)
        status = owner.status()
        print(f"Exporting checkpoint {status['version']} ({status['epochs_completed']} completed epochs)")
    finally:
        owner.close()
    inference = mu.compile(model, device="cpu" if args.cpu_only else args.device)
    prediction = inference(x).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mu.save(inference, args.output)
    if args.onnx:
        from munet.interop import to_onnx
        to_onnx(inference, args.output.with_suffix(".onnx"))
    print(f"MSE: {np.mean((prediction-y)**2):.7f}; saved {args.output}")


if __name__ == "__main__":
    main()
