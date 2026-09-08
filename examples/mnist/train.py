"""Train a small CNN on downloaded MNIST, or --synthetic for an offline check."""
import argparse
from pathlib import Path
import numpy as np
import munet as mu
from examples.common import Trainer, add_training_args, cross_entropy, export, fingerprint, image_grid, positive_int, progress, write_json
from .data import load, normalize, synthetic
from .model import DigitCNN


def evaluate(model, images, labels, device, batch_size):
    predict = mu.compile(model.eval(), device=device)
    all_logits = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        padded = np.zeros((batch_size, 1, 28, 28), np.float32)
        padded[:len(batch)] = normalize(batch)
        all_logits.append(predict(padded).numpy()[:len(batch)].copy())
    logits = np.concatenate(all_logits)
    shifted = logits - logits.max(-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    return {"loss": float(-logp[np.arange(len(labels)), labels].mean()),
            "accuracy": float((logits.argmax(-1) == labels).mean())}, logits.argmax(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_args(parser, output="artifacts/mnist", batch_size=32)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/mnist"))
    parser.add_argument("--limit", type=positive_int, default=10000)
    parser.add_argument("--eval-samples", type=positive_int, default=512)
    parser.add_argument("--width", type=positive_int, default=8)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        training, validation = synthetic(args.limit, args.seed), synthetic(args.eval_samples, args.seed + 1)
    else:
        training, validation = load(args.data_dir, args.limit, args.eval_samples)
    x, y = training
    config = {"model": {"width": args.width, "seed": args.seed}, "batch_size": args.batch_size,
              "lr": args.lr, "dataset": "synthetic-seven-segment" if args.synthetic else "MNIST",
              "data_hash": fingerprint(x, y), "seed": args.seed}
    model = DigitCNN(**config["model"])
    trainer = Trainer(model, cross_entropy, example="mnist", config=config, device=args.device, lr=args.lr, seed=args.seed)
    if args.resume:
        trainer.resume(args.resume)
    initial, _ = evaluate(model, *validation, args.device, args.batch_size)
    for step in range(args.steps):
        indices = trainer.rng.integers(len(x), size=args.batch_size)
        loss = trainer.step(normalize(x[indices]), y[indices])
        progress(step, args.steps, loss)
    final, predictions = evaluate(model, *validation, args.device, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save(args.output / "training.mnet")
    export(model, normalize(validation[0][:1]), args.output, args.device)
    from PIL import Image
    Image.fromarray(validation[0][0, 0]).save(args.output / "sample.png")
    n = min(16, len(predictions))
    image_grid([a[0] / 255.0 for a in validation[0][:n]],
               [f"true {int(t)} / pred {int(p)}" for t, p in zip(validation[1][:n], predictions[:n])], args.output / "predictions.png")
    metrics = {"example": "mnist", "config": config, "steps": trainer.steps, "validation_samples": len(validation[1]),
               "initial": initial, "final": final, "runtime": trainer.update.stats()}
    write_json(args.output / "metrics.json", metrics)
    print(f"held-out accuracy: {initial['accuracy']:.1%} -> {final['accuracy']:.1%}; saved to {args.output}")


if __name__ == "__main__":
    main()
