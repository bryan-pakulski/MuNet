"""Train a tiny causal Transformer on Tiny Shakespeare or a UTF-8 text file."""
import argparse
import math
from pathlib import Path
import numpy as np
import munet as mu
from examples.common import Trainer, add_training_args, cross_entropy, export, positive_int, progress, write_json
from .data import batch, corpus, split
from .generate import sample
from .model import TinyGPT


def evaluate(model, tokens, device, batch_size):
    model.eval()
    run = mu.compile(lambda x, y: cross_entropy(model(x), y), device=device)
    rng = np.random.default_rng(123)
    loss = float(np.mean([run(*batch(tokens, batch_size, model.context, rng)).item() for _ in range(4)]))
    return {"loss": loss, "perplexity": math.exp(loss)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_args(parser, output="artifacts/language-model", batch_size=8, lr=.001)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/datasets/language-model"))
    parser.add_argument("--text", type=Path, help="use your own UTF-8 text instead of downloading")
    parser.add_argument("--synthetic", action="store_true", help="use generated repetitive text for an offline smoke test")
    parser.add_argument("--context", type=positive_int, default=32)
    parser.add_argument("--width", type=positive_int, default=32)
    parser.add_argument("--heads", type=positive_int, default=4)
    parser.add_argument("--layers", type=positive_int, default=2)
    parser.add_argument("--sample-tokens", type=positive_int, default=100)
    args = parser.parse_args()
    if args.text and args.synthetic:
        parser.error("choose --text or --synthetic")
    text, name, digest = corpus(args.data_dir, text_path=args.text, synthetic=args.synthetic)
    training, validation, vocabulary = split(text, args.context)
    config = {"model": {"vocab_size": len(vocabulary), "context": args.context, "width": args.width,
                        "heads": args.heads, "layers": args.layers, "seed": args.seed},
              "dataset": name, "data_hash": digest, "vocabulary": vocabulary,
              "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed}
    model = TinyGPT(**config["model"])
    trainer = Trainer(model, cross_entropy, example="language_model", config=config, device=args.device, lr=args.lr, seed=args.seed)
    if args.resume:
        trainer.resume(args.resume)
    initial = evaluate(model, validation, args.device, args.batch_size)
    for step in range(args.steps):
        loss = trainer.step(*batch(training, args.batch_size, args.context, trainer.rng))
        progress(step, args.steps, loss)
    final = evaluate(model, validation, args.device, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save(args.output / "training.mnet")
    export(model, training[:args.context][None], args.output, args.device)
    generated = sample(model, vocabulary, text[:min(16, args.context)], tokens=args.sample_tokens, device=args.device)
    (args.output / "sample.txt").write_text(generated, encoding="utf-8")
    write_json(args.output / "metrics.json", {"example": "language_model", "config": config,
               "steps": trainer.steps, "split_characters": {"train": len(training), "validation": len(validation)},
               "initial": initial, "final": final, "runtime": trainer.update.stats()})
    print(f"held-out loss: {initial['loss']:.3f} -> {final['loss']:.3f}; saved to {args.output}")
    print(generated)


if __name__ == "__main__":
    main()
