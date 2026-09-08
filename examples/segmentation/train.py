"""Learn foreground masks from generated circles and rectangles."""
import argparse
import numpy as np
import munet as mu
from examples.common import Trainer, add_training_args, export, fingerprint, image_grid, positive_int, progress, write_json
from .data import shapes, scores
from .model import TinyUNet, mask_loss


def evaluate(model, images, masks, device, batch_size):
    model.eval()
    predict = mu.compile(lambda x: model(x).sigmoid(), device=device)
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        padded = np.zeros((batch_size, *images.shape[1:]), np.float32)
        padded[:len(batch)] = batch
        outputs.append(predict(padded).numpy()[:len(batch)].copy())
    probabilities = np.concatenate(outputs)
    return scores(probabilities, masks), probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_args(parser, output="artifacts/segmentation", steps=150, batch_size=8)
    parser.add_argument("--samples", type=positive_int, default=256)
    parser.add_argument("--eval-samples", type=positive_int, default=64)
    parser.add_argument("--size", type=positive_int, default=32)
    parser.add_argument("--width", type=positive_int, default=4)
    args = parser.parse_args()
    x, masks = shapes(args.samples, args.size, args.seed)
    validation = shapes(args.eval_samples, args.size, args.seed + 1)
    config = {"model": {"width": args.width, "seed": args.seed}, "size": args.size,
              "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
              "dataset": "generated-shapes-v1", "data_hash": fingerprint(x, masks)}
    model = TinyUNet(**config["model"])
    trainer = Trainer(model, mask_loss, example="segmentation", config=config, device=args.device, lr=args.lr, seed=args.seed)
    if args.resume:
        trainer.resume(args.resume)
    initial, _ = evaluate(model, *validation, args.device, args.batch_size)
    for step in range(args.steps):
        indices = trainer.rng.integers(len(x), size=args.batch_size)
        loss = trainer.step(x[indices], masks[indices])
        progress(step, args.steps, loss)
    final, probabilities = evaluate(model, *validation, args.device, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save(args.output / "training.mnet")
    export(model, validation[0][:1], args.output, args.device)
    from PIL import Image
    Image.fromarray(np.uint8(validation[0][0].transpose(1, 2, 0) * 255)).save(args.output / "sample.png")
    panels, labels = [], []
    for image, target, probability in zip(validation[0][:4], validation[1][:4], probabilities[:4]):
        rgb = image.transpose(1, 2, 0)
        overlay = rgb.copy()
        foreground = probability[0] >= .5
        overlay[foreground] = .5 * overlay[foreground] + .5 * np.array([0, 1, 0])
        panels.extend([rgb, target[0], probability[0], overlay])
        labels.extend(["input", "true mask", "probability", "predicted mask"])
    image_grid(panels, labels, args.output / "masks.png")
    write_json(args.output / "metrics.json", {"example": "segmentation", "config": config,
               "steps": trainer.steps, "validation_samples": len(validation[0]),
               "initial": initial, "final": final, "runtime": trainer.update.stats()})
    print(f"held-out IoU: {initial['iou']:.3f} -> {final['iou']:.3f}; saved to {args.output}")


if __name__ == "__main__":
    main()
