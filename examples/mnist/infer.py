"""Classify a light digit on a dark background using an example checkpoint."""
import argparse
import json
import numpy as np
from PIL import Image, ImageOps
import munet as mu
from examples.common import read_checkpoint
from .model import DigitCNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--invert", action="store_true", help="use for a dark digit on a light background")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    state = read_checkpoint(args.checkpoint, "mnist")
    model = DigitCNN(**state["config"]["model"]).eval()
    model.load_state_dict(state["model"])
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize((28, 28))
        if args.invert:
            image = ImageOps.invert(image)
        inputs = np.asarray(image, np.float32)[None, None] / 255
    logits = mu.compile(model, device=args.device)(inputs).numpy()[0]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    print(json.dumps({"digit": int(probabilities.argmax()), "probabilities": probabilities.tolist()}, indent=2))


if __name__ == "__main__":
    main()
