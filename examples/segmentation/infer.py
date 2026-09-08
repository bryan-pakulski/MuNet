"""Write foreground probability and binary-mask PNGs from an RGB image."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import munet as mu
from examples.common import read_checkpoint
from .model import TinyUNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/segmentation/prediction"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    state = read_checkpoint(args.checkpoint, "segmentation")
    size = state["config"]["size"]
    model = TinyUNet(**state["config"]["model"]).eval()
    model.load_state_dict(state["model"])
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        original_size = image.size
        inputs = np.asarray(image.resize((size, size)), np.float32).transpose(2, 0, 1)[None] / 255
    probabilities = mu.compile(lambda x: model(x).sigmoid(), device=args.device)(inputs).numpy()[0, 0]
    args.output.mkdir(parents=True, exist_ok=True)
    probability = Image.fromarray(np.uint8(np.clip(probabilities, 0, 1) * 255)).resize(original_size, Image.Resampling.BILINEAR)
    mask = Image.fromarray(np.uint8(probabilities >= .5) * 255).resize(original_size, Image.Resampling.NEAREST)
    probability.save(args.output / "probability.png")
    mask.save(args.output / "mask.png")
    print(f"Saved probability.png and mask.png to {args.output}")


if __name__ == "__main__":
    main()
