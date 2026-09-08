"""Deterministic RGB shapes with exact binary masks; no download needed."""
import numpy as np


def shapes(count=256, size=32, seed=7):
    if count <= 0 or size < 16 or size % 2:
        raise ValueError("use a positive count and an even image size of at least 16")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    images = np.empty((count, 3, size, size), np.float32)
    masks = np.zeros((count, 1, size, size), np.float32)
    for i in range(count):
        image = rng.uniform(.05, .2, (3, 1, 1)) + rng.normal(0, .03, (3, size, size))
        for _ in range(int(rng.integers(1, 4))):
            cx, cy = rng.integers(size // 5, size - size // 5, 2)
            rx, ry = rng.integers(max(2, size // 10), size // 4 + 1, 2)
            if rng.random() < .5:
                region = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
            else:
                region = (abs(xx - cx) <= rx) & (abs(yy - cy) <= ry)
            color = rng.uniform(.5, .95, (3, 1))
            image[:, region] = color + rng.normal(0, .03, (3, int(region.sum())))
            masks[i, 0, region] = 1
        images[i] = np.clip(image, 0, 1)
    return images, masks


def scores(probabilities, targets):
    predicted, truth = probabilities >= .5, targets >= .5
    intersection = int((predicted & truth).sum())
    union = int((predicted | truth).sum())
    total = int(predicted.sum() + truth.sum())
    return {"iou": intersection / union if union else 1.0,
            "dice": 2 * intersection / total if total else 1.0}
