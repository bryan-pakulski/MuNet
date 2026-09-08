"""MNIST IDX decoding without torchvision, plus a labeled offline smoke dataset."""
import gzip
import math
import struct
import numpy as np
from examples.common import download


def read_idx(path):
    with gzip.open(path, "rb") as stream:
        data = stream.read(64 * 1024 * 1024 + 1)
    if len(data) > 64 * 1024 * 1024 or len(data) < 8:
        raise ValueError("invalid IDX size")
    zero, kind, rank = struct.unpack(">HBB", data[:4])
    if zero != 0 or kind != 8 or rank not in (1, 3) or len(data) < 4 + rank * 4:
        raise ValueError("expected uint8 image/label IDX data")
    shape = struct.unpack(">" + "I" * rank, data[4:4 + rank * 4])
    if any(n == 0 for n in shape) or math.prod(shape) != len(data) - 4 - rank * 4:
        raise ValueError("IDX shape/length mismatch")
    return np.frombuffer(data, np.uint8, offset=4 + rank * 4).reshape(shape)


def load(directory, limit=10000, eval_samples=512):
    if limit <= 0 or eval_samples <= 0:
        raise ValueError("dataset limits must be positive")
    arrays = []
    for prefix, count in (("train", 60000), ("t10k", 10000)):
        images = read_idx(download(prefix + "-images-idx3-ubyte.gz", directory))
        labels = read_idx(download(prefix + "-labels-idx1-ubyte.gz", directory))
        if images.shape != (count, 28, 28) or labels.shape != (count,) or np.any(labels > 9):
            raise ValueError("unexpected MNIST image/label layout")
        n = min(limit if prefix == "train" else eval_samples, count)
        arrays.append((images[:n, None], labels[:n]))
    return arrays


def synthetic(count, seed):
    """Seven-segment digits for offline execution checks; these are not MNIST."""
    rng = np.random.default_rng(seed)
    if count <= 0:
        raise ValueError("sample count must be positive")
    labels = (np.arange(count) % 10).astype(np.uint8)
    rng.shuffle(labels)
    images = np.zeros((count, 1, 28, 28), np.float32)
    segments = ["abcedf", "bc", "abged", "abgcd", "fgbc", "afgcd", "afgecd", "abc", "abcdefg", "abfgcd"]
    regions = {"a": (4, 7, 8, 20), "b": (6, 14, 19, 22), "c": (14, 23, 19, 22),
               "d": (22, 25, 8, 20), "e": (14, 23, 5, 8), "f": (6, 14, 5, 8), "g": (12, 15, 8, 20)}
    for i, label in enumerate(labels):
        for segment in segments[label]:
            y0, y1, x0, x1 = regions[segment]
            images[i, 0, y0:y1, x0:x1] = rng.uniform(.7, 1)
        images[i, 0] = np.roll(images[i, 0], tuple(rng.integers(-2, 3, 2)), axis=(0, 1))
    images += rng.normal(0, .035, images.shape)
    return np.uint8(np.clip(images, 0, 1) * 255), labels


def normalize(images):
    return images.astype(np.float32) / 255.0
