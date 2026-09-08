import hashlib
from pathlib import Path
import numpy as np
from examples.common import download


def corpus(directory, *, text_path=None, synthetic=False):
    if text_path is not None:
        path = Path(text_path)
    elif synthetic:
        text = "the sea is blue. the waves roll in. we watch the shore.\n" * 256
        return text, "generated-text-v1", hashlib.sha256(text.encode()).hexdigest()
    else:
        path = download("tinyshakespeare.txt", directory)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("this small example supports text files up to 32 MiB")
    data = path.read_bytes()
    return data.decode("utf-8"), "custom-text" if text_path else "tiny-shakespeare", hashlib.sha256(data).hexdigest()


def encode(text, vocabulary):
    lookup = {char: index for index, char in enumerate(vocabulary)}
    return np.asarray([lookup.get(char, 0) for char in text], np.float32)


def split(text, context):
    boundary = int(len(text) * .9)
    if min(boundary, len(text) - boundary) <= context:
        raise ValueError("both the 90% training and 10% validation split need more characters than the context")
    vocabulary = ["<unk>", *sorted(set(text[:boundary]))]
    return encode(text[:boundary], vocabulary), encode(text[boundary:], vocabulary), vocabulary


def batch(tokens, count, context, rng):
    if count <= 0 or context <= 0 or len(tokens) <= context:
        raise ValueError("invalid next-character batch dimensions")
    starts = rng.integers(0, len(tokens) - context, size=count)
    indices = starts[:, None] + np.arange(context)[None]
    return tokens[indices], tokens[indices + 1]
