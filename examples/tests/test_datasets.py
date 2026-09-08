import gzip
import hashlib
import io
import json
import struct
import numpy as np
import pytest
from examples import common
from examples.mnist.data import read_idx
from examples.segmentation.data import shapes, scores
from examples.language_model.data import batch, split


def test_download_cache_integrity_and_atomic_failure(tmp_path, monkeypatch):
    payload = b"known dataset contents\n"
    (tmp_path / "datasets.json").write_text(json.dumps({"sample.txt": {
        "url": "https://example.invalid/sample.txt", "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest()}}))
    monkeypatch.setattr(common, "__file__", str(tmp_path / "common.py"))
    requests = []
    def fetch(url, timeout):
        requests.append(url)
        return io.BytesIO(payload)
    monkeypatch.setattr(common.urllib.request, "urlopen", fetch)
    cache = tmp_path / "cache"
    path = common.download("sample.txt", cache)
    assert path.read_bytes() == payload
    assert common.download("sample.txt", cache) == path
    assert len(requests) == 1
    path.write_bytes(b"x" * len(payload))
    common.download("sample.txt", cache)
    assert path.read_bytes() == payload and len(requests) == 2
    path.write_bytes(b"previous incomplete cache")
    monkeypatch.setattr(common.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"x" * len(payload)))
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        common.download("sample.txt", cache)
    assert path.read_bytes() == b"previous incomplete cache"
    assert list(cache.iterdir()) == [path]  # No partial download left behind.


def test_idx_decodes_and_rejects_truncation(tmp_path):
    path = tmp_path / "images.gz"
    array = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    header = struct.pack(">HBBIII", 0, 8, 3, *array.shape)
    path.write_bytes(gzip.compress(header + array.tobytes()))
    np.testing.assert_array_equal(read_idx(path), array)
    path.write_bytes(gzip.compress(header + array.tobytes()[:-1]))
    with pytest.raises(ValueError, match="shape/length"):
        read_idx(path)
    path.write_bytes(gzip.compress(struct.pack(">HBBI", 0, 11, 1, 2) + b"00"))
    with pytest.raises(ValueError, match="uint8"):
        read_idx(path)


def test_generated_masks_match_foreground_and_are_reproducible():
    images, masks = shapes(8, 16, 7)
    repeated = shapes(8, 16, 7)
    np.testing.assert_array_equal(images, repeated[0])
    np.testing.assert_array_equal(masks, repeated[1])
    np.testing.assert_array_equal(images.max(1, keepdims=True) > .35, masks.astype(bool))
    assert not np.array_equal(images, shapes(8, 16, 8)[0])
    assert scores(masks, masks) == {"iou": 1., "dice": 1.}
    assert scores(1 - masks, masks) == {"iou": 0., "dice": 0.}


def test_text_split_and_next_character_targets():
    train, validation, vocabulary = split("abc" * 100 + "z" * 30, context=8)
    assert len(train) == 297 and len(validation) == 33
    assert vocabulary == ["<unk>", "a", "b", "c"]
    assert (validation[-30:] == 0).all()  # Validation cannot expand training vocabulary.
    tokens = np.arange(40, dtype=np.float32)
    x, y = batch(tokens, 20, 8, np.random.default_rng(7))
    np.testing.assert_array_equal(y, x + 1)
    assert x.shape == y.shape == (20, 8) and y.max() < len(tokens)
    with pytest.raises(ValueError, match="more characters"):
        split("short", context=8)
