"""Small conveniences shared by source examples, excluded from library wheels."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import munet as mu
from munet.checkpoint import load_state, save_state


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def download(name, directory):
    """Fetch a known dataset atomically, verifying both fresh and cached bytes."""
    resources = json.loads(Path(__file__).with_name("datasets.json").read_text())
    spec = resources[name]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    def valid(path):
        return path.stat().st_size == spec["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"]
    if target.exists() and valid(target):
        return target
    fd, temporary = tempfile.mkstemp(prefix=name + ".", suffix=".part", dir=directory)
    try:
        print(f"Downloading {name} ({spec['bytes']:,} bytes)", flush=True)
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(spec["url"], timeout=60) as response:
            total = 0
            while chunk := response.read(64 * 1024):
                total += len(chunk)
                if total > spec["bytes"]:
                    raise ValueError(f"unexpected dataset size: {name}")
                out.write(chunk)
        if not valid(Path(temporary)):
            raise ValueError(f"dataset checksum/size mismatch: {name}")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def cross_entropy(logits, labels):
    """Class-index cross entropy; the last logits axis holds the classes."""
    shifted = logits - logits.amax(-1, keepdim=True).detach()
    log_probabilities = shifted - shifted.exp().sum(-1, keepdim=True).log()
    return -log_probabilities.gather(-1, labels.unsqueeze(-1)).mean()


class Trainer:
    def __init__(self, model, loss, *, example, config, device="cpu", lr=0.003, seed=7):
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("learning rate must be positive and finite")
        self.model, self.example, self.config = model, example, config
        self.optimizer = mu.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.rng = np.random.default_rng(seed)
        self.steps = 0

        @mu.compile(device=device)
        def update(inputs, targets):
            self.optimizer.zero_grad()
            value = loss(model(inputs), targets)
            value.backward()
            mu.optim.clip_grad_norm_(model.parameters(), 1.0)
            self.optimizer.step()
            return value
        self.update = update

    def step(self, inputs, targets):
        self.model.train()
        loss = self.update(np.asarray(inputs, np.float32), np.asarray(targets, np.float32)).item()
        if not np.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        self.steps += 1
        return loss

    def save(self, path):
        save_state({"format": "munet-example-v1", "example": self.example, "config": self.config,
                    "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "rng": copy.deepcopy(self.rng.bit_generator.state), "steps": self.steps}, path)

    def resume(self, path):
        state = read_checkpoint(path, self.example)
        if state["config"] != self.config:
            raise ValueError("resume requires the same model, dataset, batch size, seed and learning rate")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.bit_generator.state = state["rng"]
        self.steps = state["steps"]


def read_checkpoint(path, example):
    state = load_state(path)
    if state.get("format") != "munet-example-v1" or state.get("example") != example:
        raise ValueError(f"expected a {example} example checkpoint")
    return state


def export(model, inputs, directory, device):
    program = mu.compile(model.eval(), device=device)
    expected = program(np.asarray(inputs, np.float32)).numpy()
    path = Path(directory) / "inference.mnet"
    mu.save(program, path)
    restored = mu.load(path, device=device)
    np.testing.assert_allclose(restored(inputs).numpy(), expected, rtol=3e-4, atol=3e-5)
    return path


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def image_grid(images, labels, path, columns=4):
    from PIL import Image, ImageDraw
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * 102, rows * 122), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (array, label) in enumerate(zip(images, labels)):
        tile = Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255)).convert("RGB")
        tile = tile.resize((96, 96), Image.Resampling.NEAREST)
        x, y = (i % columns) * 102, (i // columns) * 122
        canvas.paste(tile, (x, y))
        draw.text((x, y + 99), label, fill="black")
    canvas.save(path)


def add_training_args(parser, *, output, steps=200, batch_size=16, lr=0.003):
    parser.add_argument("--device", default="cpu", help="cpu, vulkan or vulkan:N")
    parser.add_argument("--steps", type=positive_int, default=steps, help="additional optimizer updates")
    parser.add_argument("--batch-size", type=positive_int, default=batch_size)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path(output))
    parser.add_argument("--resume", type=Path)


def progress(step, count, loss):
    if step == 0 or (step + 1) % 25 == 0 or step + 1 == count:
        print(f"update {step + 1}/{count}: loss={loss:.4f}", flush=True)
