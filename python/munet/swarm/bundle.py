"""Export static forward/backward profiles and a deterministic sample manifest."""
from pathlib import Path
import uuid
import numpy as np
from ..core import Compiled, grad, _compile_shaders
from ..nn import functional
from .artifacts import ABI, Artifacts, atomic_write, encode, digest, tensor


def create_job(model, x, target, directory, *, global_batch_size=32,
               micro_batch_size=8, epochs=1, lr=0.01, seed=0,
               include_vulkan=True):
    """Create a new MSE/SGD job. Each epoch visits every sample once.

    Global batches define SGD semantics; workers subdivide them into independent
    microbatches. v1 deliberately fixes the loss to mean squared error. Models
    must be deterministic and independent across the leading sample dimension.
    No BatchNorm, dropout, batch-coupled loss, or local optimizer is supported.
    """
    for name, value in (("global_batch_size", global_batch_size),
                        ("micro_batch_size", micro_batch_size), ("epochs", epochs)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(lr) or not 0 < lr <= 1e3:
        raise ValueError("lr must be finite and positive (at most 1000)")
    arrays = [np.asarray(a) for a in (x, target)]
    if any(a.dtype != np.float32 or a.ndim < 2 or not all(a.shape)
           or not np.isfinite(a).all() for a in arrays):
        raise ValueError("dataset must be finite float32 arrays of rank >= 2")
    if arrays[0].shape[0] != arrays[1].shape[0]:
        raise ValueError("input and target sample counts differ")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "job.json").exists() or (root / "owner.sqlite").exists():
        raise FileExistsError("use a new job directory; existing jobs are immutable")
    store = Artifacts(root / "artifacts")
    named = list(model.named_parameters())
    if not named or sum(p.numpy().size for _, p in named) > 100_000:
        raise ValueError("prototype jobs require 1..100,000 parameter values")
    initial = {name: tensor(p.numpy()) for name, p in named}
    if any(not np.isfinite(p.numpy()).all() for _, p in named):
        raise ValueError("parameters must be finite")
    checkpoint = store.put({"abi": ABI, "version": 0, "parameters": initial})
    profiles = {}

    def profile(count):
        if count in profiles:
            return profiles[count]
        def backward(a, b):
            prediction = model(a)
            if prediction.shape != b.shape:
                raise ValueError("MSE prediction and target shapes must match exactly")
            loss = functional.mse_loss(prediction, b)
            return [loss] + grad(loss, [p for _, p in named])
        compiled = Compiled(backward, device="cpu")
        compiled._capture([a[:count] for a in arrays])
        nodes = compiled._plan.graph.nodes()
        # Parameters are supplied by each round's immutable checkpoint.
        parameters = []
        for name, p in named:
            ident = compiled._parameters[p]
            nodes[ident]["data"] = []
            parameters.append({"name": name, "id": ident, "shape": list(p.shape)})
        shaders = compiled._plan.shaders()
        program = {"abi": ABI, "nodes": nodes, "outputs": compiled._plan.outputs,
                   "feed_indices": compiled._feed_indices, "parameters": parameters,
                   "shader_hashes": [digest(s.encode()) for s in shaders],
                   "spirv": _compile_shaders(shaders) if include_vulkan else [],
                   "arena_bytes": compiled.stats()["arena_bytes"]}
        key = store.put(program)
        profiles[count] = {"program": key, "arena_bytes": program["arena_bytes"],
                           "vulkan": include_vulkan}
        return profiles[count]

    rounds = []
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        order = rng.permutation(len(arrays[0])).tolist()
        for start in range(0, len(order), global_batch_size):
            ids = order[start:start + global_batch_size]
            chunks = []
            for offset in range(0, len(ids), micro_batch_size):
                sample_ids = ids[offset:offset + micro_batch_size]
                data = store.put({"abi": ABI, "sample_ids": sample_ids,
                                  "inputs": [tensor(a[sample_ids]) for a in arrays]})
                chunks.append({"id": f"r{len(rounds)}c{len(chunks)}", "samples": len(sample_ids),
                               "data": data, **profile(len(sample_ids))})
            rounds.append({"epoch": epoch, "samples": len(ids), "chunks": chunks})
    job = {"abi": ABI, "id": uuid.uuid4().hex, "algorithm": "mean-mse-sgd",
           "epochs": epochs, "samples_per_epoch": len(arrays[0]), "lr": float(lr),
           "seed": seed, "global_batch_size": global_batch_size,
           "micro_batch_size": micro_batch_size, "initial_checkpoint": checkpoint,
           "rounds": rounds}
    atomic_write(root / "job.json", encode(job))
    return root
