"""Reproducible local measurements; this does not compare against vendor libraries."""
import argparse
import json
import os
import tempfile
import time
import numpy as np
import munet as mu

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="vulkan")
parser.add_argument("--runs", type=int, default=50)
args = parser.parse_args()
if args.runs <= 0: parser.error("--runs must be positive")
rng = np.random.default_rng(11)
model = mu.nn.Sequential(mu.nn.Linear(32, 64, rng=rng), mu.nn.ReLU(), mu.nn.Linear(64, 16, rng=rng))
x = rng.normal(size=(16, 32)).astype(np.float32)
results = {}
with tempfile.TemporaryDirectory() as cache:
    os.environ["MUNET_CACHE_DIR"] = cache
    for fuse in [False, True]:
        program = mu.compile(model, device=args.device, fuse=fuse)
        start = time.perf_counter(); program(x); program.synchronize()
        first = (time.perf_counter() - start) * 1000
        timings = []
        for _ in range(args.runs):
            start = time.perf_counter(); program(x); program.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
        results["fused" if fuse else "unfused"] = {
            "first_call_ms": first, "warm_median_ms": float(np.median(timings)),
            "warm_p95_ms": float(np.percentile(timings, 95)), **program.stats(),
        }
print(json.dumps({"note": "First call includes graph capture/compile/pipeline creation. Driver cache state is uncontrolled. Warm measurements include input staging and completion; no output readback.", "results": results}, indent=2))
