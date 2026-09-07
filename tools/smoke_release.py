"""Exercise real installed/frozen executables on one complete training round."""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
import numpy as np
import munet as mu
from munet.swarm import create_job, Owner


def exercise(node, server):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(MUNET_SWARM_TOKEN=secrets.token_urlsafe(32),
               MUNET_VULKAN_LIBRARY="/not-installed-on-cpu-owner/libvulkan.so.1",
               MUNET_GLSLANG="/not-installed-on-node/glslangValidator")
    for executable in (node, server):
        result = subprocess.run([str(executable), "--version"], env=env, capture_output=True, text=True, check=True, timeout=20)
        assert mu.__version__ in result.stdout, result.stdout
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(5)
        model = mu.nn.Linear(2, 1, rng=rng)
        x = np.ones((5, 2), np.float32)
        y = np.zeros((5, 1), np.float32)
        before = np.mean(mu.compile(model, device="cpu")(x).numpy()**2)
        create_job(model, x, y, root / "job", global_batch_size=5, micro_batch_size=3, include_vulkan=False)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with (root / "server.log").open("w+") as log:
            process = subprocess.Popen([str(server), str(root / "job"), "--port", str(port)],
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                url = f"http://127.0.0.1:{port}/v1/status"
                def status():
                    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + env["MUNET_SWARM_TOKEN"]})
                    with urllib.request.urlopen(req, timeout=1) as response:
                        return json.load(response)
                for _ in range(100):
                    try:
                        status(); break
                    except (OSError, urllib.error.URLError):
                        if process.poll() is not None:
                            log.seek(0)
                            raise RuntimeError("server exited: " + log.read())
                        time.sleep(0.1)
                else:
                    raise RuntimeError("server did not become ready")
                worker = subprocess.run([str(node), "--owner", url.removesuffix("/v1/status"),
                                         "--device", "cpu", "--state", str(root / "worker"), "--poll-seconds", "0.01"],
                                        env=env, capture_output=True, text=True, timeout=30)
                assert worker.returncode == 0, worker.stdout + worker.stderr
                progress = status()
                assert progress["state"] == "completed" and progress["version"] == 1 and progress["samples_accepted"] == 5
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
        owner = Owner(root / "job")
        try:
            owner.load_into(model)
        finally:
            owner.close()
        after = np.mean(mu.compile(model, device="cpu")(x).numpy()**2)
        assert after < before
    print("Installed server/node: version checks, headless startup, training, journal recovery and checkpoint update passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    args = parser.parse_args()
    exercise(args.node.resolve(), args.server.resolve())
