"""Gradient equivalence and crash/retry semantics, including the real worker binary."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import threading
import urllib.request
import urllib.error
import numpy as np
import pytest
import munet as mu
from munet.swarm import create_job, Owner, ProtocolError
from munet.swarm.artifacts import ABI, atomic_write, encode, tensor
from munet.swarm.artifacts import Artifacts
from munet.swarm.server import make_server

TOKEN = "test-only-" + "t" * 40
NODE = Path(os.environ.get("MUNET_SWARM_NODE", Path(__file__).resolve().parents[1] / "build" / "munet-node"))


def make_job(root, *, n=11, batch=7, micro=3, epochs=2, vulkan=False, deep=False):
    rng = np.random.default_rng(77)
    model = (mu.nn.Sequential(mu.nn.Linear(3, 5, rng=rng), mu.nn.ReLU(),
                              mu.nn.Linear(5, 2, rng=rng)) if deep else mu.nn.Linear(3, 2, rng=rng))
    x, y = [rng.normal(size=(n, dim)).astype(np.float32) for dim in (3, 2)]
    create_job(model, x, y, root, global_batch_size=batch, micro_batch_size=micro,
               epochs=epochs, lr=0.03, seed=9, include_vulkan=vulkan)
    return model, x, y


def register(owner, name="a", **changes):
    caps = dict(abi=ABI, backend="cpu", dtype="float32", memory_bytes=2**30,
                max_buffer_bytes=2**30, max_batch=1024, samples_per_second=100)
    caps.update(changes)
    return owner.register({"worker_id": name, "capabilities": caps})


def lease(owner, name="a"):
    return owner.lease({"worker_id": name})["assignments"]


def result_for(owner, a):
    # Independent analytical gradient for Linear + MSE, without MuNet autodiff.
    checkpoint = owner.artifacts.get(a["checkpoint"])["parameters"]
    data = owner.artifacts.get(a["data"])["inputs"]
    x, y = [np.array(t["data"], np.float32).reshape(t["shape"]) for t in data]
    weight, bias = [np.array(checkpoint[k]["data"], np.float32).reshape(checkpoint[k]["shape"])
                    for k in ("weight", "bias")]
    error = x @ weight.T + bias - y
    delta = 2 * error / error.size
    gradients = {"weight": tensor(delta.T @ x), "bias": tensor(delta.sum(axis=0))}
    return {**{k: v for k, v in a.items() if k != "deadline"}, "loss": float((error**2).mean()),
            "seconds": 0.1, "gradients": gradients}


def drain(owner, name="a"):
    while owner.status()["state"] != "completed":
        work = lease(owner, name)
        assert work, "test worker cannot obtain work"
        for a in reversed(work):
            owner.result(result_for(owner, a))


@contextmanager
def serving(owner, port=0, context=None):
    server = make_server(owner, ("127.0.0.1", port), TOKEN)
    if context:
        server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield f"{'https' if context else 'http'}://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def node_command(url, state, device="cpu", once=False):
    if not NODE.is_file() and os.environ.get("MUNET_TEST_SWARM") != "1":
        pytest.skip("build munet-node or set MUNET_TEST_SWARM=1 to require the integration gate")
    assert NODE.is_file(), "build munet-node with -DMUNET_SWARM_NODE=ON before running swarm integration tests"
    command = [str(NODE), "--owner", url, "--state", str(state), "--device", device,
               "--poll-seconds", "0.01", "--samples-per-second", "50"]
    return command + (["--once"] if once else [])


def node_env():
    # A standalone native node must not depend on Python or an on-node GLSL compiler.
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(MUNET_SWARM_TOKEN=TOKEN, MUNET_GLSLANG="/missing/not-needed-on-node")
    return env


def test_swarm_uneven_weighting_and_epoch_coverage(tmp_path):
    model, x, y = make_job(tmp_path)
    reference = model.state_dict()
    owner = Owner(tmp_path)
    try:
        register(owner)
        for info in owner.job["rounds"]:
            ids = sum([owner.artifacts.get(c["data"])["sample_ids"] for c in info["chunks"]], [])
            error = x[ids] @ reference["weight"].T + reference["bias"] - y[ids]
            delta = 2 * error / error.size
            reference["weight"] -= np.float32(0.03) * (delta.T @ x[ids])
            reference["bias"] -= np.float32(0.03) * delta.sum(axis=0)
        drain(owner)
        owner.load_into(model)
        for k, v in model.state_dict().items():
            np.testing.assert_allclose(v, reference[k], atol=1e-7, rtol=1e-6)
        status = owner.status()
        assert status["epochs_completed"] == 2 and status["samples_accepted"] == 22
        assert status["version"] == 4 and len(status["commits"]) == 4
        for epoch in range(2):
            samples = [i for r in owner.job["rounds"] if r["epoch"] == epoch
                       for c in r["chunks"] for i in owner.artifacts.get(c["data"])["sample_ids"]]
            assert sorted(samples) == list(range(11))
    finally:
        owner.close()


def test_swarm_lease_retry_late_result_and_duplicate_race(tmp_path):
    make_job(tmp_path, n=3, batch=3, micro=3, epochs=1)
    now = [100.0]
    owner = Owner(tmp_path, lease_seconds=10, clock=lambda: now[0])
    try:
        register(owner, "lost"); register(owner, "replacement")
        first = lease(owner, "lost")[0]
        assert not lease(owner, "replacement")
        now[0] += 11
        second = lease(owner, "replacement")[0]
        assert first["chunk_id"] == second["chunk_id"] and first["attempt"] != second["attempt"]
        assert first["checkpoint"] == second["checkpoint"] and first["data"] == second["data"]
        payload = result_for(owner, first)
        with ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(pool.map(lambda _: owner.result(payload)["status"], range(4)))
        assert statuses.count("accepted") == 1 and statuses.count("duplicate") == 3
        assert owner.result(result_for(owner, second))["status"] == "already_completed"
        assert owner.status()["samples_accepted"] == 3 and owner.status()["version"] == 1
        bad = copy.deepcopy(payload); bad["loss"] += 1
        with pytest.raises(ProtocolError, match="payload changed"):
            owner.result(bad)
    finally:
        owner.close()


def test_swarm_owner_restart_and_atomic_commit_failure(tmp_path, monkeypatch):
    make_job(tmp_path, n=5, batch=5, micro=3, epochs=1)
    owner = Owner(tmp_path)
    register(owner)
    first, last = lease(owner)
    owner.result(result_for(owner, first))
    final = result_for(owner, last)
    original_put = owner.artifacts.put
    def crash(value):
        if value.get("version") == 1:
            original_put(value)  # Simulate a crash after writing the new checkpoint, before DB commit.
            raise OSError("injected crash")
        return original_put(value)
    monkeypatch.setattr(owner.artifacts, "put", crash)
    with pytest.raises(OSError, match="injected crash"):
        owner.result(final)
    assert owner.status()["version"] == 0 and owner.status()["samples_accepted"] == 3
    owner.close()
    owner = Owner(tmp_path)
    assert owner.result(final)["status"] == "accepted"
    before = owner.checkpoint()
    owner.close()
    owner = Owner(tmp_path)
    try:
        assert owner.result(final)["status"] == "duplicate"
        assert owner.checkpoint() == before and owner.status()["samples_accepted"] == 5
        with pytest.raises(RuntimeError, match="another owner"):
            Owner(tmp_path)
    finally:
        owner.close()


@pytest.mark.parametrize("mutation", ["checkpoint", "version", "samples", "shape", "nan", "extra", "data", "attempt"])
def test_swarm_rejects_incompatible_results(tmp_path, mutation):
    make_job(tmp_path, n=3, batch=3, micro=3, epochs=1)
    owner = Owner(tmp_path)
    try:
        register(owner)
        a = lease(owner)[0]; result = result_for(owner, a)
        if mutation == "checkpoint": result["checkpoint"] = "0" * 64
        if mutation == "version": result["base_version"] += 1
        if mutation == "samples": result["samples"] += 1
        if mutation == "shape": result["gradients"]["bias"]["shape"] = [1, 2]
        if mutation == "nan": result["gradients"]["bias"]["data"][0] = float("nan")
        if mutation == "extra": result["gradients"]["unknown"] = result["gradients"]["bias"]
        if mutation == "data": result["data"] = "0" * 64
        if mutation == "attempt": result["attempt"] = "0" * 48
        with pytest.raises((ProtocolError, ValueError)):
            owner.result(result)
        assert owner.status()["samples_accepted"] == 0 and owner.status()["version"] == 0
        assert owner.result(result_for(owner, a))["status"] == "accepted"
    finally:
        owner.close()


def test_swarm_capability_scheduling_and_heartbeat(tmp_path):
    make_job(tmp_path, n=11, batch=11, micro=3, epochs=1)
    now = [100.0]
    owner = Owner(tmp_path, lease_seconds=10, target_seconds=2, clock=lambda: now[0])
    try:
        register(owner, "tiny", memory_bytes=1)
        assert not lease(owner, "tiny")
        with pytest.raises(ProtocolError, match="dtype"):
            register(owner, "bad", dtype="float16")
        register(owner, "slow", samples_per_second=0)
        slow = lease(owner, "slow"); assert len(slow) == 1
        now[0] += 9
        assert owner.heartbeat({"worker_id": "slow", "attempts": [slow[0]["attempt"]]})["renewed"]
        now[0] += 2
        register(owner, "fast", samples_per_second=100)
        fast = lease(owner, "fast"); assert len(fast) == 3
        assert slow[0]["chunk_id"] not in [a["chunk_id"] for a in fast]
        assert lease(owner, "fast") == fast
    finally:
        owner.close()


@pytest.mark.parametrize("devices", [("cpu", "cpu"), ("cpu", "vulkan:0")])
def test_native_swarm_matches_pytorch(tmp_path, devices):
    if "vulkan:0" in devices and os.environ.get("MUNET_TEST_VULKAN") != "1":
        pytest.skip("set MUNET_TEST_VULKAN=1 to require Vulkan")
    import torch
    job = tmp_path / "job"
    model, x, y = make_job(job, n=23, batch=11, micro=4, epochs=3, vulkan="vulkan:0" in devices, deep=True)
    reference = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))
    reference.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in model.state_dict().items()})
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
    owner = Owner(job, max_bundle=1)
    try:
        for r in owner.job["rounds"]:
            ids = sum([owner.artifacts.get(c["data"])["sample_ids"] for c in r["chunks"]], [])
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(reference(torch.from_numpy(x[ids])), torch.from_numpy(y[ids]))
            loss.backward(); optimizer.step()
        with serving(owner) as url:
            workers = [subprocess.Popen(node_command(url, tmp_path / f"node{i}", d), env=node_env(),
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                       for i, d in enumerate(devices)]
            try:
                for w in workers:
                    output, _ = w.communicate(timeout=60)
                    print(output)
                    assert w.returncode == 0, output
            finally:
                for w in workers:
                    if w.poll() is None:
                        w.kill(); w.wait()
        owner.load_into(model)
        for k, v in model.state_dict().items():
            np.testing.assert_allclose(v, reference.state_dict()[k].numpy(), rtol=2e-5, atol=2e-6)
        assert owner.status()["samples_accepted"] == 69 and owner.status()["version"] == 9
        # Both replicas actually contributed, not merely registered.
        assert owner.db.execute("SELECT COUNT(DISTINCT a.worker) FROM attempts a JOIN chunks c ON c.winner=a.id").fetchone()[0] == 2
        assert owner.db.execute("""SELECT COUNT(*) FROM (SELECT c.round FROM chunks c JOIN attempts a
            ON c.winner=a.id GROUP BY c.round HAVING COUNT(DISTINCT a.worker)=2)""").fetchone()[0] > 0
    finally:
        owner.close()


def test_native_offline_outbox_owner_restart_and_lost_ack(tmp_path):
    job, state = tmp_path / "job", tmp_path / "node"
    make_job(job, n=3, batch=3, micro=3, epochs=1)
    owner = Owner(job)
    with serving(owner) as url:
        register(owner, "offline")
        assignment = lease(owner, "offline")[0]
    port = int(url.rsplit(":", 1)[1])
    # Durable state at the end of prefetch, immediately before losing the owner.
    atomic_write(state / "worker.json", encode({"owner": url, "worker_id": "offline", "job_id": owner.job["id"]}))
    atomic_write(state / "assignments" / (assignment["attempt"] + ".json"), encode(assignment))
    for key in ("program", "checkpoint", "data"):
        atomic_write(state / "cache" / assignment[key], owner.artifacts.path(assignment[key]).read_bytes())
    worker = subprocess.run(node_command(url, state, once=True), env=node_env(), capture_output=True, text=True, timeout=30)
    assert worker.returncode == 2, worker.stdout + worker.stderr
    assert len(list((state / "outbox").glob("*.json"))) == 1
    assert owner.status()["version"] == 0
    owner.close()
    owner = Owner(job)
    original_result = owner.result
    def lose_ack(message):
        original_result(message)
        raise ConnectionResetError("injected lost acknowledgement after COMMIT")
    owner.result = lose_ack
    with serving(owner, port) as url:
        worker = subprocess.run(node_command(url, state, once=True), env=node_env(), capture_output=True, text=True, timeout=30)
        assert worker.returncode == 2 and owner.status()["version"] == 1
        assert len(list((state / "outbox").glob("*.json"))) == 1
    owner.close()
    owner = Owner(job)
    try:
        with serving(owner, port) as url:
            worker = subprocess.run(node_command(url, state, once=True), env=node_env(), capture_output=True, text=True, timeout=30)
            assert worker.returncode == 0, worker.stdout + worker.stderr
            assert "duplicate" in worker.stdout and not list((state / "outbox").glob("*.json"))
        assert owner.status()["samples_accepted"] == 3 and owner.status()["version"] == 1
    finally:
        owner.close()


def test_native_resumes_partial_artifact_and_checks_auth(tmp_path):
    job, state = tmp_path / "job", tmp_path / "node"
    make_job(job, n=3, batch=3, micro=3, epochs=1)
    owner = Owner(job)
    key = owner.job["rounds"][0]["chunks"][0]["program"]
    raw = owner.artifacts.path(key).read_bytes()
    atomic_write(state / "cache" / (key + ".part"), raw[:117])
    try:
        with serving(owner) as url:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(url + "/v1/status")
            assert error.value.code == 401
            request = urllib.request.Request(url + "/v1/artifacts/" + key,
                                            headers={"Authorization": "Bearer " + TOKEN, "Range": "bytes=117-"})
            with urllib.request.urlopen(request) as response:
                assert response.status == 206 and response.read() == raw[117:]
            worker = subprocess.run(node_command(url, state), env=node_env(), capture_output=True, text=True, timeout=30)
            assert worker.returncode == 0, worker.stdout + worker.stderr
        assert (state / "cache" / key).read_bytes() == raw
        assert not (state / "cache" / (key + ".part")).exists()
        assert owner.status()["state"] == "completed"
    finally:
        owner.close()


def test_native_killed_worker_lease_is_retried(tmp_path):
    job = tmp_path / "job"
    make_job(job, n=3, batch=3, micro=3, epochs=1)
    now = [100.0]
    owner = Owner(job, lease_seconds=10, clock=lambda: now[0])
    leased, unblock = threading.Event(), threading.Event()
    original_lease = owner.lease
    issued = []
    def interrupted_lease(message):
        reply = original_lease(message)
        if not issued and reply["assignments"]:
            issued.extend(reply["assignments"])
            leased.set()
            unblock.wait(timeout=10)
        return reply
    owner.lease = interrupted_lease
    try:
        with serving(owner) as url:
            worker = subprocess.Popen(node_command(url, tmp_path / "doomed"), env=node_env(),
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                assert leased.wait(timeout=10), "worker never requested a lease"
                worker.kill(); worker.communicate(timeout=5)
                assert owner.status()["version"] == 0
                now[0] += 11
                unblock.set()
                survivor = subprocess.run(node_command(url, tmp_path / "survivor"), env=node_env(),
                                          capture_output=True, text=True, timeout=30)
                assert survivor.returncode == 0, survivor.stdout + survivor.stderr
                assert owner.status()["version"] == 1 and owner.status()["samples_accepted"] == 3
                assert owner.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2
            finally:
                unblock.set()
                if worker.poll() is None:
                    worker.kill(); worker.wait()
    finally:
        owner.close()


def test_native_compiler_mismatch_does_not_contribute(tmp_path):
    job, state = tmp_path / "job", tmp_path / "node"
    make_job(job, n=3, batch=3, micro=3, epochs=1)
    manifest = json.loads((job / "job.json").read_text())
    artifacts = Artifacts(job / "artifacts")
    chunk = manifest["rounds"][0]["chunks"][0]
    program = artifacts.get(chunk["program"])
    program["shader_hashes"][0] = "0" * 64
    chunk["program"] = artifacts.put(program)
    atomic_write(job / "job.json", encode(manifest))
    owner = Owner(job)
    try:
        with serving(owner) as url:
            worker = subprocess.run(node_command(url, state, once=True), env=node_env(), capture_output=True, text=True, timeout=30)
            assert worker.returncode == 1 and "compiler shader ABI mismatch" in worker.stderr
            assert len(list((state / "failed").glob("*.json"))) == 1
            assert owner.status()["version"] == 0 and owner.status()["samples_accepted"] == 0
    finally:
        owner.close()


def test_native_tls_ca_and_hostname_verification(tmp_path):
    job = tmp_path / "job"
    make_job(job, n=3, batch=3, micro=3, epochs=1)
    cert, key = tmp_path / "server.crt", tmp_path / "server.key"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-keyout", str(key), "-out", str(cert), "-subj", "/CN=MuNet test",
                    "-addext", "subjectAltName=IP:127.0.0.1"], check=True, capture_output=True)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    owner = Owner(job)
    try:
        with serving(owner, context=context) as url:
            untrusted = subprocess.run(node_command(url, tmp_path / "untrusted", once=True), env=node_env(),
                                       capture_output=True, text=True, timeout=20)
            assert untrusted.returncode == 2 and "SSL" in untrusted.stderr
            env = dict(node_env(), MUNET_CA_BUNDLE=str(cert))
            wrong_host = subprocess.run(node_command(url.replace("127.0.0.1", "localhost"), tmp_path / "wrong-host", once=True),
                                        env=env, capture_output=True, text=True, timeout=20)
            assert wrong_host.returncode == 2 and "SSL" in wrong_host.stderr
            assert owner.db.execute("SELECT COUNT(*) FROM workers").fetchone()[0] == 0
            trusted = subprocess.run(node_command(url, tmp_path / "trusted"), env=env,
                                     capture_output=True, text=True, timeout=30)
            assert trusted.returncode == 0, trusted.stdout + trusted.stderr
            assert owner.status()["state"] == "completed"
    finally:
        owner.close()
