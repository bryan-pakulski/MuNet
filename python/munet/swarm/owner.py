"""One authoritative owner, durable leases, and exactly-once SGD commits."""
from contextlib import contextmanager
import fcntl
import math
from pathlib import Path
import secrets
import sqlite3
import threading
import time
import numpy as np
from .artifacts import ABI, Artifacts, decode, digest, encode, tensor


class ProtocolError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class Owner:
    def __init__(self, directory, *, lease_seconds=120, target_seconds=30,
                 max_bundle=4, clock=time.time):
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be finite and positive")
        if not math.isfinite(target_seconds) or target_seconds <= 0 or type(max_bundle) is not int or not 1 <= max_bundle <= 64:
            raise ValueError("invalid scheduler settings")
        self.root = Path(directory)
        self.job_bytes = (self.root / "job.json").read_bytes()
        self.job = decode(self.job_bytes)
        if self.job["abi"] != ABI or not self.job["rounds"]:
            raise ValueError("unsupported or empty swarm job")
        self.artifacts = Artifacts(self.root / "artifacts")
        self.clock, self.lease_seconds = clock, lease_seconds
        self.target_seconds, self.max_bundle = target_seconds, max_bundle
        self.lock = threading.RLock()
        self.file_lock = (self.root / "owner.lock").open("a+")
        try:
            fcntl.flock(self.file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file_lock.close()
            raise RuntimeError("another owner holds this job; only one owner is allowed")
        try:
            self.db = sqlite3.connect(self.root / "owner.sqlite", check_same_thread=False,
                                      isolation_level=None)
            self.db.row_factory = sqlite3.Row
            self.db.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1),
                    manifest TEXT, round INTEGER, checkpoint TEXT, accepted INTEGER);
                CREATE TABLE IF NOT EXISTS workers (id TEXT PRIMARY KEY, caps TEXT,
                    rate REAL, seen REAL);
                CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, round INTEGER,
                    ordinal INTEGER, result TEXT, winner TEXT);
                CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, chunk TEXT,
                    worker TEXT, deadline REAL, result_hash TEXT);
                CREATE INDEX IF NOT EXISTS attempts_chunk ON attempts(chunk, deadline);
                CREATE TABLE IF NOT EXISTS commits (round INTEGER PRIMARY KEY,
                    epoch INTEGER, samples INTEGER, loss REAL, checkpoint TEXT);
            """)
            with self.transaction():
                row = self.db.execute("SELECT * FROM state").fetchone()
                if row and row["manifest"] != digest(self.job_bytes):
                    raise ValueError("job manifest changed; cannot resume")
                if not row:
                    self.db.execute("INSERT INTO state VALUES(1,?,?,?,0)",
                                    (digest(self.job_bytes), 0, self.job["initial_checkpoint"]))
                    for r, info in enumerate(self.job["rounds"]):
                        for ordinal, chunk in enumerate(info["chunks"]):
                            self.db.execute("INSERT INTO chunks VALUES(?,?,?,NULL,NULL)",
                                            (chunk["id"], r, ordinal))
                self.artifacts.get(self.state()["checkpoint"])
        except Exception:
            if hasattr(self, "db"):
                self.db.close()
            self.file_lock.close()
            raise

    def close(self):
        with self.lock:
            self.db.close()
            self.file_lock.close()

    @contextmanager
    def transaction(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise

    def state(self):
        return self.db.execute("SELECT * FROM state WHERE id=1").fetchone()

    def status(self):
        with self.lock:
            s = self.state()
            commits = [dict(r) for r in self.db.execute("SELECT * FROM commits ORDER BY round")]
            done = s["round"] == len(self.job["rounds"])
            finished_epochs = self.job["epochs"] if done else self.job["rounds"][s["round"]]["epoch"]
            return {"job_id": self.job["id"], "state": "completed" if done else "running",
                    "version": s["round"], "rounds": len(self.job["rounds"]),
                    "epochs_completed": finished_epochs, "epochs": self.job["epochs"],
                    "samples_accepted": s["accepted"], "checkpoint": s["checkpoint"],
                    "commits": commits,
                    "workers": [{"id": r["id"], "last_seen": r["seen"],
                                 "samples_per_second": r["rate"], "capabilities": decode(r["caps"])}
                                for r in self.db.execute("SELECT * FROM workers ORDER BY id")]}

    def checkpoint(self):
        with self.lock:
            return self.artifacts.get(self.state()["checkpoint"])

    def load_into(self, model):
        model.load_state_dict({name: np.asarray(t["data"], dtype=np.float32).reshape(t["shape"])
                               for name, t in self.checkpoint()["parameters"].items()})

    def register(self, message):
        ident, caps = message.get("worker_id"), message.get("capabilities", {})
        if not isinstance(ident, str) or not 1 <= len(ident) <= 128:
            raise ProtocolError("invalid worker identity")
        if caps.get("abi") != ABI or caps.get("backend") not in ("cpu", "vulkan") or caps.get("dtype") != "float32":
            raise ProtocolError("unsupported worker runtime/backend/dtype")
        for key in ("memory_bytes", "max_buffer_bytes", "max_batch"):
            if type(caps.get(key)) is not int or not 0 < caps[key] <= 2**60:
                raise ProtocolError(f"invalid capability: {key}")
        rate = caps.get("samples_per_second", 0)
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 <= rate <= 1e12:
            raise ProtocolError("invalid throughput estimate")
        with self.transaction():
            self.db.execute("INSERT INTO workers VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET caps=excluded.caps,seen=excluded.seen",
                            (ident, encode(caps).decode(), rate, self.clock()))
        return {"abi": ABI, "job_id": self.job["id"], "lease_seconds": self.lease_seconds}

    def assignment(self, attempt, chunk, state):
        return {"abi": ABI, "job_id": self.job["id"], "attempt": attempt["id"],
                "worker_id": attempt["worker"], "chunk_id": chunk["id"],
                "base_version": state["round"], "checkpoint": state["checkpoint"],
                "program": chunk["program"], "data": chunk["data"], "samples": chunk["samples"],
                "deadline": attempt["deadline"]}

    def lease(self, message):
        ident = message.get("worker_id")
        with self.transaction():
            worker = self.db.execute("SELECT * FROM workers WHERE id=?", (ident,)).fetchone()
            if not worker:
                raise ProtocolError("register worker first", 409)
            self.db.execute("UPDATE workers SET seen=? WHERE id=?", (self.clock(), ident))
            s = self.state()
            if s["round"] == len(self.job["rounds"]):
                return {"state": "completed", "assignments": []}
            chunks = self.job["rounds"][s["round"]]["chunks"]
            caps = decode(worker["caps"])
            eligible = [c for c in chunks if c["samples"] <= caps["max_batch"]
                        and c["arena_bytes"] <= caps["max_buffer_bytes"]
                        and c["arena_bytes"] * 2 <= caps["memory_bytes"]
                        and (caps["backend"] == "cpu" or c["vulkan"])]
            if not eligible:
                return {"state": "waiting", "reason": "no profile fits worker capabilities", "assignments": []}
            # Repeating a lease request returns still-active assignments, not extra work.
            active = list(self.db.execute("""SELECT a.* FROM attempts a JOIN chunks c ON c.id=a.chunk
                WHERE a.worker=? AND a.deadline>? AND c.round=? AND c.result IS NULL""",
                                          (ident, self.clock(), s["round"])))
            by_id = {c["id"]: c for c in chunks}
            if active:
                return {"state": "work", "assignments": [self.assignment(a, by_id[a["chunk"]], s) for a in active]}
            count = max(1, min(self.max_bundle, math.ceil(worker["rate"] * self.target_seconds /
                                                        max(c["samples"] for c in eligible))))
            work = []
            for c in eligible:
                row = self.db.execute("SELECT result FROM chunks WHERE id=?", (c["id"],)).fetchone()
                live = self.db.execute("SELECT 1 FROM attempts WHERE chunk=? AND deadline>? LIMIT 1",
                                       (c["id"], self.clock())).fetchone()
                if row["result"] is not None or live:
                    continue
                a = {"id": secrets.token_hex(24), "worker": ident, "deadline": self.clock() + self.lease_seconds}
                self.db.execute("INSERT INTO attempts VALUES(?,?,?,?,NULL)",
                                (a["id"], c["id"], ident, a["deadline"]))
                work.append(self.assignment(a, c, s))
                if len(work) == count:
                    break
            return {"state": "work" if work else "waiting", "assignments": work}

    def heartbeat(self, message):
        with self.transaction():
            renewed = []
            for ident in message.get("attempts", [])[:64]:
                row = self.db.execute("""SELECT a.*, c.result FROM attempts a JOIN chunks c ON c.id=a.chunk
                    WHERE a.id=? AND a.worker=?""", (ident, message.get("worker_id"))).fetchone()
                # Expired attempts may still return a result but cannot revoke a replacement lease.
                if row and row["result"] is None and row["deadline"] > self.clock():
                    self.db.execute("UPDATE attempts SET deadline=? WHERE id=?", (self.clock() + self.lease_seconds, ident))
                    renewed.append(ident)
            return {"renewed": renewed}

    def result(self, message):
        # JSON hash makes a retry of one attempt immutable, including after owner restart.
        result_hash = digest(encode(message))
        with self.transaction():
            a = self.db.execute("""SELECT a.*, c.round, c.ordinal, c.result FROM attempts a
                JOIN chunks c ON c.id=a.chunk WHERE a.id=?""", (message.get("attempt"),)).fetchone()
            if not a or a["worker"] != message.get("worker_id") or a["chunk"] != message.get("chunk_id"):
                raise ProtocolError("unknown attempt or worker", 409)
            if message.get("abi") != ABI or message.get("job_id") != self.job["id"] or message.get("base_version") != a["round"]:
                raise ProtocolError("wrong job or checkpoint version", 409)
            if a["result_hash"]:
                if a["result_hash"] != result_hash:
                    raise ProtocolError("attempt payload changed after acceptance", 409)
                return {"status": "duplicate"}
            if a["result"] is not None:
                return {"status": "already_completed"}
            s = self.state()
            if a["round"] != s["round"] or message.get("checkpoint") != s["checkpoint"]:
                raise ProtocolError("stale checkpoint", 409)
            c = self.job["rounds"][s["round"]]["chunks"][a["ordinal"]]
            if message.get("samples") != c["samples"] or message.get("data") != c["data"] or message.get("program") != c["program"]:
                raise ProtocolError("result does not match assigned samples/program", 409)
            loss, duration = message.get("loss"), message.get("seconds")
            if not isinstance(loss, (float, int)) or not math.isfinite(loss) or abs(loss) > float(np.finfo(np.float32).max):
                raise ProtocolError("non-finite loss")
            if not isinstance(duration, (float, int)) or not math.isfinite(duration) or not 0 < duration < 1e9:
                raise ProtocolError("invalid compute duration")
            checkpoint = self.artifacts.get(s["checkpoint"])
            gradients = message.get("gradients")
            if not isinstance(gradients, dict) or gradients.keys() != checkpoint["parameters"].keys():
                raise ProtocolError("gradient parameter names differ")
            for name, p in checkpoint["parameters"].items():
                g = gradients[name]
                if not isinstance(g, dict) or g.get("shape") != p["shape"] or not isinstance(g.get("data"), list):
                    raise ProtocolError(f"gradient shape mismatch: {name}")
                try:
                    array = np.asarray(g["data"], dtype=np.float32)
                except (ValueError, TypeError, OverflowError):
                    raise ProtocolError(f"invalid gradient: {name}")
                if array.ndim != 1 or array.size != len(p["data"]) or not np.isfinite(array).all():
                    raise ProtocolError(f"invalid gradient size/value: {name}")
            key = self.artifacts.put(message)  # fsynced before a DB pointer can refer to it
            self.db.execute("UPDATE chunks SET result=?,winner=? WHERE id=?", (key, a["id"], c["id"]))
            self.db.execute("UPDATE attempts SET result_hash=? WHERE id=?", (result_hash, a["id"]))
            self.db.execute("UPDATE state SET accepted=accepted+? WHERE id=1", (c["samples"],))
            rate = min(c["samples"] / duration, 1e12)
            self.db.execute("UPDATE workers SET rate=CASE WHEN rate=0 THEN ? ELSE 0.8*rate+0.2*? END,seen=? WHERE id=?",
                            (rate, rate, self.clock(), a["worker"]))
            rows = list(self.db.execute("SELECT result FROM chunks WHERE round=? ORDER BY ordinal", (s["round"],)))
            if all(r["result"] for r in rows):
                info = self.job["rounds"][s["round"]]
                results = [self.artifacts.get(r["result"]) for r in rows]
                updated = {}
                for name, p in checkpoint["parameters"].items():
                    total = np.zeros(len(p["data"]), dtype=np.float64)
                    for r in results:
                        total += np.asarray(r["gradients"][name]["data"], dtype=np.float64) * r["samples"]
                    mean = (total / info["samples"]).astype(np.float32)
                    values = np.asarray(p["data"], dtype=np.float32) - np.float32(self.job["lr"]) * mean
                    if not np.isfinite(values).all():
                        raise ProtocolError("SGD update is non-finite; result was not committed", 422)
                    updated[name] = tensor(values.reshape(p["shape"]))
                new_key = self.artifacts.put({"abi": ABI, "version": s["round"] + 1, "parameters": updated})
                mean_loss = sum(r["loss"] * (r["samples"] / info["samples"]) for r in results)
                self.db.execute("INSERT INTO commits VALUES(?,?,?,?,?)",
                                (s["round"], info["epoch"], info["samples"], mean_loss, new_key))
                self.db.execute("UPDATE state SET round=round+1,checkpoint=? WHERE id=1", (new_key,))
            return {"status": "accepted", "version": self.state()["round"]}
