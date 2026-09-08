"""Exercise the exact C++ loader/runtime through a thin internal binding.

tools/smoke_sdk.py separately compiles a Python-free installed-SDK consumer.
"""
import json
import zipfile
import numpy as np
import pytest
import munet as mu
from munet import _native


def rewrite(path, modify):
    with zipfile.ZipFile(path) as z: entries = {n: z.read(n) for n in z.namelist()}
    manifest = json.loads(entries["manifest.json"])
    modify(manifest, entries)
    entries["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        for name, data in entries.items(): z.writestr(name, data)


def test_cpp_multioutput_dead_input_and_repeatability(tmp_path, device):
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    unused = np.zeros((1,), np.float32)
    bias = mu.Parameter(np.array([.5, 1, 2], np.float32))
    def forward(x, unused): return {"logits": x + bias, "pooled": (x.mean(), "metadata")}
    path = mu.export(forward, tmp_path / "model.mnet", (x, unused), input_names=["features", "unused"],
                     output_names=["logits", "mean"], include_vulkan=device != "cpu")
    cpp = _native.InferenceModel(str(path), device)
    assert cpp.inputs() == [{"name": "features", "shape": [2, 3]}, {"name": "unused", "shape": [1]}]
    assert cpp.outputs()[1] == {"name": "mean", "shape": []}
    outputs = cpp.run([x, unused])
    expected = mu.load(path, device=device).predict(x, unused)
    np.testing.assert_allclose(outputs[0], expected["logits"], rtol=1e-5)
    np.testing.assert_allclose(outputs[1], expected["pooled"][0], rtol=1e-5)
    cpp.run([x + 1, unused])
    np.testing.assert_array_equal(outputs[0], expected["logits"])
    assert cpp.stats()["runs"] == 2
    with pytest.raises(RuntimeError, match="expected 2 input"): cpp.run([x])
    with pytest.raises(RuntimeError, match="features.*expected shape"): cpp.run([x.reshape(3, 2), unused])


def test_cpp_loads_existing_save_and_rejects_training(tmp_path):
    x = np.ones((1, 2), np.float32)
    model = mu.nn.Linear(2, 1)
    f = model.eval().compile(device="cpu")
    expected = f.predict(x)
    path = tmp_path / "old.mnet"
    f.save(path)
    # Legacy v1 files have no names, fusion setting or Vulkan deployment metadata.
    rewrite(path, lambda m, e: [m.pop(k) for k in ("input_names", "output_names", "fuse")])
    np.testing.assert_allclose(_native.InferenceModel(str(path)).run([x])[0], expected)
    step = mu.train_step(model, mu.optim.SGD(model.parameters()), mu.nn.MSELoss())
    step.prepare(x, np.ones((1, 1), np.float32)).save(path)
    with pytest.raises(RuntimeError, match="training/state-update"): _native.InferenceModel(str(path))


@pytest.mark.parametrize("mutation,match", [
    (lambda m, e: m.update(dtype="float16"), "dtype"),
    (lambda m, e: m.update(outputs=[1000000]), "invalid value"),
    (lambda m, e: m.update(feed_indices=[8]), "index out of range"),
    (lambda m, e: m.update(input_names=["a", "a"]), "input names"),
    (lambda m, e: m["nodes"][-1].update(shape=[4, 4]), "inferred shape"),
    (lambda m, e: m["nodes"][-1].update(inputs=[-1]), "topologically"),
    (lambda m, e: e.update({"unexpected.txt": b"unexpected"}), "unexpected entries"),
    (lambda m, e: e.update({"tensors/1.npy": e["tensors/1.npy"][:-1]}), "payload length"),
    (lambda m, e: m.update(output_tree=["tensor", 50]), "output tensor index"),
])
def test_cpp_rejects_malformed_artifacts(tmp_path, mutation, match):
    path = mu.export(lambda x: x + 1, tmp_path / "bad.mnet", np.ones((2,), np.float32))
    rewrite(path, mutation)
    with pytest.raises(RuntimeError, match=match): _native.InferenceModel(str(path))


def test_cpp_archive_crc_memory_limit_and_missing_shaders(tmp_path):
    path = mu.export(lambda x: x + 1, tmp_path / "model.mnet", np.ones((2,), np.float32))
    with pytest.raises(RuntimeError, match="oversized model"): _native.InferenceModel(str(path), "cpu", 32)
    with pytest.raises(RuntimeError, match="include_vulkan=True"): _native.InferenceModel(str(path), "vulkan")
    data = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as z: at = z.getinfo("tensors/1.npy").header_offset
    # Corrupt tensor bytes while leaving the directory checksum untouched.
    size_name = int.from_bytes(data[at + 26:at + 28], "little")
    size_extra = int.from_bytes(data[at + 28:at + 30], "little")
    data[at + 30 + size_name + size_extra + 20] ^= 1
    path.write_bytes(data)
    with pytest.raises(RuntimeError, match="checksum mismatch"): _native.InferenceModel(str(path))


def test_cpp_arena_limit_and_zip64_directory(tmp_path):
    import struct
    path = mu.export(lambda x: x + 1, tmp_path / "model.mnet", np.ones((10000,), np.float32))
    with pytest.raises(RuntimeError, match="planned arena"):
        _native.InferenceModel(str(path), "cpu", 10000)
    data = path.read_bytes()
    end = data.rfind(b"PK\x05\x06")
    footer = bytearray(data[end:])
    count = int.from_bytes(footer[10:12], "little")
    directory_size, directory_at = struct.unpack_from("<II", footer, 12)
    record = struct.pack("<IQHHIIQQQQ", 0x06064b50, 44, 45, 45, 0, 0, count, count, directory_size, directory_at)
    locator = struct.pack("<IIQI", 0x07064b50, 0, end, 1)
    struct.pack_into("<HH", footer, 8, 65535, 65535)
    path.write_bytes(data[:end] + record + locator + footer)
    x = np.arange(10000, dtype=np.float32)
    np.testing.assert_array_equal(_native.InferenceModel(str(path)).run([x])[0], x + 1)


def test_shared_model_returns_each_callers_outputs(tmp_path, device):
    from concurrent.futures import ThreadPoolExecutor
    path = mu.export(lambda x: x * 2 + 1, tmp_path / "threaded.mnet", np.ones((2,), np.float32),
                     include_vulkan=device != "cpu")
    cpp = _native.InferenceModel(str(path), device)
    python = mu.load(path, device=device)
    def infer(index):
        x = np.full((2,), index, np.float32)
        return cpp.run([x])[0], python.predict(x)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(infer, range(12)))
    for index, outputs in enumerate(results):
        for output in outputs: np.testing.assert_array_equal(output, np.full((2,), 2 * index + 1, np.float32))


def test_embedded_vulkan_load_does_not_invoke_shader_compiler(tmp_path, device, monkeypatch):
    if device == "cpu": pytest.skip("requires Vulkan backend")
    path = mu.export(lambda x: (x + 1).relu(), tmp_path / "gpu.mnet", np.ones((2,), np.float32), include_vulkan=True)
    def forbidden(*a, **k): raise AssertionError("deployment attempted shader compilation")
    monkeypatch.setattr(mu.core, "_compile_shaders", forbidden)
    x = np.array([-2, 3], np.float32)
    np.testing.assert_array_equal(mu.load(path, device=device).predict(x), [0, 4])
    np.testing.assert_array_equal(_native.InferenceModel(str(path), device).run([x])[0], [0, 4])
