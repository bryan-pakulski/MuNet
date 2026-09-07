import io
import json
import zipfile
import numpy as np
import pytest
import munet as mu
from munet.interop import from_onnx, to_onnx, from_torch, UnsupportedOperatorError


def test_native_onnx_native_roundtrip_and_onnxruntime(device, tmp_path):
    import onnxruntime as ort
    ort.disable_telemetry_events()
    rng = np.random.default_rng(33)
    model = mu.nn.Sequential(mu.nn.Linear(4, 6, rng=rng), mu.nn.Sigmoid(), mu.nn.Linear(6, 2, rng=rng))
    x = rng.normal(size=(5, 4)).astype(np.float32)
    inference = mu.compile(model, device=device)
    expected = inference(x).numpy()
    native_path, onnx_path, returned = tmp_path / "one.mnet", tmp_path / "one.onnx", tmp_path / "two.mnet"
    mu.save(inference, native_path)
    native = mu.load(native_path, device=device)
    to_onnx(native, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: x})[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    imported = from_onnx(onnx_path, device=device)
    mu.save(imported, returned)
    np.testing.assert_allclose(mu.load(returned, device=device)(x).numpy(), expected, rtol=2e-5, atol=2e-5)


def test_reduction_broadcast_and_shape_roundtrip(device, tmp_path):
    import onnxruntime as ort
    ort.disable_telemetry_events()
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 12
    program = mu.compile(lambda a: ((a + 1).mean((0, 2))).reshape(3, 1).T, device=device)
    expected = program(x).numpy()
    path = tmp_path / "reduce.onnx"
    to_onnx(program, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(session.run(None, {session.get_inputs()[0].name:x})[0], expected, atol=2e-6)
    np.testing.assert_allclose(from_onnx(path, device=device)(x).numpy(), expected, atol=2e-6)


def test_training_checkpoint_resumes_exact_sgd_state(device, tmp_path):
    p = mu.Parameter(np.asarray([0.4, 0.8], np.float32))
    optimizer = mu.optim.SGD([p], lr=0.05)
    @mu.compile(device=device)
    def step(x):
        optimizer.zero_grad(); loss = (p - x).square().mean()
        loss.backward(); optimizer.step(); return loss
    x = np.asarray([1.0, -1.0], np.float32)
    step(x)
    path = tmp_path / "checkpoint.mnet"
    mu.save(step, path)
    loaded = mu.load(path, device=device)
    for _ in range(3):
        np.testing.assert_allclose(loaded(x).item(), step(x).item(), rtol=2e-6, atol=2e-6)
    with pytest.raises(UnsupportedOperatorError, match="inference graph"):
        to_onnx(step, tmp_path / "training.onnx")


def test_from_torch_modern_export(device):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxscript")
    torch.manual_seed(10)
    model = torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.ReLU(), torch.nn.Linear(7, 3)).eval()
    x = torch.randn(2, 4)
    imported = from_torch(model, (x,), device=device)
    np.testing.assert_allclose(imported(x.numpy()).numpy(), model(x).detach().numpy(), rtol=2e-5, atol=2e-5)


def test_unsupported_onnx_is_explicit_and_never_executes_fallback(tmp_path):
    import onnx
    from onnx import helper as h, TensorProto as T
    graph = h.make_graph([h.make_node("Softmax", ["x"], ["y"], name="attention_softmax")], "unsupported",
                        [h.make_tensor_value_info("x", T.FLOAT, [2, 3])], [h.make_tensor_value_info("y", T.FLOAT, [2, 3])])
    path = tmp_path / "unsupported.onnx"
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=8), path)
    with pytest.raises(UnsupportedOperatorError, match="attention_softmax"):
        from_onnx(path, device="cpu")
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    graph.node[0].op_type = "Relu"
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=8), path)
    with pytest.raises(UnsupportedOperatorError, match="static positive"):
        from_onnx(path, device="cpu")


@pytest.mark.parametrize("mutation", ["version", "shape", "cycle", "extra", "duplicate"])
def test_malformed_native_files_fail_closed(tmp_path, mutation):
    program = mu.compile(lambda x: (x + 1).square(), device="cpu")
    program(np.ones((2, 2), np.float32))
    source, bad = tmp_path / "ok.mnet", tmp_path / "bad.mnet"
    mu.save(program, source)
    with zipfile.ZipFile(source) as z: files = {name:z.read(name) for name in z.namelist()}
    manifest = json.loads(files["manifest.json"])
    if mutation == "version": manifest["version"] = 999
    if mutation == "shape": manifest["nodes"][-1]["shape"] = [3, 3]
    if mutation == "cycle": manifest["nodes"][-1]["inputs"] = [len(manifest["nodes"])-1] * 2
    if mutation == "extra": files["unexpected.py"] = b"print('must never execute')"
    files["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(bad, "w") as z:
        for name, data in files.items(): z.writestr(name, data)
        if mutation == "duplicate":
            with pytest.warns(UserWarning): z.writestr("manifest.json", files["manifest.json"])
    with pytest.raises(ValueError): mu.load(bad, device="cpu")
