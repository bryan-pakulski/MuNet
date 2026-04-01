import pytest

try:
    import munet_nn as munet
except Exception as exc:  # pragma: no cover
    pytest.skip(f"munet import unavailable: {exc}", allow_module_level=True)


def test_tensor_training_surface_exposed():
    required = {
        "backward",
        "grad",
        "has_grad",
        "zero_grad",
        "register_gradient_hook",
        "all_reduce",
        "step",
        "clone",
        "detach",
        "fill_",
        "uniform_",
        "to",
        "to_options",
        "layer_norm",
        "batch_norm",
        "mse_loss",
        "cross_entropy",
    }
    missing = sorted(name for name in required if not hasattr(munet.Tensor, name))
    assert not missing, f"Tensor binding missing training/inference methods: {missing}"


def test_module_optimizer_inference_surface_exposed():
    assert hasattr(munet.nn.Module, "train")
    assert hasattr(munet.nn.Module, "eval")
    assert hasattr(munet.nn.Module, "parameters")
    assert hasattr(munet.optim, "SGD")
    assert hasattr(munet.optim, "Adam")
    assert hasattr(munet, "inference")
    assert hasattr(munet.inference, "Engine")


def test_backend_probe_apis_exposed_and_cpu_default():
    assert hasattr(munet, "list_available_backends")
    assert hasattr(munet, "backend_status")

    backends = set(munet.list_available_backends())
    assert "cpu" in backends

    status = munet.backend_status()
    assert isinstance(status, dict)
    assert "statuses" in status
    assert "summary" in status
    assert "No accelerators loaded" in status["summary"] or status["accelerator_loaded"] is True


def test_backend_status_reason_codes_stable_tokens():
    status = munet.backend_status()
    entries = status.get("statuses", [])
    assert any(item.get("name") == "cpu" and item.get("reason_code") == "ok" for item in entries)

    valid_reason_codes = {
        "ok",
        "plugin_not_found",
        "abi_mismatch",
        "runtime_dependency_missing",
        "invalid_plugin_binary",
        "plugin_dlopen_failed",
        "not_compiled",
    }
    for item in entries:
        assert item.get("reason_code") in valid_reason_codes


def test_backend_status_reports_abi_mismatch_for_incompatible_plugin(tmp_path, monkeypatch):
    c_src = tmp_path / "plugin_bad_abi.c"
    so_path = tmp_path / "libmunet_backend_badabi.so"
    c_src.write_text(
        """
        #include <stdint.h>
        uint32_t munet_backend_plugin_abi_version(void) { return 999u; }
        const char* munet_backend_plugin_name(void) { return \"badabi\"; }
        const char* munet_backend_plugin_device_type(void) { return \"cuda\"; }
        uint64_t munet_backend_plugin_capability_flags(void) { return 0u; }
        const char* munet_backend_plugin_probe(void) { return 0; }
        """
    )

    import shutil
    import subprocess

    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("C compiler unavailable for plugin ABI mismatch test")

    subprocess.check_call([cc, "-shared", "-fPIC", str(c_src), "-o", str(so_path)])
    monkeypatch.setenv("MUNET_BACKEND_PLUGIN_PATH", str(tmp_path))

    status = munet.backend_status()
    entries = status.get("statuses", [])
    badabi_entries = [e for e in entries if e.get("name") == "badabi"]
    assert badabi_entries, "Expected loader to discover badabi plugin"
    assert badabi_entries[0].get("reason_code") == "abi_mismatch"


def test_backend_status_reports_valid_plugin_and_smoke_op(tmp_path, monkeypatch):
    c_src = tmp_path / "plugin_ok.c"
    so_path = tmp_path / "libmunet_backend_good.so"
    c_src.write_text(
        """
        #include <stdint.h>
        uint32_t munet_backend_plugin_abi_version(void) { return 1u; }
        const char* munet_backend_plugin_name(void) { return "goodplugin"; }
        const char* munet_backend_plugin_device_type(void) { return "cuda"; }
        uint64_t munet_backend_plugin_capability_flags(void) { return 7u; }
        const char* munet_backend_plugin_probe(void) { return 0; }
        """
    )

    import shutil
    import subprocess

    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("C compiler unavailable for plugin load test")

    subprocess.check_call([cc, "-shared", "-fPIC", str(c_src), "-o", str(so_path)])
    monkeypatch.setenv("MUNET_BACKEND_PLUGIN_PATH", str(tmp_path))

    backends = set(munet.list_available_backends())
    assert "goodplugin" in backends

    status = munet.backend_status()
    entries = status.get("statuses", [])
    good_entries = [e for e in entries if e.get("name") == "goodplugin"]
    assert good_entries and good_entries[0].get("reason_code") == "ok"

    # Smoke op must still execute while plugin is active.
    a = munet.ones([2, 2])
    b = munet.ones([2, 2])
    out = munet.matmul(a, b)
    assert out.shape == [2, 2]
