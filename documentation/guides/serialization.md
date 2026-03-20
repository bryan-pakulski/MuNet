# Serialization & Model Loading

## Supported flows

1. **Full model reconstruction**
   - Save with `munet.save(model, "model.npz")`
   - Load with `munet.load("model.npz")`

2. **Weights-only restore**
   - `munet.load(existing_model, "model.npz")`
   - or `munet.load_weights(existing_model, "model.npz")`

3. **Deploy-first inference loading**
   - `munet.load_for_inference("model.npz", device=...)`
   - `munet.load_weights_for_inference(existing_model, "model.npz", device=...)`
   - `munet.inference.load_serialized(...)` / `munet.inference.load_weights_serialized(...)`

## Compatibility contract

MuNet writes explicit deploy-oriented serialization metadata alongside the tensor payload:

- `__format_name__ = "munet_model"`
- `__format_revision__ = 1`
- `__format_version__ = "munet_model_v1"` (legacy compatibility tag)
- `__producer__ = "munet"`
- `__artifact_kind__ = "deploy_model"`
- `__artifact_scope__ = "runtime_only"`
- `__default_load_mode__ = "eval"`
- `__contains_training_state__ = false`
- `__device_policy__ = "caller_specified"`
- `__dtype_policy__ = "per_tensor"`
- `__recommended_loader__ = "load_for_inference"`
- `__compile_contract_policy__ = "external"`
- `__tensor_names__ = [...]`

Use `munet.serialization_format_info()` to inspect the supported format contract in the current build, and `munet.serialization_metadata(path)` to inspect a saved artifact before loading it.

### Compatibility expectations

- Revision `1` artifacts are expected to load in current builds.
- The legacy `munet_model_v1` tag remains accepted for compatibility with earlier checkpoints.
- Forward compatibility is **not** guaranteed across future format revisions; newer checkpoints should fail with a targeted error instead of loading partially.
- Inference-only deployment paths should save/load model structure + tensor state only; optimizer/master-weight/scaler state belongs in trainer-side checkpoint formats, not this artifact.

## Deploy behavior

- `munet.save(...)` produces a **deploy artifact**, not a training checkpoint.
- Deploy artifacts are validated as **runtime-only** payloads: architecture config, tensor payload, and deploy metadata are allowed; training/checkpoint payload keys are rejected.
- `munet.load_for_inference(...)` and `munet.inference.load_serialized(...)` normalize the loaded module for deployment by:
  - validating the deploy manifest
  - optionally moving the module to the requested device
  - forcing `eval()` before runtime use
- Shape contracts and warm/compile behavior stay external to the artifact (`compile_contract_policy = "external"`), so deployment can choose fixed or dynamic contracts at `Engine.compile(...)` time instead of inheriting trainer-side assumptions.
- Use plain `munet.load(...)` when you explicitly want generic reconstruction/restore semantics outside a deploy path.

## Best practices

- Keep serialization tests in CI for every new module you add.
- For deployment, pair a serialized model with an inference compile contract.
- When `munet.inference.compile_onnx(...)` lowers to a native sequential MuNet module, prefer `output_path="model.npz"` so the result crosses into the same deploy-artifact path as `munet.save(...)`.
- Validate serialization metadata during deployment promotion so unsupported revisions or non-deploy artifacts fail before runtime rollout.
