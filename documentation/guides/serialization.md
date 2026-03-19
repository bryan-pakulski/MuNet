# Serialization & Model Loading

## Supported flows

1. **Full model reconstruction**
   - Save with `munet.save(model, "model.npz")`
   - Load with `munet.load("model.npz")`

2. **Weights-only restore**
   - `munet.load(existing_model, "model.npz")`
   - or `munet.load_weights(existing_model, "model.npz")`

## Compatibility contract

MuNet now writes explicit serialization metadata alongside the tensor payload:

- `__format_name__ = "munet_model"`
- `__format_revision__ = 1`
- `__format_version__ = "munet_model_v1"` (legacy compatibility tag)
- `__producer__ = "munet"`

Use `munet.serialization_format_info()` to inspect the supported format contract in the current build, and `munet.serialization_metadata(path)` to inspect a saved artifact before loading it.

### Compatibility expectations

- Revision `1` artifacts are expected to load in current builds.
- The legacy `munet_model_v1` tag remains accepted for compatibility with earlier checkpoints.
- Forward compatibility is **not** guaranteed across future format revisions; newer checkpoints should fail with a targeted error instead of loading partially.
- Inference-only deployment paths should save/load model structure + tensor state only; optimizer/master-weight/scaler state belongs in trainer-side checkpoint formats, not this artifact.

## Best practices

- Keep serialization tests in CI for every new module you add.
- For deployment, pair a serialized model with an inference compile contract.
- Validate serialization metadata during deployment promotion so unsupported revisions fail before runtime rollout.
