# Serialization & Model Loading

## 1) Artifact types and intended loaders (contract clarity)

MuNet now has two explicit artifact APIs:

1. **Deploy artifact** (runtime-only, lean)
   - Save: `munet.save_deploy(model, "model.npz")`
   - Backward alias: `munet.save(...)`
   - Load (Python): `munet.load_deploy(...)` or `munet.load_for_inference(...)`
   - Load (C++): `munet::inference::load_serialized(...)`
   - Contract goal: minimal runtime structure + tensor state for inference.

2. **Checkpoint artifact** (training/inference round-trip)
   - Save: `munet.save_checkpoint(model, "checkpoint.npz")`
   - Load: `munet.load_checkpoint(...)`
   - Backward alias: `munet.load(...)`
   - Contract goal: support model reconstruction (including custom Python classes), weights restore, and continued training.

## 2) Metadata contract

Shared fields:

- `__format_name__ = "munet_model"`
- `__format_revision__ = 1`
- `__format_version__ = "munet_model_v1"`
- `__producer__ = "munet"`
- `__contains_training_state__ = false` (optimizer/scaler/etc. are out of scope for this artifact)
- `__device_policy__ = "caller_specified"`
- `__dtype_policy__ = "per_tensor"`
- `__tensor_names__ = [...]`

Deploy-only values:

- `__artifact_kind__ = "deploy_model"`
- `__artifact_scope__ = "runtime_only"`
- `__default_load_mode__ = "eval"`
- `__recommended_loader__ = "load_for_inference"`
- `__compile_contract_policy__ = "external"`

Checkpoint-only values:

- `__artifact_kind__ = "training_checkpoint"`
- `__artifact_scope__ = "training+inference"`
- `__default_load_mode__ = "train"`
- `__recommended_loader__ = "load_checkpoint"`
- `__compile_contract_policy__ = "dynamic"`

Custom-checkpoint hybrid payload marker:

- `__format__ = "munet_hybrid_v1"` (+ `__shell__`) when class/source fallback payload is embedded.

## 3) Behavioral guarantees matrix

| Flow | Artifact kind | Supported? | Notes |
|---|---|---|---|
| Python full load (`load_deploy`) | Deploy | ✅ | Built-in module configs only; strict runtime metadata validation. |
| Python full load (`load_checkpoint`) | Checkpoint | ✅ | Supports built-ins and custom classes (custom fallback depends on trust policy). |
| Python weights-only (`load_weights`) | Deploy / Checkpoint | ✅ | Existing in-code model definition required. |
| Python inference normalize (`load_for_inference`) | Deploy | ✅ | Enforces eval mode and optional device move. |
| C++ `inference::load_serialized` | Deploy | ✅ | Strict deploy contract. |
| C++ `inference::load_serialized` on checkpoint/custom | Checkpoint/custom | ❌ | Rejected by design (wrong artifact kind / unsupported custom type). |
| ONNX compile output -> deploy load | Deploy | ✅ | Treated as deploy artifact path. |

## 4) Safety policy (trusted vs untrusted)

Checkpoint custom-class reconstruction may require executing embedded source.

- `munet.load_checkpoint(..., trusted=False)` (**default**) does **not** execute embedded source fallback.
- `munet.load_checkpoint(..., trusted=True)` allows source execution fallback for custom classes when import resolution fails.

Use `trusted=True` only for artifacts from trusted producers.

## 5) Compatibility expectations

- Revision `1` artifacts are expected to load in current builds.
- Legacy tag `munet_model_v1` remains accepted.
- Newer future revisions must fail clearly rather than partial-load.

## 6) Practical usage

- Use `save_deploy` + `load_for_inference` for production runtime artifacts.
- Use `save_checkpoint` + `load_checkpoint` for training workflows, custom classes, and iterative experimentation.
- Use `load_weights` when architecture remains in code and only tensor state should be restored.
