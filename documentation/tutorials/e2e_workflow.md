# Tutorial: End-to-end Training → Save → Load → Inference

See runnable demo:

- `demos/inference/e2e_train_save_load_infer.py`

## Workflow

1. Train small model in Python with `munet.nn` + `munet.optim`.
2. Serialize model with `munet.save`.
3. Reconstruct with `munet.load`.
4. Initialize `munet.inference.Engine` and `load` model.
5. `compile` with wildcard shape contract (`-1` dynamic dims).
6. `run` inference for different batch sizes / resolutions.
