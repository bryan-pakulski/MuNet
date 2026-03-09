# Serialization & Model Loading

## Supported flows

1. **Full model reconstruction**
   - Save with `munet.save(model, "model.npz")`
   - Load with `munet.load("model.npz")`

2. **Weights-only restore**
   - `munet.load(existing_model, "model.npz")`
   - or `munet.load_weights(existing_model, "model.npz")`

## Best practices

- Keep serialization tests in CI for every new module you add.
- For deployment, pair serialized model with an inference compile contract.
