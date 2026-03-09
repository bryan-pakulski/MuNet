# Common Design Patterns

## 1) Training model, then export for inference

- train in `munet.nn`
- save with `munet.save`
- load into inference runtime
- compile with shape contract and run

## 2) Dynamic batch serving

- compile with `expected_input_shape=[-1, feature_dim]`
- keep strict shape checks on to validate non-dynamic dimensions

## 3) Dynamic resolution vision serving

- compile with `[-1, C, -1, -1]`
- verify output contract includes dynamic spatial dimensions

## 4) Weights-only updates

- keep model definition in code
- roll out parameter updates with `load_weights`
