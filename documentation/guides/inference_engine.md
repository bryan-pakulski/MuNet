# Inference Engine Guide

## Goal

Provide a stable deploy runtime with shape contracts and warmup/compile behavior.

## Compile contracts

Use `compile(...)` with optional shape contracts:

- `expected_input_shape`
- `expected_output_shape`

Use `-1` for dynamic dims:

- Dynamic batch MLP: `[-1, 4] -> [-1, 2]`
- Dynamic resolution conv: `[-1, 3, -1, -1] -> [-1, 2, -1, -1]`

## Strict vs non-strict checks

- strict mode validates compiled/expected shapes at runtime.
- non-strict mode allows mismatches (for experimentation).
