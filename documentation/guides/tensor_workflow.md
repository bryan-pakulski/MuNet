# Working with Tensors in MuNet (Recommended Usage)

This guide shows practical ways to create, populate, move, and inspect `munet.Tensor` objects.

## Quick answer: populate ONNX inputs

For your case:

```python
import munet
import numpy as np

# Example HWC image from PIL/OpenCV/etc. converted to float32 in [0,1]
# image_hwc shape: [1280, 1280, 3]
image_hwc = np.random.rand(1280, 1280, 3).astype(np.float32)

# Convert HWC -> NCHW and add batch dim => [1, 3, 1280, 1280]
image_nchw = np.transpose(image_hwc, (2, 0, 1))[None, ...]

# Preferred: create tensor from NumPy data
input_data = munet.from_numpy(image_nchw)

# Shape/dimension input for ONNX (int64 recommended for shape tensors)
# Example [width, height] or [height, width] based on model contract.
# Always match what your ONNX model expects.
input_size_np = np.array([[1280, 1280]], dtype=np.int64)

# Graph runtime accepts NumPy arrays too, so you can pass this directly.
# If you want a tensor object anyway, you can still create one:
input_size_tensor = munet.from_numpy(input_size_np.astype(np.float32))
```

For a multi-input ONNX graph, prefer named inputs to avoid ordering mistakes:

```python
out = model.forward({
    "image": input_data,
    "image_size": input_size_np,  # or input_size_tensor
})
```

---

## Recommended tensor creation patterns

### 1) `munet.from_numpy(...)` (best for real data)

Use this when data already exists in NumPy:

```python
x_np = np.random.randn(4, 3, 224, 224).astype(np.float32)
x = munet.from_numpy(x_np)
```

Why this is preferred:
- Shape and values are explicit.
- Easy interop with preprocessing pipelines.
- Less error-prone than constructing an uninitialized tensor then filling.

### 2) Factory helpers for initialization

Use these when you need known initialization values:

```python
z = munet.zeros([2, 3])
o = munet.ones([2, 3])
r = munet.rand([2, 3])
```

### 3) `munet.Tensor(shape, ...)` (advanced/manual path)

`munet.Tensor([..])` allocates tensor memory by shape. Treat it as **uninitialized** unless you fill/copy it yourself.

```python
t = munet.Tensor([1, 3, 1280, 1280], requires_grad=False)
t.uniform_(0.0, 1.0)  # fills with random uniform values
```

Use this style only when you intentionally control a subsequent fill operation.

---

## Copying values into an existing tensor

If you already allocated a CPU tensor and want to copy NumPy data in-place:

```python
dst = munet.Tensor([1, 3, 1280, 1280], requires_grad=False)
arr = np.random.rand(1, 3, 1280, 1280).astype(np.float32)
munet.copy_from_numpy(dst, arr)
```

Notes:
- `copy_from_numpy` target must be a CPU tensor.
- Source and destination must have the same number of elements/bytes.

---

## Device movement (CPU/GPU)

Typical workflow:

```python
cpu = munet.Device(munet.DeviceType.CPU, 0)
cuda0 = munet.Device(munet.DeviceType.CUDA, 0)

x = munet.from_numpy(np.random.randn(2, 3).astype(np.float32))  # CPU
x_gpu = x.to(cuda0)
x_back = x_gpu.to(cpu)
```

When converting to NumPy, first ensure CPU and (if autograd tensor) detach:

```python
x_np = x_gpu.detach().to(cpu).numpy()
```

---

## Dtype recommendations

- Most MuNet ops are `float32`-centric; use `np.float32` for main model tensors.
- ONNX shape/index helper inputs are often `int64` in model definitions. Passing NumPy `int64` arrays is generally the cleanest for those graph inputs.
- If your graph op expects integer semantics (shape/axes/indices), keep those arrays integer on the Python side.

---

## End-to-end example for your script

```python
import munet
import numpy as np

# image_preprocessed should be float32 NCHW [1,3,H,W]
image_preprocessed = np.random.rand(1, 3, 1280, 1280).astype(np.float32)
size_input = np.array([[1280, 1280]], dtype=np.int64)

input_data = munet.from_numpy(image_preprocessed)

model = munet.inference.compile_onnx("rockfishing-tiny.onnx")
out = model.forward({
    "images": input_data,
    "image_size": size_input,
})
```

Replace `"images"` / `"image_size"` with your actual ONNX input names.
