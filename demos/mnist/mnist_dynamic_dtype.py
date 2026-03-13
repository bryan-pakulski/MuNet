import argparse
import os
import sys
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../build"))
import munet


def generate_shapes(num_samples=200, size=32):
    images = np.zeros((num_samples, 3, size, size), dtype=np.float32)
    masks = np.zeros((num_samples, 1, size, size), dtype=np.float32)

    for i in range(num_samples):
        images[i, :, 10:20, 10:20] = 1.0
        masks[i, 0, 10:20, 10:20] = 1.0
    return images, masks


def create_model():
    return munet.nn.Sequential(
        [
            munet.nn.Conv2d(3, 16, 3, padding=1),
            munet.nn.BatchNorm2d(16),
            munet.nn.ReLU(),
            munet.nn.MaxPool2d(2, 2),
            munet.nn.Conv2d(16, 32, 3, padding=1),
            munet.nn.BatchNorm2d(32),
            munet.nn.ReLU(),
            munet.nn.MaxPool2d(2, 2),
            munet.nn.Conv2d(32, 64, 3, padding=1),
            munet.nn.BatchNorm2d(64),
            munet.nn.ReLU(),
            munet.nn.Upsample(2),
            munet.nn.Conv2d(64, 32, 3, padding=1),
            munet.nn.BatchNorm2d(32),
            munet.nn.ReLU(),
            munet.nn.Upsample(2),
            munet.nn.Conv2d(32, 16, 3, padding=1),
            munet.nn.BatchNorm2d(16),
            munet.nn.ReLU(),
            munet.nn.Conv2d(16, 1, 1),
            munet.nn.Sigmoid(),
        ]
    )


def create_quant_benchmark_model():
    # Conservative op set for current quantized CPU fallback paths.
    return munet.nn.Sequential(
        [
            munet.nn.Flatten(),
            munet.nn.Linear(3 * 32 * 32, 128),
            munet.nn.ReLU(),
            munet.nn.Linear(128, 1 * 32 * 32),
            munet.nn.Sigmoid(),
        ]
    )


def parse_dtype(name: str):
    table = {
        "fp32": munet.DataType.Float32,
        "fp16": munet.DataType.Float16,
        "bf16": munet.DataType.BFloat16,
        "int8": munet.DataType.Int8,
        "int4": munet.DataType.Int4,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from: {', '.join(table.keys())}")
    return table[name]


def resolve_param_dtype(data_dtype_name: str, override: str):
    if override == "auto":
        # Default policy: for trainable float modes, store params in the same
        # dtype; for quantized modes, keep fp32 params for stability.
        if data_dtype_name in ("fp16", "bf16", "fp32"):
            return parse_dtype(data_dtype_name)
        return munet.DataType.Float32
    return parse_dtype(override)


def run_epoch(model, optimizer, x_train, y_train, batch_size, compute_dtype,
              trainable=True):
    losses = []
    start = time.perf_counter()

    dispatch_cfg = munet.DTypeDispatchConfig()
    dispatch_cfg.has_compute_dtype = compute_dtype is not None
    if compute_dtype is not None:
        dispatch_cfg.compute_dtype = compute_dtype
    dispatch_cfg.fallback_mode = munet.KernelFallbackMode.WarnAndUpcast

    for i in range(0, len(x_train), batch_size):
        if i + batch_size > len(x_train):
            break

        bx = munet.from_numpy(x_train[i : i + batch_size])
        by = munet.from_numpy(y_train[i : i + batch_size])

        if trainable:
            optimizer.zero_grad()

            with munet.precision_dispatch(dispatch_cfg):
                preds = model.forward(bx)
                target = by.reshape(preds.shape) if preds.shape != by.shape else by
                loss = preds.mse_loss(target)

            loss.backward()
            optimizer.step()
        else:
            with munet.no_grad():
                with munet.precision_dispatch(dispatch_cfg):
                    preds = model.forward(bx)
                    target = by.reshape(preds.shape) if preds.shape != by.shape else by
                    loss = preds.mse_loss(target)
        losses.append(loss.item())

    elapsed = time.perf_counter() - start
    return float(np.mean(losses)), elapsed


def main():
    parser = argparse.ArgumentParser(description="Dynamic dtype training demo (CPU)")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16", "int8", "int4"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--param-dtype",
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16", "int8", "int4"],
        help="Model parameter storage dtype. 'auto' picks dtype for fp32/fp16/bf16 runs and fp32 for int8/int4 runs.",
    )
    args = parser.parse_args()

    data_dtype = parse_dtype(args.dtype)
    compute_dtype = None if args.dtype == "fp32" else data_dtype
    trainable = args.dtype in ("fp32", "fp16", "bf16")
    param_dtype = resolve_param_dtype(args.dtype, args.param_dtype)

    print(f"Running dynamic dtype demo on CPU with dtype={args.dtype}")
    if compute_dtype is not None:
        print("Mode: precision_dispatch compute dtype")
    else:
        print("Mode: baseline fp32")
    if not trainable:
        print("Note: int8/int4 path keeps tensor storage in fp32 and benchmarks precision_dispatch compute dtype only.")

    x_train, y_train = generate_shapes(args.samples)
    model = create_quant_benchmark_model()
    model.to(munet.Device(munet.DeviceType.CPU, 0))
    model.to_dtype(param_dtype)
    if trainable and param_dtype != munet.DataType.Float32:
        optimizer = munet.amp.FP32MasterSGD(model.parameters(), 0.1)
        print("Optimizer: FP32MasterSGD (low-precision model params)")
    else:
        optimizer = munet.optim.SGD(model.parameters(), lr=0.1)
        print("Optimizer: SGD")

    param_dtypes = {}
    total_params = 0
    for _, p in model.named_parameters().items():
        k = str(p.dtype)
        param_dtypes[k] = param_dtypes.get(k, 0) + int(np.prod(np.array(p.shape)))
        total_params += int(np.prod(np.array(p.shape)))
    print(f"Model parameter elements: {total_params}")
    print("Model parameter dtype distribution:")
    for k, v in sorted(param_dtypes.items()):
        print(f"  {k}: {v}")
    print(f"Requested param dtype: {param_dtype}")

    for epoch in range(args.epochs):
        avg_loss, elapsed = run_epoch(
            model,
            optimizer,
            x_train,
            y_train,
            args.batch_size,
            compute_dtype,
            trainable,
        )
        samples_per_sec = args.samples / max(elapsed, 1e-9)
        print(
            f"Epoch {epoch:02d} | avg_loss={avg_loss:.6f} | time={elapsed:.3f}s | samples/s={samples_per_sec:.1f}"
        )


if __name__ == "__main__":
    main()
