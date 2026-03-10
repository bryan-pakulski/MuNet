"""MuNet ONNX integration helpers loaded by bindings at import time."""


class ONNXEngine:
    """ONNX Runtime-backed inference helper exposed as munet.inference.ONNXEngine.

    It accepts MuNet Tensor or NumPy input and returns MuNet Tensor outputs
    on the configured MuNet device.
    """

    def __init__(self, model_path, device=None, providers=None):
        import numpy as np
        import munet

        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError(
                "ONNX support requires `onnxruntime` to be installed. "
                "Please install it (e.g. `pip install onnxruntime` or "
                "`onnxruntime-gpu`)."
            ) from e

        self._np = np
        self._m = munet
        self._ort = ort
        self._model_path = model_path
        self._device = (
            device if device is not None else munet.Device(munet.DeviceType.CPU, 0)
        )

        if providers is None:
            available = set(ort.get_available_providers())
            chosen = []
            if (
                self._device.type == munet.DeviceType.CUDA
                and "CUDAExecutionProvider" in available
            ):
                chosen.append("CUDAExecutionProvider")
            if "CPUExecutionProvider" in available:
                chosen.append("CPUExecutionProvider")
            providers = chosen if chosen else ["CPUExecutionProvider"]

        self._providers = list(providers)
        try:
            self._session = ort.InferenceSession(model_path, providers=self._providers)
        except Exception as e:
            msg = str(e)
            if "Unsupported model IR version" in msg:
                raise RuntimeError(
                    "ONNXRuntime failed to load model due unsupported IR version. "
                    "Re-export ONNX with lower IR/opset (e.g. opset<=11 for older ORT) "
                    "or upgrade onnxruntime. Original error: " + msg
                ) from e
            raise

        self._inputs = self._session.get_inputs()
        self._outputs = self._session.get_outputs()

    @property
    def providers(self):
        return list(self._providers)

    @property
    def input_names(self):
        return [i.name for i in self._inputs]

    @property
    def output_names(self):
        return [o.name for o in self._outputs]

    def _to_numpy(self, x):
        if isinstance(x, self._m.Tensor):
            return (
                x.detach()
                .to(self._m.Device(self._m.DeviceType.CPU, 0))
                .numpy()
                .astype(self._np.float32, copy=False)
            )
        return self._np.asarray(x, dtype=self._np.float32)

    def run(self, input_data, output_device=None):
        out_dev = output_device if output_device is not None else self._device

        if isinstance(input_data, dict):
            feed = {k: self._to_numpy(v) for k, v in input_data.items()}
        else:
            if len(self._inputs) != 1:
                raise ValueError(
                    f"ONNX model expects {len(self._inputs)} inputs ({self.input_names}); "
                    "pass a dict{name: tensor_or_array}."
                )
            feed = {self._inputs[0].name: self._to_numpy(input_data)}

        outs = self._session.run(None, feed)
        tensors = []
        for arr in outs:
            t = self._m.from_numpy(self._np.asarray(arr, dtype=self._np.float32))
            if out_dev.type != self._m.DeviceType.CPU:
                t = t.to(out_dev)
            tensors.append(t)

        return tensors[0] if len(tensors) == 1 else tensors


# Foundation for ONNX->MuNet native conversion.
# status values:
# - lowered: op is currently lowered to a MuNet module/layer.
# - pass_through: op advances stream but does not emit a layer.
# - planned: intended for future native lowering.
# - unsupported: currently not lowered.
ONNX_NATIVE_CONVERSION_MAP = {
    "Identity": {"status": "pass_through", "munet": "stream_identity"},
    "Cast": {"status": "pass_through", "munet": "stream_cast_noop"},
    "Constant": {"status": "pass_through", "munet": "const_capture"},
    "Gemm": {"status": "lowered", "munet": "nn.Linear"},
    "MatMul": {"status": "lowered", "munet": "nn.Linear"},
    "Conv": {"status": "lowered", "munet": "nn.Conv2d"},
    "MaxPool": {"status": "lowered", "munet": "nn.MaxPool2d"},
    "Relu": {"status": "lowered", "munet": "nn.ReLU"},
    "Sigmoid": {"status": "lowered", "munet": "nn.Sigmoid"},
    "Tanh": {"status": "lowered", "munet": "nn.Tanh"},
    "Flatten": {"status": "lowered", "munet": "nn.Flatten"},
    "LeakyRelu": {"status": "lowered", "munet": "nn.LeakyReLU"},
    "Gelu": {"status": "lowered", "munet": "nn.GELU"},
    "GlobalAveragePool": {"status": "lowered", "munet": "nn.GlobalAvgPool2d"},
    "Add": {"status": "planned", "munet": "binary_const/add"},
    "Sub": {"status": "planned", "munet": "binary_const/sub"},
    "Mul": {"status": "planned", "munet": "binary_const/mul"},
    "Div": {"status": "planned", "munet": "binary_const/div"},
    "Softmax": {"status": "unsupported", "munet": None},
    "Reshape": {"status": "unsupported", "munet": None},
    "Transpose": {"status": "unsupported", "munet": None},
    "Unsqueeze": {"status": "unsupported", "munet": None},
    "Squeeze": {"status": "unsupported", "munet": None},
    "Concat": {"status": "unsupported", "munet": None},
}


def onnx_native_conversion_map():
    """Return ONNX op conversion metadata for native ONNX->MuNet lowering."""
    return {k: dict(v) for k, v in ONNX_NATIVE_CONVERSION_MAP.items()}


class _ConversionContext:
    def __init__(self, graph, munet_module, np_module, onnx_module, numpy_helper, debug=False):
        self.graph = graph
        self.munet = munet_module
        self.np = np_module
        self.onnx = onnx_module
        self.numpy_helper = numpy_helper
        self.debug = debug

        self.consts = {
            init.name: numpy_helper.to_array(init).astype(np_module.float32)
            for init in graph.initializer
        }
        self.unsupported_ops = set()
        self.seq_layers = []
        self.lowered_count = 0
        self.stream_name = graph.input[0].name if len(graph.input) > 0 else None

    def log(self, msg):
        if self.debug:
            print(f"[compile_onnx][debug] {msg}")

    def unsupported(self, op, ignore_unsupported=False, report_only=False):
        self.unsupported_ops.add(op)
        self.log(f"unsupported op: {op}")
        if not ignore_unsupported and not report_only:
            raise ValueError(
                f"compile_onnx: unsupported op '{op}'. "
                "Use munet.inference.load_onnx for full ONNXRuntime execution."
            )

    def const(self, name):
        arr = self.consts.get(name)
        if arr is None:
            return None
        return self.munet.from_numpy(self.np.asarray(arr, dtype=self.np.float32))

    def get_attr(self, node, name, default=None):
        for a in node.attribute:
            if a.name != name:
                continue
            if a.type == self.onnx.AttributeProto.INT:
                return int(a.i)
            if a.type == self.onnx.AttributeProto.FLOAT:
                return float(a.f)
            if a.type == self.onnx.AttributeProto.INTS:
                return [int(v) for v in a.ints]
            if a.type == self.onnx.AttributeProto.FLOATS:
                return [float(v) for v in a.floats]
        return default


def _capture_constant_node(node, ctx):
    v = None
    for a in node.attribute:
        if a.name == "value" and a.type == ctx.onnx.AttributeProto.TENSOR:
            v = ctx.numpy_helper.to_array(a.t).astype(ctx.np.float32)
            break
    if v is not None and len(node.output) > 0:
        ctx.consts[node.output[0]] = v


def _lower_gemm(node, ctx):
    W = ctx.const(node.input[1])
    if W is None:
        return False

    transB = ctx.get_attr(node, "transB", 0)
    transA = ctx.get_attr(node, "transA", 0)
    alpha = ctx.get_attr(node, "alpha", 1.0)
    beta = ctx.get_attr(node, "beta", 1.0)

    if transA != 0 or alpha != 1.0 or beta != 1.0:
        return False

    if transB == 0:
        in_features = int(W.shape[0])
        out_features = int(W.shape[1])
        weight_native = W
    else:
        in_features = int(W.shape[1])
        out_features = int(W.shape[0])
        weight_native = W.transpose(0, 1).contiguous()

    layer = ctx.munet.nn.Linear(in_features, out_features, len(node.input) >= 3)
    layer.weight.replace_(weight_native)
    if len(node.input) >= 3:
        B = ctx.const(node.input[2])
        if B is None:
            return False
        layer.bias.replace_(B.reshape([out_features]))
    ctx.seq_layers.append(layer)
    return True


def _lower_matmul(node, ctx):
    B = ctx.const(node.input[1])
    if B is None:
        return False
    in_features = int(B.shape[0])
    out_features = int(B.shape[1])
    layer = ctx.munet.nn.Linear(in_features, out_features, False)
    layer.weight.replace_(B)
    ctx.seq_layers.append(layer)
    return True


def _lower_conv(node, ctx):
    W = ctx.const(node.input[1])
    if W is None:
        return False

    B = ctx.const(node.input[2]) if len(node.input) > 2 else None
    strides = ctx.get_attr(node, "strides", [1, 1])
    pads = ctx.get_attr(node, "pads", [0, 0, 0, 0])
    dil = ctx.get_attr(node, "dilations", [1, 1])
    group = ctx.get_attr(node, "group", 1)
    if group != 1 or dil != [1, 1] or pads[0] != pads[2] or pads[1] != pads[3]:
        return False

    oc, ic, kh, kw = [int(v) for v in W.shape]
    if kh != kw:
        return False

    layer = ctx.munet.nn.Conv2d(ic, oc, kh, strides[0], pads[0])
    layer.weight.replace_(W)
    if B is not None:
        layer.bias.replace_(B.reshape([oc]))
    ctx.seq_layers.append(layer)
    return True


def _lower_maxpool(node, ctx):
    ks = ctx.get_attr(node, "kernel_shape", None)
    st = ctx.get_attr(node, "strides", ks)
    pd = ctx.get_attr(node, "pads", [0, 0, 0, 0])
    if (
        ks is None
        or len(ks) != 2
        or ks[0] != ks[1]
        or st[0] != st[1]
        or pd[0] != pd[2]
        or pd[1] != pd[3]
    ):
        return False

    ctx.seq_layers.append(ctx.munet.nn.MaxPool2d(int(ks[0]), int(st[0]), int(pd[0])))
    return True


def _lower_relu(node, ctx):
    ctx.seq_layers.append(ctx.munet.nn.ReLU())
    return True


def _lower_sigmoid(node, ctx):
    ctx.seq_layers.append(ctx.munet.nn.Sigmoid())
    return True


def _lower_tanh(node, ctx):
    ctx.seq_layers.append(ctx.munet.nn.Tanh())
    return True


def _lower_flatten(node, ctx):
    ctx.seq_layers.append(ctx.munet.nn.Flatten())
    return True


def _lower_leaky_relu(node, ctx):
    alpha = float(ctx.get_attr(node, "alpha", 0.01))
    ctx.seq_layers.append(ctx.munet.nn.LeakyReLU(alpha))
    return True


def _lower_gelu(node, ctx):
    # ONNX Gelu supports optional approximation mode; MuNet GELU currently
    # provides a fast approximation implementation, so we map directly.
    ctx.seq_layers.append(ctx.munet.nn.GELU())
    return True


def _lower_global_average_pool(node, ctx):
    ctx.seq_layers.append(ctx.munet.nn.GlobalAvgPool2d())
    return True


_NODE_LOWERING_DISPATCH = {
    "Gemm": _lower_gemm,
    "MatMul": _lower_matmul,
    "Conv": _lower_conv,
    "MaxPool": _lower_maxpool,
    "Relu": _lower_relu,
    "Sigmoid": _lower_sigmoid,
    "Tanh": _lower_tanh,
    "Flatten": _lower_flatten,
    "LeakyRelu": _lower_leaky_relu,
    "Gelu": _lower_gelu,
    "GlobalAveragePool": _lower_global_average_pool,
}

_PASS_THROUGH_OPS = {"Identity", "Cast"}


def load_onnx(model_path, device=None, providers=None):
    """Create an ONNXRuntime-backed inference engine.

    Args:
        model_path: Path to ONNX file.
        device: Optional MuNet device for output tensors.
        providers: Optional explicit ORT provider list.
    """
    return ONNXEngine(model_path, device=device, providers=providers)


def compile_onnx(
    model_path,
    output_path=None,
    ignore_unsupported=False,
    report_only=False,
    allow_partial=False,
    debug=False,
):
    """Compile a supported ONNX graph into a MuNet model module.

    Native lowering currently targets single-input forward graphs where each node
    can be expressed as either:
      1) a unary transformation of the running tensor, or
      2) a binary op with a constant tensor (initializer/Constant node).

    Args:
        model_path: Path to .onnx model
        output_path: Optional .npz destination for compiled MuNet model
        ignore_unsupported: If True, skip unsupported ops and continue scan.
        report_only: If True, return sorted unique unsupported op names.
        allow_partial: If True and ignore_unsupported=True, allow emitting partial model.
        debug: If True, print per-node lowering decisions and summary.
    """
    import numpy as np
    import munet

    try:
        import onnx
        from onnx import numpy_helper
    except Exception as e:
        raise RuntimeError(
            "compile_onnx requires `onnx` package installed (pip install onnx)."
        ) from e

    if report_only and output_path is not None:
        raise ValueError(
            "compile_onnx: report_only=True does not compile/save. Remove output_path to get unsupported-op report."
        )

    model = onnx.load(model_path)
    graph = model.graph
    ctx = _ConversionContext(graph, munet, np, onnx, numpy_helper, debug=debug)

    for node in graph.node:
        op = node.op_type
        ctx.log(f"node op={op} inputs={list(node.input)} outputs={list(node.output)}")

        if op == "Constant":
            _capture_constant_node(node, ctx)
            continue

        if op in _PASS_THROUGH_OPS:
            if len(node.output) > 0:
                ctx.stream_name = node.output[0]
            continue

        if ctx.stream_name is None or len(node.input) == 0:
            ctx.unsupported(op, ignore_unsupported=ignore_unsupported, report_only=report_only)
            continue

        if node.input[0] != ctx.stream_name:
            # non-linear dataflow/branch currently unsupported for native lowering
            ctx.unsupported(op, ignore_unsupported=ignore_unsupported, report_only=report_only)
            continue

        lower_fn = _NODE_LOWERING_DISPATCH.get(op)
        lowered = False
        if lower_fn is not None:
            lowered = lower_fn(node, ctx)

        if not lowered:
            ctx.unsupported(op, ignore_unsupported=ignore_unsupported, report_only=report_only)
            continue

        ctx.lowered_count += 1
        if len(node.output) > 0:
            ctx.stream_name = node.output[0]

    if report_only:
        if debug:
            print(
                f"[compile_onnx] report_only unsupported_ops={sorted(list(ctx.unsupported_ops))}"
            )
        return sorted(list(ctx.unsupported_ops))

    if ctx.unsupported_ops:
        skipped = sorted(list(ctx.unsupported_ops))
        if ignore_unsupported:
            if not allow_partial:
                raise ValueError(
                    "compile_onnx: graph contains unsupported ops; refusing to emit partial model. "
                    f"Unsupported ops: {skipped}. "
                    "Use report_only=True to inspect, or set allow_partial=True to emit best-effort partial model."
                )
            print(
                f"[compile_onnx] warning: emitting partial model; skipped unsupported ops: {skipped}"
            )

    if not ctx.seq_layers:
        raise ValueError("compile_onnx: no supported nodes found in graph")

    if debug:
        print(
            f"[compile_onnx] lowered_nodes={ctx.lowered_count} total_nodes={len(graph.node)} layers={len(ctx.seq_layers)}"
        )

    module = munet.nn.Sequential(ctx.seq_layers)
    if output_path is not None:
        munet.save(module, output_path)
    return module


def report_onnx_unsupported_ops(model_path):
    """Return a sorted unique list of unsupported ONNX ops for native compile."""
    return compile_onnx(
        model_path,
        output_path=None,
        ignore_unsupported=True,
        report_only=True,
        allow_partial=True,
        debug=False,
    )


def compare_onnx_native_to_ort(model_path, input_data, output_device=None, providers=None):
    """Run ONNXRuntime and native MuNet lowering and report numeric drift.

    The model must be fully supported by `compile_onnx` (no partial lowering).
    """
    import numpy as np

    ort_engine = load_onnx(model_path, device=output_device, providers=providers)
    native_module = compile_onnx(model_path)

    ort_out = ort_engine.run(input_data, output_device=output_device)
    native_out = native_module.forward(input_data)

    ort_np = np.asarray(ort_out.detach().to(ort_engine._m.Device(ort_engine._m.DeviceType.CPU, 0)).numpy(), dtype=np.float32)
    native_np = np.asarray(native_out.detach().to(ort_engine._m.Device(ort_engine._m.DeviceType.CPU, 0)).numpy(), dtype=np.float32)

    if ort_np.shape != native_np.shape:
        raise ValueError(
            f"compare_onnx_native_to_ort: output shape mismatch ORT={ort_np.shape} native={native_np.shape}"
        )

    diff = native_np - ort_np
    abs_diff = np.abs(diff)
    return {
        "shape": list(ort_np.shape),
        "max_abs_error": float(abs_diff.max()) if abs_diff.size > 0 else 0.0,
        "mean_abs_error": float(abs_diff.mean()) if abs_diff.size > 0 else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if abs_diff.size > 0 else 0.0,
    }


inference.ONNXEngine = ONNXEngine
inference.load_onnx = load_onnx
inference.compile_onnx = compile_onnx
inference.report_onnx_unsupported_ops = report_onnx_unsupported_ops
inference.onnx_native_conversion_map = onnx_native_conversion_map
inference.compare_onnx_native_to_ort = compare_onnx_native_to_ort
