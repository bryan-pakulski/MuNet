"""MuNet ONNX integration helpers loaded by bindings at import time."""


def _munet_dtype_from_numpy(arr, munet_module):
    import numpy as np

    dt = np.asarray(arr).dtype
    if dt == np.float32:
        return munet_module.DataType.Float32
    if dt == np.float16:
        return munet_module.DataType.Float16
    if dt == np.int32:
        return munet_module.DataType.Int32
    raise RuntimeError(f"Unsupported NumPy dtype for MuNet interop: {dt}")


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
            return x.detach().to(self._m.Device(self._m.DeviceType.CPU, 0)).numpy()

        arr = self._np.asarray(x)
        _munet_dtype_from_numpy(arr, self._m)
        return arr

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
            arr = self._np.asarray(arr)
            _munet_dtype_from_numpy(arr, self._m)
            t = self._m.from_numpy(arr)
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
    "Cast": {"status": "lowered", "munet": "graph/cast"},
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
    "Add": {"status": "lowered", "munet": "graph/add"},
    "Sub": {"status": "lowered", "munet": "graph/sub"},
    "Mul": {"status": "lowered", "munet": "graph/mul"},
    "Div": {"status": "lowered", "munet": "graph/div"},
    "Softmax": {"status": "unsupported", "munet": None},
    "Squeeze": {"status": "lowered", "munet": "graph/squeeze"},
    "Expand": {"status": "lowered", "munet": "graph/expand"},
    "Tile": {"status": "lowered", "munet": "graph/tile"},
    "ConstantOfShape": {"status": "lowered", "munet": "graph/constant_of_shape"},
    "Gather": {"status": "lowered", "munet": "graph/gather"},
    "Concat": {"status": "lowered", "munet": "graph/concat"},
    "Reshape": {"status": "lowered", "munet": "graph/reshape"},
    "Transpose": {"status": "lowered", "munet": "graph/transpose"},
    "Unsqueeze": {"status": "lowered", "munet": "graph/unsqueeze"},
    "Shape": {"status": "lowered", "munet": "graph/shape"},
    "Slice": {"status": "lowered", "munet": "graph/slice"},
    "Split": {"status": "lowered", "munet": "graph/split"},
    "Resize": {"status": "lowered", "munet": "graph/resize_nearest"},
    "Pow": {"status": "lowered", "munet": "graph/pow"},
    "Floor": {"status": "lowered", "munet": "graph/floor"},
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
            init.name: numpy_helper.to_array(init)
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
        _munet_dtype_from_numpy(arr, self.munet)
        return self.munet.from_numpy(self.np.asarray(arr))

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
            v = ctx.numpy_helper.to_array(a.t)
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




class _ONNXGraphModule:
    """General ONNX graph interpreter backed by MuNet tensor ops + safe numpy fallbacks."""

    def __init__(self, model, munet_module, np_module):
        self._model = model
        self._graph = model.graph
        self._m = munet_module
        self._np = np_module

        self._consts = {}
        for init in self._graph.initializer:
            self._consts[init.name] = self._onnx_numpy_helper.to_array(init)

        for node in self._graph.node:
            if node.op_type == "Constant" and len(node.output) > 0:
                for a in node.attribute:
                    if a.name == "value" and a.type == self._onnx.AttributeProto.TENSOR:
                        self._consts[node.output[0]] = self._onnx_numpy_helper.to_array(a.t)
                        break

    @property
    def _onnx(self):
        import onnx
        return onnx

    @property
    def _onnx_numpy_helper(self):
        from onnx import numpy_helper
        return numpy_helper

    def _get_attr(self, node, name, default=None):
        for a in node.attribute:
            if a.name != name:
                continue
            if a.type == self._onnx.AttributeProto.INT:
                return int(a.i)
            if a.type == self._onnx.AttributeProto.FLOAT:
                return float(a.f)
            if a.type == self._onnx.AttributeProto.INTS:
                return [int(v) for v in a.ints]
            if a.type == self._onnx.AttributeProto.FLOATS:
                return [float(v) for v in a.floats]
            if a.type == self._onnx.AttributeProto.STRING:
                return a.s.decode("utf-8")
        return default

    def _numpy_dtype_for_munet_dtype(self, dtype):
        if dtype == self._m.DataType.Float32:
            return self._np.float32
        if dtype == self._m.DataType.Float16:
            return self._np.float16
        if dtype == self._m.DataType.Int32:
            return self._np.int32
        raise RuntimeError(f"Unsupported MuNet tensor dtype in ONNX graph runtime: {dtype}")

    def _tensor_numpy_dtype(self, tensor):
        return self._numpy_dtype_for_munet_dtype(tensor.dtype)

    def _munet_dtype_from_onnx_tensor_type(self, tensor_type):
        tp = self._onnx.TensorProto
        if tensor_type == tp.FLOAT:
            return self._m.DataType.Float32
        if tensor_type == tp.FLOAT16:
            return self._m.DataType.Float16
        if tensor_type == tp.INT32:
            return self._m.DataType.Int32
        raise RuntimeError(f"Unsupported ONNX tensor dtype in graph runtime: {tensor_type}")

    def _as_tensor(self, v, ref_device=None):
        if isinstance(v, self._m.Tensor):
            return v
        arr = self._np.asarray(v)
        _munet_dtype_from_numpy(arr, self._m)
        t = self._m.from_numpy(arr)
        if ref_device is not None and ref_device.type != self._m.DeviceType.CPU:
            t = t.to(ref_device)
        return t

    def _as_numpy(self, v):
        if isinstance(v, self._m.Tensor):
            cpu = self._m.Device(self._m.DeviceType.CPU, 0)
            return self._np.asarray(v.detach().to(cpu).numpy())
        return self._np.asarray(v)

    def _from_numpy_like(self, arr, like=None):
        arr_np = self._np.asarray(arr)
        if like is not None and isinstance(like, self._m.Tensor):
            arr_np = arr_np.astype(self._tensor_numpy_dtype(like), copy=False)
        else:
            _munet_dtype_from_numpy(arr_np, self._m)
        t = self._m.from_numpy(arr_np)
        if like is not None and isinstance(like, self._m.Tensor):
            if like.device.type != self._m.DeviceType.CPU:
                t = t.to(like.device)
        return t

    def _scalar_tensor(self, value, ref):
        t = self._m.Tensor([1], ref.device, ref.dtype, False)
        t.uniform_(float(value), float(value))
        return t

    def _value(self, env, name):
        if name in env:
            return env[name]
        if name in self._consts:
            return self._consts[name]
        raise KeyError(f"Missing ONNX value: {name}")

    def forward(self, x):
        env = {}
        if len(self._graph.input) != 1:
            raise ValueError("_ONNXGraphModule currently supports single-input graphs")
        env[self._graph.input[0].name] = x

        for node in self._graph.node:
            op = node.op_type
            if op == "Constant":
                continue

            ins = [self._value(env, n) for n in node.input if n != ""]

            if op == "Identity":
                out = ins[0]
            elif op == "Cast":
                target_dtype = self._munet_dtype_from_onnx_tensor_type(
                    int(self._get_attr(node, "to"))
                )
                if isinstance(ins[0], self._m.Tensor):
                    out = self._as_tensor(ins[0]).to(target_dtype)
                else:
                    out = self._as_numpy(ins[0]).astype(
                        self._numpy_dtype_for_munet_dtype(target_dtype), copy=False
                    )
            elif op == "Conv":
                data = self._as_tensor(ins[0])
                W = self._as_tensor(ins[1], data.device)
                B = self._as_tensor(ins[2], data.device) if len(ins) > 2 else self._m.Tensor()
                strides = self._get_attr(node, "strides", [1, 1])
                pads = self._get_attr(node, "pads", [0, 0, 0, 0])
                out = data.conv2d(W, B, int(strides[0]), int(pads[0]))
            elif op == "MaxPool":
                data = self._as_tensor(ins[0])
                ks = self._get_attr(node, "kernel_shape", [2, 2])
                st = self._get_attr(node, "strides", ks)
                pd = self._get_attr(node, "pads", [0, 0, 0, 0])
                out = data.max_pool2d(int(ks[0]), int(st[0]), int(pd[0]))
            elif op == "Sigmoid":
                out = self._as_tensor(ins[0]).sigmoid()
            elif op == "Relu":
                out = self._as_tensor(ins[0]).relu()
            elif op == "LeakyRelu":
                x = self._as_tensor(ins[0])
                alpha = float(self._get_attr(node, "alpha", 0.01))
                out = self._m.nn.LeakyReLU(alpha).forward(x)
            elif op == "GlobalAveragePool":
                x = self._as_tensor(ins[0])
                out = self._m.nn.GlobalAvgPool2d().forward(x)
            elif op == "Gemm":
                A = self._as_tensor(ins[0])
                B = self._as_tensor(ins[1], A.device)
                C = self._as_tensor(ins[2], A.device) if len(ins) > 2 else None
                transA = int(self._get_attr(node, "transA", 0))
                transB = int(self._get_attr(node, "transB", 0))
                alpha = float(self._get_attr(node, "alpha", 1.0))
                beta = float(self._get_attr(node, "beta", 1.0))
                if transA != 0:
                    A = A.permute([1, 0]).contiguous()
                if transB != 0:
                    B = B.permute([1, 0]).contiguous()
                out = A.__matmul__(B)
                if alpha != 1.0:
                    out = out * self._scalar_tensor(alpha, out)
                if C is not None:
                    c_term = C if beta == 1.0 else C * self._scalar_tensor(beta, C)
                    out = out + c_term
            elif op == "Add":
                a = self._as_tensor(ins[0])
                b = self._as_tensor(ins[1], a.device)
                out = a + b
            elif op == "Mul":
                a = self._as_tensor(ins[0])
                b = self._as_tensor(ins[1], a.device)
                out = a * b
            elif op == "Sub":
                a = self._as_tensor(ins[0])
                b = self._as_tensor(ins[1], a.device)
                out = a - b
            elif op == "Div":
                a = self._as_tensor(ins[0])
                b = self._as_tensor(ins[1], a.device)
                out = a / b
            elif op == "Concat":
                axis = int(self._get_attr(node, "axis", 0))
                base = self._as_tensor(ins[0])
                ts = [base]
                for v in ins[1:]:
                    ts.append(self._as_tensor(v, base.device))
                out = self._m.Tensor.cat(ts, axis)
            elif op == "Reshape":
                data = self._as_tensor(ins[0])
                new_shape = [int(v) for v in self._as_numpy(ins[1]).reshape(-1).tolist()]
                out = data.reshape(new_shape)
            elif op == "Transpose":
                data = self._as_tensor(ins[0])
                perm = self._get_attr(node, "perm", list(range(len(data.shape) - 1, -1, -1)))
                out = data.permute([int(v) for v in perm]).contiguous()
            elif op == "Unsqueeze":
                data = self._as_tensor(ins[0])
                axes = self._get_attr(node, "axes", None)
                if axes is None and len(ins) > 1:
                    axes = [int(v) for v in self._as_numpy(ins[1]).reshape(-1).tolist()]
                old_shape = list(data.shape)
                rank = len(old_shape)
                norm = sorted([a if a >= 0 else a + rank + len(axes) for a in axes])
                for ax in norm:
                    old_shape.insert(ax, 1)
                out = data.reshape(old_shape)
            elif op == "Squeeze":
                data = self._as_tensor(ins[0])
                axes = self._get_attr(node, "axes", None)
                if axes is None and len(ins) > 1:
                    axes = [int(v) for v in self._as_numpy(ins[1]).reshape(-1).tolist()]
                shape = list(data.shape)
                if axes is None:
                    new_shape = [d for d in shape if int(d) != 1]
                    if not new_shape:
                        new_shape = [1]
                else:
                    rank = len(shape)
                    norm_axes = sorted(set((a if a >= 0 else a + rank) for a in axes), reverse=True)
                    for ax in norm_axes:
                        if shape[ax] != 1:
                            raise ValueError(f"Squeeze axis {ax} has dim {shape[ax]} != 1")
                        shape.pop(ax)
                    new_shape = shape if shape else [1]
                out = data.reshape(new_shape)
            elif op == "ConstantOfShape":
                shp = self._as_numpy(ins[0]).astype(self._np.int64).reshape(-1).tolist()
                value = 0.0
                value_dtype = self._np.float32
                for a in node.attribute:
                    if a.name == "value" and a.type == self._onnx.AttributeProto.TENSOR:
                        v = self._onnx_numpy_helper.to_array(a.t)
                        value = float(v.reshape(-1)[0]) if v.size > 0 else 0.0
                        try:
                            _munet_dtype_from_numpy(v, self._m)
                            value_dtype = v.dtype
                        except RuntimeError:
                            value_dtype = self._np.float32
                out = self._from_numpy_like(self._np.full(shp, value, dtype=value_dtype))
            elif op == "Expand":
                data_np = self._as_numpy(ins[0])
                shape = [int(v) for v in self._as_numpy(ins[1]).astype(self._np.int64).reshape(-1).tolist()]
                out = self._from_numpy_like(self._np.broadcast_to(data_np, shape).copy(), ins[0])
            elif op == "Tile":
                data_np = self._as_numpy(ins[0])
                reps = [int(v) for v in self._as_numpy(ins[1]).astype(self._np.int64).reshape(-1).tolist()]
                out = self._from_numpy_like(self._np.tile(data_np, reps), ins[0])
            elif op == "Gather":
                data_np = self._as_numpy(ins[0])
                idx_np = self._as_numpy(ins[1]).astype(self._np.int64)
                axis = int(self._get_attr(node, "axis", 0))
                gathered = self._np.take(data_np, idx_np, axis=axis)
                if isinstance(ins[0], self._m.Tensor):
                    out = self._from_numpy_like(gathered, ins[0])
                else:
                    out = gathered
            elif op == "Shape":
                out = self._np.asarray(self._as_tensor(ins[0]).shape, dtype=self._np.int64)
            elif op == "Slice":
                arr = self._as_numpy(ins[0])
                starts = self._as_numpy(ins[1]).astype(self._np.int64).reshape(-1)
                ends = self._as_numpy(ins[2]).astype(self._np.int64).reshape(-1)
                axes = self._as_numpy(ins[3]).astype(self._np.int64).reshape(-1) if len(ins) > 3 else self._np.arange(len(starts), dtype=self._np.int64)
                steps = self._as_numpy(ins[4]).astype(self._np.int64).reshape(-1) if len(ins) > 4 else self._np.ones_like(starts)
                sl = [slice(None)] * arr.ndim
                for i, ax in enumerate(axes.tolist()):
                    sl[int(ax)] = slice(int(starts[i]), int(ends[i]), int(steps[i]))
                out = arr[tuple(sl)]
            elif op == "Split":
                axis = int(self._get_attr(node, "axis", 0))
                arr = self._as_numpy(ins[0])
                split = self._get_attr(node, "split", None)
                if split is None:
                    parts = self._np.array_split(arr, len(node.output), axis=axis)
                else:
                    idx = self._np.cumsum(split)[:-1]
                    parts = self._np.split(arr, idx, axis=axis)
                for out_name, part in zip(node.output, parts):
                    env[out_name] = self._from_numpy_like(part, ins[0] if isinstance(ins[0], self._m.Tensor) else None)
                continue
            elif op == "Resize":
                data = self._as_tensor(ins[0])
                scales = None
                if len(ins) >= 3 and self._as_numpy(ins[2]).size > 0:
                    scales = self._as_numpy(ins[2]).astype(self._np.float32)
                mode = self._get_attr(node, "mode", "nearest")
                if mode != "nearest":
                    raise ValueError("Resize only supports nearest mode")
                if scales is not None and len(scales) >= 4:
                    sf_h = int(round(float(scales[-2])))
                    sf_w = int(round(float(scales[-1])))
                    if sf_h == sf_w and sf_h >= 1:
                        out = data.upsample2d(sf_h)
                    else:
                        raise ValueError("Resize only supports equal integer spatial scale")
                else:
                    raise ValueError("Resize requires scales input")
            elif op == "Pow":
                a = self._as_numpy(ins[0])
                b = self._as_numpy(ins[1])
                out = self._from_numpy_like(self._np.power(a, b), ins[0])
            elif op == "Floor":
                arr = self._as_numpy(ins[0])
                out = self._from_numpy_like(self._np.floor(arr), ins[0])
            else:
                raise ValueError(f"Unsupported op in graph runtime: {op}")

            if len(node.output) != 1:
                raise ValueError(f"Unsupported output arity for op {op}: {len(node.output)}")
            env[node.output[0]] = out

        outs = [env[o.name] for o in self._graph.output]
        return outs[0] if len(outs) == 1 else outs


def _compile_onnx_graph_module(model_path):
    import numpy as np
    import munet
    import onnx
    model = onnx.load(model_path)
    return _ONNXGraphModule(model, munet, np)

def load_onnx(model_path, device=None, providers=None):
    """Deprecated runtime path.

    MuNet now uses strict native conversion (`compile_onnx`) and does not
    provide ONNX Runtime fallback execution from this helper.
    """
    (void_device, void_providers) = (device, providers)
    raise RuntimeError(
        "munet.inference.load_onnx is deprecated. "
        "Use munet.inference.compile_onnx(model_path) for strict native conversion."
    )


_GRAPH_RUNTIME_SUPPORTED_OPS = {
    "Constant",
    "Identity",
    "Cast",
    "Conv",
    "MaxPool",
    "Sigmoid",
    "Relu",
    "LeakyRelu",
    "GlobalAveragePool",
    "Gemm",
    "Add",
    "Mul",
    "Sub",
    "Div",
    "Concat",
    "Reshape",
    "Transpose",
    "Unsqueeze",
    "Squeeze",
    "ConstantOfShape",
    "Expand",
    "Tile",
    "Gather",
    "Shape",
    "Slice",
    "Split",
    "Resize",
    "Pow",
    "Floor",
}


def _collect_runtime_missing_ops(graph):
    from collections import Counter

    op_counts = Counter(node.op_type for node in graph.node)
    missing_unique = []
    missing_total = 0
    for op, count in sorted(op_counts.items()):
        if op not in _GRAPH_RUNTIME_SUPPORTED_OPS:
            missing_unique.append(op)
            missing_total += int(count)
    return {
        "missing_runtime_unique": missing_unique,
        "missing_runtime_total": int(missing_total),
    }


def _collect_conversion_failures(graph):
    from collections import Counter

    op_counts = Counter(node.op_type for node in graph.node)
    unsupported_unique = []
    unsupported_total = 0

    for op, count in sorted(op_counts.items()):
        entry = ONNX_NATIVE_CONVERSION_MAP.get(op)
        status = entry["status"] if entry is not None else "unmapped"
        if status not in ("lowered", "pass_through"):
            unsupported_unique.append(op)
            unsupported_total += int(count)

    return {
        "op_counts": dict(sorted(op_counts.items())),
        "unsupported_unique": unsupported_unique,
        "unsupported_total": int(unsupported_total),
    }


def _format_conversion_failure(model_path, fail_info):
    return (
        "compile_onnx: conversion failed. "
        f"model='{model_path}', "
        f"unsupported_unique={fail_info['unsupported_unique']}, "
        f"unsupported_total={fail_info['unsupported_total']}, "
        f"op_counts={fail_info['op_counts']}"
    )


def compile_onnx(model_path, output_path=None, debug=False):
    """Strictly convert an ONNX model to a MuNet-native module.

    Behavior:
      - Success: full conversion, optional save to output_path.
      - Failure: raise with unsupported unique ops + total unsupported count.

    No fallback execution/runtime path is used.
    """
    import onnx

    model = onnx.load(model_path)
    fail_info = _collect_conversion_failures(model.graph)
    runtime_info = _collect_runtime_missing_ops(model.graph)

    if fail_info["unsupported_total"] > 0:
        raise ValueError(_format_conversion_failure(model_path, fail_info))
    if runtime_info["missing_runtime_total"] > 0:
        raise ValueError(
            "compile_onnx: conversion failed due missing graph-runtime op implementations. "
            f"model='{model_path}', "
            f"missing_runtime_unique={runtime_info['missing_runtime_unique']}, "
            f"missing_runtime_total={runtime_info['missing_runtime_total']}"
        )

    module = _compile_onnx_graph_module(model_path)

    if output_path is not None:
        raise ValueError(
            "compile_onnx: output_path save is not yet supported for graph-runtime modules. "
            "Convert first and run inference directly with returned module."
        )

    if debug:
        print(
            f"[compile_onnx] success model={model_path} "
            f"nodes={len(model.graph.node)} unique_ops={len(fail_info['op_counts'])}"
        )

    return module


def report_onnx_unsupported_ops(model_path):
    """Return sorted unique list of ops that block strict native conversion."""
    import onnx

    model = onnx.load(model_path)
    fail_info = _collect_conversion_failures(model.graph)
    return fail_info["unsupported_unique"]


def onnx_conversion_coverage_report(model_path):
    """Return ONNX operator coverage against MuNet native conversion map."""
    import onnx

    model = onnx.load(model_path)
    fail_info = _collect_conversion_failures(model.graph)

    coverage = {
        "lowered": [],
        "pass_through": [],
        "planned": [],
        "unsupported": [],
        "unmapped": [],
    }

    for op in sorted(fail_info["op_counts"].keys()):
        entry = ONNX_NATIVE_CONVERSION_MAP.get(op)
        if entry is None:
            coverage["unmapped"].append(op)
        else:
            coverage[entry["status"]].append(op)

    return {
        "model_path": model_path,
        "total_nodes": int(sum(fail_info["op_counts"].values())),
        "unique_ops": sorted(fail_info["op_counts"].keys()),
        "op_counts": fail_info["op_counts"],
        "coverage": coverage,
        "unsupported_unique": fail_info["unsupported_unique"],
        "unsupported_total": fail_info["unsupported_total"],
        "fully_lowerable": fail_info["unsupported_total"] == 0,
    }


def download_yolov5n_onnx(destination_path):
    """Download public yolov5n ONNX model used for conversion coverage checks."""
    import urllib.request

    url = (
        "https://github.com/yakhyo/yolov5-onnx-inference/releases/download/"
        "v0.0.1/yolov5n.onnx"
    )
    urllib.request.urlretrieve(url, destination_path)
    return destination_path


inference.ONNXEngine = ONNXEngine
inference.load_onnx = load_onnx
inference.compile_onnx = compile_onnx
inference.report_onnx_unsupported_ops = report_onnx_unsupported_ops
inference.onnx_native_conversion_map = onnx_native_conversion_map
inference.onnx_conversion_coverage_report = onnx_conversion_coverage_report
inference.download_yolov5n_onnx = download_yolov5n_onnx
