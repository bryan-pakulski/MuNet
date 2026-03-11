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
    "Add": {"status": "lowered", "munet": "graph/add"},
    "Sub": {"status": "lowered", "munet": "graph/sub"},
    "Mul": {"status": "lowered", "munet": "graph/mul"},
    "Div": {"status": "lowered", "munet": "graph/div"},
    "Softmax": {"status": "lowered", "munet": "graph/softmax"},
    "ReduceSum": {"status": "lowered", "munet": "graph/reduce_sum"},
    "ReduceMean": {"status": "lowered", "munet": "graph/reduce_mean"},
    "ReduceMax": {"status": "lowered", "munet": "graph/reduce_max"},
    "Log": {"status": "lowered", "munet": "graph/log"},
    "Sqrt": {"status": "lowered", "munet": "graph/sqrt"},
    "Clip": {"status": "lowered", "munet": "graph/clip"},
    "Erf": {"status": "lowered", "munet": "graph/erf"},
    "Pad": {"status": "lowered", "munet": "graph/pad_constant"},
    "GatherElements": {"status": "lowered", "munet": "graph/gather_elements"},
    "TopK": {"status": "lowered", "munet": "graph/topk"},
    "GridSample": {"status": "lowered", "munet": "graph/grid_sample"},
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




class _ONNXGraphModule:
    """General ONNX graph interpreter backed by MuNet tensor ops + safe numpy fallbacks."""

    def __init__(self, model, munet_module, np_module):
        import os

        self._model = model
        self._graph = model.graph
        self._m = munet_module
        self._np = np_module
        self._execution_device = self._m.Device(self._m.DeviceType.CPU, 0)
        self._trace_nonfinite = str(os.getenv("MUNET_ONNX_TRACE_NONFINITE", "0")).strip().lower() in (
            "1", "true", "yes", "on"
        )
        self._trace_nonfinite_inputs = str(
            os.getenv("MUNET_ONNX_TRACE_NONFINITE_INPUTS", "1")
        ).strip().lower() in ("1", "true", "yes", "on")
        self._pow_clamp_finite = str(
            os.getenv("MUNET_ONNX_POW_CLAMP_FINITE", "1")
        ).strip().lower() in ("1", "true", "yes", "on")
        self._sqrt_clamp_nonneg = str(
            os.getenv("MUNET_ONNX_SQRT_CLAMP_NONNEG", "1")
        ).strip().lower() in ("1", "true", "yes", "on")

        self._opset = 13
        for imp in model.opset_import:
            if getattr(imp, "domain", "") in ("", None):
                self._opset = int(imp.version)
                break

        self._consts = {}
        for init in self._graph.initializer:
            self._consts[init.name] = self._onnx_numpy_helper.to_array(init)

        self._input_names = [
            value.name for value in self._graph.input if value.name not in self._consts
        ]

        for node_idx, node in enumerate(self._graph.node):
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

    @property
    def device(self):
        return self._execution_device

    def to(self, device):
        self._execution_device = device
        return self

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

    def _cast_numpy(self, arr, to):
        tp = self._onnx.TensorProto
        dtype_map = {
            tp.FLOAT: self._np.float32,
            tp.DOUBLE: self._np.float64,
            tp.FLOAT16: self._np.float16,
            tp.BFLOAT16: self._np.float32,
            tp.INT64: self._np.int64,
            tp.INT32: self._np.int32,
            tp.INT16: self._np.int16,
            tp.INT8: self._np.int8,
            tp.UINT8: self._np.uint8,
            tp.BOOL: self._np.bool_,
        }
        if to not in dtype_map:
            raise ValueError(f"Cast target dtype not supported: {to}")

        src = self._np.asarray(arr)
        tgt = dtype_map[to]

        if self._np.issubdtype(tgt, self._np.integer):
            srcf = self._np.asarray(src, dtype=self._np.float64)
            if not self._np.all(self._np.isfinite(srcf)):
                raise ValueError(f"Cast received non-finite values for integer target: {src}")
            if not self._np.all(self._np.isclose(srcf, self._np.round(srcf), atol=1e-6)):
                raise ValueError(f"Cast received non-integer values for integer target: {src}")
            info = self._np.iinfo(tgt)
            if not self._np.all((srcf >= info.min) & (srcf <= info.max)):
                raise ValueError(f"Cast integer overflow: target={tgt}, values={src}")
            return self._np.round(srcf).astype(tgt, copy=False)

        if tgt == self._np.bool_:
            return self._np.asarray(src).astype(self._np.bool_, copy=False)

        return src.astype(tgt, copy=False)

    def _as_tensor(self, v, ref_device=None):
        target_device = ref_device if ref_device is not None else self._execution_device

        if isinstance(v, self._m.Tensor):
            if target_device is not None and v.device != target_device:
                return v.to(target_device)
            return v

        arr = self._np.asarray(v, dtype=self._np.float32)
        t = self._m.from_numpy(arr)
        if target_device is not None and target_device.type != self._m.DeviceType.CPU:
            t = t.to(target_device)
        return t

    def _as_numpy(self, v):
        if isinstance(v, self._m.Tensor):
            cpu = self._m.Device(self._m.DeviceType.CPU, 0)
            # `.numpy()` shares tensor storage; ensure returned array owns memory
            # so callers never observe dangling/temporary tensor buffers.
            return self._np.array(v.detach().to(cpu).numpy(), copy=True)
        return self._np.asarray(v)

    def _from_numpy_like(self, arr, like=None):
        arr_np = self._np.asarray(arr)
        if like is None or not isinstance(like, self._m.Tensor):
            return arr_np

        t = self._m.from_numpy(self._np.asarray(arr_np, dtype=self._np.float32))
        if like.device.type != self._m.DeviceType.CPU:
            t = t.to(like.device)
        return t

    def _pad_tensor_constant(self, data, pads_begin, pads_end, value=0.0):
        rank = len(data.shape)
        if len(pads_begin) != rank or len(pads_end) != rank:
            raise ValueError(
                f"pad rank mismatch: rank={rank}, pads_begin={pads_begin}, pads_end={pads_end}"
            )

        pads_begin = [int(v) for v in pads_begin]
        pads_end = [int(v) for v in pads_end]

        # ONNX Pad allows negative pads, which means cropping.
        if any(v < 0 for v in pads_begin + pads_end):
            arr = self._as_numpy(data).astype(self._np.float32)

            sl = []
            for ax in range(rank):
                crop_before = max(-pads_begin[ax], 0)
                crop_after = max(-pads_end[ax], 0)
                end = arr.shape[ax] - crop_after
                if crop_before > end:
                    raise ValueError(
                        f"Pad crop exceeds dimension at axis {ax}: "
                        f"shape={arr.shape[ax]}, pads_begin={pads_begin[ax]}, pads_end={pads_end[ax]}"
                    )
                sl.append(slice(crop_before, end))
            arr = arr[tuple(sl)]

            np_pads = [
                (max(pads_begin[ax], 0), max(pads_end[ax], 0)) for ax in range(rank)
            ]
            if any(pb > 0 or pe > 0 for pb, pe in np_pads):
                arr = self._np.pad(arr, np_pads, mode="constant", constant_values=float(value))

            return self._from_numpy_like(arr.astype(self._np.float32), data)

        cur = data
        for ax in range(rank):
            pb = pads_begin[ax]
            pe = pads_end[ax]

            parts = []
            if pb > 0:
                left_shape = list(cur.shape)
                left_shape[ax] = pb
                left = self._m.Tensor(left_shape, cur.device, cur.dtype, False)
                left.uniform_(float(value), float(value))
                parts.append(left)

            parts.append(cur)

            if pe > 0:
                right_shape = list(cur.shape)
                right_shape[ax] = pe
                right = self._m.Tensor(right_shape, cur.device, cur.dtype, False)
                right.uniform_(float(value), float(value))
                parts.append(right)

            if len(parts) > 1:
                cur = self._m.Tensor.cat(parts, ax)

        return cur

    def _conv2d_numpy_grouped(self, x_np, w_np, b_np, stride, groups):
        # x: [N,C,H,W], w: [OC, IC_per_group, KH, KW]
        n, c, h, w = [int(v) for v in x_np.shape]
        oc, icpg, kh, kw = [int(v) for v in w_np.shape]

        if groups <= 0 or c % groups != 0 or oc % groups != 0:
            raise ValueError(
                f"Invalid grouped conv config: in_channels={c}, out_channels={oc}, groups={groups}"
            )
        icg = c // groups
        ocg = oc // groups
        if icpg != icg:
            raise ValueError(
                f"Grouped conv weight/input mismatch: weight_ic_per_group={icpg}, expected={icg}"
            )

        oh = (h - kh) // int(stride) + 1
        ow = (w - kw) // int(stride) + 1
        if oh <= 0 or ow <= 0:
            raise ValueError(
                f"Grouped conv invalid output shape for input={[n,c,h,w]}, kernel={[kh,kw]}, stride={stride}"
            )

        out = self._np.zeros((n, oc, oh, ow), dtype=self._np.float32)
        for bn in range(n):
            for g in range(groups):
                in_base = g * icg
                out_base = g * ocg
                for oc_i in range(ocg):
                    oc_idx = out_base + oc_i
                    for oy in range(oh):
                        iy = oy * int(stride)
                        for ox in range(ow):
                            ix = ox * int(stride)
                            acc = 0.0
                            for ic_i in range(icg):
                                for ky in range(kh):
                                    for kx in range(kw):
                                        acc += float(x_np[bn, in_base + ic_i, iy + ky, ix + kx]) * float(
                                            w_np[oc_idx, ic_i, ky, kx]
                                        )
                            if b_np is not None:
                                acc += float(b_np[oc_idx])
                            out[bn, oc_idx, oy, ox] = acc
        return out

    def _resize_nearest_numpy(self, x_np, out_h, out_w):
        if x_np.ndim != 4:
            raise ValueError(f"Resize currently supports only 4D tensors, got rank={x_np.ndim}")

        n, c, in_h, in_w = x_np.shape
        out_h = int(out_h)
        out_w = int(out_w)
        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"Resize target spatial size must be positive, got H={out_h}, W={out_w}")

        if in_h == 0 or in_w == 0:
            raise ValueError(f"Resize input has empty spatial dims: H={in_h}, W={in_w}")

        ys = self._np.floor(self._np.arange(out_h, dtype=self._np.float64) * (float(in_h) / float(out_h))).astype(self._np.int64)
        xs = self._np.floor(self._np.arange(out_w, dtype=self._np.float64) * (float(in_w) / float(out_w))).astype(self._np.int64)
        ys = self._np.clip(ys, 0, in_h - 1)
        xs = self._np.clip(xs, 0, in_w - 1)

        return x_np[:, :, ys[:, None], xs[None, :]]

    def _resolve_2d_auto_pad(self, auto_pad, in_hw, kernel_hw, strides_hw, dilations_hw=(1, 1)):
        mode = str(auto_pad or "NOTSET").upper()
        if mode in ("", "NOTSET"):
            return [0, 0, 0, 0]
        if mode == "VALID":
            return [0, 0, 0, 0]
        if mode not in ("SAME_UPPER", "SAME_LOWER"):
            raise ValueError(f"Unsupported auto_pad mode: {auto_pad}")

        pads = []
        for i in range(2):
            in_dim = int(in_hw[i])
            k = int(kernel_hw[i])
            s = int(strides_hw[i])
            d = int(dilations_hw[i])
            eff_k = (k - 1) * d + 1
            out_dim = int(self._np.ceil(float(in_dim) / float(s)))
            total = max((out_dim - 1) * s + eff_k - in_dim, 0)
            if mode == "SAME_UPPER":
                p0 = total // 2
                p1 = total - p0
            else:
                p1 = total // 2
                p0 = total - p1
            pads.extend([p0, p1])
        return [pads[0], pads[2], pads[1], pads[3]]


    def _align_binary_pair_min(self, a, b):
        if not (isinstance(a, self._m.Tensor) and isinstance(b, self._m.Tensor)):
            return a, b

        ra, rb = len(a.shape), len(b.shape)
        if ra != rb:
            return a, b

        # NCHW residual path: align H/W when N/C match.
        if ra == 4 and int(a.shape[0]) == int(b.shape[0]) and int(a.shape[1]) == int(b.shape[1]):
            ah, aw = int(a.shape[2]), int(a.shape[3])
            bh, bw = int(b.shape[2]), int(b.shape[3])
            if ah == bh and aw == bw:
                return a, b
            th, tw = min(ah, bh), min(aw, bw)
            if th <= 0 or tw <= 0:
                return a, b

            def _crop4(x, th, tw):
                if int(x.shape[2]) == th and int(x.shape[3]) == tw:
                    return x
                arr = self._as_numpy(x).astype(self._np.float32)
                arr = arr[:, :, :th, :tw]
                return self._from_numpy_like(arr, x)

            return _crop4(a, th, tw), _crop4(b, th, tw)

        # Sequence path [B, T, C]: align T when B/C match.
        if ra == 3 and int(a.shape[0]) == int(b.shape[0]) and int(a.shape[2]) == int(b.shape[2]):
            ta, tb = int(a.shape[1]), int(b.shape[1])
            if ta == tb:
                return a, b
            tt = min(ta, tb)
            if tt <= 0:
                return a, b

            def _crop3(x, tt):
                if int(x.shape[1]) == tt:
                    return x
                arr = self._as_numpy(x).astype(self._np.float32)
                arr = arr[:, :tt, :]
                return self._from_numpy_like(arr, x)

            return _crop3(a, tt), _crop3(b, tt)

        return a, b

    def _binary_tensor_op(self, op_name, a, b, tensor_fn, numpy_fn):
        try:
            return tensor_fn(a, b)
        except RuntimeError as e:
            # Fall back to NumPy ONNX-style broadcasting when MuNet backend
            # broadcast constraints are stricter than ONNX semantics.
            a_np = self._as_numpy(a).astype(self._np.float32)
            b_np = self._as_numpy(b).astype(self._np.float32)
            try:
                y_np = numpy_fn(a_np, b_np)
            except ValueError:
                # Last-resort alignment used elsewhere in this runtime for
                # mismatched residual-style feature maps.
                if (
                    a_np.ndim == b_np.ndim == 4
                    and int(a_np.shape[0]) == int(b_np.shape[0])
                    and int(a_np.shape[1]) == int(b_np.shape[1])
                ):
                    h = min(int(a_np.shape[2]), int(b_np.shape[2]))
                    w = min(int(a_np.shape[3]), int(b_np.shape[3]))
                    y_np = numpy_fn(a_np[:, :, :h, :w], b_np[:, :, :h, :w])
                elif (
                    a_np.ndim == b_np.ndim == 3
                    and int(a_np.shape[0]) == int(b_np.shape[0])
                    and int(a_np.shape[2]) == int(b_np.shape[2])
                ):
                    t = min(int(a_np.shape[1]), int(b_np.shape[1]))
                    y_np = numpy_fn(a_np[:, :t, :], b_np[:, :t, :])
                elif (
                    a_np.ndim == b_np.ndim == 3
                    and int(a_np.shape[0]) == int(b_np.shape[0])
                    and (
                        int(a_np.shape[2]) == int(b_np.shape[2])
                        or int(a_np.shape[2]) == 1
                        or int(b_np.shape[2]) == 1
                    )
                ):
                    # Sequence-style fallback where token lengths diverge but
                    # final dim is still broadcast-compatible (e.g. [B,T,1]
                    # with [B,T2,C]). Align token axis to shared minimum.
                    t = min(int(a_np.shape[1]), int(b_np.shape[1]))
                    y_np = numpy_fn(a_np[:, :t, :], b_np[:, :t, :])
                else:
                    raise RuntimeError(
                        f"{op_name}: shape mismatch a_shape={list(a.shape)} b_shape={list(b.shape)}"
                    ) from e
            return self._from_numpy_like(y_np.astype(self._np.float32), a)

    def _summarize_value(self, name, value):
        arr = self._as_numpy(value)
        shape = list(arr.shape)
        dtype = str(arr.dtype)
        if arr.size == 0:
            return f"{name}: shape={shape}, dtype={dtype}, empty"

        if not self._np.issubdtype(arr.dtype, self._np.number):
            return f"{name}: shape={shape}, dtype={dtype}, non-numeric"

        finite = self._np.isfinite(arr)
        finite_count = int(finite.sum())
        total = int(arr.size)
        if finite_count == 0:
            return f"{name}: shape={shape}, dtype={dtype}, finite=0/{total}"

        vals = arr[finite]
        min_v = float(vals.min())
        max_v = float(vals.max())
        mean_v = float(vals.mean())
        return (
            f"{name}: shape={shape}, dtype={dtype}, finite={finite_count}/{total}, "
            f"min={min_v:.6g}, max={max_v:.6g}, mean={mean_v:.6g}"
        )

    def _check_nonfinite_output(self, node, value, output_name, env=None):
        if not self._trace_nonfinite:
            return

        arr = self._as_numpy(value)
        if not self._np.issubdtype(arr.dtype, self._np.floating):
            return

        bad = ~self._np.isfinite(arr)
        if not bad.any():
            return

        flat = bad.reshape(-1)
        first = int(self._np.argmax(flat))
        bad_count = int(bad.sum())
        total = int(arr.size)
        node_name = node.name if getattr(node, "name", "") else "<unnamed>"

        details = []
        if self._trace_nonfinite_inputs and env is not None:
            for in_name in [n for n in node.input if n != ""]:
                try:
                    in_val = self._value(env, in_name)
                    details.append(self._summarize_value(in_name, in_val))
                except Exception:
                    details.append(f"{in_name}: <unavailable>")

        details.append(self._summarize_value(output_name, value))

        raise ValueError(
            "Non-finite detected in ONNX graph runtime output: "
            f"op={node.op_type}, node={node_name}, output={output_name}, "
            f"first_bad_flat_index={first}, bad_count={bad_count}, total={total}; "
            f"stats=[{' | '.join(details)}]"
        )

    def _scalar_tensor(self, value, ref):
        t = self._m.Tensor([1], ref.device, ref.dtype, False)
        t.uniform_(float(value), float(value))
        return t

    def export_npz(self, output_path):
        import json

        payload = {
            "__format__": self._np.asarray(["munet.onnx_graph_module.npz.v1"]),
            "__opset__": self._np.asarray([int(self._opset)], dtype=self._np.int64),
            "__input_names__": self._np.asarray(self._input_names, dtype=object),
            "__output_names__": self._np.asarray([o.name for o in self._graph.output], dtype=object),
        }

        graph_meta = {
            "graph_name": getattr(self._graph, "name", ""),
            "nodes": [
                {
                    "name": node.name,
                    "op_type": node.op_type,
                    "input": list(node.input),
                    "output": list(node.output),
                }
                for node in self._graph.node
            ],
        }
        payload["__graph_json__"] = self._np.asarray([json.dumps(graph_meta)], dtype=object)

        for name, value in self._consts.items():
            payload[f"tensor/{name}"] = self._np.asarray(value)

        self._np.savez_compressed(output_path, **payload)
        return output_path

    def _value(self, env, name):
        if name in env:
            return env[name]
        if name in self._consts:
            return self._consts[name]
        raise KeyError(f"Missing ONNX value: {name}")

    def _bind_inputs(self, inputs, named_inputs):
        expected_names = list(self._input_names)
        expected_count = len(expected_names)

        if named_inputs:
            if len(inputs) > 0:
                raise ValueError("Use either positional inputs or keyword inputs, not both")
            provided = dict(named_inputs)
        elif len(inputs) == 1:
            arg = inputs[0]
            if isinstance(arg, dict):
                provided = dict(arg)
            elif isinstance(arg, (tuple, list)):
                if len(arg) != expected_count:
                    raise ValueError(
                        f"ONNX graph expects {expected_count} inputs ({expected_names}), "
                        f"but got {len(arg)} positional values"
                    )
                provided = {name: value for name, value in zip(expected_names, arg)}
            else:
                if expected_count != 1:
                    raise ValueError(
                        f"ONNX graph expects {expected_count} inputs ({expected_names}). "
                        "Pass a dict{name: value}, a tuple/list of positional inputs, "
                        "or multiple positional arguments."
                    )
                provided = {expected_names[0]: arg}
        else:
            if len(inputs) != expected_count:
                raise ValueError(
                    f"ONNX graph expects {expected_count} inputs ({expected_names}), "
                    f"but got {len(inputs)} positional values"
                )
            provided = {name: value for name, value in zip(expected_names, inputs)}

        missing = [name for name in expected_names if name not in provided]
        if missing:
            raise ValueError(f"Missing required ONNX inputs: {missing}")

        unknown = [name for name in provided.keys() if name not in expected_names]
        if unknown:
            raise ValueError(f"Unknown ONNX inputs provided: {unknown}")

        return {name: provided[name] for name in expected_names}

    def forward(self, *inputs, **named_inputs):
        env = self._bind_inputs(inputs, named_inputs)

        for node_idx, node in enumerate(self._graph.node):
            op = node.op_type
            if op == "Constant":
                continue

            ins = [self._value(env, n) for n in node.input if n != ""]

            if op == "Identity":
                out = ins[0]
            elif op == "Cast":
                to = int(self._get_attr(node, "to", -1))
                if to < 0:
                    raise ValueError("Cast requires 'to' attribute")
                src = ins[0]
                casted = self._cast_numpy(self._as_numpy(src), to)

                # Keep integer/bool casts as numpy to preserve exact meta values.
                if self._np.issubdtype(casted.dtype, self._np.integer) or casted.dtype == self._np.bool_:
                    out = casted
                elif isinstance(src, self._m.Tensor):
                    out = self._from_numpy_like(casted, src)
                else:
                    out = casted
            elif op == "Conv":
                data = self._as_tensor(ins[0])
                W = self._as_tensor(ins[1], data.device)
                B = self._as_tensor(ins[2], data.device) if len(ins) > 2 else None

                groups = int(self._get_attr(node, "group", 1))

                # ONNX Conv is typically NCHW. If caller passes NHWC by mistake and it
                # is unambiguous (last dim matches Conv in-channels), transpose to NCHW.
                if len(data.shape) == 4 and len(W.shape) == 4:
                    expected_in_ch = int(W.shape[1]) * groups
                    if int(data.shape[1]) != expected_in_ch and int(data.shape[3]) == expected_in_ch:
                        data = data.permute([0, 3, 1, 2]).contiguous()
                    elif int(data.shape[1]) > expected_in_ch:
                        # Some converted graphs may produce an extra channel from
                        # dynamic shape/pad flows; align to expected Conv channels.
                        arr = self._as_numpy(data)
                        arr = arr[:, :expected_in_ch, :, :]
                        data = self._from_numpy_like(arr.astype(self._np.float32), data)

                strides = [int(v) for v in self._get_attr(node, "strides", [1, 1])]
                pads = [int(v) for v in self._get_attr(node, "pads", [0, 0, 0, 0])]
                dilations = [int(v) for v in self._get_attr(node, "dilations", [1, 1])]
                auto_pad = self._get_attr(node, "auto_pad", "NOTSET")

                if len(strides) != 2:
                    raise ValueError(f"Conv only supports 2D strides, got {strides}")
                if len(pads) != 4:
                    raise ValueError(f"Conv only supports 2D pads [t,l,b,r], got {pads}")
                if len(dilations) != 2 or dilations != [1, 1]:
                    raise ValueError(f"Conv currently supports dilation=[1,1], got {dilations}")
                if strides[0] != strides[1]:
                    raise ValueError(f"Conv currently supports equal spatial stride, got {strides}")

                if str(auto_pad).upper() != "NOTSET":
                    pads = self._resolve_2d_auto_pad(auto_pad, [int(data.shape[2]), int(data.shape[3])], [int(W.shape[2]), int(W.shape[3])], strides, dilations)

                pt, pl, pb, pr = pads
                if any(v < 0 for v in (pt, pl, pb, pr)):
                    raise ValueError(f"Conv does not support negative padding, got {pads}")

                if any(v != 0 for v in (pt, pl, pb, pr)) and not (pt == pb and pl == pr and pt == pl):
                    data = self._pad_tensor_constant(data, [0, 0, pt, pl], [0, 0, pb, pr], value=0.0)
                    conv_padding = 0
                else:
                    conv_padding = int(pt)

                if groups == 1:
                    if B is None:
                        out = data.conv2d(W, stride=int(strides[0]), padding=conv_padding)
                    else:
                        out = data.conv2d(W, B, int(strides[0]), conv_padding)
                else:
                    x_np = self._as_numpy(data).astype(self._np.float32)
                    w_np = self._as_numpy(W).astype(self._np.float32)
                    b_np = None if B is None else self._as_numpy(B).reshape(-1).astype(self._np.float32)
                    y_np = self._conv2d_numpy_grouped(x_np, w_np, b_np, int(strides[0]), groups)
                    out = self._from_numpy_like(y_np, data)
            elif op == "MaxPool":
                data = self._as_tensor(ins[0])
                ks = self._get_attr(node, "kernel_shape", [2, 2])
                st = self._get_attr(node, "strides", ks)
                pd = self._get_attr(node, "pads", [0, 0, 0, 0])
                auto_pad = self._get_attr(node, "auto_pad", "NOTSET")
                ceil_mode = int(self._get_attr(node, "ceil_mode", 0))
                ks = [int(ks[0]), int(ks[1])]
                st = [int(st[0]), int(st[1])]
                pd = [int(v) for v in pd]

                if str(auto_pad).upper() != "NOTSET":
                    pd = self._resolve_2d_auto_pad(auto_pad, [int(data.shape[2]), int(data.shape[3])], ks, st)

                if (
                    st[0] == st[1]
                    and ks[0] == ks[1]
                    and pd[0] == pd[1] == pd[2] == pd[3]
                    and ceil_mode == 0
                ):
                    out = data.max_pool2d(int(ks[0]), int(st[0]), int(pd[0]))
                else:
                    x = self._as_numpy(data).astype(self._np.float32)
                    n, c, h, w = x.shape
                    pt, pl, pb, pr = pd
                    x = self._np.pad(
                        x,
                        ((0, 0), (0, 0), (pt, pb), (pl, pr)),
                        mode="constant",
                        constant_values=-self._np.inf,
                    )
                    out_h_nom = (h + pt + pb - ks[0]) / float(st[0]) + 1.0
                    out_w_nom = (w + pl + pr - ks[1]) / float(st[1]) + 1.0
                    out_h = int(self._np.ceil(out_h_nom) if ceil_mode else self._np.floor(out_h_nom))
                    out_w = int(self._np.ceil(out_w_nom) if ceil_mode else self._np.floor(out_w_nom))
                    out_h = max(out_h, 0)
                    out_w = max(out_w, 0)
                    y = self._np.empty((n, c, out_h, out_w), dtype=self._np.float32)
                    for oy in range(out_h):
                        sy = oy * st[0]
                        ey = sy + ks[0]
                        for ox in range(out_w):
                            sx = ox * st[1]
                            ex = sx + ks[1]
                            window = x[:, :, sy:ey, sx:ex]
                            y[:, :, oy, ox] = window.max(axis=(2, 3))
                    out = self._from_numpy_like(y, data)
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
            elif op == "Flatten":
                data = self._as_tensor(ins[0])
                axis = int(self._get_attr(node, "axis", 1))
                rank = len(data.shape)
                axis = axis if axis >= 0 else axis + rank
                if axis < 0 or axis > rank:
                    raise ValueError(f"Flatten axis out of range: axis={axis}, rank={rank}")
                shape = list(data.shape)
                d0 = 1
                for v in shape[:axis]:
                    d0 *= int(v)
                d1 = 1
                for v in shape[axis:]:
                    d1 *= int(v)
                out = data.reshape([d0, d1])
            elif op == "MatMul":
                a = self._as_tensor(ins[0])
                b = self._as_tensor(ins[1], a.device)
                out = a.__matmul__(b)
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
                if isinstance(ins[0], self._m.Tensor) or isinstance(ins[1], self._m.Tensor):
                    a = self._as_tensor(ins[0])
                    b = self._as_tensor(ins[1], a.device)
                    if len(a.shape) == len(b.shape) and len(a.shape) in (3, 4):
                        a, b = self._align_binary_pair_min(a, b)
                    out = a + b
                else:
                    out = self._as_numpy(ins[0]) + self._as_numpy(ins[1])
            elif op == "Mul":
                if isinstance(ins[0], self._m.Tensor) or isinstance(ins[1], self._m.Tensor):
                    a = self._as_tensor(ins[0])
                    b = self._as_tensor(ins[1], a.device)
                    if len(a.shape) == len(b.shape) and len(a.shape) in (3, 4):
                        a, b = self._align_binary_pair_min(a, b)
                    out = self._binary_tensor_op("Mul", a, b, lambda x, y: x * y, lambda x, y: x * y)
                else:
                    out = self._as_numpy(ins[0]) * self._as_numpy(ins[1])
            elif op == "Sub":
                if isinstance(ins[0], self._m.Tensor) or isinstance(ins[1], self._m.Tensor):
                    a = self._as_tensor(ins[0])
                    b = self._as_tensor(ins[1], a.device)
                    out = a - b
                else:
                    out = self._as_numpy(ins[0]) - self._as_numpy(ins[1])
            elif op == "Div":
                if isinstance(ins[0], self._m.Tensor) or isinstance(ins[1], self._m.Tensor):
                    a = self._as_tensor(ins[0])
                    b = self._as_tensor(ins[1], a.device)
                    out = a / b
                else:
                    out = self._as_numpy(ins[0]) / self._as_numpy(ins[1])
            elif op == "Softmax":
                x = self._as_tensor(ins[0])
                rank = len(x.shape)
                if rank == 0:
                    raise ValueError("Softmax requires rank >= 1")
                default_axis = -1 if self._opset >= 13 else 1
                axis = int(self._get_attr(node, "axis", default_axis))
                axis = axis if axis >= 0 else axis + rank
                if axis < 0 or axis >= rank:
                    raise ValueError(f"Softmax axis out of range: axis={axis}, rank={rank}")

                if axis == rank - 1:
                    out = x.softmax(-1)
                else:
                    perm = [d for d in range(rank) if d != axis] + [axis]
                    inv_perm = [0] * rank
                    for i, p in enumerate(perm):
                        inv_perm[p] = i
                    moved = x.permute(perm).contiguous()
                    sm = moved.softmax(-1)
                    out = sm.permute(inv_perm).contiguous()
            elif op in ("ReduceSum", "ReduceMean", "ReduceMax"):
                data = self._as_tensor(ins[0])
                in_shape = list(data.shape)
                rank = len(in_shape)

                if len(ins) > 1 and self._as_numpy(ins[1]).size > 0:
                    axes_raw = [int(v) for v in self._as_numpy(ins[1]).astype(self._np.int64).reshape(-1).tolist()]
                else:
                    axes_raw = self._get_attr(node, "axes", None)

                noop_with_empty_axes = int(self._get_attr(node, "noop_with_empty_axes", 0))
                keepdims = int(self._get_attr(node, "keepdims", 1))

                if axes_raw is None:
                    axes = list(range(rank))
                elif len(axes_raw) == 0:
                    axes = [] if noop_with_empty_axes == 1 else list(range(rank))
                else:
                    axes = sorted(set((ax if ax >= 0 else ax + rank) for ax in axes_raw))

                for ax in axes:
                    if ax < 0 or ax >= rank:
                        raise ValueError(f"{op} axis out of range: axis={ax}, rank={rank}")

                if len(axes) == 0:
                    out = data
                else:
                    keep_shape = [1 if i in axes else int(in_shape[i]) for i in range(rank)]
                    if op == "ReduceMax":
                        out = data
                        # reduce one axis at a time (descending to keep axis positions stable)
                        for ax in sorted(axes, reverse=True):
                            vals, _ = out.topk(1, ax, True, True)
                            out = vals
                    else:
                        out = data.sum_to_shape(keep_shape)

                        if op == "ReduceMean":
                            count = 1
                            for ax in axes:
                                count *= int(in_shape[ax])
                            out = out / self._scalar_tensor(float(count), out)

                    if keepdims == 0:
                        final_shape = [int(in_shape[i]) for i in range(rank) if i not in axes]
                        if len(final_shape) == 0:
                            final_shape = [1]
                        out = out.reshape(final_shape)
            elif op == "Concat":
                axis = int(self._get_attr(node, "axis", 0))
                all_numpy = not any(isinstance(v, self._m.Tensor) for v in ins)

                if all_numpy:
                    arrs = [self._as_numpy(v) for v in ins]
                    rank = arrs[0].ndim
                    axis = axis if axis >= 0 else axis + rank
                    if axis < 0 or axis >= rank:
                        raise ValueError(f"Concat axis out of range: axis={axis}, rank={rank}")

                    mismatch = False
                    for d in range(rank):
                        if d == axis:
                            continue
                        vals = {int(a.shape[d]) for a in arrs}
                        if len(vals) > 1:
                            mismatch = True
                            break

                    if mismatch and rank == 4 and axis == 1:
                        target = [None] * rank
                        for d in range(rank):
                            if d == axis:
                                continue
                            target[d] = min(int(a.shape[d]) for a in arrs)
                        cropped = []
                        for a in arrs:
                            sl = [slice(None)] * rank
                            for d in range(rank):
                                if d == axis:
                                    continue
                                sl[d] = slice(0, int(target[d]))
                            cropped.append(a[tuple(sl)])
                        arrs = cropped

                    out = self._np.concatenate(arrs, axis=axis)
                else:
                    base = self._as_tensor(ins[0])
                    ts = [base]
                    for v in ins[1:]:
                        ts.append(self._as_tensor(v, base.device))

                    rank = len(base.shape)
                    axis = axis if axis >= 0 else axis + rank
                    if axis < 0 or axis >= rank:
                        raise ValueError(f"Concat axis out of range: axis={axis}, rank={rank}")

                    shapes = [[int(d) for d in t.shape] for t in ts]
                    mismatch = False
                    for d in range(rank):
                        if d == axis:
                            continue
                        vals = {shape[d] for shape in shapes}
                        if len(vals) > 1:
                            mismatch = True
                            break

                    if mismatch and rank == 4 and axis == 1:
                        target = [None] * rank
                        for d in range(rank):
                            if d == axis:
                                continue
                            target[d] = min(shape[d] for shape in shapes)
                        cropped = []
                        for t in ts:
                            arr = self._as_numpy(t)
                            sl = [slice(None)] * rank
                            for d in range(rank):
                                if d == axis:
                                    continue
                                sl[d] = slice(0, int(target[d]))
                            cropped.append(self._from_numpy_like(arr[tuple(sl)].astype(self._np.float32), t))
                        ts = cropped
                        shapes = [[int(d) for d in t.shape] for t in ts]

                    for i, tshape in enumerate(shapes[1:], start=1):
                        for d in range(rank):
                            if d != axis and tshape[d] != shapes[0][d]:
                                raise ValueError(
                                    "Concat shape mismatch: "
                                    f"axis={axis}, input0_shape={shapes[0]}, input{i}_shape={tshape}"
                                )

                    out = self._m.Tensor.cat(ts, axis)
            elif op == "Reshape":
                data = self._as_tensor(ins[0])
                raw_shape = [int(v) for v in self._as_numpy(ins[1]).reshape(-1).tolist()]
                in_shape = [int(v) for v in data.shape]
                allowzero = int(self._get_attr(node, "allowzero", 0))

                if len(raw_shape) == 0:
                    raise ValueError("Reshape requires non-empty target shape")

                resolved_shape = []
                infer_idx = -1
                known_product = 1

                for i, d in enumerate(raw_shape):
                    if d == -1:
                        if infer_idx != -1:
                            raise ValueError("Reshape allows at most one -1 dimension")
                        infer_idx = i
                        resolved_shape.append(-1)
                        continue

                    if d == 0 and allowzero == 0:
                        if i >= len(in_shape):
                            raise ValueError(
                                f"Reshape uses 0 for dim copy at axis {i}, but input rank is {len(in_shape)}"
                            )
                        d = in_shape[i]

                    if d < 0:
                        raise ValueError(f"Reshape has invalid negative dim {d} at axis {i}")

                    resolved_shape.append(int(d))
                    known_product *= int(d)

                input_elems = 1
                for d in in_shape:
                    input_elems *= int(d)

                if infer_idx != -1:
                    if known_product == 0:
                        raise ValueError("Reshape with -1 cannot infer when known product is zero")
                    if input_elems % known_product != 0:
                        raise ValueError(
                            f"Reshape cannot infer -1 dimension: input_elems={input_elems}, known_product={known_product}"
                        )
                    resolved_shape[infer_idx] = int(input_elems // known_product)
                else:
                    if known_product != input_elems:
                        raise ValueError(
                            f"Reshape: element count mismatch (input={input_elems}, target={known_product})"
                        )

                if isinstance(ins[0], self._m.Tensor):
                    out = data.reshape(resolved_shape)
                else:
                    out = self._as_numpy(ins[0]).reshape(resolved_shape)
            elif op == "Transpose":
                data = self._as_tensor(ins[0])
                perm = self._get_attr(node, "perm", list(range(len(data.shape) - 1, -1, -1)))
                out = data.permute([int(v) for v in perm]).contiguous()
            elif op == "Unsqueeze":
                base = ins[0]
                axes = self._get_attr(node, "axes", None)
                if axes is None and len(ins) > 1:
                    axes = [int(v) for v in self._as_numpy(ins[1]).reshape(-1).tolist()]

                if isinstance(base, self._m.Tensor):
                    data = self._as_tensor(base)
                    old_shape = list(data.shape)
                    rank = len(old_shape)
                    norm = sorted([a if a >= 0 else a + rank + len(axes) for a in axes])
                    for ax in norm:
                        old_shape.insert(ax, 1)
                    out = data.reshape(old_shape)
                else:
                    arr = self._as_numpy(base)
                    rank = arr.ndim
                    norm = sorted([a if a >= 0 else a + rank + len(axes) for a in axes])
                    new_shape = list(arr.shape)
                    for ax in norm:
                        new_shape.insert(ax, 1)
                    out = arr.reshape(new_shape)
            elif op == "Squeeze":
                base = ins[0]
                axes = self._get_attr(node, "axes", None)
                if axes is None and len(ins) > 1:
                    axes = [int(v) for v in self._as_numpy(ins[1]).reshape(-1).tolist()]

                if isinstance(base, self._m.Tensor):
                    shape = list(self._as_tensor(base).shape)
                else:
                    shape = list(self._as_numpy(base).shape)

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

                if isinstance(base, self._m.Tensor):
                    out = self._as_tensor(base).reshape(new_shape)
                else:
                    out = self._as_numpy(base).reshape(new_shape)
            elif op == "ConstantOfShape":
                shp = self._as_numpy(ins[0]).astype(self._np.int64).reshape(-1).tolist()
                value = 0.0
                for a in node.attribute:
                    if a.name == "value" and a.type == self._onnx.AttributeProto.TENSOR:
                        v = self._onnx_numpy_helper.to_array(a.t)
                        value = float(v.reshape(-1)[0]) if v.size > 0 else 0.0
                out = self._from_numpy_like(self._np.full(shp, value, dtype=self._np.float32))
            elif op == "Expand":
                data_np = self._as_numpy(ins[0]).astype(self._np.float32)
                shape = [int(v) for v in self._as_numpy(ins[1]).astype(self._np.int64).reshape(-1).tolist()]
                out = self._from_numpy_like(self._np.broadcast_to(data_np, shape).copy(), ins[0])
            elif op == "Tile":
                data_np = self._as_numpy(ins[0]).astype(self._np.float32)
                reps = [int(v) for v in self._as_numpy(ins[1]).astype(self._np.int64).reshape(-1).tolist()]
                out = self._from_numpy_like(self._np.tile(data_np, reps), ins[0])
            elif op == "Gather":
                data_np = self._as_numpy(ins[0])
                idx_np = self._as_numpy(ins[1]).astype(self._np.int64)
                axis = int(self._get_attr(node, "axis", 0))
                gathered = self._np.take(data_np, idx_np, axis=axis)
                if isinstance(ins[0], self._m.Tensor):
                    out = self._from_numpy_like(gathered.astype(self._np.float32), ins[0])
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
                    if isinstance(ins[0], self._m.Tensor):
                        env[out_name] = self._m.from_numpy(self._np.asarray(part, dtype=self._np.float32))
                    else:
                        env[out_name] = self._np.asarray(part)
                continue
            elif op == "Resize":
                data = self._as_tensor(ins[0])
                mode = str(self._get_attr(node, "mode", "nearest"))
                if mode != "nearest":
                    raise ValueError("Resize only supports nearest mode")

                def _optional_input(slot_idx):
                    if slot_idx >= len(node.input):
                        return None
                    name = node.input[slot_idx]
                    if name == "":
                        return None
                    return self._value(env, name)

                scales = None
                sizes = None
                scales_in = _optional_input(2)
                sizes_in = _optional_input(3)
                if scales_in is not None and self._as_numpy(scales_in).size > 0:
                    scales = self._as_numpy(scales_in).astype(self._np.float32).reshape(-1)
                if sizes_in is not None and self._as_numpy(sizes_in).size > 0:
                    sizes = self._as_numpy(sizes_in).astype(self._np.int64).reshape(-1)

                in_shape = [int(v) for v in data.shape]
                if len(in_shape) != 4:
                    raise ValueError(f"Resize currently supports only 4D tensors, got shape={in_shape}")

                if sizes is not None and len(sizes) >= 4:
                    out_h = int(sizes[-2])
                    out_w = int(sizes[-1])
                elif scales is not None and len(scales) >= 4:
                    out_h = int(round(float(in_shape[-2]) * float(scales[-2])))
                    out_w = int(round(float(in_shape[-1]) * float(scales[-1])))
                else:
                    raise ValueError("Resize requires scales or sizes input")

                if out_h <= 0 or out_w <= 0:
                    raise ValueError(f"Resize target spatial size must be positive, got H={out_h}, W={out_w}")

                sf_h = out_h / float(in_shape[-2])
                sf_w = out_w / float(in_shape[-1])
                if abs(sf_h - sf_w) < 1e-6 and sf_h >= 1.0 and abs(sf_h - round(sf_h)) < 1e-6:
                    out = data.upsample2d(int(round(sf_h)))
                else:
                    x_np = self._as_numpy(data).astype(self._np.float32)
                    y_np = self._resize_nearest_numpy(x_np, out_h, out_w)
                    out = self._from_numpy_like(y_np.astype(self._np.float32), data)
            elif op == "Log":
                x = self._as_tensor(ins[0])
                out = x.log()
            elif op == "Sqrt":
                x = self._as_tensor(ins[0])
                if self._sqrt_clamp_nonneg:
                    max_f = self._np.finfo(self._np.float32).max
                    x = x.clip(0.0, float(max_f))
                out = x.sqrt()
            elif op == "Clip":
                x = self._as_tensor(ins[0])
                if len(ins) > 1 and self._as_numpy(ins[1]).size > 0:
                    min_v = float(self._as_numpy(ins[1]).reshape(-1)[0])
                else:
                    min_v = float(self._get_attr(node, "min", -3.4028235e38))
                if len(ins) > 2 and self._as_numpy(ins[2]).size > 0:
                    max_v = float(self._as_numpy(ins[2]).reshape(-1)[0])
                else:
                    max_v = float(self._get_attr(node, "max", 3.4028235e38))
                out = x.clip(min_v, max_v)
            elif op == "Erf":
                x = self._as_tensor(ins[0])
                out = x.erf()
            elif op == "GatherElements":
                data = self._as_tensor(ins[0])
                idx = self._as_tensor(ins[1], data.device)
                axis = int(self._get_attr(node, "axis", 0))
                out = data.gather_elements(idx, axis)
            elif op == "TopK":
                x = self._as_tensor(ins[0])
                if len(ins) < 2:
                    raise ValueError("TopK requires k input")
                k = int(self._as_numpy(ins[1]).astype(self._np.int64).reshape(-1)[0])
                largest = int(self._get_attr(node, "largest", 1)) != 0
                sorted_flag = int(self._get_attr(node, "sorted", 1)) != 0
                axis = int(self._get_attr(node, "axis", -1))
                values, indices = x.topk(k, axis, largest, sorted_flag)
                if len(node.output) != 2:
                    raise ValueError(f"Unsupported output arity for TopK: {len(node.output)}")
                self._check_nonfinite_output(node, values, node.output[0], env)
                self._check_nonfinite_output(node, indices, node.output[1], env)
                env[node.output[0]] = values
                env[node.output[1]] = indices
                continue
            elif op == "GridSample":
                x = self._as_tensor(ins[0])
                grid = self._as_tensor(ins[1], x.device)
                mode = self._get_attr(node, "mode", "bilinear")
                padding_mode = self._get_attr(node, "padding_mode", "zeros")
                align_corners = int(self._get_attr(node, "align_corners", 0)) != 0
                if padding_mode != "zeros":
                    raise ValueError("GridSample currently supports only padding_mode='zeros'")
                if mode not in ("bilinear", "nearest"):
                    raise ValueError("GridSample currently supports mode in {'bilinear','nearest'}")
                out = x.grid_sample(grid, mode, align_corners)
            elif op == "Pad":
                data = self._as_tensor(ins[0])
                mode = self._get_attr(node, "mode", "constant")
                if mode != "constant":
                    raise ValueError("Pad currently supports only constant mode")

                if len(ins) < 2:
                    raise ValueError("Pad requires pads input")
                pads_arr = self._as_numpy(ins[1]).reshape(-1)
                if self._np.issubdtype(pads_arr.dtype, self._np.floating):
                    if not self._np.all(self._np.isfinite(pads_arr)):
                        raise ValueError(f"Pad received non-finite pads values: {pads_arr.tolist()}")
                    if not self._np.all(self._np.isclose(pads_arr, self._np.round(pads_arr), atol=1e-5)):
                        raise ValueError(f"Pad received non-integer pads values: {pads_arr.tolist()}")
                pads = [int(v) for v in pads_arr.astype(self._np.int64).tolist()]
                rank = len(data.shape)
                if len(pads) != 2 * rank:
                    raise ValueError(f"Pad expects pads length {2*rank}, got {len(pads)}")

                value = 0.0
                if len(ins) > 2 and self._as_numpy(ins[2]).size > 0:
                    value = float(self._as_numpy(ins[2]).reshape(-1)[0])
                elif len(ins) > 3 and self._as_numpy(ins[3]).size > 0:
                    value = float(self._as_numpy(ins[3]).reshape(-1)[0])
                else:
                    value = float(self._get_attr(node, "value", 0.0))

                pads_begin = pads[:rank]
                pads_end = pads[rank:]
                out = self._pad_tensor_constant(data, pads_begin, pads_end, value=value)
            elif op == "Pow":
                a = self._as_numpy(ins[0]).astype(self._np.float64)
                b = self._as_numpy(ins[1]).astype(self._np.float64)
                with self._np.errstate(over="ignore", invalid="ignore"):
                    y = self._np.power(a, b)

                if self._pow_clamp_finite and self._np.issubdtype(y.dtype, self._np.floating):
                    finfo = self._np.finfo(self._np.float32)
                    y = self._np.nan_to_num(y, nan=0.0, posinf=finfo.max, neginf=-finfo.max)
                    y = self._np.clip(y, -finfo.max, finfo.max)

                out = self._from_numpy_like(y.astype(self._np.float32), ins[0])
            elif op == "Floor":
                arr = self._as_numpy(ins[0]).astype(self._np.float32)
                out = self._from_numpy_like(self._np.floor(arr).astype(self._np.float32), ins[0])
            else:
                raise ValueError(f"Unsupported op in graph runtime: {op}")

            if len(node.output) != 1:
                raise ValueError(f"Unsupported output arity for op {op}: {len(node.output)}")
            self._check_nonfinite_output(node, out, node.output[0], env)
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
    "Softmax",
    "ReduceSum",
    "ReduceMean",
    "ReduceMax",
    "Log",
    "Sqrt",
    "Clip",
    "Erf",
    "Pad",
    "TopK",
    "GridSample",
    "Relu",
    "LeakyRelu",
    "GlobalAveragePool",
    "Gemm",
    "MatMul",
    "Flatten",
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
    "GatherElements",
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
            "compile_onnx: conversion failed due to missing graph-runtime op implementations. "
            f"model='{model_path}', "
            f"missing_runtime_unique={runtime_info['missing_runtime_unique']}, "
            f"missing_runtime_total={runtime_info['missing_runtime_total']}"
        )

    module = _compile_onnx_graph_module(model_path)

    if output_path is not None:
        module.export_npz(output_path)

    if debug:
        print(
            f"[compile_onnx] success model={model_path} "
            f"nodes={len(model.graph.node)} unique_ops={len(fail_info['op_counts'])}"
        )

    return module


def export_onnx_npz(model_path, output_path, debug=False):
    """Compile ONNX graph module and export it as a compressed NPZ bundle."""
    module = compile_onnx(model_path, output_path=output_path, debug=debug)
    return output_path


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
            coverage.setdefault(entry["status"], []).append(op)

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
inference.export_onnx_npz = export_onnx_npz
inference.report_onnx_unsupported_ops = report_onnx_unsupported_ops
inference.onnx_native_conversion_map = onnx_native_conversion_map
inference.onnx_conversion_coverage_report = onnx_conversion_coverage_report
inference.download_yolov5n_onnx = download_yolov5n_onnx
