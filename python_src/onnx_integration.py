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
        self._device = device if device is not None else munet.Device(munet.DeviceType.CPU, 0)

        if providers is None:
            available = set(ort.get_available_providers())
            chosen = []
            if self._device.type == munet.DeviceType.CUDA and "CUDAExecutionProvider" in available:
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
            return x.detach().to(self._m.Device(self._m.DeviceType.CPU, 0)).numpy().astype(self._np.float32, copy=False)
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

def load_onnx(model_path, device=None, providers=None):
    """Create an ONNXRuntime-backed inference engine.

    Args:
        model_path: Path to ONNX file.
        device: Optional MuNet device for output tensors.
        providers: Optional explicit ORT provider list.
    """
    return ONNXEngine(model_path, device=device, providers=providers)

def compile_onnx(model_path, output_path=None, ignore_unsupported=False, report_only=False, allow_partial=False, debug=False):
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
        raise ValueError("compile_onnx: report_only=True does not compile/save. Remove output_path to get unsupported-op report.")

    model = onnx.load(model_path)
    graph = model.graph

    consts = {init.name: numpy_helper.to_array(init).astype(np.float32) for init in graph.initializer}
    unsupported_ops = set()
    seq_layers = []
    lowered_count = 0

    # Track simple single-stream tensor name through graph.
    stream_name = graph.input[0].name if len(graph.input) > 0 else None

    def _log(msg):
        if debug:
            print(f"[compile_onnx][debug] {msg}")

    def _unsupported(op):
        unsupported_ops.add(op)
        _log(f"unsupported op: {op}")
        if not ignore_unsupported and not report_only:
            raise ValueError(
                f"compile_onnx: unsupported op '{op}'. "
                "Use munet.inference.load_onnx for full ONNXRuntime execution."
            )

    def _const(name):
        arr = consts.get(name)
        if arr is None:
            return None
        return munet.from_numpy(np.asarray(arr, dtype=np.float32))

    def _get_attr(node, name, default=None):
        for a in node.attribute:
            if a.name != name:
                continue
            if a.type == onnx.AttributeProto.INT:
                return int(a.i)
            if a.type == onnx.AttributeProto.FLOAT:
                return float(a.f)
            if a.type == onnx.AttributeProto.INTS:
                return [int(v) for v in a.ints]
            if a.type == onnx.AttributeProto.FLOATS:
                return [float(v) for v in a.floats]
        return default

    class _TensorOp(munet.nn.Module):
        def __init__(self, fn):
            super().__init__()
            self._fn = fn
        def forward(self, x):
            return self._fn(x)

    for node in graph.node:
        op = node.op_type
        _log(f"node op={op} inputs={list(node.input)} outputs={list(node.output)}")

        if op == "Constant":
            # Capture constant outputs for downstream constant-fed ops.
            v = None
            for a in node.attribute:
                if a.name == "value" and a.type == onnx.AttributeProto.TENSOR:
                    v = numpy_helper.to_array(a.t).astype(np.float32)
                    break
            if v is not None and len(node.output) > 0:
                consts[node.output[0]] = v
            continue

        if op in ("Identity", "Cast"):
            if len(node.output) > 0:
                stream_name = node.output[0]
            continue

        if stream_name is None or len(node.input) == 0:
            _unsupported(op)
            continue

        if node.input[0] != stream_name:
            # non-linear dataflow/branch currently unsupported for native lowering
            _unsupported(op)
            continue

        if op == "Gemm":
            W = _const(node.input[1])
            if W is None:
                _unsupported("Gemm")
                continue

            transB = _get_attr(node, "transB", 0)
            transA = _get_attr(node, "transA", 0)
            alpha = _get_attr(node, "alpha", 1.0)
            beta = _get_attr(node, "beta", 1.0)

            if transA != 0 or alpha != 1.0 or beta != 1.0:
                _unsupported("Gemm")
                continue

            if transB == 0:
                in_features = int(W.shape[0]); out_features = int(W.shape[1]); weight_native = W
            else:
                in_features = int(W.shape[1]); out_features = int(W.shape[0]); weight_native = W.transpose(0, 1).contiguous()

            layer = munet.nn.Linear(in_features, out_features, len(node.input) >= 3)
            layer.weight.replace_(weight_native)
            if len(node.input) >= 3:
                B = _const(node.input[2])
                if B is None:
                    _unsupported("Gemm")
                    continue
                layer.bias.replace_(B.reshape([out_features]))
            seq_layers.append(layer)

        elif op == "MatMul":
            B = _const(node.input[1])
            if B is None:
                _unsupported("MatMul")
                continue
            in_features = int(B.shape[0]); out_features = int(B.shape[1])
            layer = munet.nn.Linear(in_features, out_features, False)
            layer.weight.replace_(B)
            seq_layers.append(layer)

        elif op in ("Add", "Sub", "Mul", "Div"):
            c = _const(node.input[1])
            if c is None:
                _unsupported(op)
                continue
            if op == "Add":
                seq_layers.append(_TensorOp(lambda x, c=c: x + c))
            elif op == "Sub":
                seq_layers.append(_TensorOp(lambda x, c=c: x - c))
            elif op == "Div":
                seq_layers.append(_TensorOp(lambda x, c=c: x / c))
            else:
                seq_layers.append(_TensorOp(lambda x, c=c: x * c))

        elif op == "Relu":
            seq_layers.append(munet.nn.ReLU())
        elif op == "Sigmoid":
            seq_layers.append(munet.nn.Sigmoid())
        elif op == "Tanh":
            seq_layers.append(munet.nn.Tanh())
        elif op == "Flatten":
            seq_layers.append(munet.nn.Flatten())
        elif op == "Softmax":
            axis = _get_attr(node, "axis", -1)
            seq_layers.append(_TensorOp(lambda x, axis=axis: x.softmax(axis)))
        elif op == "Reshape":
            shp = consts.get(node.input[1])
            if shp is None:
                _unsupported("Reshape")
                continue
            shp_list = [int(v) for v in np.asarray(shp).tolist()]
            seq_layers.append(_TensorOp(lambda x, shp_list=shp_list: x.reshape(shp_list)))
        elif op == "Transpose":
            perm = _get_attr(node, "perm", None)
            if perm is None:
                _unsupported("Transpose")
                continue
            seq_layers.append(_TensorOp(lambda x, perm=perm: x.permute(perm)))
        elif op == "Unsqueeze":
            axes = _get_attr(node, "axes", None)
            if axes is None and len(node.input) > 1 and node.input[1] in consts:
                axes = [int(v) for v in np.asarray(consts[node.input[1]]).tolist()]
            if axes is None:
                _unsupported("Unsqueeze")
                continue
            def _unsq(x, axes=axes):
                shape = list(x.shape)
                for a in sorted([(ax if ax >= 0 else ax + len(shape) + 1) for ax in axes]):
                    shape.insert(a, 1)
                return x.reshape(shape)
            seq_layers.append(_TensorOp(_unsq))
        elif op == "Squeeze":
            axes = _get_attr(node, "axes", None)
            if axes is None and len(node.input) > 1 and node.input[1] in consts:
                axes = [int(v) for v in np.asarray(consts[node.input[1]]).tolist()]
            if axes is None:
                _unsupported("Squeeze")
                continue
            def _sq(x, axes=axes):
                shape = list(x.shape)
                rm = sorted([(ax if ax >= 0 else ax + len(shape)) for ax in axes], reverse=True)
                for a in rm:
                    if shape[a] != 1:
                        raise RuntimeError("Squeeze: axis dim must be 1")
                    shape.pop(a)
                return x.reshape(shape)
            seq_layers.append(_TensorOp(_sq))
        elif op == "Concat":
            axis = _get_attr(node, "axis", 0)
            others = []
            ok = True
            for inp in node.input[1:]:
                t = _const(inp)
                if t is None:
                    ok = False
                    break
                others.append(t)
            if not ok:
                _unsupported("Concat")
                continue
            seq_layers.append(_TensorOp(lambda x, others=others, axis=axis: munet.Tensor.cat([x] + others, axis)))
        elif op == "Conv":
            W = _const(node.input[1])
            if W is None:
                _unsupported("Conv")
                continue
            B = _const(node.input[2]) if len(node.input) > 2 else None
            strides = _get_attr(node, "strides", [1, 1])
            pads = _get_attr(node, "pads", [0, 0, 0, 0])
            dil = _get_attr(node, "dilations", [1, 1])
            group = _get_attr(node, "group", 1)
            if group != 1 or dil != [1, 1] or pads[0] != pads[2] or pads[1] != pads[3]:
                _unsupported("Conv")
                continue
            oc, ic, kh, kw = [int(v) for v in W.shape]
            if kh != kw:
                _unsupported("Conv")
                continue
            layer = munet.nn.Conv2d(ic, oc, kh, strides[0], pads[0])
            layer.weight.replace_(W)
            if B is not None:
                layer.bias.replace_(B.reshape([oc]))
            seq_layers.append(layer)
        elif op == "MaxPool":
            ks = _get_attr(node, "kernel_shape", None)
            st = _get_attr(node, "strides", ks)
            pd = _get_attr(node, "pads", [0, 0, 0, 0])
            if ks is None or len(ks) != 2 or ks[0] != ks[1] or st[0] != st[1] or pd[0] != pd[2] or pd[1] != pd[3]:
                _unsupported("MaxPool")
                continue
            seq_layers.append(munet.nn.MaxPool2d(int(ks[0]), int(st[0]), int(pd[0])))
        else:
            _unsupported(op)

        lowered_count += 1

        if len(node.output) > 0:
            stream_name = node.output[0]

    if report_only:
        if debug:
            print(f"[compile_onnx] report_only unsupported_ops={sorted(list(unsupported_ops))}")
        return sorted(list(unsupported_ops))

    if unsupported_ops:
        skipped = sorted(list(unsupported_ops))
        if ignore_unsupported:
            if not allow_partial:
                raise ValueError(
                    "compile_onnx: graph contains unsupported ops; refusing to emit partial model. "
                    f"Unsupported ops: {skipped}. "
                    "Use report_only=True to inspect, or set allow_partial=True to emit best-effort partial model."
                )
            print(f"[compile_onnx] warning: emitting partial model; skipped unsupported ops: {skipped}")
        # non-ignore case already raised at first unsupported op

    if not seq_layers:
        raise ValueError("compile_onnx: no supported nodes found in graph")

    if debug:
        print(f"[compile_onnx] lowered_nodes={lowered_count} total_nodes={len(graph.node)} layers={len(seq_layers)}")

    module = munet.nn.Sequential(seq_layers)
    if output_path is not None:
        munet.save(module, output_path)
    return module


def report_onnx_unsupported_ops(model_path):
    """Return a sorted unique list of unsupported ONNX ops for native compile."""
    return compile_onnx(model_path, output_path=None, ignore_unsupported=True, report_only=True, allow_partial=True, debug=False)

inference.ONNXEngine = ONNXEngine
inference.load_onnx = load_onnx
inference.compile_onnx = compile_onnx
inference.report_onnx_unsupported_ops = report_onnx_unsupported_ops
