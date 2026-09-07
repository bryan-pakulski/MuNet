"""Strict ONNX 13..18 interchange. Imports execute through MuNet's native graph."""
import tempfile
from pathlib import Path
import numpy as np
from . import _native
from .serialization import from_graph


class UnsupportedOperatorError(ValueError):
    pass


SUPPORTED_ONNX = {
    "Add", "Sub", "Mul", "Div", "Neg", "MatMul", "Gemm", "Relu", "Sigmoid",
    "Exp", "Log", "Sqrt", "Transpose", "Reshape", "Expand", "ReduceSum", "ReduceMean",
    "Identity", "Constant",
}


def from_onnx(path, *, device="vulkan", fuse=True):
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    model = onnx.load(str(path), load_external_data=False)
    if model.functions or model.training_info or model.graph.sparse_initializer:
        raise UnsupportedOperatorError("ONNX functions, training_info, and sparse initializers are outside the v0 contract")
    if any(t.data_location == TensorProto.EXTERNAL for t in model.graph.initializer):
        raise UnsupportedOperatorError("external tensor data is unsupported in v0; save a self-contained ONNX model")
    imports = {x.domain: x.version for x in model.opset_import}
    if set(imports) != {""} or not 13 <= imports[""] <= 18:
        raise UnsupportedOperatorError(f"supported contract is default-domain ONNX opset 13..18; received {imports}")
    unsupported = [(n.name or f"node_{i}", n.domain, n.op_type) for i, n in enumerate(model.graph.node)
                   if n.domain or n.op_type not in SUPPORTED_ONNX or len(n.output) != 1]
    if unsupported:
        raise UnsupportedOperatorError(f"unsupported ONNX nodes: {unsupported}")
    onnx.checker.check_model(model, full_check=True)
    graph = _native.Graph()
    values, constants = {}, {}
    specs, input_values = [], []
    initializers = {x.name: x for x in model.graph.initializer}
    if set(initializers) & {v.name for v in model.graph.input}:
        raise UnsupportedOperatorError("overridable initializers are unsupported; remove them from graph inputs")

    def tensor(name, array):
        array = np.asarray(array)
        constants[name] = array
        if array.dtype == np.float32:
            values[name] = graph.leaf("parameter", name, array.shape, array.ravel().tolist())
        elif array.dtype not in (np.int32, np.int64):
            raise UnsupportedOperatorError(f"unsupported tensor dtype {array.dtype}: {name}")

    def scalar(value):
        return graph.leaf("constant", f"__munet_scalar_{len(graph.nodes())}", [], [float(value)])

    def integer(name):
        if name not in constants or constants[name].dtype not in (np.int32, np.int64):
            raise UnsupportedOperatorError(f"expected a constant integer shape/axes tensor: {name}")
        return constants[name].reshape(-1).tolist()

    for value in model.graph.input:
        t = value.type.tensor_type
        if t.elem_type != TensorProto.FLOAT or any(not d.HasField("dim_value") or d.dim_value <= 0 for d in t.shape.dim):
            raise UnsupportedOperatorError(f"input {value.name} needs a static positive float32 shape")
        shape = [d.dim_value for d in t.shape.dim]
        values[value.name] = graph.leaf("input", value.name, shape)
        input_values.append(values[value.name]); specs.append(shape)
    for name, value in initializers.items():
        tensor(name, numpy_helper.to_array(value))
    simple = {"Add":"add", "Sub":"sub", "Mul":"mul", "Div":"div", "Neg":"neg", "MatMul":"matmul",
              "Relu":"relu", "Sigmoid":"sigmoid", "Exp":"exp", "Log":"log", "Sqrt":"sqrt", "Identity":"identity"}
    for node in model.graph.node:
        op = node.op_type
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        name = node.output[0]
        if op == "Constant":
            if set(attrs) != {"value"}:
                raise UnsupportedOperatorError("v0 Constant requires a tensor-valued 'value' attribute")
            tensor(name, numpy_helper.to_array(attrs["value"]))
            continue
        try:
            if op in simple:
                value = graph.op(simple[op], [values[x] for x in node.input])
            elif op == "Gemm":
                a, b = [values[x] for x in node.input[:2]]
                if attrs.get("transA", 0): a = graph.op("transpose", [a])
                if attrs.get("transB", 0): b = graph.op("transpose", [b])
                value = graph.op("matmul", [a, b])
                if attrs.get("alpha", 1.0) != 1.0: value = graph.op("mul", [value, scalar(attrs["alpha"])])
                if len(node.input) > 2 and node.input[2]:
                    c = values[node.input[2]]
                    if attrs.get("beta", 1.0) != 1.0: c = graph.op("mul", [c, scalar(attrs["beta"])])
                    value = graph.op("add", [value, c])
            elif op == "Transpose":
                a = values[node.input[0]]
                if len(graph.shape(a)) != 2 or attrs.get("perm", [1, 0]) != [1, 0]:
                    raise UnsupportedOperatorError("v0 Transpose only supports rank-2 perm=[1,0]")
                value = graph.op("transpose", [a])
            elif op in ("Reshape", "Expand"):
                a = values[node.input[0]]
                shape = integer(node.input[1])
                if op == "Reshape":
                    original = graph.shape(a)
                    if not attrs.get("allowzero", 0):
                        shape = [original[i] if d == 0 else d for i, d in enumerate(shape)]
                    if shape.count(-1) == 1:
                        count = int(np.prod([d for d in shape if d != -1]))
                        if count <= 0 or int(np.prod(original)) % count:
                            raise UnsupportedOperatorError("invalid inferred reshape")
                        shape[shape.index(-1)] = int(np.prod(original)) // count
                value = graph.op("reshape" if op == "Reshape" else "broadcast", [a], shape)
            elif op in ("ReduceSum", "ReduceMean"):
                a = values[node.input[0]]
                shape = graph.shape(a)
                axes = integer(node.input[1]) if len(node.input) > 1 and node.input[1] else attrs.get("axes", [])
                if not axes and not attrs.get("noop_with_empty_axes", 0): axes = list(range(len(shape)))
                axes = [x + len(shape) if x < 0 else x for x in axes]
                value = graph.op("sum", [a], axes)
                if op == "ReduceMean": value = graph.op("div", [value, scalar(np.prod([shape[x] for x in axes]))])
                if not attrs.get("keepdims", 1): value = graph.op("reshape", [value], [d for i, d in enumerate(shape) if i not in axes])
            else:
                raise UnsupportedOperatorError(f"no lowering for {op}")
        except (KeyError, IndexError, ValueError) as exc:
            raise UnsupportedOperatorError(f"cannot lower {node.name or name} ({op}): {exc}") from exc
        values[name] = value
    outputs = []
    for value in model.graph.output:
        if value.type.tensor_type.elem_type != TensorProto.FLOAT or value.name not in values:
            raise UnsupportedOperatorError("v0 only supports float32 tensor outputs")
        result = values[value.name]
        declared = value.type.tensor_type.shape.dim
        actual = graph.shape(result)
        if len(declared) != len(actual) or any(d.HasField("dim_value") and d.dim_value != a for d, a in zip(declared, actual)):
            raise ValueError(f"declared output shape differs from native shape: {value.name}")
        outputs.append(result)
    # Match only live input bindings after native dead-code elimination.
    plan = _native.Plan(graph, outputs, [], fuse)
    feeds = [input_values.index(i) for i in plan.inputs]
    return from_graph(graph, outputs, [], specs, feeds, len(outputs) == 1, device=device, fuse=fuse)


def to_onnx(program, path):
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    if program._plan is None:
        raise ValueError("call the compiled program before exporting")
    if program._plan.updates:
        raise UnsupportedOperatorError("ONNX export is an inference graph; compile the trained model separately before exporting")
    with program._lock:
        for p, value in program._parameters.items():
            if p._version != program._seen_versions[p]:
                program._plan.write(value, np.ascontiguousarray(p.numpy()))
                program._seen_versions[p] = p._version
        nodes = program._plan.graph.nodes()
        live = set()
        def visit(i):
            if i in live: return
            live.add(i)
            for parent in nodes[i]["inputs"]: visit(parent)
        for i in program._plan.outputs: visit(i)
        inputs, initializers, operations = [], [], []
        names = [f"v{i}" for i in range(len(nodes))]
        simple = {"add":"Add", "sub":"Sub", "mul":"Mul", "div":"Div", "neg":"Neg", "matmul":"MatMul",
                  "relu":"Relu", "sigmoid":"Sigmoid", "exp":"Exp", "log":"Log", "sqrt":"Sqrt", "identity":"Identity"}
        for i, node in enumerate(nodes):
            if i not in live: continue
            op, name = node["op"], names[i]
            if op == "input":
                inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, node["shape"]))
            elif op in ("parameter", "constant"):
                initializers.append(numpy_helper.from_array(program._plan.read(i), name))
            else:
                args = [names[a] for a in node["inputs"]]
                attrs = {}
                if op in simple: kind = simple[op]
                elif op == "transpose": kind, attrs = "Transpose", {"perm": [1, 0]}
                elif op in ("reshape", "broadcast", "sum"):
                    kind = {"reshape":"Reshape", "broadcast":"Expand", "sum":"ReduceSum"}[op]
                    const_name = f"shape_{i}"
                    initializers.append(numpy_helper.from_array(np.asarray(node["attrs"], dtype=np.int64), const_name))
                    args.append(const_name)
                    if op == "sum": attrs = {"keepdims": 1, "noop_with_empty_axes": 1}
                else: raise UnsupportedOperatorError(f"no standard ONNX export for native op {op}")
                operations.append(helper.make_node(kind, args, [name], name=f"node_{i}", **attrs))
        outputs = [helper.make_tensor_value_info(names[i], TensorProto.FLOAT, nodes[i]["shape"]) for i in program._plan.outputs]
        graph = helper.make_graph(operations, "MuNet", inputs, outputs, initializers)
        model = helper.make_model(graph, producer_name="munet-next", opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
        onnx.checker.check_model(model, full_check=True)
        onnx.save_model(model, str(path))


def from_torch(model, example_inputs, *, device="vulkan", fuse=True):
    """Import an explicitly eval-mode torch module through torch.export's ONNX path.

    The exported inference graph can preserve weights and supported ops, but cannot
    reconstruct arbitrary Python behavior, optimizers, or training state.
    """
    import torch
    if model.training:
        raise ValueError("call model.eval() before importing an inference model")
    if not isinstance(example_inputs, tuple): example_inputs = (example_inputs,)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        torch.onnx.export(model, example_inputs, str(path), dynamo=True, opset_version=18, external_data=False)
        return from_onnx(path, device=device, fuse=fuse)
