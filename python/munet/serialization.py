"""Versioned data-only .mnet programs. No pickle and no embedded executable code."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from . import _native
from .core import Compiled, _device

FORMAT_VERSION = 1
MAX_BYTES = 2 * 1024 * 1024 * 1024
MAX_NODES = 100_000


def validate_output_tree(tree, count):
    """Validate metadata before using it to reconstruct public output containers."""
    def visit(node, depth=0):
        if depth>64 or not isinstance(node,list) or len(node)!=2: raise ValueError("invalid output tree")
        kind,value=node
        if kind=='tensor':
            if type(value) is not int or not 0<=value<count: raise ValueError("invalid output tensor index")
        elif kind=='constant':
            if value is not None and type(value) not in (str,int,float,bool): raise ValueError("invalid output constant")
        elif kind in ('tuple','list'):
            if not isinstance(value,list): raise ValueError("invalid output sequence")
            for v in value: visit(v,depth+1)
        elif kind=='dict':
            if not isinstance(value,list): raise ValueError("invalid output dictionary")
            keys=set()
            for pair in value:
                if not isinstance(pair,list) or len(pair)!=2 or not isinstance(pair[0],str) or pair[0] in keys: raise ValueError("invalid output dictionary key")
                keys.add(pair[0]);visit(pair[1],depth+1)
        else: raise ValueError("unknown output container")
    visit(tree)
    return tree


def _read_tensor(payload, declared_shape):
    """Check the header and byte count before constructing any tensor allocation."""
    import math
    buf = io.BytesIO(payload)
    version = np.lib.format.read_magic(buf)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(buf)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(buf)
    else:
        raise ValueError("unsupported tensor encoding version")
    if dtype != np.dtype("<f4") or list(shape) != declared_shape or fortran:
        raise ValueError("tensor dtype/shape/layout does not match graph")
    if len(shape) > 8 or any(d <= 0 for d in shape):
        raise ValueError("invalid tensor shape")
    size = math.prod(shape) * 4
    if size > MAX_BYTES or len(payload) - buf.tell() != size:
        raise ValueError("tensor payload size does not match header")
    return np.frombuffer(payload, dtype="<f4", offset=buf.tell()).reshape(shape)


def save(program, path):
    if not isinstance(program, Compiled) or program._plan is None:
        raise ValueError("save requires a compiled program that has been called at least once")
    with program._lock:
        for p, value in program._parameters.items():
            if p._version != program._seen_versions[p]:
                program._plan.write(value, np.ascontiguousarray(p.numpy()))
                program._seen_versions[p] = p._version
        nodes = program._plan.graph.nodes(data=False)
        tensors = []
        for i, node in enumerate(nodes):
            node.pop("data")
            if node["op"] in ("parameter", "constant"):
                name = f"tensors/{i}.npy"
                node["tensor"] = name
                tensors.append((i,name))
        manifest = {
            "format": "munet-program", "version": FORMAT_VERSION, "dtype": "float32",
            "nodes": nodes, "outputs": program._plan.outputs, "updates": program._plan.updates,
            "input_specs": [list(s) for s in program._input_specs], "feed_indices": program._feed_indices,
            "single_output": program._single,
            "output_tree": getattr(program, "_output_tree", None),
        }
        # Validate the producer against the same limits the loader applies.
        encoded = json.dumps(manifest, allow_nan=False, separators=(",", ":")).encode()
        if len(nodes) > MAX_NODES or len(encoded) + sum(int(np.prod(nodes[i]["shape"]))*4+256 for i,_ in tensors) > MAX_BYTES:
            raise ValueError("program exceeds the v0 format size limit")
        path = Path(path)
        import os, tempfile
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(fd)
        try:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
                z.writestr("manifest.json", encoded)
                for i,name in tensors:
                    with z.open(name,"w",force_zip64=True) as out:
                        np.lib.format.write_array(out,program._plan.read(i).astype("<f4",copy=False),allow_pickle=False)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)


def load(path, *, device="vulkan", fuse=True):
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive entries")
        if sum(i.file_size for i in infos) > MAX_BYTES or len(names) > MAX_NODES + 1:
            raise ValueError("archive exceeds the v0 format size limit")
        if "manifest.json" not in names or z.getinfo("manifest.json").file_size > 32 * 1024 * 1024:
            raise ValueError("missing or oversized manifest")
        manifest = json.loads(z.read("manifest.json"))
        if manifest.get("format") != "munet-program" or manifest.get("version") != FORMAT_VERSION:
            raise ValueError("unsupported MuNet format/version")
        if manifest.get("dtype") != "float32":
            raise ValueError("unsupported program dtype")
        nodes = manifest["nodes"]
        if not isinstance(nodes, list) or len(nodes) > MAX_NODES:
            raise ValueError("invalid node list")
        graph = _native.Graph()
        expected = {"manifest.json"}
        for i, node in enumerate(nodes):
            op = node["op"]
            if op in ("input", "parameter", "constant"):
                if node["inputs"] or node["attrs"]:
                    raise ValueError("leaf contains inputs or attributes")
                data = []
                if op != "input":
                    name = f"tensors/{i}.npy"
                    if node.get("tensor") != name:
                        raise ValueError("invalid tensor entry name")
                    expected.add(name)
                    array = _read_tensor(z.read(name), node["shape"])
                    graph.leaf_array(op, node["name"], array)
                else:
                    graph.leaf(op, node["name"], node["shape"], data)
            else:
                if any(type(x) is not int or x < 0 or x >= i for x in node["inputs"]):
                    raise ValueError("graph is not topologically ordered")
                value = graph.op(op, node["inputs"], node["attrs"])
                if graph.shape(value) != node["shape"]:
                    raise ValueError("declared node shape differs from inferred shape")
        if set(names) != expected:
            raise ValueError("unexpected entries in MuNet archive")
    program = from_graph(graph, manifest["outputs"], manifest["updates"],
                      manifest["input_specs"], manifest["feed_indices"],
                      manifest["single_output"], device=device, fuse=fuse)
    if manifest.get("output_tree") is not None:
        program._output_tree = validate_output_tree(manifest["output_tree"],len(manifest["outputs"]))
    return program


def from_graph(graph, outputs, updates, input_specs, feed_indices, single_output, *, device="vulkan", fuse=True):
    program = Compiled(device=device, fuse=fuse)
    program._plan = _native.Plan(graph, outputs, updates, fuse)
    program._single = bool(single_output)
    if program._single and len(outputs) != 1:
        raise ValueError("single-output contract has multiple outputs")
    specs = [tuple(s) for s in input_specs]
    if any(any(type(d) is not int or d <= 0 for d in s) or len(s) > 8 for s in specs):
        raise ValueError("invalid runtime input shape")
    if len(feed_indices) != len(program._plan.inputs) or len(set(feed_indices)) != len(feed_indices):
        raise ValueError("invalid runtime input mapping")
    for i, node in zip(feed_indices, program._plan.inputs):
        if type(i) is not int or not 0 <= i < len(specs) or tuple(graph.shape(node)) != specs[i]:
            raise ValueError("runtime input mapping does not match graph")
    program._input_specs = specs
    program._feed_indices = feed_indices
    _device(program._plan, device)
    return program
