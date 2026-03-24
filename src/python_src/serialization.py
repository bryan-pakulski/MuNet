"""
MuNet Hybrid Pickle + Tensor Serialization

This module provides serialization that works with any Python module.
It uses pickle for object structure and NPZ for tensor storage.

Key features:
- Works with any Python class without registration
- Efficient binary tensor storage
- Preserves module hierarchy and attributes
- No whitelist requirements
"""

import numpy as np
import pickle
import json

SERIALIZATION_FORMAT_NAME = "munet"
SERIALIZATION_FORMAT_REVISION = 1
SERIALIZATION_LEGACY_TAG = "munet:v1"
SERIALIZATION_ARTIFACT_KIND = "model"
SERIALIZATION_ARTIFACT_SCOPE = "inference+training"
SERIALIZATION_DEFAULT_LOAD_MODE = "full_reconstruction"
SERIALIZATION_CONTAINS_TRAINING_STATE = "true"
SERIALIZATION_DEVICE_POLICY = "preserve"
SERIALIZATION_DTYPE_POLICY = "preserve"
SERIALIZATION_RECOMMENDED_LOADER = "munet.load"
SERIALIZATION_COMPILE_CONTRACT_POLICY = "dynamic"


def _get_munet():
    """Get the munet module."""
    return __import__("munet")


def _tensor_dtype_name(t):
    """Get dtype name from tensor."""
    dtype_map = {
        0: 'float32',
        1: 'float64',
        2: 'int32',
        3: 'int64',
        4: 'float16',
    }
    return dtype_map.get(int(t.dtype), 'float32')


def _tensor_options_for_dtype(dtype_name):
    """Get tensor options for dtype."""
    m = _get_munet()
    dtype_map = {
        'float32': m.DType.Float32,
        'float64': m.DType.Float64,
        'int32': m.DType.Int32,
        'int64': m.DType.Int64,
        'float16': m.DType.Float16,
    }
    return m.TensorOptions().dtype(dtype_map.get(dtype_name, m.DType.Float32))


def _tensor_to_numpy(t):
    """Convert a MuNet tensor to numpy array (CPU copy)."""
    m = _get_munet()
    cpu = m.Device(m.DeviceType.CPU, 0)
    td = t.detach()
    if td.device.type != m.DeviceType.CPU:
        td = td.to(cpu)
    return np.array(td, copy=False).copy()


def _numpy_to_tensor(arr, device=None):
    """Convert numpy array to MuNet tensor."""
    m = _get_munet()
    t = m.from_numpy(np.ascontiguousarray(arr))
    if device is not None:
        t = t.to(device)
    return t


def _iter_named_tensors(module):
    """Iterate over all tensors in a module with their names."""
    items = []
    
    def _direct_named_members(mod):
        result = []
        if hasattr(mod, '_parameters'):
            for name, param in mod._parameters.items():
                if param is not None:
                    result.append(('param', name, param))
        if hasattr(mod, '_buffers'):
            for name, buf in mod._buffers.items():
                if buf is not None:
                    result.append(('buffer', name, buf))
        return result
    
    def _collect(mod, prefix=''):
        for kind, name, tensor in _direct_named_members(mod):
            full_name = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            items.append((full_name, tensor))
        if hasattr(mod, '_modules'):
            for child_name, child in mod._modules.items():
                if child is not None:
                    new_prefix = f"{prefix}{child_name}." if not prefix else f"{prefix}.{child_name}."
                    _collect(child, new_prefix)
    
    _collect(module)
    return items


def _extract_tensors_and_shell(obj, path=""):
    """
    Recursively extract all tensors and create a pickle-able shell.
    
    Returns: (tensors_dict, shell_object)
    """
    m = _get_munet()
    tensors = {}
    tensor_counter = [0]  # Use list for mutability in nested function
    
    def extract(o, p):
        if isinstance(o, m.Tensor):
            # Extract tensor to numpy and store reference
            tensor_name = f"__tensor_{tensor_counter[0]}__"
            tensor_counter[0] += 1
            tensors[tensor_name] = _tensor_to_numpy(o)
            return {"__tensor_ref__": tensor_name, "__path__": p}
        
        elif isinstance(o, dict):
            return {k: extract(v, f"{p}.{k}" if p else k) for k, v in o.items()}
        
        elif isinstance(o, (list, tuple)):
            result = [extract(v, f"{p}[{i}]" if p else str(i)) for i, v in enumerate(o)]
            # Preserve tuple type
            if isinstance(o, tuple):
                return tuple(result)
            return result
        
        elif hasattr(o, '__dict__') and not isinstance(o, (type, type(lambda: None))):
            # It's an object with attributes (like nn.Module)
            shell = {
                "__class_module__": type(o).__module__,
                "__class_qualname__": type(o).__qualname__,
            }
            
            # Handle common nn.Module attributes
            if hasattr(o, '__class__'):
                shell["__class_name__"] = type(o).__name__
            
            # Store all attributes
            for k, v in o.__dict__.items():
                shell[k] = extract(v, f"{p}.{k}" if p else k)
            
            return shell
        
        elif isinstance(o, (int, float, str, bool, type(None))):
            return o
        
        else:
            # For other types, try to store as primitive
            try:
                # Test if it's picklable
                pickle.dumps(o)
                return o
            except:
                return repr(o)
    
    shell = extract(obj, path)
    return tensors, shell


def _rebuild_from_shell(shell, tensors, device=None):
    """
    Rebuild object from shell and tensors.
    
    Returns: reconstructed object
    """
    m = _get_munet()
    
    def rebuild(o):
        if isinstance(o, dict):
            if "__tensor_ref__" in o:
                # This is a tensor reference
                tensor_name = o["__tensor_ref__"]
                if tensor_name in tensors:
                    t = _numpy_to_tensor(tensors[tensor_name], device)
                    return t
                else:
                    raise ValueError(f"Missing tensor: {tensor_name}")
            
            elif "__class_module__" in o:
                # This is a class instance
                class_module = o["__class_module__"]
                class_qualname = o["__class_qualname__"]
                
                # Try to import the class
                try:
                    import importlib
                    module = importlib.import_module(class_module)
                    parts = class_qualname.split('.')
                    cls = module
                    for part in parts:
                        cls = getattr(cls, part)
                    
                    # Create instance
                    instance = cls.__new__(cls)
                    
                    # Restore attributes
                    for k, v in o.items():
                        if k not in ("__class_module__", "__class_qualname__", "__class_name__"):
                            setattr(instance, k, rebuild(v))
                    
                    return instance
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to reconstruct {class_module}.{class_qualname}: {e}\n"
                        f"Make sure the class is importable from {class_module}"
                    )
            
            else:
                return {k: rebuild(v) for k, v in o.items()}
        
        elif isinstance(o, (list, tuple)):
            result = [rebuild(v) for v in o]
            if isinstance(o, tuple):
                return tuple(result)
            return result
        
        elif isinstance(o, (int, float, str, bool, type(None))):
            return o
        
        else:
            return o
    
    return rebuild(shell)


def save(module, filename):
    """
    Saves a module architecture + parameters/buffers to a .npz file.
    
    Uses a hybrid pickle + tensor approach:
    - Pickles the Python object structure (class, attributes, submodules)
    - Stores tensors separately in NPZ format
    
    This works with any Python module without registration.
    
    Args:
        module: The module to save (any Python object with MuNet tensors)
        filename: Path to save the file
    """
    # Extract tensors and create shell
    tensors, shell = _extract_tensors_and_shell(module)
    
    # Build the NPZ state
    state = {}
    
    # Store the pickled shell
    shell_bytes = pickle.dumps(shell, protocol=pickle.HIGHEST_PROTOCOL)
    state['__shell__'] = np.frombuffer(shell_bytes, dtype=np.uint8)
    
    # Store tensors
    for name, arr in tensors.items():
        state[name] = arr
    
    # Store metadata
    state['__format__'] = np.array('munet_hybrid_v1')
    state['__producer__'] = np.array('munet')
    
    # Store legacy metadata for compatibility
    state['__config__'] = np.array('{"type": "HybridPickle"}')
    state['__format_name__'] = np.array(SERIALIZATION_FORMAT_NAME)
    state['__format_revision__'] = np.array(SERIALIZATION_FORMAT_REVISION)
    state['__format_version__'] = np.array(SERIALIZATION_LEGACY_TAG)
    state['__artifact_kind__'] = np.array(SERIALIZATION_ARTIFACT_KIND)
    state['__artifact_scope__'] = np.array(SERIALIZATION_ARTIFACT_SCOPE)
    state['__default_load_mode__'] = np.array(SERIALIZATION_DEFAULT_LOAD_MODE)
    state['__contains_training_state__'] = np.array(SERIALIZATION_CONTAINS_TRAINING_STATE)
    state['__device_policy__'] = np.array(SERIALIZATION_DEVICE_POLICY)
    state['__dtype_policy__'] = np.array(SERIALIZATION_DTYPE_POLICY)
    state['__recommended_loader__'] = np.array(SERIALIZATION_RECOMMENDED_LOADER)
    state['__compile_contract_policy__'] = np.array(SERIALIZATION_COMPILE_CONTRACT_POLICY)
    
    # Also save tensors with named paths for debugging
    state['__tensor_names__'] = np.array(json.dumps(sorted(tensors.keys())))
    
    np.savez(filename, **state)


def load(arg, filename=None, device=None):
    """
    Loads a previously saved module state.
    
    Supports:
      - load("model.npz") -> reconstruct full model from file (any Python class)
      - load(module, "model.npz") -> load weights/buffers into existing model (legacy)
    
    Args:
        arg: Either filename (str) or module to load into
        filename: If first arg is module, this is the filename
        device: Optional device to load tensors to
    
    Returns:
        Reconstructed module
    """
    m = _get_munet()
    
    if filename is None:
        # Load from file
        path = arg
        
        with np.load(path, allow_pickle=True) as state:
            # Validate metadata
            if '__format__' in state and str(state['__format__']) == 'munet_hybrid_v1':
                # New hybrid format
                shell_bytes = state['__shell__'].tobytes()
                shell = pickle.loads(shell_bytes)
                
                # Collect tensors
                tensors = {}
                for name in state.files:
                    if name.startswith('__tensor_') and name.endswith('__'):
                        tensors[name] = state[name]
                
                return _rebuild_from_shell(shell, tensors, device)
            
            else:
                # Legacy format - try to load with old method
                if '__config__' not in state:
                    raise ValueError(
                        "File does not contain architecture config. "
                        "Use `load(module, filename)` for weights-only restore."
                    )
                
                config = json.loads(str(state['__config__']))
                
                def build_module(cfg):
                    t = cfg['type']
                    opts = _tensor_options_for_dtype(cfg.get('dtype', 'float32'))
                    
                    # Built-in module constructors
                    if t == 'Sequential':
                        return m.nn.Sequential([build_module(c) for c in cfg['layers']])
                    elif t == 'Linear':
                        return m.nn.Linear(cfg['in_features'], cfg['out_features'], cfg['bias'], opts)
                    elif t == 'Conv2d':
                        return m.nn.Conv2d(cfg['in_channels'], cfg['out_channels'], 
                                          cfg['kernel_size'], cfg['stride'], cfg['padding'], opts)
                    elif t == 'MaxPool2d':
                        return m.nn.MaxPool2d(cfg['kernel_size'], cfg['stride'], cfg['padding'])
                    elif t == 'BatchNorm2d':
                        return m.nn.BatchNorm2d(cfg['num_features'], cfg['eps'], cfg['momentum'], opts)
                    elif t == 'Upsample':
                        return m.nn.Upsample(cfg['scale_factor'])
                    elif t == 'GlobalAvgPool2d':
                        return m.nn.GlobalAvgPool2d()
                    elif t == 'ReLU':
                        return m.nn.ReLU()
                    elif t == 'Sigmoid':
                        return m.nn.Sigmoid()
                    elif t == 'Tanh':
                        return m.nn.Tanh()
                    elif t == 'GELU':
                        return m.nn.GELU()
                    elif t == 'LeakyReLU':
                        return m.nn.LeakyReLU(cfg.get('negative_slope', 0.01))
                    elif t == 'Dropout':
                        return m.nn.Dropout(cfg.get('p', 0.5))
                    elif t == 'Embedding':
                        return m.nn.Embedding(cfg['num_embeddings'], cfg['embedding_dim'], opts)
                    elif t == 'LayerNorm':
                        return m.nn.LayerNorm(cfg['normalized_shape'], cfg.get('eps', 1e-5), opts)
                    elif t == 'RMSNorm':
                        return m.nn.RMSNorm(cfg['normalized_shape'], cfg.get('eps', 1e-5), opts)
                    elif t == 'MultiHeadAttention':
                        return m.nn.MultiHeadAttention(cfg['embed_dim'], cfg['num_heads'], 
                                                      cfg.get('causal', True), opts)
                    elif t == 'Flatten':
                        return m.nn.Flatten()
                    else:
                        raise ValueError(f"Unsupported saved module type: {t}")
                
                module = build_module(config)
                
                # Apply state
                def copy_numpy_into_tensor(t, arr):
                    req = bool(t.requires_grad)
                    target = t.device
                    src = m.from_numpy(np.ascontiguousarray(arr))
                    if src.dtype != t.dtype:
                        src = src.to(t.dtype)
                    if target.type != m.DeviceType.CPU:
                        src = src.to(target)
                    t.replace_(src)
                    t.requires_grad = req
                
                for name, tensor in _iter_named_tensors(module):
                    if name in state:
                        copy_numpy_into_tensor(tensor, state[name])
                
                return module
    else:
        # Load into existing module (weights-only)
        module = arg
        
        with np.load(filename, allow_pickle=True) as state:
            def copy_numpy_into_tensor(t, arr):
                req = bool(t.requires_grad)
                target = t.device
                src = m.from_numpy(np.ascontiguousarray(arr))
                if src.dtype != t.dtype:
                    src = src.to(t.dtype)
                if target.type != m.DeviceType.CPU:
                    src = src.to(target)
                t.replace_(src)
                t.requires_grad = req
            
            for name, tensor in _iter_named_tensors(module):
                if name in state:
                    copy_numpy_into_tensor(tensor, state[name])
            
            return module


def load_for_inference(arg, filename=None, device=None):
    """
    Load a deploy artifact and normalize the result for inference execution.
    
    Usage:
      - load_for_inference("model.npz", device=None) -> reconstruct + eval-safe module.
      - load_for_inference(module, "model.npz", device=None) -> apply state into existing module, move if requested, then eval().
    """
    module = load(arg, filename, device) if filename is not None else load(arg, device=device)
    
    if device is not None:
        module.to(device)
    module.eval()
    
    return module


def load_weights(module, filename):
    """Alias for `load(module, filename)` to explicitly do weights-only restore."""
    return load(module, filename)


def load_weights_for_inference(module, filename, device=None):
    """Weights-only restore that also normalizes the module for inference execution."""
    module = load_weights(module, filename)
    if device is not None:
        module.to(device)
    module.eval()
    return module