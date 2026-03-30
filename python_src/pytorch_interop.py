from __future__ import annotations

"""
PyTorch <-> MuNet Interoperability Module

This module provides utilities for converting between PyTorch and MuNet models.
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Try to import torch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = Any
    TORCH_AVAILABLE = False

# Add build directory to path for MuNet import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

try:
    import munet_nn as munet
    MUNET_AVAILABLE = True
except ImportError:
    MUNET_AVAILABLE = False


# Layer type mapping from PyTorch to MuNet
PYTORCH_TO_MUNET_LAYER_MAP = {
    'Linear': 'Linear',
    'Conv2d': 'Conv2d',
    'BatchNorm2d': 'BatchNorm2d',
    'MaxPool2d': 'MaxPool2d',
    'AvgPool2d': 'AvgPool2d',
    'ReLU': 'ReLU',
    'LeakyReLU': 'LeakyReLU',
    'Sigmoid': 'Sigmoid',
    'Tanh': 'Tanh',
    'GELU': 'GELU',
    'Softmax': 'Softmax',
    'Dropout': 'Dropout',
    'Flatten': 'Flatten',
    'Upsample': 'Upsample',
    'Embedding': 'Embedding',
    'LayerNorm': 'LayerNorm',
    'MultiheadAttention': 'MultiHeadAttention',
}


def get_layer_config(layer: nn.Module) -> Optional[Dict[str, Any]]:
    """
    Extract configuration from a PyTorch layer.
    
    Args:
        layer: PyTorch nn.Module layer
        
    Returns:
        Dictionary with layer configuration or None if layer type not supported
    """
    layer_type = layer.__class__.__name__
    
    if layer_type == 'Linear':
        return {
            'type': 'Linear',
            'in_features': layer.in_features,
            'out_features': layer.out_features,
            'bias': layer.bias is not None
        }
    elif layer_type == 'Conv2d':
        return {
            'type': 'Conv2d',
            'in_channels': layer.in_channels,
            'out_channels': layer.out_channels,
            'kernel_size': layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size,
            'stride': layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride,
            'padding': layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding,
            'bias': layer.bias is not None
        }
    elif layer_type == 'BatchNorm2d':
        return {
            'type': 'BatchNorm2d',
            'num_features': layer.num_features,
            'eps': layer.eps,
            'momentum': layer.momentum
        }
    elif layer_type == 'MaxPool2d':
        return {
            'type': 'MaxPool2d',
            'kernel_size': layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0],
            'stride': layer.stride if isinstance(layer.stride, int) else (layer.stride[0] if layer.stride else layer.kernel_size),
            'padding': layer.padding if isinstance(layer.padding, int) else (layer.padding[0] if layer.padding else 0)
        }
    elif layer_type == 'ReLU':
        return {'type': 'ReLU'}
    elif layer_type == 'LeakyReLU':
        return {
            'type': 'LeakyReLU',
            'negative_slope': layer.negative_slope
        }
    elif layer_type == 'Sigmoid':
        return {'type': 'Sigmoid'}
    elif layer_type == 'Tanh':
        return {'type': 'Tanh'}
    elif layer_type == 'GELU':
        return {'type': 'GELU'}
    elif layer_type == 'Softmax':
        return {
            'type': 'Softmax',
            'dim': layer.dim if hasattr(layer, 'dim') else -1
        }
    elif layer_type == 'Dropout':
        return {
            'type': 'Dropout',
            'p': layer.p
        }
    elif layer_type == 'Flatten':
        return {'type': 'Flatten'}
    elif layer_type == 'Upsample':
        return {
            'type': 'Upsample',
            'scale_factor': layer.scale_factor if hasattr(layer, 'scale_factor') else 2
        }
    elif layer_type == 'Embedding':
        return {
            'type': 'Embedding',
            'num_embeddings': layer.num_embeddings,
            'embedding_dim': layer.embedding_dim
        }
    elif layer_type == 'LayerNorm':
        return {
            'type': 'LayerNorm',
            'normalized_shape': layer.normalized_shape[0] if isinstance(layer.normalized_shape, tuple) else layer.normalized_shape,
            'eps': layer.eps
        }
    elif layer_type == 'MultiheadAttention':
        return {
            'type': 'MultiHeadAttention',
            'embed_dim': layer.embed_dim,
            'num_heads': layer.num_heads
        }
    
    return None


def build_architecture_config(model: nn.Module) -> Dict[str, Any]:
    """
    Build MuNet architecture configuration from PyTorch model.
    
    Args:
        model: PyTorch nn.Module
        
    Returns:
        Dictionary with architecture configuration
    """
    # Check if it's a Sequential
    if isinstance(model, nn.Sequential):
        layers = []
        for i, layer in enumerate(model):
            layer_config = get_layer_config(layer)
            if layer_config:
                layers.append(layer_config)
            else:
                raise ValueError(f"Unsupported layer type: {layer.__class__.__name__}")
        
        return {
            'type': 'Sequential',
            'layers': layers
        }
    
    # For custom modules, we need to build a graph
    # For now, we support Sequential and simple modules
    raise ValueError(
        f"Custom module architecture export not yet supported for {model.__class__.__name__}. "
        "Please use nn.Sequential or manually specify the architecture."
    )


def convert_state_dict_to_numpy(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """
    Convert PyTorch state dict to NumPy arrays.
    
    Args:
        state_dict: PyTorch state dictionary
        
    Returns:
        Dictionary with NumPy arrays
    """
    numpy_dict = {}
    for name, param in state_dict.items():
        numpy_dict[name] = param.detach().cpu().numpy()
    return numpy_dict


def convert_numpy_to_state_dict(
    numpy_dict: Dict[str, np.ndarray],
    model: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Convert NumPy arrays to PyTorch state dict.
    
    Args:
        numpy_dict: Dictionary with NumPy arrays
        model: PyTorch model for reference
        
    Returns:
        PyTorch state dictionary
    """
    state_dict = {}
    for name, param in model.state_dict().items():
        if name in numpy_dict:
            state_dict[name] = torch.from_numpy(numpy_dict[name])
        else:
            state_dict[name] = param
    return state_dict


def pytorch_to_munet(
    model: nn.Module,
    output_path: str,
    input_shape: Optional[Tuple[int, ...]] = None
) -> Dict[str, Any]:
    """
    Convert a PyTorch model to MuNet format and save to file.
    
    Args:
        model: PyTorch nn.Module
        output_path: Path to save the MuNet model (.npz)
        input_shape: Optional input shape for validation
        
    Returns:
        Dictionary with conversion metadata
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for pytorch_to_munet")
    
    model.eval()
    
    # Build architecture config
    config = build_architecture_config(model)
    
    # Convert weights
    state_dict = model.state_dict()
    weights_dict = convert_state_dict_to_numpy(state_dict)
    
    # Save using MuNet format
    if MUNET_AVAILABLE:
        # Use MuNet's serialization
        # First we need to build a MuNet model from config
        munet_model = build_munet_model_from_config(config)
        
        # Copy weights
        copy_weights_to_munet(weights_dict, munet_model)
        
        # Save
        munet.save(munet_model, output_path)
    else:
        # Manual NPZ save for deployment without MuNet
        save_as_npz(config, weights_dict, output_path)
    
    return {
        'config': config,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'output_path': output_path
    }


def build_munet_model_from_config(config: Dict[str, Any]):
    """
    Build a MuNet model from configuration dictionary.
    
    Args:
        config: Architecture configuration
        
    Returns:
        MuNet model instance
    """
    if not MUNET_AVAILABLE:
        raise ImportError("MuNet is required for build_munet_model_from_config")
    
    layer_type = config['type']
    
    if layer_type == 'Sequential':
        model = munet.nn.Sequential()
        for layer_config in config['layers']:
            layer = create_munet_layer(layer_config)
            model.add(layer)
        return model
    
    raise ValueError(f"Unsupported architecture type: {layer_type}")


def create_munet_layer(config: Dict[str, Any]):
    """
    Create a MuNet layer from configuration.
    
    Args:
        config: Layer configuration
        
    Returns:
        MuNet layer instance
    """
    if not MUNET_AVAILABLE:
        raise ImportError("MuNet is required for create_munet_layer")
    
    layer_type = config['type']
    
    if layer_type == 'Linear':
        return munet.nn.Linear(
            config['in_features'],
            config['out_features'],
            bias=config.get('bias', True)
        )
    elif layer_type == 'Conv2d':
        # C++ binding requires positional args: (in_channels, out_channels, kernel_size, stride, padding)
        return munet.nn.Conv2d(
            config['in_channels'],
            config['out_channels'],
            config['kernel_size'],
            config.get('stride', 1),
            config.get('padding', 0)
        )
    elif layer_type == 'BatchNorm2d':
        return munet.nn.BatchNorm2d(
            config['num_features'],
            eps=config.get('eps', 1e-5),
            momentum=config.get('momentum', 0.1)
        )
    elif layer_type == 'MaxPool2d':
        return munet.nn.MaxPool2d(
            kernel_size=config['kernel_size'],
            stride=config.get('stride', config['kernel_size']),
            padding=config.get('padding', 0)
        )
    elif layer_type == 'ReLU':
        return munet.nn.ReLU()
    elif layer_type == 'LeakyReLU':
        return munet.nn.LeakyReLU(negative_slope=config.get('negative_slope', 0.01))
    elif layer_type == 'Sigmoid':
        return munet.nn.Sigmoid()
    elif layer_type == 'Tanh':
        return munet.nn.Tanh()
    elif layer_type == 'GELU':
        return munet.nn.GELU()
    elif layer_type == 'Softmax':
        dim = config.get('dim', -1)
        return munet.nn.Softmax(dim=dim)
    elif layer_type == 'Dropout':
        return munet.nn.Dropout(p=config.get('p', 0.5))
    elif layer_type == 'Flatten':
        return munet.nn.Flatten()
    elif layer_type == 'Upsample':
        return munet.nn.Upsample(scale_factor=config.get('scale_factor', 2))
    elif layer_type == 'Embedding':
        return munet.nn.Embedding(
            config['num_embeddings'],
            config['embedding_dim']
        )
    elif layer_type == 'LayerNorm':
        return munet.nn.LayerNorm(
            config['normalized_shape'],
            eps=config.get('eps', 1e-5)
        )
    elif layer_type == 'MultiHeadAttention':
        return munet.nn.MultiHeadAttention(
            config['embed_dim'],
            config['num_heads']
        )
    
    raise ValueError(f"Unsupported layer type: {layer_type}")

def copy_weights_to_munet(
    weights_dict: Dict[str, np.ndarray],
    munet_model,
    strict: bool = True
):
    """
    Copy weights from NumPy dict to MuNet model.
    
    Note: MuNet's named_parameters() returns both parameters and buffers,
    so we only need to iterate over named_parameters().
    
    Args:
        weights_dict: Dictionary with NumPy arrays
        munet_model: MuNet model instance
    """
    if not MUNET_AVAILABLE:
        raise ImportError("MuNet is required for copy_weights_to_munet")
    
    named_params = dict(munet_model.named_parameters())

    missing_in_source = []
    for name, param in named_params.items():
        if name not in weights_dict:
            missing_in_source.append(name)
            continue

        numpy_array = weights_dict[name]
        # Transpose Linear layer weights (PyTorch: [out, in], MuNet: [in, out])
        if '.weight' in name and len(numpy_array.shape) == 2:
            numpy_array = numpy_array.T

        if list(numpy_array.shape) != list(param.shape):
            raise ValueError(
                f"Shape mismatch for parameter '{name}': "
                f"source {list(numpy_array.shape)} vs target {list(param.shape)}"
            )

        param.copy_from_numpy(numpy_array)

    extra_in_source = sorted(set(weights_dict.keys()) - set(named_params.keys()))
    if strict and (missing_in_source or extra_in_source):
        details = []
        if missing_in_source:
            details.append(
                "missing source weights for target params: "
                + ", ".join(sorted(missing_in_source))
            )
        if extra_in_source:
            details.append(
                "unused source weights: " + ", ".join(extra_in_source)
            )
        raise ValueError("Weight mapping mismatch: " + "; ".join(details))


def save_as_npz(
    config: Dict[str, Any],
    weights_dict: Dict[str, np.ndarray],
    output_path: str
):
    """
    Save model as NPZ file in MuNet format.
    
    Args:
        config: Architecture configuration
        weights_dict: Dictionary with NumPy arrays
        output_path: Output file path
    """
    # Create metadata
    metadata = {
        '__format_name__': np.array('munet_model'),
        '__format_revision__': np.array(1, dtype=np.int64),
        '__producer__': np.array('munet_pytorch_interop'),
        '__artifact_kind__': np.array('deploy_model'),
        '__artifact_scope__': np.array('runtime_only'),
        '__default_load_mode__': np.array('eval'),
        '__contains_training_state__': np.array(False),
        '__recommended_loader__': np.array('load_for_inference'),
        '__compile_contract_policy__': np.array('external'),
        '__config__': np.array(json.dumps(config)),
    }
    
    # Add tensor names manifest
    tensor_names = sorted(weights_dict.keys())
    metadata['__tensor_names__'] = np.array(json.dumps(tensor_names))
    
    # Combine metadata and weights
    save_dict = {**metadata, **weights_dict}
    
    # Save as NPZ
    np.savez(output_path, **save_dict)


def munet_to_pytorch(
    model_path: str,
    pytorch_model: nn.Module
) -> nn.Module:
    """
    Load MuNet model weights into a PyTorch model.
    
    Args:
        model_path: Path to MuNet model file (.npz)
        pytorch_model: PyTorch model instance to load weights into
        
    Returns:
        PyTorch model with loaded weights
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for munet_to_pytorch")
    
    # Load NPZ file
    data = np.load(model_path, allow_pickle=True)
    
    # Extract weights
    weights_dict = {}
    for key in data.files:
        if not key.startswith('__'):
            weights_dict[key] = data[key]
    
    # Convert to PyTorch state dict
    state_dict = convert_numpy_to_state_dict(weights_dict, pytorch_model)
    
    # Load into model
    pytorch_model.load_state_dict(state_dict)
    
    return pytorch_model


def get_pytorch_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a PyTorch model.
    
    Args:
        model: PyTorch nn.Module
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    layer_info = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            config = get_layer_config(module)
            if config:
                layer_info.append({
                    'name': name,
                    'type': config['type'],
                    'config': config
                })
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'num_layers': len(layer_info),
        'layers': layer_info
    }


class PyTorchInterop:
    """Compatibility wrapper used by tests."""

    @staticmethod
    def _named_params(module) -> Dict[str, Any]:
        return dict(module.named_parameters()) if hasattr(module, "named_parameters") else {}

    def save_weights(self, module) -> Dict[str, np.ndarray]:
        state: Dict[str, np.ndarray] = {}
        for name, param in self._named_params(module).items():
            arr = np.array(param.detach().numpy(), copy=True)
            # MuNet Linear stores weight as [in, out]; PyTorch uses [out, in].
            if (name.endswith(".weight") or name == "weight") and arr.ndim == 2:
                arr = arr.T
            state[name] = arr
        if "weight" not in state and hasattr(module, "weight"):
            arr = np.array(module.weight.detach().numpy(), copy=True)
            if arr.ndim == 2:
                arr = arr.T
            state["weight"] = arr
        if "bias" not in state and hasattr(module, "bias") and module.bias is not None:
            state["bias"] = np.array(module.bias.detach().numpy(), copy=True)
        return state

    def load_weights(self, module, state_dict: Dict[str, np.ndarray]) -> None:
        named = self._named_params(module)
        for name, arr in state_dict.items():
            value = np.asarray(arr, dtype=np.float32)
            if name in named:
                # Handle Linear convention mismatch when shapes are transposed.
                if value.ndim == 2 and list(value.shape) == list(reversed(named[name].shape)):
                    value = value.T
                named[name].copy_from_numpy(value)
            elif name == "weight" and hasattr(module, "weight"):
                if value.ndim == 2 and list(value.shape) == list(reversed(module.weight.shape)):
                    value = value.T
                module.weight.copy_from_numpy(value)
            elif name == "bias" and hasattr(module, "bias") and module.bias is not None:
                module.bias.copy_from_numpy(value)

    def save_to_file(self, module, path: str) -> None:
        state = self.save_weights(module)
        if TORCH_AVAILABLE:
            tensor_state = {k: torch.from_numpy(v) for k, v in state.items()}
            torch.save(tensor_state, path)
        else:
            np.savez(path, **state)

    def load_from_file(self, module, path: str) -> None:
        if TORCH_AVAILABLE:
            state = torch.load(path, weights_only=False)
            state = {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v) for k, v in state.items()}
        else:
            with np.load(path, allow_pickle=True) as data:
                state = {k: data[k] for k in data.files}
        self.load_weights(module, state)
