"""
MuNet to PyTorch Import Demo

This script demonstrates how to load a MuNet model into PyTorch.
"""

import sys

import numpy as np

try:
    import munet_nn as munet
except ImportError:
    print("MuNet not available. Install it with 'python -m pip install munet-nn'")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. This demo requires PyTorch.")
    sys.exit(1)

from pytorch_interop import (
    munet_to_pytorch,
    get_pytorch_model_info,
    save_as_npz
)


def create_matching_pytorch_model():
    """Create a PyTorch model that matches the exported MuNet architecture."""
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 10)
    )


def demo_round_trip():
    """Demonstrate round-trip conversion: PyTorch -> MuNet -> PyTorch."""
    print("=" * 60)
    print("PyTorch <-> MuNet Round-Trip Demo")
    print("=" * 60)
    
    # Create original model
    print("\n[1] Creating original PyTorch model...")
    original_model = create_matching_pytorch_model()
    original_model.eval()
    
    # Initialize with random weights
    torch.manual_seed(42)
    for param in original_model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    
    # Test forward pass
    print("\n[2] Testing original model...")
    test_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        original_output = original_model(test_input)
    print(f"    Original output shape: {original_output.shape}")
    
    # Save to NPZ (MuNet format)
    output_path = "/tmp/test_roundtrip.npz"
    print(f"\n[3] Saving to {output_path}...")
    
    config = {
        'type': 'Sequential',
        'layers': [
            {'type': 'Conv2d', 'in_channels': 3, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True},
            {'type': 'BatchNorm2d', 'num_features': 32, 'eps': 1e-5, 'momentum': 0.1},
            {'type': 'ReLU'},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2, 'padding': 0},
            {'type': 'Conv2d', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True},
            {'type': 'BatchNorm2d', 'num_features': 64, 'eps': 1e-5, 'momentum': 0.1},
            {'type': 'ReLU'},
            {'type': 'MaxPool2d', 'kernel_size': 2, 'stride': 2, 'padding': 0},
            {'type': 'Flatten'},
            {'type': 'Linear', 'in_features': 64 * 8 * 8, 'out_features': 10, 'bias': True}
        ]
    }
    
    state_dict = original_model.state_dict()
    weights_dict = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
    save_as_npz(config, weights_dict, output_path)
    print("    Saved successfully")
    
    # Create new model and load weights
    print("\n[4] Creating new model and loading weights...")
    loaded_model = create_matching_pytorch_model()
    loaded_model = munet_to_pytorch(output_path, loaded_model)
    loaded_model.eval()
    
    # Test loaded model
    print("\n[5] Testing loaded model...")
    with torch.no_grad():
        loaded_output = loaded_model(test_input)
    print(f"    Loaded output shape: {loaded_output.shape}")
    
    # Compare outputs
    print("\n[6] Comparing outputs...")
    max_diff = torch.max(torch.abs(original_output - loaded_output)).item()
    print(f"    Maximum difference: {max_diff:.6e}")
    
    if max_diff < 1e-6:
        print("    ✓ Round-trip successful! Outputs match within tolerance.")
    else:
        print("    ⚠ Outputs differ - check weight loading")
        print(f"    Original output sample: {original_output[0, :5]}")
        print(f"    Loaded output sample: {loaded_output[0, :5]}")
    
    # Compare weights
    print("\n[7] Comparing weights...")
    for (name1, param1), (name2, param2) in zip(
        original_model.named_parameters(),
        loaded_model.named_parameters()
    ):
        max_weight_diff = torch.max(torch.abs(param1 - param2)).item()
        if max_weight_diff > 1e-10:
            print(f"    ⚠ {name1}: max diff = {max_weight_diff:.6e}")
    
    print("\n" + "=" * 60)
    print("Round-trip demo complete!")
    print("=" * 60)


def demo_batchnorm_loading():
    """Demonstrate BatchNorm running statistics loading."""
    print("\n" + "=" * 60)
    print("BatchNorm Running Statistics Demo")
    print("=" * 60)
    
    # Create model with BatchNorm
    print("\n[1] Creating model with BatchNorm...")
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 32 * 32, 10)
    )
    
    # Train for a few iterations to set running stats
    print("\n[2] Training to set running statistics...")
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for _ in range(10):
        x = torch.randn(4, 3, 32, 32)
        y = torch.randn(4, 10)
        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.mse_loss(out, y)
        loss.backward()
        optimizer.step()
    
    # Get running stats
    bn_layer = model[1]
    running_mean = bn_layer.running_mean.clone()
    running_var = bn_layer.running_var.clone()
    
    print(f"    Running mean: {running_mean[:5]}")
    print(f"    Running var: {running_var[:5]}")
    
    # Save model
    output_path = "/tmp/test_batchnorm.npz"
    print(f"\n[3] Saving model...")
    
    config = {
        'type': 'Sequential',
        'layers': [
            {'type': 'Conv2d', 'in_channels': 3, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True},
            {'type': 'BatchNorm2d', 'num_features': 16, 'eps': 1e-5, 'momentum': 0.1},
            {'type': 'ReLU'},
            {'type': 'Flatten'},
            {'type': 'Linear', 'in_features': 16 * 32 * 32, 'out_features': 10, 'bias': True}
        ]
    }
    
    state_dict = model.state_dict()
    weights_dict = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
    save_as_npz(config, weights_dict, output_path)
    
    # Load into new model
    print("\n[4] Loading into new model...")
    new_model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 32 * 32, 10)
    )
    new_model = munet_to_pytorch(output_path, new_model)
    
    # Check running stats preserved
    new_bn_layer = new_model[1]
    loaded_mean = new_bn_layer.running_mean
    loaded_var = new_bn_layer.running_var
    
    print("\n[5] Checking running statistics...")
    print(f"    Loaded running mean: {loaded_mean[:5]}")
    print(f"    Loaded running var: {loaded_var[:5]}")
    
    mean_diff = torch.max(torch.abs(running_mean - loaded_mean)).item()
    var_diff = torch.max(torch.abs(running_var - loaded_var)).item()
    
    print(f"    Mean max difference: {mean_diff:.6e}")
    print(f"    Var max difference: {var_diff:.6e}")
    
    if mean_diff < 1e-6 and var_diff < 1e-6:
        print("    ✓ Running statistics preserved!")
    else:
        print("    ⚠ Running statistics not fully preserved")


if __name__ == "__main__":
    demo_round_trip()
    demo_batchnorm_loading()
