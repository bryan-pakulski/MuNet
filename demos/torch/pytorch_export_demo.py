"""
PyTorch to MuNet Export Demo

This script demonstrates how to export a PyTorch model to MuNet format.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python_src'))

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. This demo requires PyTorch.")
    sys.exit(1)

try:
    import munet
    MUNET_AVAILABLE = True
except ImportError:
    MUNET_AVAILABLE = False
    print("MuNet not available. Build MuNet first with 'make build-release'")

from pytorch_interop import (
    pytorch_to_munet,
    munet_to_pytorch,
    get_pytorch_model_info,
    build_architecture_config,
    save_as_npz
)


# Define a simple test model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Define a sequential model for export
def create_sequential_model():
    """Create a Sequential model that can be exported."""
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


def demo_pytorch_export():
    """Demonstrate PyTorch to MuNet export."""
    print("=" * 60)
    print("PyTorch to MuNet Export Demo")
    print("=" * 60)
    
    # Create model
    print("\n[1] Creating PyTorch Sequential model...")
    model = create_sequential_model()
    model.eval()
    
    # Get model info
    print("\n[2] Getting model info...")
    info = get_pytorch_model_info(model)
    print(f"    Total parameters: {info['total_parameters']:,}")
    print(f"    Trainable parameters: {info['trainable_parameters']:,}")
    print(f"    Number of layers: {info['num_layers']}")
    
    # Test forward pass
    print("\n[3] Testing PyTorch forward pass...")
    test_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        pytorch_output = model(test_input)
    print(f"    Input shape: {test_input.shape}")
    print(f"    Output shape: {pytorch_output.shape}")
    
    # Build architecture config
    print("\n[4] Building architecture configuration...")
    config = build_architecture_config(model)
    print(f"    Architecture type: {config['type']}")
    print(f"    Number of layers: {len(config['layers'])}")
    
    # Save model
    output_path = "/tmp/test_model.npz"
    print(f"\n[5] Saving model to {output_path}...")
    
    if MUNET_AVAILABLE:
        # Use MuNet serialization
        pytorch_to_munet(model, output_path)
        print("    Saved using MuNet serialization")
    else:
        # Manual save
        state_dict = model.state_dict()
        weights_dict = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
        save_as_npz(config, weights_dict, output_path)
        print("    Saved using manual NPZ serialization")
    
    # Verify file exists
    import os
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"    File size: {file_size:,} bytes")
        
        # Load and inspect
        print("\n[6] Inspecting saved model...")
        data = np.load(output_path, allow_pickle=True)
        print("    Keys in NPZ file:")
        for key in sorted(data.files):
            if not key.startswith('__'):
                print(f"      - {key}: {data[key].shape}")
            else:
                print(f"      - {key}: (metadata)")
    
    # Test round-trip conversion
    if MUNET_AVAILABLE:
        print("\n[7] Testing round-trip conversion...")
        print("    Loading MuNet model...")
        
        # Load as MuNet model
        munet_model = munet.inference.load_serialized(output_path)
        
        print("    Running inference with MuNet...")
        device = munet.Device(munet.DeviceType.CUDA if torch.cuda.is_available() else munet.DeviceType.CPU, 0)
        config = munet.inference.EngineConfig()
        config.device = device
        engine = munet.inference.Engine(config)
        engine.load(munet_model)
        
        # Prepare input
        input_np = test_input.numpy()
        input_tensor = munet.from_numpy(input_np[0]).to(device)
        input_tensor = munet.from_numpy(input_np).to(device)
        
        engine.compile(input_tensor)
        output_tensor = engine.run(input_tensor)
        output_np = output_tensor.to(munet.Device(munet.DeviceType.CPU, 0)).detach().numpy()
        
        print(f"    MuNet output shape: {output_np.shape}")
        
        # Compare outputs
        pytorch_output_np = pytorch_output.detach().numpy()[0]
        max_diff = np.max(np.abs(pytorch_output_np - output_np))
        print(f"    Max difference (PyTorch vs MuNet): {max_diff:.6e}")
        
        if max_diff < 1e-5:
            print("    ✓ Outputs match within tolerance!")
        else:
            print("    ⚠ Outputs differ - check weight conversion")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo_pytorch_export()
