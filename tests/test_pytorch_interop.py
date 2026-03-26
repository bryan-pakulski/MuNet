"""
Test PyTorch interoperability - save/load and weight conversion.

Tests verify:
- Loading PyTorch weights into MuNet modules
- Saving MuNet weights to PyTorch-compatible format
- Round-trip conversion integrity
"""

import pytest
import numpy as np
import tempfile
import os
import sys
import subprocess

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    pytestmark = pytest.mark.skip(reason="PyTorch not installed")

# Import MuNet
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(repo_root, "python_src"))
for build_dir in (
    os.path.join(repo_root, "build", "debug"),
    os.path.join(repo_root, "build", "release"),
    os.path.join(repo_root, "build"),
):
    if os.path.isdir(build_dir):
        sys.path.insert(0, build_dir)

try:
    import munet
except ImportError:
    subprocess.run(["make", "build-debug", "-j4"], cwd=repo_root, check=True)
    for build_dir in (
        os.path.join(repo_root, "build", "debug"),
        os.path.join(repo_root, "build", "release"),
        os.path.join(repo_root, "build"),
    ):
        if os.path.isdir(build_dir):
            sys.path.insert(0, build_dir)
    import munet
from munet import nn as munet_nn
from pytorch_interop import PyTorchInterop


@pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch not installed")
class TestPyTorchInteropBasic:
    """Test basic PyTorch interop functionality."""
    
    def test_pytorch_interop_initialization(self):
        """Test that PyTorchInterop can be initialized."""
        interop = PyTorchInterop()
        assert interop is not None
    
    def test_save_linear_weights(self):
        """Test saving MuNet Linear weights to PyTorch format."""
        # Create MuNet Linear layer
        munet_linear = munet_nn.Linear(10, 5)
        
        # Initialize with known weights
        weight_data = np.random.randn(5, 10).astype(np.float32)
        bias_data = np.random.randn(5).astype(np.float32)
        munet_linear.weight.copy_from_numpy(weight_data.T.copy())
        munet_linear.bias.copy_from_numpy(bias_data.copy())
        
        # Save to PyTorch format
        interop = PyTorchInterop()
        state_dict = interop.save_weights(munet_linear)
        
        # Verify state dict structure
        assert 'weight' in state_dict
        assert 'bias' in state_dict
        np.testing.assert_array_almost_equal(state_dict['weight'], weight_data)
        np.testing.assert_array_almost_equal(state_dict['bias'], bias_data)
    
    def test_load_linear_weights(self):
        """Test loading PyTorch Linear weights into MuNet."""
        # Create PyTorch Linear layer
        torch_linear = nn.Linear(10, 5)
        torch_linear.weight.data = torch.randn(5, 10)
        torch_linear.bias.data = torch.randn(5)
        
        # Create MuNet Linear layer
        munet_linear = munet_nn.Linear(10, 5)
        
        # Load weights
        interop = PyTorchInterop()
        state_dict = {
            'weight': torch_linear.weight.detach().numpy(),
            'bias': torch_linear.bias.detach().numpy()
        }
        interop.load_weights(munet_linear, state_dict)
        
        # Verify weights loaded correctly
        np.testing.assert_array_almost_equal(
            np.array(munet_linear.weight.detach(), copy=False),
            torch_linear.weight.detach().numpy().T
        )
        np.testing.assert_array_almost_equal(
            np.array(munet_linear.bias.detach(), copy=False),
            torch_linear.bias.detach().numpy()
        )


@pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch not installed")
class TestPyTorchInteropRoundTrip:
    """Test round-trip weight conversion between PyTorch and MuNet."""
    
    def test_linear_roundtrip(self):
        """Test round-trip conversion for Linear layers."""
        # Create PyTorch model
        torch_model = nn.Linear(20, 10)
        original_weight = torch_model.weight.detach().numpy().copy()
        original_bias = torch_model.bias.detach().numpy().copy()
        
        # Convert to MuNet
        interop = PyTorchInterop()
        munet_model = munet_nn.Linear(20, 10)
        interop.load_weights(munet_model, {
            'weight': original_weight,
            'bias': original_bias
        })
        
        # Convert back to PyTorch format
        state_dict = interop.save_weights(munet_model)
        
        # Verify round-trip integrity
        np.testing.assert_array_almost_equal(state_dict['weight'], original_weight)
        np.testing.assert_array_almost_equal(state_dict['bias'], original_bias)
    
    def test_conv2d_roundtrip(self):
        """Test round-trip conversion for Conv2d layers."""
        # Create PyTorch Conv2d
        torch_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        original_weight = torch_conv.weight.detach().numpy().copy()
        original_bias = torch_conv.bias.detach().numpy().copy()
        
        # Convert to MuNet
        interop = PyTorchInterop()
        munet_conv = munet_nn.Conv2d(3, 16, kernel_size=3, padding=1)
        interop.load_weights(munet_conv, {
            'weight': original_weight,
            'bias': original_bias
        })
        
        # Convert back
        state_dict = interop.save_weights(munet_conv)
        
        # Verify round-trip
        np.testing.assert_array_almost_equal(state_dict['weight'], original_weight)
        np.testing.assert_array_almost_equal(state_dict['bias'], original_bias)
    
    def test_multi_layer_roundtrip(self):
        """Test round-trip for multi-layer model."""
        # Create PyTorch sequential model
        torch_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        # Extract weights
        original_weights = {
            '0.weight': torch_model[0].weight.detach().numpy().copy(),
            '0.bias': torch_model[0].bias.detach().numpy().copy(),
            '2.weight': torch_model[2].weight.detach().numpy().copy(),
            '2.bias': torch_model[2].bias.detach().numpy().copy(),
        }
        
        # Create MuNet equivalent
        munet_model = munet_nn.Sequential(
            munet_nn.Linear(10, 20),
            munet_nn.ReLU(),
            munet_nn.Linear(20, 10)
        )
        
        # Load weights
        interop = PyTorchInterop()
        interop.load_weights(munet_model, original_weights)
        
        # Save back
        state_dict = interop.save_weights(munet_model)
        
        # Verify all weights
        for key in original_weights:
            np.testing.assert_array_almost_equal(
                state_dict[key], original_weights[key]
            )


@pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch not installed")
class TestPyTorchInteropFileIO:
    """Test file-based save/load operations."""
    
    def test_save_to_file(self):
        """Test saving MuNet weights to PyTorch .pt file."""
        munet_linear = munet_nn.Linear(10, 5)
        weight_data = np.random.randn(5, 10).astype(np.float32)
        bias_data = np.random.randn(5).astype(np.float32)
        munet_linear.weight.copy_from_numpy(weight_data.T.copy())
        munet_linear.bias.copy_from_numpy(bias_data.copy())
        
        interop = PyTorchInterop()
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to file
            interop.save_to_file(munet_linear, temp_path)
            
            # Load with PyTorch
            loaded = torch.load(temp_path)
            
            np.testing.assert_array_almost_equal(loaded['weight'], weight_data)
            np.testing.assert_array_almost_equal(loaded['bias'], bias_data)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_load_from_file(self):
        """Test loading PyTorch weights from .pt file."""
        # Create and save PyTorch model
        torch_linear = nn.Linear(10, 5)
        torch_linear.weight.data = torch.randn(5, 10)
        torch_linear.bias.data = torch.randn(5)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            torch.save({
                'weight': torch_linear.weight.detach().numpy(),
                'bias': torch_linear.bias.detach().numpy()
            }, temp_path)
            
            # Load into MuNet
            munet_linear = munet_nn.Linear(10, 5)
            interop = PyTorchInterop()
            interop.load_from_file(munet_linear, temp_path)
            
            # Verify
            np.testing.assert_array_almost_equal(
                np.array(munet_linear.weight.detach(), copy=False),
                torch_linear.weight.detach().numpy().T
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch not installed")
class TestPyTorchInteropBatchNorm:
    """Test BatchNorm weight conversion."""
    
    def test_batchnorm1d_weight_conversion(self):
        """Test BatchNorm1d weight conversion."""
        torch_bn = nn.BatchNorm2d(10)
        torch_bn.weight.data = torch.randn(10)
        torch_bn.bias.data = torch.randn(10)
        torch_bn.running_mean = torch.randn(10)
        torch_bn.running_var = torch.abs(torch.randn(10)) + 0.1
        
        munet_bn = munet_nn.BatchNorm2d(10)
        
        interop = PyTorchInterop()
        interop.load_weights(munet_bn, {
            'weight': torch_bn.weight.detach().numpy(),
            'bias': torch_bn.bias.detach().numpy(),
            'running_mean': torch_bn.running_mean.detach().numpy(),
            'running_var': torch_bn.running_var.detach().numpy(),
        })
        
        # Verify
        np.testing.assert_array_almost_equal(
            np.array(munet_bn.weight.detach(), copy=False), torch_bn.weight.detach().numpy()
        )
        np.testing.assert_array_almost_equal(
            np.array(munet_bn.bias.detach(), copy=False), torch_bn.bias.detach().numpy()
        )


@pytest.mark.skipif(not HAS_PYTORCH, reason="PyTorch not installed")  
class TestPyTorchInteropInference:
    """Test inference consistency between PyTorch and MuNet models."""
    
    def test_linear_inference_consistency(self):
        """Test that loaded MuNet model produces same output as PyTorch."""
        # Create PyTorch model
        torch_linear = nn.Linear(10, 5)
        torch_linear.eval()
        
        # Create MuNet model and load weights
        munet_linear = munet_nn.Linear(10, 5)
        interop = PyTorchInterop()
        interop.load_weights(munet_linear, {
            'weight': torch_linear.weight.detach().numpy(),
            'bias': torch_linear.bias.detach().numpy()
        })
        
        # Test input
        test_input = np.random.randn(1, 10).astype(np.float32)
        
        # PyTorch inference
        torch_output = torch_linear(torch.from_numpy(test_input)).detach().numpy()
        
        # MuNet inference
        munet_output = munet_linear(munet.from_numpy(test_input)).detach().numpy()
        
        # Compare
        np.testing.assert_array_almost_equal(torch_output, munet_output, decimal=5)
    
    def test_conv2d_inference_consistency(self):
        """Test Conv2d inference consistency."""
        # Create PyTorch Conv2d
        torch_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        torch_conv.eval()
        
        # Create MuNet Conv2d
        munet_conv = munet_nn.Conv2d(3, 8, kernel_size=3, padding=1)
        interop = PyTorchInterop()
        interop.load_weights(munet_conv, {
            'weight': torch_conv.weight.detach().numpy(),
            'bias': torch_conv.bias.detach().numpy()
        })
        
        # Test input (batch=1, channels=3, height=8, width=8)
        test_input = np.random.randn(1, 3, 8, 8).astype(np.float32)
        
        # PyTorch inference
        torch_output = torch_conv(torch.from_numpy(test_input)).detach().numpy()
        
        # MuNet inference
        munet_output = munet_conv(munet.from_numpy(test_input)).detach().numpy()
        
        # Compare
        np.testing.assert_array_almost_equal(torch_output, munet_output, decimal=4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
