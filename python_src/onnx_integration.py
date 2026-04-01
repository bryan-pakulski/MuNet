"""Compatibility shim for legacy source-tree imports.

Runtime packaging now uses `munet_nn._helpers.onnx_integration` as the source of
truth. This shim keeps source/developer workflows functional while avoiding
code duplication.
"""

from munet_nn._helpers.onnx_integration import *  # noqa: F401,F403
from munet_nn._helpers.onnx_integration import (
    register_inference_bindings as _register_inference_bindings,
)

if "inference" in globals():
    _register_inference_bindings(inference)
