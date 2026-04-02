"""MuNet Python package."""

from ._core import *  # noqa: F401,F403
from ._helpers.onnx_integration import register_inference_bindings as _register_onnx

_register_onnx(
    inference,
    load_for_inference_fn=load_for_inference,
    load_weights_for_inference_fn=load_weights_for_inference,
)
