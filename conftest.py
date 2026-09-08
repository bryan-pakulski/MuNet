import os
import pytest


@pytest.fixture(params=["cpu"] + (["vulkan"] if os.environ.get("MUNET_TEST_VULKAN") == "1" else []))
def device(request):
    # Opted-in Vulkan failures are failures, not skips or CPU fallback.
    return request.param
