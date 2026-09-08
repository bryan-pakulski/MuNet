import os
import pytest

os.environ.setdefault("MUNET_TEST_VULKAN", "1")


@pytest.fixture(params=["cpu"] + (["vulkan"] if os.environ.get("MUNET_TEST_VULKAN") == "1" else []))
def device(request):
    # Vulkan is tested by default; MUNET_TEST_VULKAN=0 explicitly selects CPU-only tests.
    return request.param
