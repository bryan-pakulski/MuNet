import io
import numpy as np
import pytest
from munet.serialization import _read_tensor


def test_tensor_header_cannot_trigger_oversized_allocation():
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(buf, {"descr":"<f4", "fortran_order":False, "shape":(2**40,)})
    with pytest.raises(ValueError, match="payload size"):
        _read_tensor(buf.getvalue(), [2**40])


def test_tensor_payload_rejects_pickle_and_truncation():
    for array in [np.asarray([object()], dtype=object), np.asarray([1, 2, 3], np.float32)]:
        buf = io.BytesIO(); np.save(buf, array)
        with pytest.raises(ValueError): _read_tensor(buf.getvalue()[:-1], list(array.shape))


def test_torch_state_dict_names_and_shapes():
    import munet as mu
    torch = pytest.importorskip("torch")
    source = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    target = mu.nn.Sequential(mu.nn.Linear(3, 4), mu.nn.ReLU(), mu.nn.Linear(4, 2))
    target.load_state_dict(source.state_dict())
    assert set(target.state_dict()) == set(source.state_dict())
    for name, tensor in source.state_dict().items():
        np.testing.assert_array_equal(target.state_dict()[name], tensor.detach().numpy())


def test_nested_sequential_parameter_names():
    import munet as mu
    class Model(mu.nn.Module):
        def __init__(self): self.encoder = mu.nn.Sequential(mu.nn.Linear(2, 3))
    assert set(Model().state_dict()) == {"encoder.0.weight", "encoder.0.bias"}
