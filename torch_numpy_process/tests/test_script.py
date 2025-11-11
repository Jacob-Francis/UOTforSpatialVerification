from torchnumpyprocess import TorchNumpyProcessing, DetachNumpy

import torch
import numpy as np
import pytest

# pylint isn't picking up on some torch functions
# pylint: disable=no-member
# pylint: disable=protected-access


@pytest.mark.parametrize(
    "device, data_form",
    [("cpu", "np"), ("cpu", "torch"), ("cuda", "np"), ("cuda", "torch")],
)
def test_torch_to_numpy_np_torch(device, data_form):
    if device == "cpu":
        tnp = TorchNumpyProcessing(set_fail=True)
    else:
        # Else we can test is cuda is avaliable to us or not and test this configuration. If cuda isn't then technically these are redundant tests.
        tnp = TorchNumpyProcessing(set_fail=False)
        if not torch.cuda.is_available():
            device = "cpu"

    if data_form == "np":
        x = np.random.rand(5, 5)
        x = tnp._torch_numpy_process(x)
        # This should work in place - if not then I need to fix this!
        assert type(x) == torch.Tensor

        # We use in rather than == due to the cuda core numbering
        assert device in str(x.device)
    else:
        x = torch.rand(5, 5, device="cpu")
        x = tnp._torch_numpy_process(x)
        # This should work in place - if not then I need to fix this!
        assert type(x) == torch.Tensor

        # We use in rather than == due to the cuda core numbering
        assert device in str(x.device)


# We need to also test if given a cuda array does it keep in cuda in place
def test_torch_to_numpy_torch_torch():
    tnp = TorchNumpyProcessing(set_fail=False)

    try:
        x = torch.rand(5, 5).type(torch.cuda.DoubleTensor)
        device = "cuda"
    except RuntimeError:
        x = torch.rand(5, 5, device="cpu").type(torch.DoubleTensor)
        device = "cpu"

    mem_location = x.data_ptr()
    x = tnp._torch_numpy_process(x)

    # It should keep the memory location the same once in hte correct form
    assert x.data_ptr() == mem_location

    # This should work in place - if not then I need to fix this!
    assert type(x) == torch.Tensor

    # We use in rather than == due to the cuda core numbering
    assert device in str(x.device)


# Test cloning version
@pytest.mark.parametrize(
    "device, data_form",
    [("cpu", "np"), ("cpu", "torch"), ("cuda", "np"), ("cuda", "torch")],
)
def test_clone_processing(device, data_form):
    if device == "cpu":
        tnp = TorchNumpyProcessing(set_fail=True)
    else:
        # Else we can test is cuda is avaliable to us or not and test this configuration. If cuda isn't then technically these are redundant tests.
        tnp = TorchNumpyProcessing(set_fail=False)
        if not torch.cuda.is_available():
            device = "cpu"

    if data_form == "np":
        x = np.random.rand(5, 5)
        x0 = tnp._clone_process(x)

        # This should work in place - if not then I need to fix this!
        assert type(x0) == torch.Tensor

        # We use in rather than == due to the cuda core numbering
        assert device in str(x0.device)
    else:
        x = torch.rand(5, 5, device="cpu")
        x0 = tnp._clone_process(x)

        # Check different memory locations
        assert x.data_ptr() != x0.data_ptr()

        # This should work in place - if not then I need to fix this!
        assert type(x0) == torch.Tensor

        # We use in rather than == due to the cuda core numbering
        assert device in str(x0.device)


# Test Detaching
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_detaching(device):

    if not torch.cuda.is_available():
        device = "cpu"
    x = torch.rand(5, 5, 1, device=device)
    dnp = DetachNumpy()
    x = dnp(x)

    assert isinstance(x, np.ndarray)


# Test reinitialising - this test only makes sense if cuda is available


# pylint: enable=no-member
# pylint: enable=protected-access

if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)