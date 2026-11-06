"""
Class to correctly initialise pytorch tensors from numpy tensor, or check that a tensor is on the correct device etc.
This has methods;
_initialise
_reinitialise
_torch_numpy_process

`TorchNumpyProcessing` is the base of the whole class chain:
TorchNumpyProcessing -> Tensorisation -> CostClass -> UnbalancedSinkhorn
-> UnbalancedOT -> DebiasedUOT. Every input (points, densities, parameters,
potentials) passes through `_torch_numpy_process` or `_clone_process`, so the
solver always works with float64 torch tensors on one chosen device.
"""

import torch


class TorchNumpyProcessing:
    """
    Class to correctly initialise pytorch tensors from numpy tensor, or check that a tensor is on the correct device etc.

    This has methods;
    _initialise
    _reinitialise
    _torch_numpy_process

    Note parameter, set_fail : bool, optional
        If True then we force use_cuda=False meaning that torch.tensors will stay on cpu, by default False. If False then we check if cuda is available and if so use cuda.

    Attributes
    ----------
    set_fail : bool
        True if the CPU was forced (set_fail=True, or cuda_device='cpu').
    device : torch.device
        Device that every processed tensor is moved to.
    dtype : torch.dtype
        Always torch.float64 (double precision).
    """

    def __init__(self, set_fail=False, cuda_device=None):
        """Init

        Parameters
        ----------
        set_fail : bool, optional
            If True then we force use_cuda=False meaning that torch.tensors will stay on cpu, by default False. If False then we check if cuda is available and if so use cuda.
        cuda_device : int, str or None, optional
            Which GPU to use when CUDA is available, by default None (meaning cuda:0).
            - int, e.g. 1        -> "cuda:1"
            - str with "cuda"    -> used as given, e.g. "cuda:1"
            - the string "cpu"   -> same as set_fail=True
            - anything else      -> "cuda:0" (see NOTE in `_initialise`)
        """
        # The string "cpu" is a shortcut for forcing the CPU.
        if cuda_device == 'cpu':
            set_fail = True
        self.set_fail = set_fail
        self._initialise(set_fail, cuda_device)

    def _initialise(self, set_fail, cuda_device):
        """Initialise the class, meaning we look if cuda is available as this will define what device to put torch.tensors on.

        Sets `self.device` and `self.dtype`. When a GPU is chosen this also calls
        `torch.cuda.set_device`, which changes PyTorch's current GPU for the whole
        Python process, not just this object.

        Parameters
        ----------
        set_fail :
            see above
        cuda_device :
            see `__init__`
        """
        # Determine whether CUDA can be used
        if set_fail:
            use_cuda = False
        else:
            use_cuda = torch.cuda.is_available()

        if use_cuda:
            # Handle optional cuda_device argument
            if cuda_device is not None:
                # cuda_device can be an int (e.g., 0) or string (e.g., 'cuda:0')
                # NOTE: a torch.device object is neither an int nor a str, so it
                # falls through to "cuda:0". Checked: torch.device("cpu") gives
                # cuda:0, and so does an unknown string such as "mps". This is why
                # DebiasedUOT passes str(self.device).
                if isinstance(cuda_device, int):
                    device_str = f"cuda:{cuda_device}"
                elif isinstance(cuda_device, str) and 'cuda' in cuda_device:
                    device_str = cuda_device
                else:
                    device_str = "cuda:0"
            else:
                device_str = "cuda:0"

            # Verify that device exists
            # NOTE: this only checks that at least one GPU exists, not that the
            # requested index exists. Asking for cuda:1 on a single-GPU machine
            # makes set_device below raise "CUDA error: invalid device ordinal".
            if torch.cuda.device_count() > 0:
                self.device = torch.device(device_str)
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cpu")
                use_cuda = False
        else:
            self.device = torch.device("cpu")


        self.dtype = torch.float64  # Double precision

    def _reinitialise(self, set_fail=False):
        """Force reassessment of whether cuda is available or not, without reinitialising the whole class.

        Not called anywhere in the package.

        Parameters
        ----------
        set_fail :
            see above
        """

        # BUG(stage 4): `_initialise` also needs `cuda_device`, so this call
        # always raises TypeError (checked).
        self._initialise(set_fail)

    def _torch_numpy_process(self, x, non_blocking=False):
        """
        Given argument x, test is its in the appropriate form, a torch.tensor on the correct device.

        - torch.Tensor already float64 on `self.device`: returned AS IS. This is the
          same object as the input, so changing the result in place also changes
          the caller's tensor. Use `_clone_process` when a private copy is needed.
        - torch.Tensor with a different dtype or device: converted (a new tensor).
        - Anything else (numpy array, list, Python number): copied into a new
          float64 tensor on `self.device`.

        Parameters
        ----------
        x : array like
            Value to convert.
        non_blocking : bool, optional
            Passed to `Tensor.to`, by default False. Only speeds up copies from
            pinned CPU memory to the GPU; otherwise it has no effect.

        Returns
        -------
        array like
            array to be processed
        """
        if isinstance(x, torch.Tensor):
            # Ensure correct dtype and device
            if x.dtype != self.dtype or x.device != self.device:
                return x.to(device=self.device, dtype=self.dtype, non_blocking=non_blocking)
            return x
        else:
            # Convert numpy array or other to torch tensor on correct device
            return torch.tensor(x, device=self.device, dtype=self.dtype)

    def _clone_process(self, x, non_blocking=False):
        """Process and clone argument, to a torch.tensor on the correct device.

        Always returns a new tensor, so the caller's data is never modified by
        later in-place updates.

        NOTE: when `_torch_numpy_process` already made a new tensor (numpy input,
        or a dtype/device change), the extra `.clone()` copies the data a second
        time. Harmless, but it briefly doubles memory for large inputs.

        Parameters
        ----------
        x : array like
            Value to convert and copy.
        non_blocking : bool, optional
            See `_torch_numpy_process`.

        Returns
        -------
        array like
            array to be processed
        """
        return self._torch_numpy_process(x, non_blocking=non_blocking).clone()
