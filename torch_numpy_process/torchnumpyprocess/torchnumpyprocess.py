"""
Class to correctly initialise pytorch tensors from numpy tensor, or check that a tensor is on the correct device etc.
This has methods;
_initialise
_reinitialise
_torch_numpy_process
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

    """

    def __init__(self, set_fail=False, cuda_device=None):
        """Init

        Parameters
        ----------
        set_fail : bool, optional
            If True then we force use_cuda=False meaning that torch.tensors will stay on cpu, by default False. If False then we check if cuda is available and if so use cuda.
        """
        self.set_fail = set_fail
        self._initialise(set_fail)
    
        if cuda_device:
            self.device = cuda_device
            torch.cuda.set_device(self.device)
        else:
            self.device = None

    def _initialise(self, set_fail):
        """Initialise the class, meaning we look if cuda is available as this will define what device to put torch.tensors on.

        Parameters
        ----------
        set_fail :
            see above
        """
        if set_fail:
            use_cuda = False
        else:
            use_cuda = torch.cuda.is_available()

        # Assign attribute to use in child classes
        # pylint: disable-next=no-member
        self.dtype = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor

    def _reinitialise(self, set_fail=False):
        """Force reassessment of whether cuda is available or not, without reinitialising the whole class.

        Parameters
        ----------
        set_fail :
            see above
        """

        self._initialise(set_fail)

    def _torch_numpy_process(self, x, non_blocking=False):
        """
        Given argument x, test is its in the appropriate form, a torch.tensor on the correct device.

        Returns
        -------
        array like
            array to be processed
        """
        if isinstance(x, torch.Tensor):

            if x.type() == self.dtype:
                # We perform this check to avoid moving tensors around in memory.
                return x
            else:
                return x.type(self.dtype, non_blocking=non_blocking)
        else:
            # pylint: disable-next=no-member
            return torch.tensor(x).type(self.dtype, non_blocking=non_blocking)

    def _clone_process(self, x, non_blocking=False):
        """Process and clone argument, to a torch.tensor on the correct device.

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        array like
            array to be processed
        """
        return self._torch_numpy_process(x, non_blocking=non_blocking).clone()
