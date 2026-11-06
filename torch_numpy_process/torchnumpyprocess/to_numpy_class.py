"""
Class to detach a pytorch tensor fully onto numpy, AND squeeze.
"""


class DetachNumpy:
    """
    Class to detach a pytorch tensor fully onto numpy, AND squeeze.

    Used as a function: `DetachNumpy()(tensor)`. Handy for plotting or saving
    results that live on the GPU or are part of an autograd graph.
    """

    def __init__(self) -> None:
        """No settings to store."""
        pass

    def __call__(self, p):
        """
        Convert a tensor to a squeezed numpy array.

        Parameters
        ----------
        p : torch.Tensor
            Tensor on any device.

        Returns
        -------
        numpy.ndarray
            Same values with all length-1 dimensions removed, e.g. (N, 1) -> (N,).
            A one-element tensor becomes a 0-d array.
        """
        # detach: drop autograd history; cpu: numpy can't read GPU memory;
        # squeeze: remove length-1 axes.
        return p.detach().cpu().numpy().squeeze()
