"""
Class to detach a pytorch tensor fully onto numpy, AND squeeze.
"""


class DetachNumpy:
    """
    Class to detach a pytorch tensor fully onto numpy, AND squeeze.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, p):
        return p.detach().cpu().numpy().squeeze()
