"""
Class `Tensorisation` for tensorisation of cost for regular cartesian grid meshes

"""
import torch
from torchnumpyprocess import TorchNumpyProcessing

# pylint: disable=no-member


class Tensorisation(TorchNumpyProcessing):
    """
    Class `Tensorisation` for tensorisation of cost for regular cartesian grid meshes
    """

    def __init__(self, set_fail=False, cuda_device=None,coordinate="eularian") -> None:
        """Class __init__

        Parameters
        ----------
        set_fail : bool, optional
            Passed to TorchNumpyProcessing for setting the device, by default False
        coordinate : str, optional
            The geometry: ['eularian', ...tbc...], by default "eularian"
        """

        super().__init__(set_fail, cuda_device)

        # Switches to figure out if a given data set is tensorisable (made up word?)
        self.tensorise = [None, None]
        self.coordinate = coordinate

    def process_kwargs(self, kwarg, n1, n2, m1, m2):
        """
        Process kwargs to return the weighting of the matrix multiplication.
        This can be ones in the case of simple miltuplication of weighted if given kwargs['weight']


        Parameters
        ----------
        kwarg : dictionary
            **kwargs given to __call__
        n1 : int
            X_x size
        n2 : int
            X_y size
        m1 : int
            Y_x size
        m2 : int
            Y_y size

        Returns
        -------
        torch.Tensor
            weighting of the matrix multiplication

        Raises
        ------
        ValueError
            "Expected **kwarg with one key from; dim, axis or weight"
        """
        # If not given weight argument then return ones of the correct dimensions
        if ("dim" in kwarg or "axis" in kwarg) and (len(kwarg) == 1):
            if 0 in kwarg.values():
                f = self._torch_numpy_process(torch.ones(n1, n2))
            elif 1 in kwarg.values():
                f = self._torch_numpy_process(torch.ones(m1, m2))
        elif ("weight" in kwarg) and (len(kwarg) == 1):
            f = self._torch_numpy_process(kwarg["weight"])
        else:
            raise ValueError("Expected **kwarg with one key from; dim, axis or weight")
        return f

    def __call__(self, X_grid_x, X_grid_y, Y_grid_x, Y_grid_y, **kwarg):
        """
        Tensorisation call, which multiples the 2D regular meshes X, Y, which can respectively be
        described by (X_x, X_y) and (Y_x, Y_y).

        Parameters
        ----------
        X_grid_x : array
            X_x points
        X_grid_y : array
            X_y points
        Y_grid_x : array
            Y_x points
        Y_grid_y : array
            Y_y points
        kwargs:
            **kwarg with one key from; dim, axis or weight
        Returns
        -------
        torch.Tensor
            multiplied tensors
        """

        n1, n2 = len(X_grid_x), len(X_grid_y)
        m1, m2 = len(Y_grid_x), len(Y_grid_y)

        # process arguments
        X_grid_x = self._torch_numpy_process(X_grid_x)
        X_grid_y = self._torch_numpy_process(X_grid_y)
        Y_grid_x = self._torch_numpy_process(Y_grid_x)
        Y_grid_y = self._torch_numpy_process(Y_grid_y)

        f = self.process_kwargs(kwarg, n1, n2, m1, m2)

        # TODO squared or not?
        C1 = torch.exp(torch.cdist(X_grid_x[:, None], Y_grid_x[:, None]) ** 2).view(
            n1, m1
        )
        C2 = torch.exp(torch.cdist(X_grid_y[:, None], Y_grid_y[:, None]) ** 2).view(
            n2, m2
        )

        return self.tensorise_f(C1, C2, f)

    def tensorise_f(self, C1, C2, f):
        """
        Perform the tensorised multiplication for regular cartesian grid

        Parameters
        ----------
        C1 : torch.Tensor
            X_x . Y_x (n1, m1)
        C2 : torch.Tensor
            X_y . Y_y (n2, m2)
        f : torch.Tensor
            weighting (n1, n2) or (m1, m2)

        Returns
        -------
        torch.Tensor
            output multiplication (m1, m2) or (n1, n2)
        """
        # Check dimensions
        N, M = f.shape

        if N == C1.shape[0] and M == C2.shape[0]:
            ind = 0
        else:
            ind = 1

        return torch.tensordot(
            torch.tensordot(C1, f, dims=([ind], [0])), C2, dims=([1], [ind])
        )

    def _process_points(self, x):
        """
        Process to figure out if to tensorise or not? What shape is the data we're given

        Parameters
        ----------
        x : (N, 2) or (n1, n2, 2) or- tuple((n1), (n2))
            input data that may be tensorisable.  If (N,2) we can't tensorise.

        Returns
        -------
        tuple/int
            shape of tensorisable object

        Raises
        ------
        ValueError
            "Expected points input (N,2), (n1, n2, 2), tuple((n1), (n2))"
        """
        # Reset
        if self.tensorise[0] is not None and self.tensorise[1] is not None:
            self.tensorise = [None, None]

        if self.tensorise[0] is None:
            ind = 0
        else:
            ind = 1

        # tuple to tensorize procedure
        if type(x) == tuple and len(x[0].shape) == 1 and len(x[1].shape) == 1:
            self.tensorise[ind] = True
            return x[0].shape[0], x[1].shape[0]
        elif type(x) == tuple:
            raise ValueError(
                "Expected points input (N,2), (n1, n2, 2), tuple((n1), (n2))"
            )

        size = x.shape

        if len(size) == 2 and size[1] == 2:  # (N, 2) usual thing
            self.tensorise[ind] = False
            return size[0], 1
        elif len(size) == 3 and size[2] == 2:  # (n1, n2, 2)
            self.tensorise[ind] = True
            return size[0], size[1]
        else:
            raise ValueError(
                "Expected points input (N,2), (n1, n2, 2), tuple((n1), (n2))"
            )


# pylint: enable=no-member
if __name__ == "__main__":
    pass
