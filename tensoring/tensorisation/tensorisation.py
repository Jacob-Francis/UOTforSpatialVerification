"""
Class `Tensorisation` for tensorisation of cost for regular cartesian grid meshes.

Why tensorise?
--------------
On a regular, axis-aligned 2D grid the squared Euclidean cost separates:

    C((x_k, y_p), (x'_r, y'_s)) = C1[k, r] + C2[p, s]

where C1 only involves the x-coordinates and C2 only the y-coordinates.
The Gibbs kernel exp(-C / eps) then factorises into exp(-C1 / eps) * exp(-C2 / eps),
so applying the full (n1*n2) x (m1*m2) kernel to a field reduces to two small
matrix products (see `tensorise_f`). That avoids ever building the full cost matrix:

    memory  O(n1*m1 + n2*m2)       instead of O(n1*n2*m1*m2)
    compute O(n1*n2*(m1 + m2))-ish instead of O(n1*n2*m1*m2)

Shape conventions used throughout
---------------------------------
- Source (observation) grid: n1 points in x, n2 points in y -> fields of shape (n1, n2).
- Target (forecast) grid:    m1 points in x, m2 points in y -> fields of shape (m1, m2).
- C1 : (n1, m1) kernel between the x-coordinates.
- C2 : (n2, m2) kernel between the y-coordinates.
"""
import torch
from torchnumpyprocess import TorchNumpyProcessing

# pylint: disable=no-member


class Tensorisation(TorchNumpyProcessing):
    """
    Class `Tensorisation` for tensorisation of cost for regular cartesian grid meshes.

    Inherits device/dtype handling from `TorchNumpyProcessing`, and is itself the base
    class of `CostClass` -> `UnbalancedSinkhorn` -> `UnbalancedOT` -> `DebiasedUOT`.

    Attributes
    ----------
    tensorise : list of [bool or None, bool or None]
        Whether the [source, target] point clouds are regular grids that can be
        tensorised. Filled in by `_process_points`; None means "not yet processed".
        The solver only uses the tensorised path when BOTH entries are True.
    coordinate : str
        Geometry label. Stored but not currently used anywhere.
    """

    def __init__(self, set_fail=False, cuda_device=None,coordinate="eularian") -> None:
        """Class __init__

        Parameters
        ----------
        set_fail : bool, optional
            Passed to TorchNumpyProcessing for setting the device, by default False.
            If True, force CPU.
        cuda_device : int, str or None, optional
            Passed to TorchNumpyProcessing, e.g. 0 or "cuda:0", by default None.
        coordinate : str, optional
            The geometry: ['eularian', ...tbc...], by default "eularian"
        """

        super().__init__(set_fail, cuda_device)

        # Switches to figure out if a given data set is tensorisable (made up word?)
        # Index 0 = source points, index 1 = target points.
        self.tensorise = [None, None]
        self.coordinate = coordinate

    def process_kwargs(self, kwarg, n1, n2, m1, m2):
        """
        Process kwargs to return the weighting of the matrix multiplication.
        This can be ones in the case of simple miltuplication of weighted if given kwargs['weight']

        Only used by `__call__`.

        Parameters
        ----------
        kwarg : dictionary
            **kwargs given to __call__. Exactly one key:
            - dim=0 or axis=0 : weight is ones of the source shape (n1, n2)
            - dim=1 or axis=1 : weight is ones of the target shape (m1, m2)
            - weight=array    : use the given array as the weight
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
            weighting of the matrix multiplication, shape (n1, n2) or (m1, m2)

        Raises
        ------
        ValueError
            "Expected **kwarg with one key from; dim, axis or weight"
        """
        # If not given weight argument then return ones of the correct dimensions
        # NOTE: if dim/axis is given a value other than 0 or 1, `f` is never assigned
        # and the `return f` below raises UnboundLocalError.
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

        Builds the two 1D kernels from the grid axes and applies them to a weight
        with `tensorise_f`. The solver does NOT use this method: it calls
        `tensorise_f` directly with its own Gibbs kernels exp(-C / eps).

        Note the kernels here are exp(+|x - y|^2), i.e. a positive exponent and no
        epsilon (see the TODO below). This is what the tests check.

        Parameters
        ----------
        X_grid_x : array
            X_x points, shape (n1,)
        X_grid_y : array
            X_y points, shape (n2,)
        Y_grid_x : array
            Y_x points, shape (m1,)
        Y_grid_y : array
            Y_y points, shape (m2,)
        kwargs:
            **kwarg with one key from; dim, axis or weight (see `process_kwargs`)
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

        # 1D kernels between the x-axes (n1, m1) and the y-axes (n2, m2).
        # [:, None] turns each axis into a column of 1D "points" for cdist.
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
        Perform the tensorised multiplication for regular cartesian grid.

        Applies the separable kernel K = C1 (x) C2 to the 2D field f without forming K.
        Which way it applies depends on f's shape:

        - f is (n1, n2), a source field -> sum over the source grid:
              out[r, s] = sum_{k, p} C1[k, r] * f[k, p] * C2[p, s]      out: (m1, m2)
          i.e. K^T f. In the solver this is the g (target) update.

        - f is (m1, m2), a target field -> sum over the target grid:
              out[k, p] = sum_{r, s} C1[k, r] * f[r, s] * C2[p, s]      out: (n1, n2)
          i.e. K f. In the solver this is the f (source) update.

        Each case is two tensordots, so the full (n1*n2) x (m1*m2) kernel is never built.

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

        # ind is the C1/C2 axis that gets contracted with f:
        #   ind = 0 -> f lives on the source grid (rows of C1, C2)
        #   ind = 1 -> f lives on the target grid (columns of C1, C2)
        # BUG(stage 4): when the source and target grids have the same shape
        # (n1, n2) == (m1, m2), a target field also matches here, so ind = 0 and the
        # f-update uses K^T instead of K. This is only correct when K is symmetric,
        # i.e. the two grids have identical coordinates. Shifted grids of the same
        # shape give wrong results (measured: primal 1.21 instead of 0.186).
        if N == C1.shape[0] and M == C2.shape[0]:
            ind = 0
        else:
            ind = 1

        # Inner tensordot: contract the x-axis of f with C1.
        # Outer tensordot: contract the y-axis with C2.
        return torch.tensordot(
            torch.tensordot(C1, f, dims=([ind], [0])), C2, dims=([1], [ind])
        )

    def _process_points(self, x):
        """
        Process to figure out if to tensorise or not? What shape is the data we're given

        Called twice per `densities()` call: first for the source points, then for
        the target points. Each call fills the next empty slot of `self.tensorise`
        (source first, then target). When both slots are already filled, they are
        reset, so a new `densities()` call starts over.

        Parameters
        ----------
        x : (N, 2) or (n1, n2, 2) or- tuple((n1), (n2)) or (N, 3)
            input data that may be tensorisable.  If (N,2) we can't tensorise.
            - tuple of two 1D arrays: the x- and y-axes of a regular grid -> tensorise
            - (n1, n2, 2): a grid of points (meshgrid stacked on the last axis) -> tensorise
            - (N, 2): a flat 2D point cloud -> no tensorisation
            - (N, 3): a flat 3D point cloud -> no tensorisation (KeOps only)

        Returns
        -------
        tuple/int
            shape of tensorisable object, used as the shape of the density:
            (n1, n2) for grids, (N, 1) for flat point clouds.

        Raises
        ------
        ValueError
            "Expected points input (N,2), (n1, n2, 2), tuple((n1), (n2))"
        """
        # Reset
        if self.tensorise[0] is not None and self.tensorise[1] is not None:
            self.tensorise = [None, None]

        # ind: which slot to fill. 0 = source (first call), 1 = target (second call).
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
            # BUG(stage 4): any (n1, n2, 2) array is treated as a regular,
            # axis-aligned grid. Curvilinear grids (e.g. lon/lat) pass this check
            # but their cost does not separate, so results are silently wrong
            # (measured: primal 0.241 instead of 0.173).
            self.tensorise[ind] = True
            return size[0], size[1]
        elif size[1] == 3:
            # NOTE: this branch does not set self.tensorise[ind], so that slot stays
            # None. With a 2D source and a 3D target this leaves [False, None], and
            # the next densities() call then fills the slots in the wrong order.
            print('3D works only for pykeops right now - so no tensorisation implemented yet')
            return size[0], 1 # shape for the density

        else:
            raise ValueError(
                "Expected points input (N,2), (n1, n2, 2), tuple((n1), (n2))"
            )


# pylint: enable=no-member
if __name__ == "__main__":
    pass
