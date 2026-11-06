"""
`CostClass`: builds the transport cost for whichever back-end will be used.

The cost is always  (cost_const / 2) * c(x, y),  where c is

- 'rigid'      : |x - y|^2                                   (the default)
- 'periodic'   : like rigid, but x (the first coordinate) wraps round with period L
- 'beta_plane' : unfinished (see the BUG notes below)

What gets built depends on the input (see `Tensorisation._process_points`):

- regular grids (tuples, or (n1, n2, 2) arrays) -> two small 1D costs, cost_1 (n1, m1)
                                                    and cost_2 (n2, m2)
- flat point clouds with pykeops=True           -> nothing; KeOps computes the cost
                                                    on the fly
- flat point clouds otherwise                   -> the full (N, M) matrix self.cost

Whenever pykeops=True, the KeOps formulas are also built, because UnbalancedOT can
use them for the energies and barycentres even when the Sinkhorn loop does not.
"""
from tensorisation import Tensorisation
from .utils import pbcost_cdist
from .pykeops_formula import PyKeOpsFormulas
import torch
from pykeops.torch import LazyTensor


class CostClass(Tensorisation):
    """
    Cost construction layer of the class chain:
    TorchNumpyProcessing -> Tensorisation -> CostClass -> UnbalancedSinkhorn -> ...

    Attributes set by `process_initial_point_clouds`
    ------------------------------------------------
    cost_kwargs : dict
        Cost options; always has 'cost_type', plus 'L' for periodic.
    X_s, Y_t : torch.Tensor or tuple
        Source and target points (tuples are stored as given).
    cost_1, cost_2 : torch.Tensor
        Separable 1D costs, regular-grid inputs only.
    cost : torch.Tensor
        Full (N, M) cost, dense path only.
    pbcost_ind : torch.Tensor
        Periodic shift index (0: -L, 1: none, 2: +L), periodic cost only.
    pykeops_formulas : PyKeOpsFormulas
        Only when pykeops=True.
    """

    def process_initial_point_clouds(
        self,
        source_points,
        target_points,
        cost_const,
        n1,
        n2,
        m1,
        m2,
        **kwargs
    ):
        """
        Process point clouds depending on the input shape and if pykeops is available or if tensorisation would be faster
        kwargs;
        cost_type: str
            'rigid', 'periodic'  ToDo: [ 'beta', 'beta_periodic']. Need to pass kwarg
        L: int
                for periodic case

        Called by `UnbalancedSinkhorn.densities`, after `_process_points` has set
        `self.tensorise` and worked out n1, n2, m1, m2.

        Parameters
        ----------
        source_points : (N, 2), (n1, n2, 2), ((n1), (n2))
            source 2D points, this is the mesh. Not weights.
        target_points : (M, 2), (m1, m2, 2), ((m1), (m2))
            target 2D points, this is the mesh. Not weights.
        cost_const : torch.Tensor
            Constant in front of the cost; the cost is (cost_const / 2) * c(x, y).
        n1, n2, m1, m2 : int
            Grid sizes from `_process_points` ((N, 1) and (M, 1) for flat inputs).
        source_density : (N) (n1, n2), optional
            weights associated to the 2D source points, by default None
            (NOTE: not actually a parameter here; densities are handled in
            `UnbalancedSinkhorn.densities`.)
        target_density : (N) (m1, m2), optional
            weights associated to the 2D target points, by default None
            (NOTE: not actually a parameter here either.)
        kwargs:
            relating to the geometry or cost type
        """

        # Process kwargs
        if 'cost_type' in kwargs.keys():
            # NOTE: this stores the caller's kwargs dict itself, not a copy.
            self.cost_kwargs = kwargs
            if 'L' in kwargs.keys():
                # BUG(stage 4): int() silently truncates a non-integer period,
                # e.g. L=1.5 becomes 1 (checked). The KeOps formula below also
                # needs an integer (IntCst), see its ToDo.
                self.cost_kwargs['L'] = int(self.cost_kwargs['L'])
        else:
            self.cost_kwargs = dict(cost_type='rigid')

        # pylint: disable=attribute-defined-outside-init
        if isinstance(source_points, tuple) and isinstance(target_points, tuple):
            # Tuple input (x-axis, y-axis): the 1D costs come straight from the axes.
            # cost_1[k, r] = (C/2) (x_k - x'_r)^2 ,  cost_2[p, s] = (C/2) (y_p - y'_s)^2
            # BUG(stage 4): cost_type is ignored here, so 'periodic' silently gives
            # the rigid cost (checked).
            self.cost_1 = (
                0.5
                * cost_const
                * torch.cdist(
                    self._clone_process(source_points[0], non_blocking=True).view(
                        -1, 1
                    ),
                    self._clone_process(target_points[0], non_blocking=True).view(
                        -1, 1
                    ),
                )
                ** 2
            )
            self.cost_2 = (
                0.5
                * cost_const
                * torch.cdist(
                    self._clone_process(source_points[1], non_blocking=True).view(
                        -1, 1
                    ),
                    self._clone_process(target_points[1], non_blocking=True).view(
                        -1, 1
                    ),
                )
                ** 2
            )

            # Save the tuples for use in forming OT cost terms
            self.X_s = source_points
            self.Y_t = target_points
        else:
            # Not necessary to clone, to save memory
            # (NOTE: despite the comment, _clone_process does make a copy.)
            self.X_s = self._clone_process(source_points, non_blocking=True)
            self.Y_t = self._clone_process(target_points, non_blocking=True)

            # Calculate cost matrices
            if self.tensorise[0] and self.tensorise[1]:

                # This means we're okay to tensorise
                # (n1, n2, 2) grids: take one row and one column of 2D points.
                # X_s[:n1, 0] are the points (x_k, y_0); X_s[0, :n2] are (x_0, y_p).
                # BUG(stage 4): cdist on these 2D points also includes the other
                # coordinate, so cost_1 picks up (C/2)(y_0 - y'_0)^2 and cost_2
                # picks up (C/2)(x_0 - x'_0)^2. That is zero only when the two grids
                # start at the same point; offset grids get an extra constant
                # (checked: primal 0.298 instead of 0.241). The tuple branch above
                # does not have this problem. cost_type is ignored here too.
                self.cost_1 = (
                    0.5
                    * cost_const
                    * torch.cdist(self.X_s[:n1, 0], self.Y_t[:m1, 0]) ** 2
                )
                self.cost_2 = (
                    0.5
                    * cost_const
                    * torch.cdist(self.X_s[0, :n2], self.Y_t[0, :m2]) ** 2
                )
            elif self.pykeops == True:
                # Run cost calculation on the fly, not creating a large cost matrix
                pass
            else:
                self._full_calculation_of_cost(cost_const)

        # if pykeops is available we need the class ready, weather we use it for updates or not
        if self.pykeops:
            if isinstance(source_points, tuple) and isinstance(target_points, tuple):
                space_dim =2
            else:
                space_dim = source_points.shape[-1]
                assert space_dim == target_points.shape[-1], "Source and target point clouds must have the same dimension"

            if  self.cost_kwargs['cost_type']== 'rigid':
                self.pykeops_formulas = PyKeOpsFormulas(cost_string="SqDist(X, Y)", space_dim=space_dim)
            elif  self.cost_kwargs['cost_type'] == 'periodic':
                # Periodic in x: min over the three shifts x - L, x + L, x of the
                # squared x-distance, plus the squared y-distance. 2D only
                # (space_dim is left at its default of 2).
                # ToDo : integers part here - use Fractionals it can be made for rationals, not reals.
                self.pykeops_formulas = PyKeOpsFormulas(cost_string=f"(Min(Concat(SqDist(Elem(X, 0) - IntCst({int(kwargs['L'])}), Elem(Y, 0)), Concat(SqDist(Elem(X, 0) + IntCst({int(kwargs['L'])}), Elem(Y, 0)), SqDist(Elem(X, 0), Elem(Y, 0)))) + SqDist(Elem(X, 1), Elem(Y, 1))))")
            elif self.cost_kwargs['cost_type']== 'beta_plane':
                # ToDo
                # BUG(stage 4): no formula is built, so self.pykeops_formulas does
                # not exist and any later KeOps call raises AttributeError.
                pass


    def _full_calculation_of_cost(self, cost_const):
        """
        Build the full (N, M) cost matrix `self.cost` (dense back-end).

        Parameters
        ----------
        cost_const : torch.Tensor
            Constant in front of the cost.
        """
        if self.cost_kwargs['cost_type'] == 'rigid':
            self.cost = 0.5 * cost_const * torch.cdist(self.X_s, self.Y_t) ** 2
        elif self.cost_kwargs['cost_type']== 'periodic':
            # Also keeps which shift was used, for the periodic barycentre correction.
            self.cost, self.pbcost_ind = pbcost_cdist(self.X_s, self.Y_t, self.cost_kwargs['L'])
            self.cost *= 0.5 * cost_const
        elif self.cost_kwargs['cost_type']== 'beta_plane':
            # We should have cos_const = self.f_constant**2
            # BUG(stage 4): self.f_constant is never set anywhere in the package,
            # so this line raises AttributeError.
            self.cost = 0.5 * (self.f_constant + self.cost_kwargs['beta']*self.Y_t[:, 1].view(1, -1))**2 * torch.cdist(self.X_s, self.Y_t) ** 2


    def _generate_periodic_ind(self,):
        """
        Set `self.pbcost_ind`: for each pair (i, j), which x-shift gives the
        shortest periodic distance (0: x - L, 1: no shift, 2: x + L).

        Uses (x - y + s)^2 - (x - y)^2 = 2 s (x - y) + s^2, so only that extra term
        is compared. Called by `UnbalancedOT.barycentre_map_of_points` on the
        KeOps path; the dense path gets the index from `pbcost_cdist` instead.
        """
        # ToDo move intto pykeops formulas
        if self.pykeops:
            L = self.cost_kwargs['L']
            X_s, Y_t = self.X_s.contiguous(), self.Y_t.contiguous()
            # NOTE: X_s_2 and Y_t_2 are built but not used here.
            X_s_2, Y_t_2 = LazyTensor(X_s[:, None, :].contiguous()), LazyTensor(Y_t[None, :, :].contiguous())
            X_s_lt, Y_t_lt = LazyTensor(X_s[:, 0][:, None, None].contiguous()), LazyTensor(Y_t[:, 0][None, :, None].contiguous())

            # Extra cost for each shift, in index order -L, 0, +L.
            cost_lt = LazyTensor.concatenate((
                -2 * L * (X_s_lt - Y_t_lt) + L**2,
                (X_s_lt - Y_t_lt - (X_s_lt - Y_t_lt)),  # hack to make it zero
                2 * L * (X_s_lt - Y_t_lt) + L**2
            ))

            # argmin over the 3 shifts for every (i, j). Nothing is computed yet:
            # the result is a symbolic (N, M) LazyTensor (checked), used inside
            # later KeOps reductions, not a stored index array.
            self.pbcost_ind = cost_lt.argmin(dim=2)
        else:
            _, self.pbcost_ind = torch.min(torch.stack((
                -2*self.cost_kwargs['L']*(self.X_s[:, 0].view(-1, 1) - self.Y_t[:, 0].view(1, -1))+ self.cost_kwargs['L']**2,
                torch.zeros(len(self.X_s), len(self.Y_t)).type_as(self.X_s),
                2*self.cost_kwargs['L']*(self.X_s[:, 0].view(-1, 1) - self.Y_t[:, 0].view(1, -1)) + self.cost_kwargs['L']**2
                ), dim=0), dim=0) # .type_as(self.X_s)


    def _cost_update(self, cost_const):
        """
        Rebuild the dense cost with a new cost_const (does nothing with KeOps).

        NOTE: not called anywhere in the package (checked with grep).
        """
        # Update cost matrix
        if self.pykeops:
            pass
        else:
            self._full_calculation_of_cost(cost_const)

