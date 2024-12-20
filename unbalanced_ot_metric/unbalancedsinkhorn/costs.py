from tensorisation import Tensorisation
from .utils import pbcost_cdist
from .pykeops_formula import PyKeOpsFormulas
import torch
from pykeops.torch import LazyTensor


class CostClass(Tensorisation):

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
        Parameters
        ----------
        source_points : (N, 2), (n1, n2, 2), ((n1), (n2))
            source 2D points, this is the mesh. Not weights.
        target_points : (M, 2), (m1, m2, 2), ((m1), (m2))
            target 2D points, this is the mesh. Not weights.
        source_density : (N) (n1, n2), optional
            weights associated to the 2D source points, by default None
        target_density : (N) (m1, m2), optional
            weights associated to the 2D target points, by default None
        kwargs:
            relating to the geometry or cost type
        """

        # Process kwargs
        if 'cost_type' in kwargs.keys():
            self.cost_kwargs = kwargs
            if 'L' in kwargs.keys():
                self.cost_kwargs['L'] = int(self.cost_kwargs['L'])
        else:
            self.cost_kwargs = dict(cost_type='rigid')

        # pylint: disable=attribute-defined-outside-init
        if isinstance(source_points, tuple) and isinstance(target_points, tuple):
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
            self.X_s = self._clone_process(source_points, non_blocking=True)
            self.Y_t = self._clone_process(target_points, non_blocking=True)

            # Calculate cost matrices
            if self.tensorise[0] and self.tensorise[1]:
                print(
                    "Tensorising from given mesh, assuming ij index and assuming mesh is regular"
                )  # need to think about
                # This means we're okay to tensorise
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
            if  self.cost_kwargs['cost_type']== 'rigid':
                self.pykeops_formulas = PyKeOpsFormulas(cost_string="SqDist(X, Y)")
            elif  self.cost_kwargs['cost_type'] == 'periodic':
                # ToDo : integers part here - use Fractionals it can be made for rationals, not reals.
                self.pykeops_formulas = PyKeOpsFormulas(cost_string=f"(Min(Concat(SqDist(Elem(X, 0) - IntCst({int(kwargs['L'])}), Elem(Y, 0)), Concat(SqDist(Elem(X, 0) + IntCst({int(kwargs['L'])}), Elem(Y, 0)), SqDist(Elem(X, 0), Elem(Y, 0)))) + SqDist(Elem(X, 1), Elem(Y, 1))))")
            elif self.cost_kwargs['cost_type']== 'beta_plane':
                # ToDo
                pass


    def _full_calculation_of_cost(self, cost_const):
        if self.cost_kwargs['cost_type'] == 'rigid':
            self.cost = 0.5 * cost_const * torch.cdist(self.X_s, self.Y_t) ** 2
        elif self.cost_kwargs['cost_type']== 'periodic':
            self.cost, self.pbcost_ind = pbcost_cdist(self.X_s, self.Y_t, self.cost_kwargs['L'])
            self.cost *= 0.5 * cost_const
        elif self.cost_kwargs['cost_type']== 'beta_plane':
            # We should have cos_const = self.f_constant**2
            self.cost = 0.5 * (self.f_constant + self.cost_kwargs['beta']*self.Y_t[:, 1].view(1, -1))**2 * torch.cdist(self.X_s, self.Y_t) ** 2

    
    def _generate_periodic_ind(self,):
        # ToDo move intto pykeops formulas
        if self.pykeops:
            L = self.cost_kwargs['L']
            X_s, Y_t = self.X_s.contiguous(), self.Y_t.contiguous()
            X_s_2, Y_t_2 = LazyTensor(X_s[:, None, :].contiguous()), LazyTensor(Y_t[None, :, :].contiguous())
            X_s_lt, Y_t_lt = LazyTensor(X_s[:, 0][:, None, None].contiguous()), LazyTensor(Y_t[:, 0][None, :, None].contiguous())

            cost_lt = LazyTensor.concatenate((
                -2 * L * (X_s_lt - Y_t_lt) + L**2,
                (X_s_lt - Y_t_lt - (X_s_lt - Y_t_lt)),  # hack to make it zero
                2 * L * (X_s_lt - Y_t_lt) + L**2
            ))

            self.pbcost_ind = cost_lt.argmin(dim=2)
        else:
            _, self.pbcost_ind = torch.min(torch.stack((
                -2*self.cost_kwargs['L']*(self.X_s[:, 0].view(-1, 1) - self.Y_t[:, 0].view(1, -1))+ self.cost_kwargs['L']**2,
                torch.zeros(len(self.X_s), len(self.Y_t)).type_as(self.X_s),
                2*self.cost_kwargs['L']*(self.X_s[:, 0].view(-1, 1) - self.Y_t[:, 0].view(1, -1)) + self.cost_kwargs['L']**2
                ), dim=0), dim=0) # .type_as(self.X_s)


    def _cost_update(self, cost_const):
        # Update cost matrix
        if self.pykeops:
            pass
        else:
            self._full_calculation_of_cost(cost_const)
            