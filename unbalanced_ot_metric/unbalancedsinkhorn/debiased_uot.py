import numpy as np
import torch
from unbalancedsinkhorn import UnbalancedOT
import pytest
from concurrent.futures import ThreadPoolExecutor


class DebiasedUOT(UnbalancedOT):
    """
    Class to calculate debiased costs. Assign
    """

    def sinkhorn_divergence(self, tol=1e-12, force_type=None, return_type='primal'):

        # ToDo This doesn't need repeating everytime if it already exists? Especially for time stepping
        # ToDo check theory for symmetric updates in non-balanced setting
        self.debias_f = UnbalancedOT(
            set_fail=self.set_fail, pykeops=self.pykeops, debias=False,  cuda_device=self.device
        )
        self.debias_g = UnbalancedOT(
            set_fail=self.set_fail, pykeops=self.pykeops, debias=False, cuda_device=self.device
        )

        self.debias_f.parameters(self.epsilon, self.rho, self.cost_const)
        self.debias_g.parameters(self.epsilon, self.rho, self.cost_const)

        self.debias_f.densities(self.X_s, self.X_s, self.α_s, self.α_s)
        self.debias_g.densities(self.Y_t, self.Y_t, self.β_t, self.β_t)

        f_update, g_update, i_sup = self.debias_f.sinkhorn_algorithm(
            tol=tol,
            verbose=False,
            left_divergence=self.left_div.print_type(),
            right_divergence=self.left_div.print_type(),
            convergence_repeats=3
        )
        if (f_update > 1e-14) or (g_update > 1e-14):
            print('f debias', f_update < 1e-14, g_update < 1e-14)

        f_update, g_update, i_sup = self.debias_g.sinkhorn_algorithm(
            tol=tol,
            verbose=False,
            left_divergence=self.right_div.print_type(),
            right_divergence=self.right_div.print_type(),
            convergence_repeats=3
        )
        if (f_update > 1e-14) or (g_update > 1e-14):
            print('g debias?', f_update < 1e-14, g_update < 1e-14)


        if return_type=='primal':
            return (
                sum(self.primal_cost(force_type=force_type))
                - (
                    sum(self.debias_f.primal_cost(force_type=force_type))
                    + sum(self.debias_g.primal_cost(force_type=force_type))
                )
                / 2
                + self.epsilon * (self.α_s.sum() - self.β_t.sum()) ** 2 / 2
            )
        elif return_type=='breakdown':
            p = self.primal_cost(force_type=force_type)
            s = (
                sum(p)
                - (
                    sum(self.debias_f.primal_cost(force_type=force_type))
                    + sum(self.debias_g.primal_cost(force_type=force_type))
                )
                / 2
                + self.epsilon * (self.α_s.sum() - self.β_t.sum()) ** 2 / 2
            )
            return s, p
        elif return_type=='dual':
            return (
                sum(self.dual_cost(force_type=force_type))
                - (
                    sum(self.debias_f.dual_cost(force_type=force_type))
                    + sum(self.debias_g.dual_cost(force_type=force_type))
                )
                / 2
                + self.epsilon * (self.α_s.sum() - self.β_t.sum()) ** 2 / 2
            )
        elif return_type==None:
            return None

    # def process_debiased_costs(
    #     self,
    #     cost_const,
    #     n1,
    #     n2,
    #     m1,
    #     m2,
    # ):

    #     # pylint: disable=attribute-defined-outside-init
    #     if isinstance(self.X_s, tuple) and isinstance(self.Y_t, tuple):
    #         self.cost_f1 = (
    #             0.5
    #             * cost_const
    #             * torch.cdist(
    #                 self._clone_process(self.X_s[0], non_blocking=True).view(-1, 1),
    #                 self._clone_process(self.X_s[0], non_blocking=True).view(-1, 1),
    #             )
    #             ** 2
    #         )
    #         self.cost_f2 = (
    #             0.5
    #             * cost_const
    #             * torch.cdist(
    #                 self._clone_process(self.X_s[1], non_blocking=True).view(-1, 1),
    #                 self._clone_process(self.X_s[1], non_blocking=True).view(-1, 1),
    #             )
    #             ** 2
    #         )

    #         self.cost_g1 = (
    #             0.5
    #             * cost_const
    #             * torch.cdist(
    #                 self._clone_process(self.Y_t[0], non_blocking=True).view(-1, 1),
    #                 self._clone_process(self.Y_t[0], non_blocking=True).view(-1, 1),
    #             )
    #             ** 2
    #         )
    #         self.cost_g2 = (
    #             0.5
    #             * cost_const
    #             * torch.cdist(
    #                 self._clone_process(self.Y_t[1], non_blocking=True).view(-1, 1),
    #                 self._clone_process(self.Y_t[1], non_blocking=True).view(-1, 1),
    #             )
    #             ** 2
    #         )
    #     else:

    #         # Calculate cost matrices
    #         if self.tensorise[0] and self.tensorise[1]:
    #             print(
    #                 "Tensorising from given mesh, assuming ij index and assuming mesh is regular"
    #             )  # need to think about
    #             # This means we're okay to tensorise
    #             self.cost_f1 = (
    #                 0.5
    #                 * cost_const
    #                 * torch.cdist(self.X_s[:n1, 0], self.X_s[:n1, 0]) ** 2
    #             )
    #             self.cost_f2 = (
    #                 0.5
    #                 * cost_const
    #                 * torch.cdist(self.X_s[0, :n2], self.X_s[0, :n2]) ** 2
    #             )
    #             self.cost_g1 = (
    #                 0.5
    #                 * cost_const
    #                 * torch.cdist(self.Y_t[:m1, 0], self.Y_t[:m1, 0]) ** 2
    #             )
    #             self.cost_g2 = (
    #                 0.5
    #                 * cost_const
    #                 * torch.cdist(self.Y_t[0, :m2], self.Y_t[0, :m2]) ** 2
    #             )
    #         elif self.pykeops == True:
    #             # Run cost calculation on the fly, not creating a large cost matrix
    #             pass
    #         else:
    #             self.cost_f = 0.5 * cost_const * torch.cdist(self.X_s, self.X_s) ** 2
    #             self.cost_g = 0.5 * cost_const * torch.cdist(self.Y_t, self.Y_t) ** 2
