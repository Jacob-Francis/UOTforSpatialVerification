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
        if (f_update > tol) or (g_update > tol):
            print('Symmetric problem 1 converged:', f_update < tol, g_update < tol)

        f_update, g_update, i_sup = self.debias_g.sinkhorn_algorithm(
            tol=tol,
            verbose=False,
            left_divergence=self.right_div.print_type(),
            right_divergence=self.right_div.print_type(),
            convergence_repeats=3
        )
        if (f_update > tol) or (g_update > tol):
            print('Symmetric problem 2 converged:', f_update < tol, g_update < tol)


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
