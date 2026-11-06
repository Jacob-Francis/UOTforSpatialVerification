"""
`DebiasedUOT`: the unbalanced Sinkhorn divergence.

Entropic OT is biased: OT_eps(alpha, alpha) is not 0. The Sinkhorn divergence
removes that bias by subtracting the two "self" problems:

    S_eps(alpha, beta) = OT_eps(alpha, beta)
                         - 1/2 OT_eps(alpha, alpha) - 1/2 OT_eps(beta, beta)
                         + eps/2 (m(alpha) - m(beta))^2

where m(.) is total mass. The last term is the mass correction for the unbalanced
case (Séjourné et al., "Sinkhorn divergences for unbalanced optimal transport").
S_eps(alpha, alpha) = 0, and S_eps is what the verification scores use.

NOTE: numpy, pytest and ThreadPoolExecutor are imported below but not used.
Importing pytest here makes pytest a runtime dependency of the package (it is
also listed in setup.py's install_requires). Stage 2 clean-up.

NOTE: `from unbalancedsinkhorn import UnbalancedOT` imports this package by its
installed name, from inside itself. It works because the package __init__ imports
unbalancedot before this module; `from .unbalancedot import UnbalancedOT` would be
the usual form.
"""
import numpy as np
import torch
from unbalancedsinkhorn import UnbalancedOT
import pytest
from concurrent.futures import ThreadPoolExecutor


class DebiasedUOT(UnbalancedOT):
    """
    Class to calculate debiased costs. Assign

    Use like UnbalancedOT (parameters -> densities -> sinkhorn_algorithm), then call
    `sinkhorn_divergence`, which solves the two self problems and combines them.

    Attributes set by sinkhorn_divergence
    -------------------------------------
    debias_f : UnbalancedOT
        The solved source self problem OT(alpha, alpha).
    debias_g : UnbalancedOT
        The solved target self problem OT(beta, beta).
    """

    def sinkhorn_divergence(self, tol=1e-12, force_type=None, return_type='primal', convergence_repeats=5):
        """
        Solve the two self problems and return the Sinkhorn divergence.

        Call `sinkhorn_algorithm` on this object first: the main problem's
        potentials and divergences are used as they are.

        Parameters
        ----------
        tol : float, optional
            Tolerance for the two self problems, by default 1e-12.
        force_type : str, optional
            Back-end for the energies; see `UnbalancedOT` (e.g. "pykeops").
        return_type : {'primal', 'breakdown', 'dual', None}, optional
            - 'primal'   : S_eps from the primal energies (default)
            - 'breakdown': (S_eps, the four primal terms of the main problem)
            - 'dual'     : S_eps from the dual energies
            - None       : solve the self problems only, return None
        convergence_repeats : int, optional
            Passed to the self problems' sinkhorn_algorithm, by default 5.

        Returns
        -------
        torch.Tensor, tuple or None
            See return_type.
        """
        # NOTE: an unknown return_type (e.g. a typo like 'primall') falls through
        # every branch and silently returns None (checked).

        # ToDo This doesn't need repeating everytime if it already exists? Especially for time stepping
        # ToDo check theory for symmetric updates in non-balanced setting
        # NOTE (efficiency, stage 5): both self problems are rebuilt and re-solved
        # on every call, even with unchanged inputs (checked: about 26 s again on a
        # repeat call, N = 900, dense). They are also solved with the two-sided
        # update (debias=False). A correct symmetric update f <- (f + aprox(T f)) / 2
        # gave the same energies to 9 digits in tests and needed 33 iterations
        # where the two-sided update needed up to 5444.
        # (cuda_device=str(...) because a torch.device object would fall back to
        # cuda:0; see TorchNumpyProcessing.)
        self.debias_f = UnbalancedOT(
            set_fail=self.set_fail, pykeops=self.pykeops, debias=False,  cuda_device=str(self.device)
        )
        self.debias_g = UnbalancedOT(
            set_fail=self.set_fail, pykeops=self.pykeops, debias=False, cuda_device=str(self.device)
        )

        self.debias_f.parameters(self.epsilon, self.rho, self.cost_const)
        self.debias_g.parameters(self.epsilon, self.rho, self.cost_const)

        # BUG(stage 4): self.cost_kwargs is not passed on, so the self problems
        # always use the rigid cost. With a periodic main problem, S(alpha, alpha)
        # came out as -0.0052 instead of 0 (checked).
        self.debias_f.densities(self.X_s, self.X_s, self.α_s, self.α_s)
        self.debias_g.densities(self.Y_t, self.Y_t, self.β_t, self.β_t)

        # NOTE: print_type() turns the divergence back into its name, so a custom
        # divergence object without print_type() would fail here.
        f_update, g_update, i_sup = self.debias_f.sinkhorn_algorithm(
            tol=tol,
            verbose=False,
            left_divergence=self.left_div.print_type(),
            right_divergence=self.left_div.print_type(),
            convergence_repeats=convergence_repeats
        )
        if (f_update > tol) or (g_update > tol):
            print('Symmetric problem 1 not converged, psuedo residal: {:.4g}'.format(max(f_update, g_update)), 'You may wish to increase convergence_repeats')

        f_update, g_update, i_sup = self.debias_g.sinkhorn_algorithm(
            tol=tol,
            verbose=False,
            left_divergence=self.right_div.print_type(),
            right_divergence=self.right_div.print_type(),
            convergence_repeats=convergence_repeats
        )
        # (This message says "problem 1" but is about problem 2, the target.)
        if (f_update > tol) or (g_update > tol):
            print('Symmetric problem 1 not converged, psuedo residal: {:.4g}'.format(max(f_update, g_update)), 'You may wish to increase convergence_repeats')


        # S = OT(a, b) - (OT(a, a) + OT(b, b)) / 2 + eps/2 (m(a) - m(b))^2
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
