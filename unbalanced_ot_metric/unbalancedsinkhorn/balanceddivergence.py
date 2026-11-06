"""
Balanced divergence class - technically just here to have desired form for application though, isn't
testing strictly for balance.

Balanced OT: the marginals of the plan must equal the inputs exactly
(pi_0 = alpha, pi_1 = beta). In the divergence framework that is the penalty
which is 0 when the marginals match and +inf otherwise, so:

- primal_cost is 0 (the constraint is assumed to hold at convergence),
- the conjugate is the identity, D*(λ) = λ,
- the aprox step does nothing, so each half-step is the plain Sinkhorn update.

Same interface as `KLDivergence` (see kldivergence.py). It does not check that
the two input masses are equal; with unequal masses balanced OT has no solution
and the iterations will not converge.
"""

import torch


class BalancedDivergence:
    """
    Balanced divergence class - technically just here to have desired form for application though, isn't
    testing strictly for balance.

    Parameters
    ----------
    rho : torch.Tensor
        Not used; stored for a common interface.
    epsilon : torch.Tensor
        Not used; stored for a common interface.
    update_function : callable, optional
        Zero-argument log-sum-exp half-step, built by
        `UnbalancedSinkhorn._process_divergences`.
    """

    def __init__(self, rho, epsilon, update_function=None) -> None:
        """
        rho is the divergence regularisation parameter
        epsilon is the blur/heat/temperature entropic regularisation parameter
        """
        self.ρ = rho
        self.ε = epsilon
        self.update_function = update_function

    def print_type(
        self,
    ):
        """Return the name "balanced", which `sinkhorn_algorithm(aprox=...)` accepts."""
        return "balanced"

    def primal_cost(self, π_k, μ):
        """Always 0: the marginal constraint is treated as satisfied."""
        return 0

    def dual_cost(self, λ):
        """Conjugate of the exact constraint: the identity, D*(λ) = λ."""
        return λ

    def aprox_function(self, λ):
        """No-op: balanced OT needs no extra step."""
        pass

    def sinkhorn_iterate(self, λ):
        """
        One half-step of Sinkhorn: just the log-sum-exp update (in place).

        Parameters
        ----------
        λ : torch.Tensor
            The potential being updated. Not used directly; `update_function`
            writes into it.
        """
        self.update_function()
        # λ *= -1, doesn't change the calculation
        # self.aprox_function(λ)
        # λ *= -1
