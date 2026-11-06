"""
Total variation divergence

Marginal penalty  rho * TV(pi_0, alpha),  where

    TV(p, q) = sum_i |p_i - q_i| .

Same interface as `KLDivergence` (see the module docstring in kldivergence.py).
"""
import torch

# pylint: disable=no-member
class TVDivergence:
    """
    Total variation divergence

    Parameters
    ----------
    rho : torch.Tensor
        Strength of the marginal penalty. Also the clamp value for the potentials:
        they are kept inside [-rho, rho].
    epsilon : torch.Tensor
        Entropic regularisation. Stored for a common interface; TV does not use it.
    update_function : callable, optional
        Zero-argument log-sum-exp half-step, built by
        `UnbalancedSinkhorn._process_divergences`.
    """

    def __init__(self, rho, epsilon, update_function=None) -> None:
        """
        rho is the divergence regularisation parameter
        epsilon is the blur/heat/emperature entropic regularisation parameter
        """

        self.ρ = rho
        self.ε = epsilon
        self.update_function = update_function

    def print_type(
        self,
    ):
        """Return the name "tv", which `sinkhorn_algorithm(aprox=...)` accepts."""
        return "tv"

    def primal_cost(self, π_k, μ):
        """
        TV(π_k, μ) = sum |π_k - μ| (not yet multiplied by rho).

        Parameters
        ----------
        π_k : torch.Tensor
            Marginal of the transport plan.
        μ : torch.Tensor
            Reference measure (the input density).

        Returns
        -------
        torch.Tensor
            Scalar.
        """

        return torch.sum(torch.abs(π_k.squeeze() - μ.squeeze()))

    def dual_cost(self, λ):
        """
        Convex conjugate of rho * TV( . , mu), per point:

            D*(λ) = max(λ, -rho)     valid for λ <= rho

        (it is +inf for λ > rho). The clamp in `aprox_function` keeps the
        potentials in [-rho, rho], so the finite formula always applies.

        Parameters
        ----------
        λ : torch.Tensor
            Dual variable, e.g. -f.

        Returns
        -------
        torch.Tensor
            Same shape as λ.
        """
        return torch.max(λ, -self.ρ)

    def aprox_function(self, λ):
        """
        The TV "aprox" step, IN PLACE: clamp λ to [-rho, rho].

        Parameters
        ----------
        λ : torch.Tensor
            Potential to clamp. Modified in place.
        """
        torch.clip(λ, -self.ρ, self.ρ, out=λ)

    def sinkhorn_iterate(self, λ):
        """
        One half-step of Sinkhorn: log-sum-exp update, then clamp, both in place.
        See `KLDivergence.sinkhorn_iterate`; `λ` must be the tensor the update
        writes into.

        Parameters
        ----------
        λ : torch.Tensor
            The potential being updated. Modified in place.
        """
        self.update_function()
        # λ *= -1, doesn't change the calculation
        self.aprox_function(λ)
        # λ *= -1


# pylint: enable=no-member
