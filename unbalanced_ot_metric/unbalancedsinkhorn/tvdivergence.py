"""
Total variation divergence 
"""
import torch

# pylint: disable=no-member
class TVDivergence:
    """
    Total variation divergence
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
        return "tv"

    def primal_cost(self, π_k, μ):

        return torch.sum(torch.abs(π_k.squeeze() - μ.squeeze()))

    def dual_cost(self, λ):
        return torch.max(λ, -self.ρ)

    def aprox_function(self, λ):
        torch.clip(λ, -self.ρ, self.ρ, out=λ)

    def sinkhorn_iterate(self, λ):
        self.update_function()
        # λ *= -1, doesn't change the calculation
        self.aprox_function(λ)
        # λ *= -1


# pylint: enable=no-member
