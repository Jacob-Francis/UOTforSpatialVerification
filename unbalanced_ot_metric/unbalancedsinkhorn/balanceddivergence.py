"""
Balanced divergence class - technically just here to have desired form for application though, isn't
testing strictly for balance.
"""

import torch


class BalancedDivergence:
    """
    Balanced divergence class - technically just here to have desired form for application though, isn't
    testing strictly for balance.
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
        return "balanced"

    def primal_cost(self, π_k, μ):
        return 0

    def dual_cost(self, λ):
        return λ

    def aprox_function(self, λ):
        pass

    def sinkhorn_iterate(self, λ):
        self.update_function()
        # λ *= -1, doesn't change the calculation
        # self.aprox_function(λ)
        # λ *= -1
