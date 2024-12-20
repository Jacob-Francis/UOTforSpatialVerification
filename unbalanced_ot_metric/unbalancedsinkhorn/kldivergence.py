"""
    Kullback Leiber divergence class for UOT class  
    """
import torch
from torch.special import xlogy

# pylint: disable=no-member
class KLDivergence:
    """
    Kullback Leiber divergence class for UOT class
    """

    def __init__(self, rho, epsilon, update_function=None) -> None:
        """
        rho is the divergence regularisation parameter
        epsilon is the blur/heat/temperature entropic regularisation parameter
        """
        self.ρ = rho
        self.ε = epsilon
        self.update_function = update_function

    def primal_cost(self, π, R):
        if 0 in R:

            return torch.sum(
                # pylint: disable-next=not-callable
                xlogy(
                    π.squeeze()[R.squeeze() > 1e-20],
                    π.squeeze()[R.squeeze() > 1e-20] / R.squeeze()[R.squeeze() > 1e-20],
                )
            ) + torch.sum(R.squeeze() - π.squeeze())
        else:

            return torch.sum(
                # pylint: disable-next=not-callable
                xlogy(π.squeeze(), π.squeeze() / R.squeeze())
                - π.squeeze()
                + R.squeeze()
            )

    def print_type(
        self,
    ):
        return "kl"

    def dual_cost(self, λ):
        return self.ρ * (torch.exp(λ / self.ρ) - 1)

    def aprox_function(self, λ):
        λ *= self.ρ / (self.ρ + self.ε)

    def sinkhorn_iterate(self, λ):
        self.update_function()
        # λ *= -1, doesn't change the calculation
        self.aprox_function(λ)
        # λ *= -1


# pylint: enable=no-member

if __name__ == "__main__":
    pass
