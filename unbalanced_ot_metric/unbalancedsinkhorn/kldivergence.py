"""
    Kullback Leiber divergence class for UOT class

Marginal penalty  rho * KL(pi_0 | alpha),  where

    KL(p | q) = sum_i  p_i log(p_i / q_i) - p_i + q_i .

Every divergence class plugs into `UnbalancedSinkhorn` through the same small
interface (see `UnbalancedSinkhorn._process_divergences`):

- primal_cost(pi_marginal, reference) : the penalty D(pi_k | mu). The solver multiplies
                                        it by rho itself.
- dual_cost(lambda)                   : the conjugate D*(lambda), used in the dual energy.
- sinkhorn_iterate(potential)         : one half-step of Sinkhorn for this side.
- print_type()                        : name string, used by DebiasedUOT to rebuild it.
"""
import torch
from torch.special import xlogy

# pylint: disable=no-member
class KLDivergence:
    """
    Kullback Leiber divergence class for UOT class

    Parameters
    ----------
    rho : torch.Tensor
        Strength of the marginal penalty (geomloss: reach^p = rho). Larger rho
        means mass creation/destruction costs more, so the problem is closer to
        balanced OT.
    epsilon : torch.Tensor
        Entropic regularisation (geomloss: blur^p = epsilon).
    update_function : callable, optional
        Zero-argument function that runs the log-sum-exp half-step and writes the
        result into the potential in place. Built by
        `UnbalancedSinkhorn._process_divergences`.
    """

    def __init__(self, rho, epsilon, update_function=None) -> None:
        """
        rho is the divergence regularisation parameter
        epsilon is the blur/heat/temperature entropic regularisation parameter
        """
        self.ρ = rho
        # NOTE: this keeps the epsilon object given at construction. During
        # epsilon annealing the solver rebinds its own self.epsilon to a new
        # tensor, so aprox_function keeps using the final (target) epsilon.
        self.ε = epsilon
        self.update_function = update_function

    def primal_cost(self, π, R):
        """
        KL(π | R) = sum π log(π / R) - π + R.

        Parameters
        ----------
        π : torch.Tensor
            Marginal of the transport plan, e.g. pi_0, any shape.
        R : torch.Tensor
            Reference measure (the input density), same number of entries as π.

        Returns
        -------
        torch.Tensor
            Scalar KL divergence (not yet multiplied by rho).
        """
        if 0 in R:
            # Where the reference has no mass, leave out the π log(π/R) term to
            # avoid log(0) / division by zero, but keep -π + R for all entries.
            # (The convention 0 log 0 = 0 is handled by xlogy.)
            # NOTE: if π > 0 where R = 0, the true KL is +inf; this returns a
            # finite value instead.

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
        """Return the name "kl", which `sinkhorn_algorithm(aprox=...)` accepts."""
        return "kl"

    def dual_cost(self, λ):
        """
        Convex conjugate of rho * KL( . | mu), per point:

            D*(λ) = rho * (exp(λ / rho) - 1)

        `UnbalancedOT.dual_cost` uses it as  -<alpha, D*(-f)>.

        Parameters
        ----------
        λ : torch.Tensor
            Dual variable, e.g. -f.

        Returns
        -------
        torch.Tensor
            Same shape as λ.
        """
        return self.ρ * (torch.exp(λ / self.ρ) - 1)

    def aprox_function(self, λ):
        """
        The KL "aprox" step, applied IN PLACE:

            λ <- rho / (rho + epsilon) * λ

        This turns the balanced log-sum-exp update into the unbalanced one
        (Séjourné et al., "Sinkhorn divergences for unbalanced optimal transport").
        As rho -> inf the factor -> 1 (balanced OT); as rho -> 0 it -> 0.

        Parameters
        ----------
        λ : torch.Tensor
            Potential to scale. Modified in place.
        """
        λ *= self.ρ / (self.ρ + self.ε)

    def sinkhorn_iterate(self, λ):
        """
        One half-step of Sinkhorn for this side.

        1. `update_function()` writes the log-sum-exp (softmin) result into the
           potential in place.
        2. `aprox_function` scales that same tensor.

        So `λ` must be the very tensor that `update_function` writes into
        (self.f or self.g of the solver).

        Parameters
        ----------
        λ : torch.Tensor
            The potential being updated. Modified in place.
        """
        # BUG(stage 4): with debias=True the solver's update_function returns the
        # AVERAGE (f + T(f)) / 2, and this then scales the whole average. The
        # correct symmetric step is (f + aprox(T(f))) / 2. With KL this converges
        # to the wrong potential (checked: primal 0.017892 instead of 0.017775).
        # TV and balanced are not affected.
        self.update_function()
        # λ *= -1, doesn't change the calculation
        self.aprox_function(λ)
        # λ *= -1


# pylint: enable=no-member

if __name__ == "__main__":
    pass
