"""
Small helper functions for the unbalanced OT solver.

Used by the package:
- `_kl`                  : discrete KL divergence, for the entropic term of the primal cost.
- `pbcost_cdist`         : periodic (in x) squared-distance cost matrix.
- `exponential_sequence` : epsilon schedule for epsilon annealing.

NOTE: the other functions in this file are not used anywhere in the package or the
tests (checked with grep). They are candidates for removal in stage 2.
"""
import torch
from torch.special import xlogy
import numpy as np

def _kl(π, R):
    """Calculate discrete KL divergence

        KL(π | R) = sum π log(π / R) - π + R

    Used by `UnbalancedOT.primal_cost` for the entropic term eps * KL(pi | alpha x beta).
    Same formula as `KLDivergence.primal_cost`.

    Parameters
    ----------
    π : (n,m), torch.Tensor
        The input measure.

    R : (n,m), torch.Tensor
        The reference measure.

    Returns
    -------
    float
        The kl-divergence.
    """
    # Should deal with 0log(0/0)
    # Where R has no mass, the π log(π/R) term is left out (avoids log(0)), but
    # -π + R is still summed over every entry. See KLDivergence.primal_cost.
    if 0 in R:
        return torch.sum(
            xlogy(
                π.squeeze()[R.squeeze() > 1e-20],
                π.squeeze()[R.squeeze() > 1e-20] / R.squeeze()[R.squeeze() > 1e-20],
            )
        ) + torch.sum(R.squeeze() - π.squeeze())
    else:
        return torch.sum(
            xlogy(π.squeeze(), π.squeeze() / R.squeeze()) - π.squeeze() + R.squeeze()
        )


def _kl_divergence_fl_transform(x, rho: int):
    """KL conjugate rho * (exp(x / rho) - 1). Unused; see KLDivergence.dual_cost."""
    return rho * (torch.exp(x / rho) - 1)


def _tv_divergence_fl_transform(x, rho: int):
    """TV conjugate max(x, -rho). Unused; see TVDivergence.dual_cost."""
    return torch.max(x, -rho)


def _tv(π_0, μ):
    """Total variation sum |π_0 - μ|. Unused; see TVDivergence.primal_cost."""
    return torch.sum(torch.abs(π_0.squeeze() - μ.squeeze()))


@torch.jit.script
def _complied_outer_3d(output, a, b, m1: int, m2: int):
    """
    Unused. Fill output[:, :, r, s] = a * b[r, s] with explicit loops.

    NOTE: the @torch.jit.script decorator compiles this when the module is
    imported. torch 2.15 prints "torch.jit.script is deprecated" for it.
    """
    for r in range(m1):
        for s in range(m2):
            output[:, :, r, s] = a * b[r, s]


@torch.jit.script
def _complied_tensor_cost(output, a, b, n1: int, n2: int, m1: int):
    """Unused. Fill output[k, p, r, :] = a[k, r] * b[p, :] with explicit loops."""
    for k in range(n1):
        for p in range(n2):
            for r in range(m1):
                output[k, p, r, :] = a[k, r] * b[p, :]


def _aprox_kl(p, rho: int, epsilon: int):
    """KL aprox step rho / (rho + eps) * p. Unused; see KLDivergence.aprox_function."""
    return rho * p / (rho + epsilon)


def _aprox_tv_clamp(p, rho: int):
    """TV aprox step clamp(p, -rho, rho). Unused; see TVDivergence.aprox_function."""
    return torch.clip(p, -rho, rho)

def pbcost_cdist(X,Y,L):
    """Calculate the periodic cost in the first coordinate given stack coordinates. i.e.
    shapes (N,2), not tensorised shapes (n,m,2). This is for squared eulcidean distance with no extra terms.
    i.e. ||x-y||^2, no half etc.

    The domain wraps around in x with period L (like longitude), but not in y:

        cost[i, j] = min over s in {-L, 0, +L} of (x_i + s - x'_j)^2  +  (y_i - y'_j)^2

    Used by `CostClass._full_calculation_of_cost` (the dense path), which then
    multiplies by 0.5 * cost_const. Assumes all x-coordinates lie within one period,
    so a shift of at most one L is enough.

    Parameters
    ----------
    X : array
        (N,2)
    Y : array
        (M,2)
    L : float
        boundary length

    Returns
    -------
    cost, ind
        minimal cost, without extra constants. The index of these costs with [0:no shift, 1:positive shift, 2: negative shift]

        CORRECTION (checked): the index actually means
            0 : x shifted by -L   (x - L was closest)
            1 : no shift
            2 : x shifted by +L   (x + L was closest)
        so (ind - 1) * L is the shift that was applied to x. Both are (N, M);
        ind is returned as floats (same dtype as X).
    """
    # Three copies of the source points: x - L, x, x + L.
    # (tile makes a new tensor, so the caller's X is not modified.)
    X_temp = X.view(1, -1, 2).tile(dims=(3, 1, 1))
    X_temp[0, :, 0] -= L
    X_temp[2, :, 0] += L
    # Squared x-distances for each shift: (3, N, M). Take the smallest over shifts.
    val1, ind1 = torch.min(torch.cdist(X_temp[:, :, 0].view(3, -1, 1),Y[:, 0].view(1, -1, 1))**2, dim=0)

    # X_temp = X.view(1, -1, 2).tile(dims=(3, 1, 1))
    # X_temp[0, :, 0] -= L
    # X_temp[2, :, 0] += L

    # dist = torch.cdist(X, Y)**2
    # n, m = dist.shape
    # Z = torch.zeros((n, m, 3)).type_as(X)
    # Z[:, :, 1] = dist

    # X1 = X.clone()
    # X1[:, 0] -= L
    # Z[:, :, 0] = torch.cdist(X1, Y)**2

    # X1 = X.clone()
    # X1[:, 0] += L
    # Z[:, :, 2] = torch.cdist(X1, Y)**2

    # # Gradient update (tau)
    # vall, temp = torch.min(Z, dim=2)
    # temp = temp.type_as(X)

    # print(torch.norm(temp-ind1, p=1))
    # print('error', torch.norm(val1+torch.cdist(X[:, 1].view(-1, 1),Y[:, 1].view(-1, 1))**2 - vall))

    # assert (temp == ind1.type_as(X)).all()
    # print('error', torch.norm(val1+torch.cdist(X[:, 1].view(-1, 1),Y[:, 1].view(-1, 1))**2 - vall))
    # assert (val1+torch.cdist(X[:, 1].view(-1, 1),Y[:, 1].view(-1, 1))**2 == vall).all()

    # Add the (non-periodic) squared y-distances.
    return val1+torch.cdist(X[:, 1].view(-1, 1),Y[:, 1].view(-1, 1))**2, ind1.type_as(X)

# THIS IS ALMOST THERE FOR THE TENORISED CASE AS WE JUST NEED THE EXTRA TERM. CHECK DIMS
def pbcost_extraterm(X,Y,L):
    """
    Unused, unfinished alternative to `pbcost_cdist`.

    Uses (x - y + s)^2 = (x - y)^2 + 2 s (x - y) + s^2: the plain cost plus the
    smallest extra term over the shifts s in {0, +L, -L}.

    NOTE: `torch.zeros(len(X), len(Y))` is float32 on the CPU, so this fails for
    float64 or GPU inputs. Its index order (0: none, 1: +L, 2: -L) also differs
    from `pbcost_cdist`.
    """
    val, ind = torch.min(torch.stack((
        torch.zeros(len(X), len(Y)),
        2*L*(X[:, 0].view(-1, 1) - Y[:, 0].view(1, -1)) + L**2,
        -2*L*(X[:, 0].view(-1, 1) - Y[:, 0].view(1, -1))+ L**2
        ), dim=0), dim=0)
    return torch.cdist(X, Y)**2+val, ind

# def tensor_outer_product(a, b):
#     n1, n2 = a.shape
#     m1, m2 = b.shape
#     outer_prod = torch.zeros(n1, n2, m1, m2)
#     for r in range(m1):
#         for s in range(m2):
#             output[:, :, r, s] = a*b[r, s]


def exponential_sequence(x, a, b):
    """
    Generate an exponential sequence starting at `x`, scaling by `a` each step, until the value exceeds `b`.

    More precisely: x, x*a, x*a^2, ... while the next value is still >= b. With
    a < 1 this is a DECREASING sequence that stops just above b.

    Used for epsilon annealing in `UnbalancedSinkhorn.sinkhorn_algorithm`, on the
    sqrt(epsilon) scale: from sqrt(2) down to sqrt(epsilon), with a = 0.99 by
    default. That is a long schedule, e.g. 264 values for epsilon = 0.01 (checked).

    BUG(stage 4): if a >= 1 (and x >= b) the values never drop below b, so the
    loop never ends (checked: still running after 3 s with a = 1).

    Parameters:
    x (float): Start value
    a (float): Scaling factor
    b (float): End value

    Returns:
    torch.Tensor: Exponentially spaced numbers
    """
    result = [x]
    while result[-1] * a >= b:
        result.append(result[-1] * a)
    return torch.tensor(result)

def sqrt_sequence(x=0.9, b=0.005, t0=0.1, max_steps=100):
    """
    Generate a sequence that follows a t^0.5 growth, starting at `x`, until the value exceeds `b`.

    Unused. NOTE: it prints `t` on every step, and despite the description it
    returns x * t with t <- sqrt(t) each step, i.e. x*t0, x*t0^(1/2), x*t0^(1/4), ...

    Parameters:
    x (float): Start value
    b (float): End value

    Returns:
    torch.Tensor: Sequence of numbers growing as t^0.5
    """
    result = [x]
    t = t0  # Starting time step
    count = 0
    while (result[-1] * t >= b) and count < max_steps:
        next_value = x * t
        result.append(next_value)
        t = t ** 0.5
        print(t)
        count += 1
    return torch.tensor(result)


if __name__ == "__main__":
    import torch

   # Example usage
    x = 0.9        # Start value
    a = 0.9        # Scaling factor
    b = 0.005    # End value

    sequence = exponential_sequence(x, a, b)
    print(sequence)

    sequence = sqrt_sequence(x, b)
    print(sequence)
