import torch
from torch.special import xlogy
import numpy as np

def _kl(π, R):
    """Calculate discrete KL divergence

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
    return rho * (torch.exp(x / rho) - 1)


def _tv_divergence_fl_transform(x, rho: int):
    return torch.max(x, -rho)


def _tv(π_0, μ):
    return torch.sum(torch.abs(π_0.squeeze() - μ.squeeze()))


@torch.jit.script
def _complied_outer_3d(output, a, b, m1: int, m2: int):
    for r in range(m1):
        for s in range(m2):
            output[:, :, r, s] = a * b[r, s]


@torch.jit.script
def _complied_tensor_cost(output, a, b, n1: int, n2: int, m1: int):
    for k in range(n1):
        for p in range(n2):
            for r in range(m1):
                output[k, p, r, :] = a[k, r] * b[p, :]


def _aprox_kl(p, rho: int, epsilon: int):
    return rho * p / (rho + epsilon)


def _aprox_tv_clamp(p, rho: int):
    return torch.clip(p, -rho, rho)

def pbcost_cdist(X,Y,L):
    """Calculate the periodic cost in the first coordinate given stack coordinates. i.e.
    shapes (N,2), not tensorised shapes (n,m,2). This is for squared eulcidean distance with no extra terms.
    i.e. ||x-y||^2, no half etc. 

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
    """
    X_temp = X.view(1, -1, 2).tile(dims=(3, 1, 1))
    X_temp[0, :, 0] -= L
    X_temp[2, :, 0] += L
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
    
    return val1+torch.cdist(X[:, 1].view(-1, 1),Y[:, 1].view(-1, 1))**2, ind1.type_as(X)

# THIS IS ALMOST THERE FOR THE TENORISED CASE AS WE JUST NEED THE EXTRA TERM. CHECK DIMS
def pbcost_extraterm(X,Y,L):
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
