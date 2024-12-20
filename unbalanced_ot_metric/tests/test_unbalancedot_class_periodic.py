import numpy as np
import torch
from unbalancedsinkhorn import UnbalancedOT
import pytest

# testing balanced case, to check our basic sinkhorn loop works, this should
# be the case for equal grids. Further test the classes ability to take both
# numpy and torch arrays/tensors.

# Testing inputs type, not tensorised, pykeops=False
@pytest.mark.parametrize(
    "n, m, t_n", [(10, 10, "torch"), (21, 19, "torch"), (10, 10, "np"), (31, 32, "np")]
)
def test_periodic_marginals_energy_cpu(n, m, t_n):
    if t_n == "torch":
        X = torch.cartesian_prod(torch.linspace(0, 1, n), torch.linspace(0, 1, n))
        Y = torch.cartesian_prod(torch.linspace(0, 1, m), torch.linspace(0, 1, m))
    else:
        X1, X2 = np.meshgrid(
            torch.linspace(0, 1, n), torch.linspace(0, 1, n), indexing="ij"
        )
        Y1, Y2 = np.meshgrid(
            torch.linspace(0, 1, m), torch.linspace(0, 1, m), indexing="ij"
        )
        X = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
        Y = np.hstack((Y1.reshape(-1, 1), Y2.reshape(-1, 1)))
    ε = 10 / np.sqrt(0.5 * (n**2 + m**2))
    Nsteps = 1000
    tol = 1e-15
    testing = UnbalancedOT(set_fail=True, pykeops=False)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y, cost_type='periodic', L=1.0)
    # print('here', testing.cost_1)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced")

    assert np.sum(testing.marginals_error()) < 1e-10
    assert testing.duality_gap() < 1e-10


# Testing inputs type, not tensorised, pykeops=False
@pytest.mark.parametrize(
    "n, m, t_n", [(10, 10, "torch"), (21, 19, "torch"), (10, 10, "np"), (31, 32, "np")]
)
def test_periodic_marginals_energy_pykeops_flat(n, m, t_n):
    if t_n == "torch":
        X = torch.cartesian_prod(torch.linspace(0, 1, n), torch.linspace(0, 1, n))
        Y = torch.cartesian_prod(torch.linspace(0, 1, m), torch.linspace(0, 1, m))
    else:
        X1, X2 = np.meshgrid(
            torch.linspace(0, 1, n), torch.linspace(0, 1, n), indexing="ij"
        )
        Y1, Y2 = np.meshgrid(
            torch.linspace(0, 1, m), torch.linspace(0, 1, m), indexing="ij"
        )
        X = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
        Y = np.hstack((Y1.reshape(-1, 1), Y2.reshape(-1, 1)))
    ε = 1 / np.sqrt(0.5 * (n**2 + m**2))
    Nsteps = 1000
    tol = 1e-15
    testing = UnbalancedOT(set_fail=True, pykeops=True)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y, cost_type='periodic', L=1.0)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced")

    assert np.sum(testing.marginals_error()) < 1e-10
    assert testing.duality_gap() < 1e-10


# Testing tensorised but not tuple input, with and wihtout pykeops
@pytest.mark.parametrize(
    "n, m, keops", [(10, 10, False), (20, 20, False), (21, 19, False), (21, 19, True)]
)
def test_periodic_marginals_tensor(n, m, keops):
    X1, X2 = np.meshgrid(
        torch.linspace(0, 1, n), torch.linspace(0, 1, n), indexing="ij"
    )
    Y1, Y2 = np.meshgrid(
        torch.linspace(0, 1, m), torch.linspace(0, 1, m), indexing="ij"
    )
    X = np.stack((X1, X2), axis=2)
    Y = np.stack((Y1, Y2), axis=2)

    ε = 10 / np.sqrt(0.5 * (n**2 + m**2))
    Nsteps = 1000
    tol = 1e-15
    testing = UnbalancedOT(set_fail=True, pykeops=keops)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y, cost_type='periodic', L=1.0)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced")

    assert np.sum(testing.marginals_error()) < 1e-10
    assert testing.duality_gap() < 1e-10


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
