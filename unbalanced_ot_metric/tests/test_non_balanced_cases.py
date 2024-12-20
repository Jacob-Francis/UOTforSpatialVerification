import numpy as np
import torch
from unbalancedsinkhorn import UnbalancedOT
import pytest

# testing balanced case, to check our basic sinkhorn loop works, this should
# be the case for equal grids. Further test the classes ability to take both
# numpy and torch arrays/tensors.

# --------------------------------- KL ----------------------------------
# Testing inputs type, not tensorised, pykeops=False


@pytest.mark.parametrize(
    "n, m, t_n", [(10, 11, "torch"), (15, 14, "torch"), (10, 10, "np"), (13, 14, "np")]
)
def test_marginals_energy_cpu_kl(n, m, t_n):
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
    ε = np.min((0.9, 10 / np.sqrt(0.5 * (n**2 + m**2))))
    Nsteps = 10000
    tol = 1e-15
    testing = UnbalancedOT(set_fail=True, pykeops=False)
    testing.parameters(ε)

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="kl")
    assert testing.duality_gap() < 1e-10


# Testing inputs type, not tensorised, pykeops=False


@pytest.mark.parametrize(
    "n, m, t_n", [(10, 10, "torch"), (21, 19, "torch"), (10, 10, "np"), (25, 23, "np")]
)
def test_marginals_energy_pykeops_flat_kl(n, m, t_n):
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
    testing = UnbalancedOT(set_fail=True, pykeops=True)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="kl")
    assert testing.duality_gap() < 1e-10


# Testing tuple input with and without pykeops


@pytest.mark.parametrize(
    "n1, n2, m1, m2, keops",
    [
        (10, 10, 10, 10, False),
        (10, 11, 12, 13, True),
        (21, 21, 19, 19, False),
        (10, 11, 10, 11, True),
    ],
)
def test_marginals_tensor_tuple_kl(n1, n2, m1, m2, keops):
    # tuple input
    X = (torch.linspace(0, 1, n1), torch.linspace(0, 1, n2))
    Y = (torch.linspace(0, 1, m1), torch.linspace(0, 1, m2))

    ε = np.min((0.9, 10 / np.sqrt(0.5 * (n1 * n2 + m1 * m2))))
    print(ε)
    Nsteps = 1000
    tol = 1e-15
    testing = UnbalancedOT(set_fail=True, pykeops=keops)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="kl")
    assert testing.duality_gap() < 1e-10


# Testing tensorised but not tuple input, with and wihtout pykeops


@pytest.mark.parametrize(
    "n, m, keops", [(10, 10, False), (20, 20, False), (21, 19, False), (21, 19, True)]
)
def test_marginals_tensor_kl(n, m, keops):
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

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="kl")
    assert testing.duality_gap() < 1e-10


# -------------------------------- TV ------------------------------------------------
# Testing inputs type, not tensorised, pykeops=False


@pytest.mark.parametrize(
    "n, m, t_n", [(10, 10, "torch"), (21, 19, "torch"), (10, 10, "np"), (31, 32, "np")]
)
def test_marginals_energy_cpu_tv(n, m, t_n):
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

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="tv")

    assert testing.duality_gap() < 1e-10


# Testing inputs type, not tensorised, pykeops=False


@pytest.mark.parametrize(
    "n, m, t_n", [(10, 10, "torch"), (21, 19, "torch"), (10, 10, "np"), (31, 32, "np")]
)
def test_marginals_energy_pykeops_flat_tv(n, m, t_n):
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
    testing = UnbalancedOT(set_fail=True, pykeops=True)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="tv")

    assert testing.duality_gap() < 1e-10


# Testing tuple input with and without pykeops


@pytest.mark.parametrize(
    "n1, n2, m1, m2, keops",
    [
        (10, 10, 10, 10, False),
        (10, 11, 12, 13, True),
        (21, 21, 19, 19, False),
        (10, 11, 10, 11, True),
    ],
)
def test_marginals_tensor_tuple_tv(n1, n2, m1, m2, keops):
    # tuple input
    X = (torch.linspace(0, 1, n1), torch.linspace(0, 1, n2))
    Y = (torch.linspace(0, 1, m1), torch.linspace(0, 1, m2))

    ε = np.min((0.9, 10 / np.sqrt(0.5 * (n1 * n2 + m1 * m2))))
    Nsteps = 1000
    tol = 1e-15
    testing = UnbalancedOT(set_fail=True, pykeops=keops)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="tv")

    assert testing.duality_gap() < 1e-10


# Testing tensorised but not tuple input, with and wihtout pykeops


@pytest.mark.parametrize(
    "n, m, keops", [(10, 10, False), (20, 20, False), (21, 19, False), (21, 19, True)]
)
def test_marginals_tensor_tv(n, m, keops):
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

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="tv")

    assert testing.duality_gap() < 1e-10


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
