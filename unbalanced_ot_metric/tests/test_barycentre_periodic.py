
import numpy as np
import torch
from unbalancedsinkhorn import UnbalancedOT
import pytest

# testing balanced case, to check our basic sinkhorn loop works, this should
# be the case for equal grids. Further test the classes ability to take both
# numpy and torch arrays/tensors.

# ------------------ Testing flat ---------------------------
# Testing inputs type, not tensorised, pykeops=False
@pytest.mark.parametrize(
    "n, m, t_n", [(10, 10, "torch"), (21, 19, "torch"), (10, 10, "np"), (31, 32, "np")]
)
def test_pbc_barycentre_cpu_flat(n, m, t_n):
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
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced")

    assert np.isclose(
        testing.barycentre(torch.ones_like(testing.α_s), sum_over="source").sum(),
        m**2,
        atol=1e-14,
        rtol=1e-16,
    )
    assert np.isclose(
        testing.barycentre(torch.ones_like(testing.β_t), sum_over="target").sum(),
        n**2,
        atol=1e-14,
        rtol=1e-16,
    )


@pytest.mark.parametrize(
    "n, m, t_n", [(10, 10, "torch"), (21, 19, "torch"), (10, 10, "np"), (31, 32, "np")]
)
def test_pbc_barycentre_gpu_flat(n, m, t_n):
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
    testing = UnbalancedOT(set_fail=False, pykeops=True)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y, cost_type='periodic', L=1.0)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced")

    assert np.isclose(
        testing.barycentre(torch.ones_like(testing.α_s), sum_over="source").sum().cpu(),
        m**2,
        atol=1e-14,
        rtol=1e-16,
    )
    assert np.isclose(
        testing.barycentre(torch.ones_like(testing.β_t), sum_over="target").sum().cpu(),
        n**2,
        atol=1e-14,
        rtol=1e-16,
    )


@pytest.mark.parametrize("n, m, t_n", [(10, 10, "torch"), (10, 10, "np")])
def test_pbc_barycentre_of_points_flat(n, m, t_n):
    if t_n == "torch":
        X = torch.cartesian_prod(torch.linspace(1/(2*n), 1 - 1/(2*n), n), torch.linspace(1/(2*n), 1 - 1/(2*n), n))
        Y = torch.cartesian_prod(torch.linspace(1/(2*m), 1 - 1/(2*m), m), torch.linspace(1/(2*m), 1 - 1/(2*m), m))
    else:
        X1, X2 = np.meshgrid(
            torch.linspace(1/(2*n), 1 - 1/(2*n), n), torch.linspace(1/(2*n), 1 - 1/(2*n), n), indexing="ij"
        )
        Y1, Y2 = np.meshgrid(
            torch.linspace(1/(2*m), 1 - 1/(2*m), m), torch.linspace(1/(2*m), 1 - 1/(2*m), m), indexing="ij"
        )
        X = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
        Y = np.hstack((Y1.reshape(-1, 1), Y2.reshape(-1, 1)))

    # Require small epsilon for the test to be valid
    ε = 0.00001
    Nsteps = 1000
    tol = 1e-15
    testing = UnbalancedOT(set_fail=False, pykeops=True)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y, cost_type='periodic', L=1.0)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced")
    print(testing.barycentre_map_of_points(which="source"))
    assert torch.isclose(
        testing.barycentre_map_of_points(which="source"), torch.zeros_like(testing.Y_t)
    ).all()
    assert torch.isclose(
        testing.barycentre_map_of_points(which="target"), torch.zeros_like(testing.X_s)
    ).all()


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
