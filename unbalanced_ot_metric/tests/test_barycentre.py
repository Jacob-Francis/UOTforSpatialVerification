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
def test_barycentre_cpu_flat(n, m, t_n):
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
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced", convergence_or_fail=False)

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
def test_barycentre_gpu_flat(n, m, t_n):
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

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced", convergence_or_fail=False)

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
def test_barycentre_of_points_flat(n, m, t_n):
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

    # Require small epsilon for the test to be valid
    ε = 0.0001
    Nsteps = 1000
    tol = 1e-15
    testing = UnbalancedOT(set_fail=False, pykeops=True)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced", convergence_or_fail=False)
    assert torch.isclose(
        testing.barycentre_map_of_points(which="source"), torch.zeros_like(testing.Y_t)
    ).all()
    assert torch.isclose(
        testing.barycentre_map_of_points(which="target"), torch.zeros_like(testing.X_s)
    ).all()


# ------------------ Testing tensorisation tuple ---------------------------
# Testing inputs type, not tensorised, pykeops=False
@pytest.mark.parametrize(
    "n1, n2, m1, m2, t_n",
    [
        (10, 10, 10, 10, "torch"),
        (7, 8, 6, 10, "torch"),
        (3, 3, 3, 3, "np"),
        (10, 9, 8, 7, "np"),
    ],
)
def test_barycentre_cpu_tuple(n1, n2, m1, m2, t_n):
    if t_n == "torch":
        X = (torch.linspace(0, 1, n1), torch.linspace(0, 1, n2))
        Y = (torch.linspace(0, 1, m1), torch.linspace(0, 1, m2))
    elif t_n == "np":
        X = (np.linspace(0, 1, n1), torch.linspace(0, 1, n2))
        Y = (np.linspace(0, 1, m1), torch.linspace(0, 1, m2))

    ε = np.min((0.5, 10 / np.sqrt(0.5 * (n1 * n2 + m1 * m2))))
    Nsteps = 1000
    tol = 1e-15
    testing = UnbalancedOT(set_fail=True, pykeops=False)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced", convergence_or_fail=False)

    assert np.isclose(
        testing.barycentre(torch.ones_like(testing.α_s), sum_over="source").sum(),
        m1 * m2,
    )
    assert np.isclose(
        testing.barycentre(torch.ones_like(testing.β_t), sum_over="target").sum(),
        n1 * n2,
    )


@pytest.mark.parametrize(
    "n1, n2, m1, m2, t_n",
    [
        (10, 10, 10, 10, "torch"),
        (7, 8, 6, 10, "torch"),
        (3, 3, 3, 3, "np"),
        (10, 9, 8, 7, "np"),
    ],
)
def test_barycentre_pykeops_tuple(n1, n2, m1, m2, t_n):
    if t_n == "torch":
        X = (torch.linspace(0, 1, n1), torch.linspace(0, 1, n2))
        Y = (torch.linspace(0, 1, m1), torch.linspace(0, 1, m2))
    elif t_n == "np":
        X = (np.linspace(0, 1, n1), torch.linspace(0, 1, n2))
        Y = (np.linspace(0, 1, m1), torch.linspace(0, 1, m2))

    ε = np.min((0.5, 10 / np.sqrt(0.5 * (n1 * n2 + m1 * m2))))
    Nsteps = 3000
    tol = 1e-15
    testing = UnbalancedOT(set_fail=False, pykeops=True)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced", convergence_or_fail=False)
    print(
        testing.barycentre(
            torch.ones_like(testing.α_s).view(-1, 1), sum_over="source"
        ).shape,
        m1 * m2,
        n1 * n2,
    )
    assert np.isclose(
        testing.barycentre(torch.ones_like(testing.α_s).view(-1, 1), sum_over="source")
        .sum()
        .cpu(),
        m1 * m2,
    )
    assert np.isclose(
        testing.barycentre(torch.ones_like(testing.β_t).view(-1, 1), sum_over="target")
        .sum()
        .cpu(),
        n1 * n2,
    )


@pytest.mark.parametrize("n, m, t_n", [(10, 10, "torch"), (10, 10, "np")])
def test_barycentre_of_points_tuple(n, m, t_n):
    if t_n == "torch":
        X = (torch.linspace(0, 1, n), torch.linspace(0, 1, n))
        Y = (torch.linspace(0, 1, m), torch.linspace(0, 1, m))
    elif t_n == "np":
        X = (np.linspace(0, 1, n), torch.linspace(0, 1, n))
        Y = (np.linspace(0, 1, m), torch.linspace(0, 1, m))

    ε = 1e-4
    Nsteps = 1000
    tol = 1e-15
    testing = UnbalancedOT(set_fail=False, pykeops=True)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced", convergence_or_fail=False)
    # ################################## This needs fixing since the output should stay in tensorised shape ############################
    assert np.isclose(
        testing.barycentre_map_of_points(which="source").cpu(), torch.zeros(m * m, 2)
    ).all()
    assert np.isclose(
        testing.barycentre_map_of_points(which="target").cpu(), torch.zeros(n * n, 2)
    ).all()


# ------------------ Testing tensorisation 2D ---------------------------
# Testing inputs type, not tensorised, pykeops=False
@pytest.mark.parametrize(
    "n1, n2, m1, m2", [(10, 10, 10, 10), (7, 8, 6, 10), (3, 3, 3, 3), (10, 9, 8, 7)]
)
def test_barycentre_cpu_2d(n1, n2, m1, m2):
    X1, X2 = np.meshgrid(
        torch.linspace(0, 1, n1), torch.linspace(0, 1, n2), indexing="ij"
    )
    Y1, Y2 = np.meshgrid(
        torch.linspace(0, 1, m1), torch.linspace(0, 1, m2), indexing="ij"
    )
    X = np.stack((X1, X2), axis=2)
    Y = np.stack((Y1, Y2), axis=2)

    ε = np.min((0.5, 10 / np.sqrt(0.5 * (n1 * n2 + m1 * m2))))
    Nsteps = 1000
    tol = 1e-15
    testing = UnbalancedOT(set_fail=True, pykeops=False)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced", convergence_or_fail=False)

    assert np.isclose(
        testing.barycentre(torch.ones_like(testing.α_s), sum_over="source").sum(),
        m1 * m2,
    )
    assert np.isclose(
        testing.barycentre(torch.ones_like(testing.β_t), sum_over="target").sum(),
        n1 * n2,
    )


@pytest.mark.parametrize(
    "n1, n2, m1, m2", [(10, 10, 10, 10), (7, 8, 6, 10), (3, 3, 3, 3), (10, 9, 8, 7)]
)
def test_barycentre_pykeops_2d(n1, n2, m1, m2):
    X1, X2 = np.meshgrid(
        torch.linspace(0, 1, n1), torch.linspace(0, 1, n2), indexing="ij"
    )
    Y1, Y2 = np.meshgrid(
        torch.linspace(0, 1, m1), torch.linspace(0, 1, m2), indexing="ij"
    )
    X = np.stack((X1, X2), axis=2)
    Y = np.stack((Y1, Y2), axis=2)

    ε = np.min((0.5, 10 / np.sqrt(0.5 * (n1 * n2 + m1 * m2))))
    Nsteps = 1000
    tol = 1e-15
    testing = UnbalancedOT(set_fail=False, pykeops=True)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced", convergence_or_fail=False)
    # ################################## This needs fixing since the input and output should stay in tensorised shape ############################
    assert np.isclose(
        testing.barycentre(torch.ones_like(testing.α_s.view(-1, 1)), sum_over="source")
        .sum()
        .cpu(),
        m1 * m2,
    )
    assert np.isclose(
        testing.barycentre(torch.ones_like(testing.β_t.view(-1, 1)), sum_over="target")
        .sum()
        .cpu(),
        n1 * n2,
    )


@pytest.mark.parametrize("n, m", [(10, 10)])
def test_barycentre_of_points_2d(n, m):
    X1, X2 = np.meshgrid(
        torch.linspace(0, 1, n), torch.linspace(0, 1, n), indexing="ij"
    )
    Y1, Y2 = np.meshgrid(
        torch.linspace(0, 1, m), torch.linspace(0, 1, m), indexing="ij"
    )
    X = np.stack((X1, X2), axis=2)
    Y = np.stack((Y1, Y2), axis=2)

    ε = 1e-4
    Nsteps = 1000
    tol = 1e-15
    testing = UnbalancedOT(set_fail=False, pykeops=True)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced", convergence_or_fail=False)
    # ################################## This needs fixing since the output should stay in tensorised shape ############################
    assert np.isclose(
        testing.barycentre_map_of_points(which="source").cpu(), torch.zeros(m * m, 2)
    ).all()
    assert np.isclose(
        testing.barycentre_map_of_points(which="target").cpu(), torch.zeros(n * n, 2)
    ).all()


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
