import numpy as np
from scipy.spatial.distance import cdist
import torch
import pytest
from unbalancedsinkhorn import UnbalancedSinkhorn

torch.set_printoptions(precision=8)

# Testing Balanced implementation against numpy
# Non-tensorised ==> pykeops


@pytest.mark.parametrize(
    "n1, n2, m1, m2, Nsteps",
    [
        (2, 2, 3, 3, 1),
        (5, 6, 7, 8, 5),
        (12, 11, 10, 9, 10),
        (6, 6, 6, 6, 50),
        (2, 3, 1, 3, 1000),
    ],
)  # noqa: E501
def test_against_numpy_flat_arrays(n1, n2, m1, m2, Nsteps):
    N, M = n1 * n2, m1 * m2
    X1, X2 = np.meshgrid(np.linspace(0, 1, n1), np.linspace(0, 1, n2), indexing="xy")
    Y1, Y2 = np.meshgrid(np.linspace(0, 1, m1), np.linspace(0, 1, m2), indexing="xy")
    X = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
    Y = np.hstack((Y1.reshape(-1, 1), Y2.reshape(-1, 1)))
    ε = 1 / np.sqrt(0.5 * (N + M))
    assert ε < 1

    # numpy steps
    cc = 100
    f = np.zeros(N)
    g = np.zeros(M)
    C = 0.5 * cc * cdist(X, Y) ** 2
    α = np.random.rand(*f.shape)
    α /= np.sum(α)
    β = np.ones_like(g) / M

    def εLSE(x, ax):
        return -ε * np.log(np.sum(np.exp(x), axis=ax))

    for _ in range(Nsteps):
        f += εLSE((f[:, None] + g[None, :] - C) / ε + np.log(β)[None, :], 1)
        g += εLSE((f[:, None] + g[None, :] - C) / ε + np.log(α)[:, None], 0)

    testing = UnbalancedSinkhorn(set_fail=True, pykeops=False)
    testing.parameters(ε, cost_const=cc)
    testing.densities(X, Y, α)
    testing.sinkhorn_algorithm(
        Nsteps, verbose=False, aprox="balanced", convergence_checks=1, convergence_or_fail=False
    )
    assert np.isclose(f, testing.f.numpy().flatten()).all()
    assert np.isclose(g, testing.g.numpy().flatten()).all()
    assert np.isclose(C, testing.cost.numpy()).all()
    assert np.isclose(α, testing.α_s.numpy().flatten()).all()
    assert np.isclose(β, testing.β_t.numpy().flatten()).all()


@pytest.mark.parametrize(
    "n1, n2, m1, m2, Nsteps",
    [
        (2, 2, 3, 3, 1),
        (5, 6, 7, 8, 5),
        (12, 11, 10, 9, 10),
        (6, 6, 6, 6, 50),
        (2, 3, 1, 3, 1000),
    ],
)  # noqa: E501
def test_pykeops_against_numpy_flat_arrays(n1, n2, m1, m2, Nsteps):
    N, M = n1 * n2, m1 * m2
    X1, X2 = np.meshgrid(np.linspace(0, 1, n1), np.linspace(0, 1, n2))
    Y1, Y2 = np.meshgrid(np.linspace(0, 1, m1), np.linspace(0, 1, m2))
    X = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
    Y = np.hstack((Y1.reshape(-1, 1), Y2.reshape(-1, 1)))
    ε = 1 / np.sqrt(0.5 * (N + M))
    assert ε < 1

    # numpy steps
    cc = 100
    f = np.zeros(N)
    g = np.zeros(M)
    C = 0.5 * cc * cdist(X, Y) ** 2
    α = np.random.rand(*f.shape)
    α /= np.sum(α)
    β = np.ones_like(g) / M

    def εLSE(x, ax):
        return -ε * np.log(np.sum(np.exp(x), axis=ax))

    for _ in range(Nsteps):
        f += εLSE((f[:, None] + g[None, :] - C) / ε + np.log(β)[None, :], 1)
        g += εLSE((f[:, None] + g[None, :] - C) / ε + np.log(α)[:, None], 0)

    testing = UnbalancedSinkhorn(set_fail=False, pykeops=True)
    testing.parameters(ε, cost_const=cc)
    testing.densities(X, Y, α)
    testing.sinkhorn_algorithm(
        Nsteps, verbose=False, aprox="balanced", convergence_checks=1, convergence_or_fail=False
    )
    try:
        assert np.isclose(f, testing.f.numpy().flatten()).all()
        assert np.isclose(g, testing.g.numpy().flatten()).all()
        assert np.isclose(α, testing.α_s.numpy().flatten()).all()
        assert np.isclose(β, testing.β_t.numpy().flatten()).all()
    except TypeError:
        # Cuda avaliable and need to convert to cpu
        assert np.isclose(f, testing.f.cpu().numpy().flatten()).all()
        assert np.isclose(g, testing.g.cpu().numpy().flatten()).all()
        assert np.isclose(α, testing.α_s.cpu().numpy().flatten()).all()
        assert np.isclose(β, testing.β_t.cpu().numpy().flatten()).all()


@pytest.mark.parametrize(
    "n1, n2, m1, m2, Nsteps",
    [
        (2, 2, 3, 3, 1),
        (5, 6, 7, 8, 5),
        (12, 11, 10, 9, 10),
        (6, 6, 6, 6, 50),
        (2, 3, 1, 3, 1000),
    ],
)  # noqa: E501
def test_tensorisation_against_numpy_2D_array(n1, n2, m1, m2, Nsteps):
    N, M = n1 * n2, m1 * m2
    X1, X2 = np.meshgrid(np.linspace(0, 1, n1), np.linspace(0, 1, n2))
    Y1, Y2 = np.meshgrid(np.linspace(0, 1, m1), np.linspace(0, 1, m2))
    X_2d = np.stack((X1, X2), axis=2)
    Y_2d = np.stack((Y1, Y2), axis=2)
    X = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
    Y = np.hstack((Y1.reshape(-1, 1), Y2.reshape(-1, 1)))
    ε = 1 / np.sqrt(0.5 * (N + M))
    assert ε < 1

    # numpy steps
    cc = 100
    f = np.zeros(N)
    g = np.zeros(M)
    C = 0.5 * cc * cdist(X, Y) ** 2
    α = np.random.rand(*f.shape)
    α /= np.sum(α)
    β = np.ones_like(g) / M

    def εLSE(x, ax):
        return -ε * np.log(np.sum(np.exp(x), axis=ax))

    for _ in range(Nsteps):
        f += εLSE((f[:, None] + g[None, :] - C) / ε + np.log(β)[None, :], 1)
        g += εLSE((f[:, None] + g[None, :] - C) / ε + np.log(α)[:, None], 0)

    # Despite pykeops being true, tensorisation should overide
    # by the given input
    testing = UnbalancedSinkhorn(set_fail=False, pykeops=True)
    testing.parameters(ε, cost_const=cc)
    testing.densities(X_2d, Y_2d, α)
    testing.sinkhorn_algorithm(
        Nsteps, verbose=False, aprox="balanced", convergence_checks=1, convergence_or_fail=False
    )

    try:
        assert np.isclose(f, testing.f.numpy().flatten()).all()
        assert np.isclose(g, testing.g.numpy().flatten()).all()
        assert np.isclose(α, testing.α_s.numpy().flatten()).all()
        assert np.isclose(β, testing.β_t.numpy().flatten()).all()
    except TypeError:
        # Cuda avaliable and need to convert to cpu
        assert np.isclose(f, testing.f.cpu().numpy().flatten()).all()
        assert np.isclose(g, testing.g.cpu().numpy().flatten()).all()
        assert np.isclose(α, testing.α_s.cpu().numpy().flatten()).all()
        assert np.isclose(β, testing.β_t.cpu().numpy().flatten()).all()


@pytest.mark.parametrize(
    "n1, n2, m1, m2, Nsteps",
    [
        (2, 2, 3, 3, 1),
        (5, 6, 7, 8, 5),
        (12, 11, 10, 9, 10),
        (6, 6, 6, 6, 50),
        (2, 3, 1, 3, 1000),
    ],
)  # noqa: E501
def test_tensorisation_against_numpy_tuple_input(n1, n2, m1, m2, Nsteps):
    N, M = n1 * n2, m1 * m2
    X_t = (np.linspace(0, 1, n1), np.linspace(0, 1, n2))
    Y_t = (np.linspace(0, 1, m1), np.linspace(0, 1, m2))
    X1, X2 = np.meshgrid(np.linspace(0, 1, n1), np.linspace(0, 1, n2), indexing="ij")
    Y1, Y2 = np.meshgrid(np.linspace(0, 1, m1), np.linspace(0, 1, m2), indexing="ij")
    X = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
    Y = np.hstack((Y1.reshape(-1, 1), Y2.reshape(-1, 1)))
    ε = 1 / np.sqrt(0.5 * (N + M))
    assert ε < 1

    # numpy steps
    cc = 100
    f = np.zeros(N)
    g = np.zeros(M)
    C = 0.5 * cc * cdist(X, Y) ** 2
    α = np.random.rand(*f.shape)
    α /= np.sum(α)
    β = np.ones_like(g) / M

    def εLSE(x, ax):
        return -ε * np.log(np.sum(np.exp(x), axis=ax))

    for _ in range(Nsteps):
        f += εLSE((f[:, None] + g[None, :] - C) / ε + np.log(β)[None, :], 1)
        g += εLSE((f[:, None] + g[None, :] - C) / ε + np.log(α)[:, None], 0)

    # Despite pykeops being true, tensorisation should overide by the
    # given input
    testing = UnbalancedSinkhorn(set_fail=False, pykeops=True)
    testing.parameters(ε, cost_const=cc)
    testing.densities(X_t, Y_t, α)
    testing.sinkhorn_algorithm(
        Nsteps, verbose=False, aprox="balanced", convergence_checks=1, convergence_or_fail=False
    )

    try:
        assert np.isclose(f, testing.f.numpy().flatten()).all()
        assert np.isclose(g, testing.g.numpy().flatten()).all()
        assert np.isclose(α, testing.α_s.numpy().flatten()).all()
        assert np.isclose(β, testing.β_t.numpy().flatten()).all()
    except TypeError:
        # Cuda avaliable and need to convert to cpu
        assert np.isclose(f, testing.f.cpu().numpy().flatten()).all()
        assert np.isclose(g, testing.g.cpu().numpy().flatten()).all()
        assert np.isclose(α, testing.α_s.cpu().numpy().flatten()).all()
        assert np.isclose(β, testing.β_t.cpu().numpy().flatten()).all()


# --------------------- Kl -----------------------------


@pytest.mark.parametrize(
    "n1, n2, m1, m2, Nsteps",
    [
        (2, 2, 3, 3, 1),
        (5, 6, 7, 8, 5),
        (12, 11, 10, 9, 10),
        (6, 6, 6, 6, 50),
        (2, 3, 1, 3, 1000),
    ],
)  # noqa: E501
def test_against_numpy_flat_arrays_kl(n1, n2, m1, m2, Nsteps):
    N, M = n1 * n2, m1 * m2
    X1, X2 = np.meshgrid(np.linspace(0, 1, n1), np.linspace(0, 1, n2))
    Y1, Y2 = np.meshgrid(np.linspace(0, 1, m1), np.linspace(0, 1, m2))
    X = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
    Y = np.hstack((Y1.reshape(-1, 1), Y2.reshape(-1, 1)))
    ε = 1 / np.sqrt(0.5 * (N + M))
    assert ε < 1
    rho = 1

    # numpy steps
    cc = 100
    f = np.zeros(N)
    g = np.zeros(M)
    C = 0.5 * cc * cdist(X, Y) ** 2
    α = np.ones_like(f) / N
    β = np.ones_like(g) / M

    def εLSE(x, ax):
        return -ε * np.log(np.sum(np.exp(x), axis=ax))

    for _ in range(Nsteps):
        f += εLSE((f[:, None] + g[None, :] - C) / ε + np.log(β)[None, :], 1)
        f = -rho * -f / (rho + ε)
        g += εLSE((f[:, None] + g[None, :] - C) / ε + np.log(α)[:, None], 0)
        g = -rho * -g / (rho + ε)

    testing = UnbalancedSinkhorn(set_fail=True, pykeops=False)
    testing.parameters(ε, cost_const=cc)
    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, verbose=True, aprox="kl", convergence_checks=1, convergence_or_fail=False)
    assert np.isclose(C, testing.cost.numpy()).all()
    assert np.isclose(α, testing.α_s.numpy().flatten()).all()
    assert np.isclose(β, testing.β_t.numpy().flatten()).all()
    assert np.isclose(f, testing.f.numpy().flatten()).all()
    assert np.isclose(g, testing.g.numpy().flatten()).all()


# -------------------- TV ------------------------------
@pytest.mark.parametrize(
    "n1, n2, m1, m2, Nsteps",
    [
        (2, 2, 3, 3, 1),
        (5, 6, 7, 8, 5),
        (12, 11, 10, 9, 10),
        (6, 6, 6, 6, 50),
        (2, 3, 1, 3, 1000),
    ],
)  # noqa: E501
def test_against_numpy_flat_arrays_tv(n1, n2, m1, m2, Nsteps):
    N, M = n1 * n2, m1 * m2
    X1, X2 = np.meshgrid(np.linspace(0, 1, n1), np.linspace(0, 1, n2))
    Y1, Y2 = np.meshgrid(np.linspace(0, 1, m1), np.linspace(0, 1, m2))
    X = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))
    Y = np.hstack((Y1.reshape(-1, 1), Y2.reshape(-1, 1)))
    ε = 1 / np.sqrt(0.5 * (N + M))
    assert ε < 1
    rho = 1

    # numpy steps
    cc = 100
    f = np.zeros(N)
    g = np.zeros(M)
    C = 0.5 * cc * cdist(X, Y) ** 2
    α = np.ones_like(f) / N
    β = np.ones_like(g) / M

    def εLSE(x, ax):
        return -ε * np.log(np.sum(np.exp(x), axis=ax))

    for _ in range(Nsteps):
        f += εLSE((f[:, None] + g[None, :] - C) / ε + np.log(β)[None, :], 1)
        f = -np.clip(-f, -rho, rho)
        g += εLSE((f[:, None] + g[None, :] - C) / ε + np.log(α)[:, None], 0)
        g = -np.clip(-g, -rho, rho)

    testing = UnbalancedSinkhorn(set_fail=True, pykeops=False)
    testing.parameters(ε, cost_const=cc)
    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, verbose=True, aprox="tv", convergence_checks=1, convergence_or_fail=False)
    assert np.isclose(C, testing.cost.numpy()).all()
    assert np.isclose(α, testing.α_s.numpy().flatten()).all()
    assert np.isclose(β, testing.β_t.numpy().flatten()).all()
    assert np.isclose(f, testing.f.numpy().flatten()).all()
    assert np.isclose(g, testing.g.numpy().flatten()).all()


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
