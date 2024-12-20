import numpy as np
import torch
from unbalancedsinkhorn import DebiasedUOT
import pytest

# testing balanced case, to check our basic sinkhorn loop works, this should
# be the case for equal grids. Further test the classes ability to take both
# numpy and torch arrays/tensors.


# ##############################################################################################
#               Test sinkhorn divergence is working and debias potentials converging
#                                          BALANCED
# ##############################################################################################

# Testing inputs type, not tensorised, pykeops=False
@pytest.mark.parametrize(
    "n, m, t_n", [(10, 10, "torch"), (21, 19, "torch"), (10, 10, "np"), (31, 32, "np")]
)
def test_sinkhorn_divergence_balanced_cpu(n, m, t_n):
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
    testing = DebiasedUOT(set_fail=True, pykeops=False)
    testing.parameters(ε)
    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced")

    assert testing.duality_gap() < 1e-10

    s_ε = testing.sinkhorn_divergence()

    # Test debiased class has equal potentials on OT_ff, OT_gg
    # assert (testing.debias_f.f == testing.debias_f.g).all()
    # assert (testing.debias_g.f == testing.debias_g.g).all()

    assert testing.duality_gap() < 1e-10
    assert testing.debias_f.duality_gap() < 1e-10
    assert testing.debias_g.duality_gap() < 1e-10


# Testing inputs type, not tensorised, pykeops=False
@pytest.mark.parametrize(
    "n, m, t_n", [(10, 10, "torch"), (21, 19, "torch"), (10, 10, "np"), (31, 32, "np")]
)
def test_sinkhorn_divergence_balanced_pykeops_flat(n, m, t_n):
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
    testing = DebiasedUOT(set_fail=True, pykeops=True)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced")

    assert testing.duality_gap() < 1e-10

    s_ε = testing.sinkhorn_divergence()

    # Test debiased class has equal potentials on OT_ff, OT_gg
    # assert (testing.debias_f.f == testing.debias_f.g).all()
    # assert (testing.debias_g.f == testing.debias_g.g).all()


    assert testing.duality_gap() < 1e-10
    assert testing.debias_f.duality_gap() < 1e-10
    assert testing.debias_g.duality_gap() < 1e-10


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
def test_sinkhorn_divergence_balanced_tensor_tuple(n1, n2, m1, m2, keops):
    # tuple input
    X = (torch.linspace(0, 1, n1), torch.linspace(0, 1, n2))
    Y = (torch.linspace(0, 1, m1), torch.linspace(0, 1, m2))

    ε = np.min((0.9, 10 / np.sqrt(0.5 * (n1 * n2 + m1 * m2))))
    Nsteps = 1000
    tol = 1e-15
    testing = DebiasedUOT(set_fail=True, pykeops=keops)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced")

    assert testing.duality_gap() < 1e-10

    s_ε = testing.sinkhorn_divergence()

    # # Test debiased class has equal potentials on OT_ff, OT_gg
    # assert (testing.debias_f.f == testing.debias_f.g).all()
    # assert (testing.debias_g.f == testing.debias_g.g).all()


    assert testing.duality_gap() < 1e-10
    assert testing.debias_f.duality_gap() < 1e-10
    assert testing.debias_g.duality_gap() < 1e-10


# Testing tensorised but not tuple input, with and wihtout pykeops
@pytest.mark.parametrize(
    "n, m, keops", [(10, 10, False), (20, 20, False), (21, 19, False), (21, 19, True)]
)
def test_sinkhorn_divergence_balanced_tensor(n, m, keops):
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
    testing = DebiasedUOT(set_fail=True, pykeops=keops)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="balanced")

    assert testing.duality_gap() < 1e-10

    s_ε = testing.sinkhorn_divergence()

    # Test debiased class has equal potentials on OT_ff, OT_gg
    # assert (testing.debias_f.f == testing.debias_f.g).all()
    # assert (testing.debias_g.f == testing.debias_g.g).all()

    assert testing.duality_gap() < 1e-10
    assert testing.debias_f.duality_gap() < 1e-10
    assert testing.debias_g.duality_gap() < 1e-10


# ##############################################################################################
#               Test sinkhorn divergence is working and debias potentials converging
#                                          KL
# ##############################################################################################

# Testing inputs type, not tensorised, pykeops=False
@pytest.mark.parametrize(
    "n, m, t_n", [(10, 10, "torch"), (21, 19, "torch"), (10, 10, "np"), (31, 32, "np")]
)
def test_sinkhorn_divergence_kl_cpu(n, m, t_n):
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
    testing = DebiasedUOT(set_fail=True, pykeops=False)
    testing.parameters(ε)
    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="kl")

    assert testing.duality_gap() < 1e-10

    s_ε = testing.sinkhorn_divergence()

    # print(testing.sinkhorn_divergence(), testing.sinkhorn_divergence(primal=False))
    # Test debiased class has equal potentials on OT_ff, OT_gg
    # assert (testing.debias_f.f == testing.debias_f.g).all()
    # assert (testing.debias_g.f == testing.debias_g.g).all()

    assert testing.duality_gap() < 1e-10
   
    assert testing.debias_f.duality_gap() < 1e-10
    assert testing.debias_g.duality_gap() < 1e-10


# Testing inputs type, not tensorised, pykeops=False
@pytest.mark.parametrize(
    "n, m, t_n", [(10, 10, "torch"), (21, 19, "torch"), (10, 10, "np"), (31, 32, "np")]
)
def test_sinkhorn_divergence_kl_pykeops_flat(n, m, t_n):
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
    testing = DebiasedUOT(set_fail=True, pykeops=True)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="kl")

    assert testing.duality_gap() < 1e-10

    s_ε = testing.sinkhorn_divergence()

    # Test debiased class has equal potentials on OT_ff, OT_gg
    # assert (testing.debias_f.f == testing.debias_f.g).all()
    # assert (testing.debias_g.f == testing.debias_g.g).all()


    assert testing.duality_gap() < 1e-10
    assert testing.debias_f.duality_gap() < 1e-10
    assert testing.debias_g.duality_gap() < 1e-10


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
def test_sinkhorn_divergence_kl_tensor_tuple(n1, n2, m1, m2, keops):
    # tuple input
    X = (torch.linspace(0, 1, n1), torch.linspace(0, 1, n2))
    Y = (torch.linspace(0, 1, m1), torch.linspace(0, 1, m2))

    ε = np.min((0.9, 10 / np.sqrt(0.5 * (n1 * n2 + m1 * m2))))
    Nsteps = 1000
    tol = 1e-15
    testing = DebiasedUOT(set_fail=True, pykeops=keops)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="kl")

    assert testing.duality_gap() < 1e-10

    s_ε = testing.sinkhorn_divergence()

    # Test debiased class has equal potentials on OT_ff, OT_gg
    # assert (testing.debias_f.f == testing.debias_f.g).all()
    # assert (testing.debias_g.f == testing.debias_g.g).all()

    assert testing.duality_gap() < 1e-10
    assert testing.debias_f.duality_gap() < 1e-10
    assert testing.debias_g.duality_gap() < 1e-10


# Testing tensorised but not tuple input, with and wihtout pykeops
@pytest.mark.parametrize(
    "n, m, keops", [(10, 10, False), (20, 20, False), (21, 19, False), (21, 19, True)]
)
def test_sinkhorn_divergence_kl_tensor(n, m, keops):
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
    testing = DebiasedUOT(set_fail=True, pykeops=keops)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="kl")

    assert testing.duality_gap() < 1e-10

    s_ε = testing.sinkhorn_divergence()

    # Test debiased class has equal potentials on OT_ff, OT_gg
    # assert (testing.debias_f.f == testing.debias_f.g).all()
    # assert (testing.debias_g.f == testing.debias_g.g).all()

    assert testing.duality_gap() < 1e-10
    assert testing.debias_f.duality_gap() < 1e-10
    assert testing.debias_g.duality_gap() < 1e-10


# ##############################################################################################
#               Test sinkhorn divergence is working and debias potentials converging
#                                          tv
# ##############################################################################################

# Testing inputs type, not tensorised, pykeops=False
@pytest.mark.parametrize(
    "n, m, t_n", [(10, 10, "torch"), (21, 19, "torch"), (10, 10, "np"), (31, 32, "np")]
)
def test_sinkhorn_divergence_tv_cpu(n, m, t_n):
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
    testing = DebiasedUOT(set_fail=True, pykeops=False)
    testing.parameters(ε)
    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="tv")

    assert testing.duality_gap() < 1e-10

    s_ε = testing.sinkhorn_divergence()

    # Test debiased class has equal potentials on OT_ff, OT_gg
    # assert (testing.debias_f.f == testing.debias_f.g).all()
    # assert (testing.debias_g.f == testing.debias_g.g).all()


    assert testing.duality_gap() < 1e-10
    assert testing.debias_f.duality_gap() < 1e-10
    assert testing.debias_g.duality_gap() < 1e-10


# Testing inputs type, not tensorised, pykeops=False
@pytest.mark.parametrize(
    "n, m, t_n", [(10, 10, "torch"), (21, 19, "torch"), (10, 10, "np"), (31, 32, "np")]
)
def test_sinkhorn_divergence_tv_pykeops_flat(n, m, t_n):
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
    testing = DebiasedUOT(set_fail=True, pykeops=True)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="tv")

    assert testing.duality_gap() < 1e-10

    s_ε = testing.sinkhorn_divergence()

    # Test debiased class has equal potentials on OT_ff, OT_gg
    # assert (testing.debias_f.f == testing.debias_f.g).all()
    # assert (testing.debias_g.f == testing.debias_g.g).all()


    assert testing.duality_gap() < 1e-10
    assert testing.debias_f.duality_gap() < 1e-10
    assert testing.debias_g.duality_gap() < 1e-10


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
def test_sinkhorn_divergence_tv_tensor_tuple(n1, n2, m1, m2, keops):
    # tuple input
    X = (torch.linspace(0, 1, n1), torch.linspace(0, 1, n2))
    Y = (torch.linspace(0, 1, m1), torch.linspace(0, 1, m2))

    ε = np.min((0.9, 10 / np.sqrt(0.5 * (n1 * n2 + m1 * m2))))
    Nsteps = 1000
    tol = 1e-15
    testing = DebiasedUOT(set_fail=True, pykeops=keops)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="tv")

    assert testing.duality_gap() < 1e-10

    s_ε = testing.sinkhorn_divergence()

    # Test debiased class has equal potentials on OT_ff, OT_gg
    # assert (testing.debias_f.f == testing.debias_f.g).all()
    # assert (testing.debias_g.f == testing.debias_g.g).all()


    assert testing.duality_gap() < 1e-10
    assert testing.debias_f.duality_gap() < 1e-10
    assert testing.debias_g.duality_gap() < 1e-10


# Testing tensorised but not tuple input, with and wihtout pykeops
@pytest.mark.parametrize(
    "n, m, keops", [(10, 10, False), (20, 20, False), (21, 19, False), (21, 19, True)]
)
def test_sinkhorn_divergence_tv_tensor(n, m, keops):
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
    testing = DebiasedUOT(set_fail=True, pykeops=keops)
    testing.parameters(ε)
    # torch test

    testing.densities(X, Y)
    testing.sinkhorn_algorithm(Nsteps, tol, verbose=False, aprox="tv")

    assert testing.duality_gap() < 1e-10

    s_ε = testing.sinkhorn_divergence()

    # Test debiased class has equal potentials on OT_ff, OT_gg
    # assert (testing.debias_f.f == testing.debias_f.g).all()
    # assert (testing.debias_g.f == testing.debias_g.g).all()


    assert testing.duality_gap() < 1e-10
    assert testing.debias_f.duality_gap() < 1e-10
    assert testing.debias_g.duality_gap() < 1e-10


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
