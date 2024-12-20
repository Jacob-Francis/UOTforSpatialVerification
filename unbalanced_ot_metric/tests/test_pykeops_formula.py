try:
    import pykeops
except ImportError:
    # There's nothing to test in this case, how can I avoid these tests?
    pass

import torch
import numpy as np
from torchnumpyprocess import TorchNumpyProcessing
from scipy.spatial.distance import cdist
from unbalancedsinkhorn import (
    PyKeOpsFormulas
)
import pytest


# Creat global PyKeOpsFormulas class object
global pykeops_class
pykeops_class = PyKeOpsFormulas(cost_string="SqDist(X, Y)")

def test_log_sum_exp():
    f = np.random.rand(100**2)
    g = np.random.rand(101**2)
    X = np.random.rand(100**2, 2)
    Y = np.random.rand(101**2, 2)
    E = np.array([0.01])
    M = np.random.rand(101**2)
    f_0 = np.array([10])

    lse = np.log(
        np.sum(
            np.exp(
                (g[None, :] + f[:, None] - 0.5 * f_0**2 * cdist(X, Y) ** 2) / E
                + np.log(M[None, :])
            ),
            axis=1,
        )
    )

    t = TorchNumpyProcessing(set_fail=True)
    tfunc = t._torch_numpy_process
    lse_temp = pykeops_class.log_sum_exp(
        tfunc(g).view(-1, 1),
        tfunc(f).view(-1, 1),
        tfunc(X),
        tfunc(Y),
        tfunc(E),
        tfunc(M).view(-1, 1),
        tfunc(f_0**2).view(-1, 1),
    )  # g, f, X, Y, E, M, f_0)
    assert np.isclose(
        lse_temp.detach().cpu().numpy().reshape(-1, 1), lse.reshape(-1, 1)
    ).all()


# --------------- Dual cost term -----------------


def test_dual_energy_kl_transform():
    f = np.random.rand(5)
    g = np.random.rand(7)
    X = np.random.rand(5, 2)
    Y = np.random.rand(7, 2)
    E = np.array([0.01])
    S = np.random.rand(5)
    M = np.random.rand(7)
    f_0 = np.array([10])

    dual_kl_term = np.sum(
        (np.exp((g[None, :] + f[:, None] - 0.5 * f_0**2 * cdist(X, Y) ** 2) / E) - 1)
        * np.outer(S, M)
    )
    t = TorchNumpyProcessing(set_fail=True)
    tfunc = t._torch_numpy_process
    dual_kl_term_t = pykeops_class.dual_energy_kl_transform(
        tfunc(g).view(-1, 1),
        tfunc(f).view(-1, 1),
        tfunc(X),
        tfunc(Y),
        tfunc(E),
        tfunc(S).view(-1, 1),
        tfunc(M).view(-1, 1),
        tfunc(f_0**2).view(-1, 1),
    )  # g, f, X, Y, E, M, f_0)

    assert np.isclose(
        dual_kl_term_t.detach().numpy().reshape(-1, 1), dual_kl_term.reshape(-1, 1)
    ).all()


# -----------Primal energy ----------------
def test_primal_energy():
    f = np.random.rand(5)
    g = np.random.rand(7)
    X = np.random.rand(5, 2)
    Y = np.random.rand(7, 2)
    E = np.array([0.1])
    S = np.random.rand(5)
    M = np.random.rand(7)
    f_0 = np.array([1])

    pi = np.exp(
        (g[None, :] + f[:, None] - 0.5 * f_0**2 * cdist(X, Y) ** 2) / E
    ) * np.outer(S, M)
    R = np.outer(S, M)
    primal = np.sum(pi * np.log(pi / R) - pi + R)
    cost_pi_nu = np.sum(pi * 0.5 * cdist(X, Y) ** 2)
    t = TorchNumpyProcessing(set_fail=True)
    tfunc = t._torch_numpy_process
    primal_t = pykeops_class.primal_energy_kl_term(
        tfunc(g).view(-1, 1),
        tfunc(f).view(-1, 1),
        tfunc(X),
        tfunc(Y),
        tfunc(E),
        tfunc(S).view(-1, 1),
        tfunc(M).view(-1, 1),
        tfunc(f_0**2).view(-1, 1),
    )
    cost_pi_t = pykeops_class.cost_pi(
        tfunc(g).view(-1, 1),
        tfunc(f).view(-1, 1),
        tfunc(X),
        tfunc(Y),
        tfunc(E),
        tfunc(S).view(-1, 1),
        tfunc(M).view(-1, 1),
        tfunc(f_0**2).view(-1, 1),
    )
    assert np.isclose(
        cost_pi_t.detach().numpy().reshape(-1, 1), cost_pi_nu.reshape(-1, 1)
    ).all()
    assert np.isclose(
        primal_t.detach().numpy().reshape(-1, 1), primal.reshape(-1, 1)
    ).all()


# --------------------------------------------------------------------------------------------------------------------------------
# Marginals of pi


def test_marginals():
    f = np.random.rand(5)
    g = np.random.rand(7)
    X = np.random.rand(5, 2)
    Y = np.random.rand(7, 2)
    E = np.array([0.01])
    S = np.random.rand(5)
    M = np.random.rand(7)
    f_0 = np.array([10])

    pi = np.outer(S, M) * np.exp(
        (g[None, :] + f[:, None] - 0.5 * f_0**2 * cdist(X, Y) ** 2) / E
    )
    marginal_uni = pi.sum(0)
    marginal_geo = pi.sum(1)
    t = TorchNumpyProcessing(set_fail=True)
    tfunc = t._torch_numpy_process
    # for k in [tfunc(g.view(-1,1)), tfunc(f.view(-1,1)), tfunc(X), tfunc(Y), tfunc(E), tfunc(M.view(-1, 1)), tfunc([f_0])]
    mar_u = pykeops_class.marginal_j(
        tfunc(g).view(-1, 1),
        tfunc(f).view(-1, 1),
        tfunc(X),
        tfunc(Y),
        tfunc(E),
        tfunc(S).view(-1, 1),
        tfunc(M).view(-1, 1),
        tfunc(f_0**2).view(-1, 1),
    )  # g, f, X, Y, E, M, f_0)
    mar_g = pykeops_class.marginal_i(
        tfunc(g).view(-1, 1),
        tfunc(f).view(-1, 1),
        tfunc(X),
        tfunc(Y),
        tfunc(E),
        tfunc(S).view(-1, 1),
        tfunc(M).view(-1, 1),
        tfunc(f_0**2).view(-1, 1),
    )  # g, f, X, Y, E, M, f_0)

    assert np.isclose(
        mar_u.detach().numpy().reshape(-1, 1), marginal_uni.reshape(-1, 1)
    ).all()
    assert np.isclose(
        mar_g.detach().numpy().reshape(-1, 1), marginal_geo.reshape(-1, 1)
    ).all()


# ----------------- Barycentres ---------------------------------------------------------------------------------------------------------------


def test_barycentres():
    # test - above
    f = np.random.rand(5)
    g = np.random.rand(7)
    X = np.random.rand(5, 2)
    Y = np.random.rand(7, 2)
    E = np.array([0.01])
    S = np.random.rand(5)
    M = np.random.rand(7)
    P_r = np.random.rand(5)
    P_l = np.random.rand(7)
    f_0 = np.array([1.5])

    pi = np.outer(S, M) * np.exp(
        (g[None, :] + f[:, None] - 0.5 * f_0**2 * cdist(X, Y) ** 2) / E
    )
    bary_left = pi @ P_l / pi.sum(1)
    bary_right = pi.T @ P_r / pi.sum(0)
    t = TorchNumpyProcessing(set_fail=True)
    tfunc = t._torch_numpy_process
    # G, F, X, Y, E, S, M, P, C=
    bary_r = pykeops_class.barycentres(
        tfunc(f).view(-1, 1),
        tfunc(g).view(-1, 1),
        tfunc(Y),
        tfunc(X),
        tfunc(E),
        tfunc(M).view(-1, 1),
        tfunc(S).view(-1, 1),
        tfunc(P_r).view(-1, 1),
        tfunc(f_0**2).view(-1, 1),
    )
    bary_l = pykeops_class.barycentres(
        tfunc(g).view(-1, 1),
        tfunc(f).view(-1, 1),
        tfunc(X),
        tfunc(Y),
        tfunc(E),
        tfunc(S).view(-1, 1),
        tfunc(M).view(-1, 1),
        tfunc(P_l).view(-1, 1),
        tfunc(f_0**2).view(-1, 1),
    )

    assert np.isclose(
        bary_l.detach().numpy().reshape(-1, 1), bary_left.reshape(-1, 1)
    ).all()
    assert np.isclose(
        bary_r.detach().numpy().reshape(-1, 1), bary_right.reshape(-1, 1)
    ).all()


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
