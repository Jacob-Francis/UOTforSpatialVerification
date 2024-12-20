from tensorisation import Tensorisation
import numpy as np
import torch
import pytest

torch.set_printoptions(precision=8)
torch.set_default_tensor_type(torch.DoubleTensor)

# Torch problem with no-member when there is
# pylint: disable=no-member


@pytest.mark.parametrize(
    "n1, n2, m1, m2, t_c",
    [
        (2, 2, 3, 3, "torch"),
        (5, 6, 7, 8, "torch"),
        (12, 11, 10, 9, "torch"),
        (6, 6, 6, 6, "np"),
        (2, 3, 1, 3, "np"),
    ],
)
def test_weight_call(n1, n2, m1, m2, t_c):
    cost = Tensorisation(set_fail=True)

    choice = torch.rand(1)
    if choice < 0.5:
        weights = torch.rand(m1, m2)
        dim = 1
    else:
        weights = torch.rand(n1, n2)
        dim = 0

    if t_c == "torch":
        test = cost(
            torch.linspace(0, 1, n1),
            torch.linspace(0, 1, n2),
            torch.linspace(0, 1, m1),
            torch.linspace(0, 1, m2),
            weight=weights,
        )
    elif t_c == "np":
        test = cost(
            np.linspace(0, 1, n1),
            np.linspace(0, 1, n2),
            np.linspace(0, 1, m1),
            np.linspace(0, 1, m2),
            weight=weights.numpy(),
        )

    # check result
    x_cloud = torch.cartesian_prod(
        torch.linspace(0, 1, n1), torch.linspace(0, 1, n2)
    ).type_as(test)
    y_cloud = torch.cartesian_prod(
        torch.linspace(0, 1, m1), torch.linspace(0, 1, m2)
    ).type_as(test)
    C_true = torch.cdist(x_cloud, y_cloud) ** 2
    C_true = C_true.type_as(test)
    weights = weights.type_as(test)
    try:
        sum_true = torch.matmul(torch.exp(C_true), weights.flatten())
    # Catch when shapes the other way around
    except RuntimeError:
        sum_true = torch.matmul(weights.flatten(), torch.exp(C_true))
    assert torch.norm(sum_true.view(-1, 1) - test.view(-1, 1)) < 1e-10


@pytest.mark.parametrize(
    "n1, n2, m1, m2, t_c",
    [
        (2, 2, 3, 3, "torch"),
        (5, 6, 7, 8, "torch"),
        (12, 11, 10, 9, "torch"),
        (6, 6, 6, 6, "np"),
        (2, 3, 1, 3, "np"),
    ],
)
def test_ones_call(n1, n2, m1, m2, t_c):
    cost = Tensorisation(set_fail=True)

    choice = torch.rand(1)
    if choice < 0.5:
        weights = torch.ones(m1, m2)
        dim = 1
    else:
        weights = torch.ones(n1, n2)
        dim = 0

    if t_c == "torch":
        test = cost(
            torch.linspace(0, 1, n1),
            torch.linspace(0, 1, n2),
            torch.linspace(0, 1, m1),
            torch.linspace(0, 1, m2),
            weight=weights,
        )
    elif t_c == "np":
        test = cost(
            np.linspace(0, 1, n1),
            np.linspace(0, 1, n2),
            np.linspace(0, 1, m1),
            np.linspace(0, 1, m2),
            weight=weights.numpy(),
        )

    # check result
    x_cloud = torch.cartesian_prod(
        torch.linspace(0, 1, n1), torch.linspace(0, 1, n2)
    ).type_as(test)
    y_cloud = torch.cartesian_prod(
        torch.linspace(0, 1, m1), torch.linspace(0, 1, m2)
    ).type_as(test)
    C_true = torch.cdist(x_cloud, y_cloud) ** 2
    C_true = C_true.type_as(test)
    weights = weights.type_as(test)
    try:
        sum_true = torch.matmul(torch.exp(C_true), weights.flatten())
    # Catch when shapes the other way around
    except RuntimeError:
        sum_true = torch.matmul(weights.flatten(), torch.exp(C_true))
    assert torch.norm(sum_true.view(-1, 1) - test.view(-1, 1)) < 1e-10


# pylint: enable=no-member

if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
