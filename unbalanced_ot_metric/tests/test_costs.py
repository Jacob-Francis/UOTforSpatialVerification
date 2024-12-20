import numpy as np
from scipy.spatial.distance import cdist
import torch
import pytest
from unbalancedsinkhorn import UnbalancedSinkhorn, pbcost_cdist

torch.set_printoptions(precision=8)

# Testing Balanced implementation against numpy
# Non-tensorised ==> pykeops


@pytest.mark.parametrize(
    "n1, n2, m1, m2, L",
    [
        (2, 2, 3, 3, 0.9),
        (5, 6, 7, 8, 3.5),
        (12, 11, 10, 9, 6.0),
        (6, 6, 6, 6, 10.0),
        (2, 3, 1, 3, 5.0),
    ],
)  # noqa: E501
def test_flat_pbcost(n1, n2, m1, m2, L):
    N, M = n1 * n2, m1 * m2
    X = torch.cartesian_prod(torch.linspace(0, L, n1), torch.linspace(0, L, n2)).type(torch.DoubleTensor)
    Y = torch.cartesian_prod(torch.linspace(0, L, m1), torch.linspace(0, L, m2)).type(torch.DoubleTensor)
    
    C, ind = pbcost_cdist(X, Y, L)

    # Full expression:
    C1 = torch.cdist(X[:,0].view(-1, 1),Y[:,0].view(-1, 1))**2
    C1_plusL = torch.cdist(X[:,0].view(-1, 1)+L,Y[:,0].view(-1, 1))**2
    C1_minusL = torch.cdist(X[:,0].view(-1, 1)-L,Y[:,0].view(-1, 1))**2
    C2 = torch.cdist(X[:,1].view(-1, 1),Y[:,1].view(-1, 1))**2
    val,ind1 = torch.min(torch.stack((C1_minusL, C1,C1_plusL), dim=0),dim=0)

    assert (ind1 == ind).all()
    assert torch.isclose((val+C2),C, atol=1e-12, rtol=1e-12).all()

if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
