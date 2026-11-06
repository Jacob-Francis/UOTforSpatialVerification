"""
unbalancedsinkhorn: unbalanced optimal transport for spatial forecast verification.

Main classes (each builds on the one before):
    UnbalancedSinkhorn  - the Sinkhorn solver (potentials f, g)
    UnbalancedOT        - energies, marginals and transport vectors from the solution
    DebiasedUOT         - the Sinkhorn divergence S_eps(alpha, beta)

NOTE: `sys` and `os.path` are imported but not used (stage 2).
"""
import sys, os.path
from .utils import pbcost_cdist
from .unbalancedsinkhorn import UnbalancedSinkhorn
from .unbalancedot import UnbalancedOT
from .debiased_uot import DebiasedUOT
from .pykeops_formula import PyKeOpsFormulas
from .costs import CostClass
from .uot_warning import ConvergenceWarning