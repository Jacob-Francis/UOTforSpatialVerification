"""
KeOps reductions for the unbalanced OT solver.

KeOps evaluates sums over all (i, j) pairs on the fly, without building the N x M
cost matrix, so memory stays O(N + M). Each formula string below is compiled by
KeOps the first time it is called and cached on disk afterwards.

Variable conventions (the same letters in every formula)
-------------------------------------------------------
    X = Vi(d)  points on the "i" side          Y = Vj(d)  points on the "j" side
    F = Vi(1)  potential on the i side         G = Vj(1)  potential on the j side
    S = Vi(1)  weights (density) on the i side M = Vj(1)  weights on the j side
    E = Pm(1)  epsilon                         C = Pm(1)  cost constant

Vi = one value per i point, Vj = one value per j point, Pm = a single parameter.
The cost used everywhere is (C / 2) * cost_string, e.g. (C / 2) |x - y|^2, and
the (scaled) transport plan is

    pi_ij = S_i M_j exp((F_i + G_j - (C/2) c(X_i, Y_j)) / E).

Which side is "i" is decided by the caller: the solver swaps the roles of source
and target when it needs a reduction over the other side (see
`UnbalancedSinkhorn._lse_pykeops` and `UnbalancedOT._barycentres_pykeops`).

Argument order: KeOps matches positional arguments to the aliases IN THE ORDER
they are declared. That is why, for example, `log_sum_exp` passes C before M.
"""
# NOTE: numpy, TorchNumpyProcessing, generic_logsumexp and cdist are imported but
# not used in this file (stage 2 clean-up).
import torch
import numpy as np
from torchnumpyprocess import TorchNumpyProcessing
from pykeops.torch import generic_logsumexp, generic_sum, Genred
from scipy.spatial.distance import cdist


class PyKeOpsFormulas:
    """
    Pre-built KeOps reductions for one cost function.

    Parameters
    ----------
    cost_string : str, optional
        KeOps expression for the cost c(X, Y) WITHOUT the 1/2 and C factors, by
        default "SqDist(X, Y)" (squared Euclidean). `CostClass` passes a different
        string for the periodic cost.
    space_dim : int, optional
        Dimension d of the points, by default 2. Use 3 for 3D point clouds.
    """

    def __init__(self, cost_string="SqDist(X, Y)", space_dim=2):
        self.cost_string = cost_string
        self.space_dim = space_dim
        # Periodic = '(Min(Concat(SqDist(Elem(X, 0) - IntCst({}), Elem(Y, 0)), Concat(SqDist(Elem(X, 0) + IntCst({}), Elem(Y, 0)), SqDist(Elem(X, 0), Elem(Y, 0)))) + SqDist(Elem(X, 1), Elem(Y, 1))) / IntCst(2))'.format(int(L), int(L))  # noqa E501
        # beta plane

        # Generate all the necessary pykeops fromulas/functions with the correct strings

        # Stable log-sum-exp over j, one result per i. "Max_SumShiftExpWeight"
        # returns two numbers per i:
        #     m_i = max_j V_ij,   s_i = sum_j M_j exp(V_ij - m_i),
        # with V_ij = (F_i + G_j - (C/2) c_ij) / E. Then
        #     log sum_j M_j exp(V_ij) = m_i + log(s_i)
        # without overflow. See `log_sum_exp` below. Aliases order: G, F, X, Y, E, C, M.
        self.log_sum_exp_max_shift_weight = Genred(
            f"((F + G - IntInv(2)*C*{self.cost_string})/E)",
            [
                "G = Vj(1)",  # Uni: 1 scalar per line
                "F = Vi(1)",  # Geo: 1 scalar per line
                f"X = Vi({space_dim})",  # Geo: 2-dim
                f"Y = Vj({space_dim})",  # Uni: 1 scalar per line
                "E = Pm(1)",  # parameter: 1 scalar per line
                "C = Pm(1)",
                "M = Vj(1)",
            ],
            reduction_op="Max_SumShiftExpWeight",
            axis=1,
            formula2="M",
        )
        # --------------- Dual cost term -----------------
        # sum over i, one value per j:  sum_i (pi_ij / (S_i M_j) - 1) S_i M_j
        # = sum_i (pi_ij - S_i M_j). Summed over j by `dual_energy_kl_transform`,
        # this is  mass(pi) - mass(alpha) * mass(beta)  =  <pi - 1, alpha x beta>.
        # Aliases order (also for the generic_sums below): G, F, X, Y, E, S, M, C.
        self._dual_energy_kl_transform = generic_sum(
            f"((Exp((F + G - IntInv(2)*C*{self.cost_string})/E) - IntCst(1))*S*M)",
            "f = Vj(1)",  # Geo: 1 scalar per line
            "G = Vj(1)",  # Uni: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            f"X = Vi({space_dim})",  # Geo: 2-dim
            f"Y = Vj({space_dim})",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",  # Geo
            "M = Vj(1)",
            "C = Pm(1)",
        )
        # -----------Primal energy ----------------
        # Transport cost: sum_i pi_ij * (C/2) c_ij, one value per j. Summed over j
        # by `cost_pi`, this is <c, pi>.
        self._cost_pi = generic_sum(
            f"(Exp((F + G - IntInv(2)*C*{self.cost_string})/E)*S*M*IntInv(2)*C*{self.cost_string})",
            "f = Vj(1)",  # output
            "G = Vj(1)",  # Uni: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            f"X = Vi({space_dim})",  # Geo: 2-dim
            f"Y = Vj({space_dim})",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",  # Geo
            "M = Vj(1)",
            "C = Pm(1)",
        )
        # Entropic term KL(pi | S x M) = sum pi log(pi / (S M)) - pi + S M.
        # Since pi / (S M) = exp(V), log(pi / (S M)) = V exactly, so no log or
        # division is needed: this is safe where S or M is zero.
        self._pi_log_term = generic_sum(
            f"(Exp((F + G - IntInv(2)*C*{self.cost_string})/E)*S*M*((F + G - IntInv(2)*C*{self.cost_string})/E - IntCst(1)) + S*M)",  # "(Exp((F + G - IntInv(2)*C*SqDist(X, Y))/E)*S*M * Log(Exp((F + G - IntInv(2)*C*SqDist(X, Y))/E)))",
            "f = Vj(1)",  # Geo: 1 scalar per line
            "G = Vj(1)",  # Uni: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            f"X = Vi({space_dim})",  # Geo: 2-dim
            f"Y = Vj({space_dim})",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",  # Geo
            "M = Vj(1)",
            "C = Pm(1)",
        )
        # --------------------------------------------------------------------------------------------------------------------------------
        # Marginals of pi

        # j-marginal: sum_i pi_ij, one value per j (output "f = Vj").
        self.marginal_j_keops = generic_sum(
            f"(Exp((F + G - IntInv(2)*C*{self.cost_string})/E)*S*M)",
            "f = Vj(1)",  # Geo: 1 scalar per line
            "G = Vj(1)",  # Uni: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            f"X = Vi({space_dim})",  # Geo: 2-dim
            f"Y = Vj({space_dim})",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",  # Geo
            "M = Vj(1)",
            "C = Pm(1)",
        )
        # i-marginal: sum_j pi_ij, one value per i (output "f = Vi"). Same formula;
        # only the output side differs.
        self.marginal_i_keops = generic_sum(
            f"(Exp((F + G - IntInv(2)*C*{self.cost_string})/E)*S*M)",
            "f = Vi(1)",  # Geo: 1 scalar per line
            "G = Vj(1)",  # Uni: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            f"X = Vi({space_dim})",  # Geo: 2-dim
            f"Y = Vj({space_dim})",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",  # Geo
            "M = Vj(1)",
            "C = Pm(1)",
        )
        # ----------------- Barycentres ---------------------------------------------------------------------------------------------------------------
        # Careful of log(density) as there may be zero's we don't demand postive support

        # NOTE: barycentres_top and barycentres_bottom are not used anywhere
        # (checked with grep). They are an older version without G and M;
        # `barycentres` below builds its own reduction instead.
        self.barycentres_top = generic_sum(
            f"(Exp((F - IntInv(2)*C*{self.cost_string})/E)*S*P )",
            "f = Vj(1)",  # Geo: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            f"X = Vi({space_dim})",  # Geo: 2-dim
            f"Y = Vj({space_dim})",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",
            "P = Vi(1)",
            "C = Pm(1)",
        )

        self.barycentres_bottom = generic_sum(
            f"(Exp((F - IntInv(2)*C*{self.cost_string})/E)*S)",
            "f = Vj(1)",  # Geo: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            f"X = Vi({space_dim})",  # Geo: 2-dim
            f"Y = Vj({space_dim})",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",
            "C = Pm(1)",
        )
        # Expected cost under the weights S: sum_i S_i (C/2) c(X_i, Y_j), one value
        # per j. Used as a starting guess for the potentials in epsilon annealing.
        self._starting_potentials = generic_sum(
            f"((IntInv(2)*C*{self.cost_string}) * S)",
            "f = Vj(1)",
            f"X = Vi({space_dim})",  # Geo: 2-dim
            f"Y = Vj({space_dim})",  # Uni: 1 scalar per line
            "S = Vi(1)",  # Geo
            "C = Pm(1)",
        )

    # NOTE: the default C=torch.tensor([1.0]) is a float32 CPU tensor, created
    # once when the module loads. Each method converts it with C.type_as(...).

    def starting_potentials(self, X, Y, S, C=torch.tensor([1.0])):
        """
        Expected cost sum_i S_i (C/2) c(X_i, Y_j) for each Y_j, shape (M, 1).

        NOTE: the only caller (epsilon annealing in `sinkhorn_algorithm`) does not
        pass C, so the solver's cost_const is ignored there.
        """
        return self._starting_potentials(X, Y, S, C.type_as(X)).view(-1, 1)

    def log_sum_exp(self, g, f, X, Y, E, M, C=torch.tensor([1.0])):
        """
        log sum_j M_j exp((f_i + g_j - (C/2) c(X_i, Y_j)) / E), one value per i.

        Despite the names, `g` is the j-side potential (alias G) and `f` the
        i-side one (alias F). `_lse_pykeops` passes the arguments so that the sum
        runs over the side being integrated out.

        Returns
        -------
        torch.Tensor
            Shape (number of X points, 1). -inf where every M_j is zero.
        """
        reduction = self.log_sum_exp_max_shift_weight(g, f, X, Y, E, C.type_as(f), M)
        # Column 0 is the max m_i, column 1 the shifted sum s_i.
        temp = reduction[:, 0] + torch.log(reduction[:, 1])
        return temp.view(-1, 1)

    def dual_energy_kl_transform(self, G, F, X, Y, E, S, M, C=torch.tensor([1.0])):
        """Scalar sum_ij (pi_ij - S_i M_j), the third term of the dual energy (before -eps)."""
        return self._dual_energy_kl_transform(G, F, X, Y, E, S, M, C.type_as(G)).sum()

    def cost_pi(self, G, F, X, Y, E, S, M, C=torch.tensor([1.0])):
        """Scalar transport cost <(C/2) c, pi>."""
        return self._cost_pi(G, F, X, Y, E, S, M, C.type_as(G)).sum()

    def primal_energy_kl_term(self, G, F, X, Y, E, S, M, C=torch.tensor([1.0])):
        """Scalar KL(pi | S x M), the entropic term of the primal energy (before eps)."""
        # mask = [R < 1e-16], need to think how to avoid this
        return self._pi_log_term(G, F, X, Y, E, S, M, C.type_as(F)).sum()

    def marginal_j(self, G, F, X, Y, E, S, M, C=torch.tensor([1.0])):
        """sum_i pi_ij for each j, shape (number of Y points, 1)."""
        return self.marginal_j_keops(G, F, X, Y, E, S, M, C.type_as(G))

    def marginal_i(self, G, F, X, Y, E, S, M, C=torch.tensor([1.0])):
        """sum_j pi_ij for each i, shape (number of X points, 1)."""
        return self.marginal_i_keops(G, F, X, Y, E, S, M, C.type_as(G))

    def barycentres(self, G, F, X, Y, E, S, M, P, C=torch.tensor([1.0])):
        """
        barycentre assuming G and P match in the first dimension, P can have a batch dimension otherwise

        Barycentric projection: for each i, the pi-weighted average of P over j,

            out_i = sum_j pi_ij P_j / sum_j pi_ij ,

        shape (number of X points, d). With P = Y this is the point that the mass
        at X_i is sent to, on average.

        NOTE: builds a new KeOps reduction on every call (because d can change).
        This is cheap: measured about 0.01 ms, and KeOps caches the compiled code.

        Parameters
        ----------
        P : torch.Tensor
            (number of Y points, d) values on the j side.
        """
        d = P.shape[-1]
        # Numerator: sum_j pi_ij P_j, output Vi(d). Aliases order: G, F, X, Y, E, S, M, C, P.
        ott_barycentres_top = generic_sum(
            f"(Exp((F + G - IntInv(2)*C*{self.cost_string})/E)*S*M*P )",
            "f = Vi(" + str(int(d)) + ")",  # Geo: 1 scalar per line
            "G = Vj(1)",  # Uni: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            "X = Vi(" + str(self.space_dim) + ")",  # Geo: 2-dim
            "Y = Vj(" + str(self.space_dim) + ")",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",  # Geo
            "M = Vj(1)",
            "C = Pm(1)",
            "P = Vj(" + str(int(d)) + ")",
        )

        # Divide by the i-marginal sum_j pi_ij.
        return ott_barycentres_top(
            G, F, X, Y, E, S, M, C.type_as(G), P
        ) / self.marginal_i_keops(G, F, X, Y, E, S, M, C.type_as(G))

    def pi_transport(self, G, F, X, Y, E, S, M, P, C=torch.tensor([1.0])):
        """
        barycentre assuming G and P match in the first dimension, P can have a batch dimension otherwise

        Apply the plan to P WITHOUT normalising: out_i = sum_j pi_ij P_j, shape
        (number of X points, d). This is the numerator of `barycentres`; the code
        is the same apart from the final division.
        """
        d = P.shape[-1]
        ott_barycentres_top = generic_sum(
            f"(Exp((F + G - IntInv(2)*C*{self.cost_string})/E)*S*M*P )",
            "f = Vi(" + str(int(d)) + ")",  # Geo: 1 scalar per line
            "G = Vj(1)",  # Uni: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            "X = Vi(" + str(self.space_dim) + ")",  # Geo: 2-dim
            "Y = Vj(" + str(self.space_dim) + ")",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",  # Geo
            "M = Vj(1)",
            "C = Pm(1)",
            "P = Vj(" + str(int(d)) + ")",
        )

        return ott_barycentres_top(
            G, F, X, Y, E, S, M, C.type_as(G), P
        )
