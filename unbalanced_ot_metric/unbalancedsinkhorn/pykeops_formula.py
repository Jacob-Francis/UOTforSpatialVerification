import torch
import numpy as np
from torchnumpyprocess import TorchNumpyProcessing
from pykeops.torch import generic_logsumexp, generic_sum, Genred
from scipy.spatial.distance import cdist


class PyKeOpsFormulas:
    def __init__(self, cost_string="SqDist(X, Y)"):
        self.cost_string = cost_string

        # Periodic = '(Min(Concat(SqDist(Elem(X, 0) - IntCst({}), Elem(Y, 0)), Concat(SqDist(Elem(X, 0) + IntCst({}), Elem(Y, 0)), SqDist(Elem(X, 0), Elem(Y, 0)))) + SqDist(Elem(X, 1), Elem(Y, 1))) / IntCst(2))'.format(int(L), int(L))  # noqa E501
        # beta plane

        # Generate all the necessary pykeops fromulas/functions with the correct strings

        self.log_sum_exp_max_shift_weight = Genred(
            f"((F + G - IntInv(2)*C*{self.cost_string})/E)",
            [
                "G = Vj(1)",  # Uni: 1 scalar per line
                "F = Vi(1)",  # Geo: 1 scalar per line
                "X = Vi(2)",  # Geo: 2-dim
                "Y = Vj(2)",  # Uni: 1 scalar per line
                "E = Pm(1)",  # parameter: 1 scalar per line
                "C = Pm(1)",
                "M = Vj(1)",
            ],
            reduction_op="Max_SumShiftExpWeight",
            axis=1,
            formula2="M",
        )
        # --------------- Dual cost term -----------------
        self._dual_energy_kl_transform = generic_sum(
            f"((Exp((F + G - IntInv(2)*C*{self.cost_string})/E) - IntCst(1))*S*M)",
            "f = Vj(1)",  # Geo: 1 scalar per line
            "G = Vj(1)",  # Uni: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            "X = Vi(2)",  # Geo: 2-dim
            "Y = Vj(2)",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",  # Geo
            "M = Vj(1)",
            "C = Pm(1)",
        )
        # -----------Primal energy ----------------
        self._cost_pi = generic_sum(
            f"(Exp((F + G - IntInv(2)*C*{self.cost_string})/E)*S*M*IntInv(2)*C*{self.cost_string})",
            "f = Vj(1)",  # output
            "G = Vj(1)",  # Uni: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            "X = Vi(2)",  # Geo: 2-dim
            "Y = Vj(2)",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",  # Geo
            "M = Vj(1)",
            "C = Pm(1)",
        )
        self._pi_log_term = generic_sum(
            f"(Exp((F + G - IntInv(2)*C*{self.cost_string})/E)*S*M*((F + G - IntInv(2)*C*{self.cost_string})/E - IntCst(1)) + S*M)",  # "(Exp((F + G - IntInv(2)*C*SqDist(X, Y))/E)*S*M * Log(Exp((F + G - IntInv(2)*C*SqDist(X, Y))/E)))",
            "f = Vj(1)",  # Geo: 1 scalar per line
            "G = Vj(1)",  # Uni: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            "X = Vi(2)",  # Geo: 2-dim
            "Y = Vj(2)",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",  # Geo
            "M = Vj(1)",
            "C = Pm(1)",
        )
        # --------------------------------------------------------------------------------------------------------------------------------
        # Marginals of pi

        self.marginal_j_keops = generic_sum(
            f"(Exp((F + G - IntInv(2)*C*{self.cost_string})/E)*S*M)",
            "f = Vj(1)",  # Geo: 1 scalar per line
            "G = Vj(1)",  # Uni: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            "X = Vi(2)",  # Geo: 2-dim
            "Y = Vj(2)",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",  # Geo
            "M = Vj(1)",
            "C = Pm(1)",
        )
        self.marginal_i_keops = generic_sum(
            f"(Exp((F + G - IntInv(2)*C*{self.cost_string})/E)*S*M)",
            "f = Vi(1)",  # Geo: 1 scalar per line
            "G = Vj(1)",  # Uni: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            "X = Vi(2)",  # Geo: 2-dim
            "Y = Vj(2)",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",  # Geo
            "M = Vj(1)",
            "C = Pm(1)",
        )
        # ----------------- Barycentres ---------------------------------------------------------------------------------------------------------------
        # Careful of log(density) as there may be zero's we don't demand postive support

        self.barycentres_top = generic_sum(
            f"(Exp((F - IntInv(2)*C*{self.cost_string})/E)*S*P )",
            "f = Vj(1)",  # Geo: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            "X = Vi(2)",  # Geo: 2-dim
            "Y = Vj(2)",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",
            "P = Vi(1)",
            "C = Pm(1)",
        )

        self.barycentres_bottom = generic_sum(
            f"(Exp((F - IntInv(2)*C*{self.cost_string})/E)*S)",
            "f = Vj(1)",  # Geo: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            "X = Vi(2)",  # Geo: 2-dim
            "Y = Vj(2)",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",
            "C = Pm(1)",
        )
        self._starting_potentials = generic_sum(
            f"((IntInv(2)*C*{self.cost_string}) * S)",
            "f = Vj(1)",
            "X = Vi(2)",  # Geo: 2-dim
            "Y = Vj(2)",  # Uni: 1 scalar per line
            "S = Vi(1)",  # Geo
            "C = Pm(1)",
        )
    
    def starting_potentials(self, X, Y, S, C=torch.tensor([1.0])):
        return self._starting_potentials(X, Y, S, C.type_as(X)).view(-1, 1)

    def log_sum_exp(self, g, f, X, Y, E, M, C=torch.tensor([1.0])):
        reduction = self.log_sum_exp_max_shift_weight(g, f, X, Y, E, C.type_as(f), M)
        temp = reduction[:, 0] + torch.log(reduction[:, 1])
        return temp.view(-1, 1)

    def dual_energy_kl_transform(self, G, F, X, Y, E, S, M, C=torch.tensor([1.0])):
        return self._dual_energy_kl_transform(G, F, X, Y, E, S, M, C.type_as(G)).sum()

    def cost_pi(self, G, F, X, Y, E, S, M, C=torch.tensor([1.0])):
        return self._cost_pi(G, F, X, Y, E, S, M, C.type_as(G)).sum()

    def primal_energy_kl_term(self, G, F, X, Y, E, S, M, C=torch.tensor([1.0])):
        # mask = [R < 1e-16], need to think how to avoid this
        return self._pi_log_term(G, F, X, Y, E, S, M, C.type_as(F)).sum()

    def marginal_j(self, G, F, X, Y, E, S, M, C=torch.tensor([1.0])):
        return self.marginal_j_keops(G, F, X, Y, E, S, M, C.type_as(G))

    def marginal_i(self, G, F, X, Y, E, S, M, C=torch.tensor([1.0])):
        return self.marginal_i_keops(G, F, X, Y, E, S, M, C.type_as(G))

    def barycentres(self, G, F, X, Y, E, S, M, P, C=torch.tensor([1.0])):
        """
        barycentre assuming G and P match in the first dimension, P can have a batch dimension otherwise
        """
        d = P.shape[-1]
        ott_barycentres_top = generic_sum(
            f"(Exp((F + G - IntInv(2)*C*{self.cost_string})/E)*S*M*P )",
            "f = Vi(" + str(int(d)) + ")",  # Geo: 1 scalar per line
            "G = Vj(1)",  # Uni: 1 scalar per line
            "F = Vi(1)",  # Geo: 1 scalar per line
            "X = Vi(2)",  # Geo: 2-dim
            "Y = Vj(2)",  # Uni: 1 scalar per line
            "E = Pm(1)",  # parameter: 1 scalar per line
            "S = Vi(1)",  # Geo
            "M = Vj(1)",
            "C = Pm(1)",
            "P = Vj(" + str(int(d)) + ")",
        )

        return ott_barycentres_top(
            G, F, X, Y, E, S, M, C.type_as(G), P
        ) / self.marginal_i_keops(G, F, X, Y, E, S, M, C.type_as(G))

            # --------------------------------------------------------------------------------------------------------------------------------
        # Marginals of pi

    # def barycentres(F, X, Y, E, S, P, C=torch.tensor([1.0])):
    #     '''
    #     Calculates barycentric mapping (Gibbs esque) with output in opposite potential to input F
    #     i.e.
    #     bary_i inputs (F_i, X_i, Y_j, E, S_i, P_i)
    #     bary_j inputs (G_j, Y_j, X_i, E, S_j, P_j)
    #     '''

    #     d = P.shape[-1]

    #     barycentres_top = generic_sum("(Exp((F + G - IntInv(2)*C*SqDist(X, Y))/E)*S*M*P )",
    #                           'f = Vi('+str(int(d))+')',   # Geo: 1 scalar per line
    #                           'G = Vj(1)',   # Uni: 1 scalar per line
    #                           'F = Vi(1)',   # Geo: 1 scalar per line
    #                           'X = Vi(2)',   # Geo: 2-dim
    #                           'Y = Vj(2)',   # Uni: 1 scalar per line
    #                           'E = Pm(1)',   # parameter: 1 scalar per line
    #                           'S = Vi(1)',   # Geo
    #                           'M = Vj(1)',
    #                           'C = Pm(1)',
    #                           'P = Vj('+str(int(d))+')'
    #                           )

    #     return barycentres_top(F, X, Y, E, S, P, C.type_as(F)) / barycentres_bottom(F, X, Y, E, S, C.type_as(F))

    # could have it having varying dimensions?
