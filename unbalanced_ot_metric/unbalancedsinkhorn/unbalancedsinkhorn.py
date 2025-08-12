"""
Main code for Sinkhorn algorithm implementation, given any *divergence class.
"""
import torch
from . pykeops_formula import PyKeOpsFormulas
from .utils import pbcost_cdist, exponential_sequence
from tensorisation import Tensorisation
from .tvdivergence import TVDivergence
from .kldivergence import KLDivergence
from .balanceddivergence import BalancedDivergence
from torchnumpyprocess import DetachNumpy
from pickle import dump
import os
from flipflop import FlipFlop
from .costs import CostClass
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .uot_warning import ConvergenceWarning
import warnings

# For any print statements
torch.set_printoptions(precision=15)

# Sinkhorn Algorithm: Sink((αi)i,(xi)i,(βj)j,(yj)j)
# Parameters :symmetric cost function C(x,y), regularizationε >0
# Input:sourceα=∑Ni=1αiδxi, targetβ=∑Mj=1βjδyj
# Output   :vectors (fi)iand (gj)j, equal to the optimal potentials
# 1)fi←zeros(M)  ;gj←zeros(N)
# 2)whileupdates>toldo
# 3)gj←−εLSENi=1[log(αi) + (fi−C(xi,yj))/ε]
# 4)gj←−aproxεφ∗(−gj)
# 5)fi←−εLSEMj=1[log(βj) + (gj−C(xi,yj))/ε]
# 6)fi←−aproxεφ∗(−fi)
# 7) return, fi, gj

# pylint: disable=no-member
class UnbalancedSinkhorn(CostClass):
    """
    Main code for Sinkhorn algorithm implementation, given any *divergence class.
    Process in order: 'self.parameters' --> 'self.densities' --> 'self.sinkhorn_algorithm' ..."
    Order of importance/efficiency:
    - Tensorisation
    - pykeops
    - neither [storing full cost matrix]
    """

    def __init__(self, set_fail=False, cuda_device=None, pykeops=False, debias=False):
        """
        __inti__

        Parameters
        ----------
        set_fail : bool, optional
            Passed to TorchNumpyProcess for device configuration, by default False. If True,
            device=='cpu'. else detect what is available.
        pykeops : bool, optional
            If True test if pykeops is available and use provided tensorisation isn't an available
            approach, by default False.
        debias : bool, optional
            If True compute a debiasing term, by default False. Note debiasing is class wise and is
            used to calculate potential which are debiased. Not extra debiasing potentials for the
            usual case.
        """

        super().__init__(set_fail, cuda_device)

        self.debias = debias

        if pykeops:
            try:
                import pykeops

                self.pykeops = True
            except ImportError:
                self.pykeops = False
        else:
            self.pykeops = False

    def parameters(self, epsilon=0.5, rho=1.0, cost_const=1.0):
        """Load in parameters of the OT problem

        Parameters
        ----------
        epsilon : float
            the entropy regularisation parameter, in geomloss blur^p=epsilon
        rho : float
            the unbalanced constraint parameter, in geomloss reach^p=rho
        cost_const : int, optional
            Constant in-front of cost term i.e <cost_cont*c(x,y), pi(x,y)> , by default 1
        """

        # Processes to correct device, torch and shape
        # pylint: disable=attribute-defined-outside-init
        self.epsilon = self._torch_numpy_process(epsilon)
        self.rho = self._torch_numpy_process(rho)
        self.cost_const = self._torch_numpy_process(cost_const).view(-1, 1)
        # pylint: enable=attribute-defined-outside-init

    def densities(
        self, source_points, target_points, source_density=None, target_density=None, **kwargs
    ):
        """
        Load densities into the class as torch tensors for speed up.
        Generate uniform densities if none is given, else use given densities
        for point cloud. Set the geometry of the domain.

        If given two tuples we assume we can tensorise and the tuples contain two lists which define the regular mesh.
        If given two 3 dimension arrays like (N,M,2) then this also defines a regular 2D mesh which can be tensorised.
        Else, if we can use pykeops we shall. Otherwise we need to compute the whole cost matrix between all points across domains.

        Parameters
        ----------
        source_points : (N, 2), (n1, n2, 2), ((n1), (n2))
            source 2D points, this is the mesh. Not weights.
        target_points : (M, 2), (m1, m2, 2), ((m1), (m2))
            target 2D points, this is the mesh. Not weights.
        source_density : (N) (n1, n2), optional
            weights associated to the 2D source points, by default None
        target_density : (N) (m1, m2), optional
            weights associated to the 2D target points, by default None
        kwargs; 
            cost_type: str
                'rigid', 'periodic'  ToDo: [ 'beta', 'beta_periodic']. Need to pass kwarg
            L: int
                 for periodic case 
        """
        global n1, n2, m1, m2
        # Process arguments depending on input, determine whether to tensorise.
        # Note n2, m2 will be one if given no-regular points.
        n1, n2 = self._process_points(source_points)
        m1, m2 = self._process_points(target_points)

        self.process_initial_point_clouds(
            source_points,
            target_points,
            self.cost_const,
            n1,
            n2,
            m1,
            m2,
            **kwargs
        )
        
        # Process associated densities - could be a utils or in torch_numpy... class
        self.α_s = self._process_inputs(source_density, n1, n2)
        self.β_t = self._process_inputs(target_density, m1, m2)

        if self.α_s.sum() == 0 or self.β_t.sum() == 0:
            raise ValueError(
                f"Source or target density is zero - only the zero plan is feasible.\n"
                f"TV Sinkhorn divergence is defined as {self.α_s.sum() + self.β_t.sum()}.\n"
                f"KL Sinkhorn divergence is defined as 0.\n"
                f"Transport vectors are not defined."
            )
                            

        if not self.α_s.is_contiguous():
            self.α_s = self.α_s.contiguous()
        if not self.β_t.is_contiguous():
            self.β_t = self.β_t.contiguous()
        # pylint: enable=attribute-defined-outside-init

    # ------------- Main Sinkhorn Loops ------------
    def _lse(self, dim):
        """
        Typical log-sum-exp calculation with hard coded cost matrix calculated.

        Parameters
        ----------
        dim : int
            If 0 then update g, if 1 then update f
        """
        if self.debias:
            if dim == 0:  # g update
                self.g += (
                    -self.epsilon
                    * torch.logsumexp(
                        (self.f + self.g.T - self.cost) / self.epsilon
                        + torch.log(self.α_s),
                        dim,
                    ).view(-1, 1)
                ) / 2
            elif dim == 1:  # f update
                self.f += (
                    -self.epsilon
                    * torch.logsumexp(
                        (self.f + self.g.T - self.cost) / self.epsilon
                        + torch.log(self.β_t.T),
                        dim,
                    ).view(-1, 1)
                ) / 2
    
        else:
            if dim == 0:  # g update
                self.g += -self.epsilon * torch.logsumexp(
                    (self.f + self.g.T - self.cost) / self.epsilon + torch.log(self.α_s),
                    dim,
                ).view(-1, 1)
            elif dim == 1:  # f update
                self.f += -self.epsilon * torch.logsumexp(
                    (self.f + self.g.T - self.cost) / self.epsilon + torch.log(self.β_t.T),
                    dim,
                ).view(-1, 1)

    def _lse_pykeops(self, dim):
        """
        Pykeops log-sum-exp calculation on the fly

        Parameters
        ----------
        dim : int
            If 0 then update g, if 1 then update f

        """
        if self.debias:
            if dim == 0:  # g update
                self.g += (
                    -self.epsilon.view(-1, 1)
                    * self.pykeops_formulas.log_sum_exp(
                        self.f,
                        self.g,
                        self.Y_t,
                        self.X_s,
                        self.epsilon.view(-1, 1),
                        self.α_s,
                        self.cost_const,
                    )
                ) / 2
            elif dim == 1:  # f update
                self.f += (
                    -self.epsilon.view(-1, 1)
                    * self.pykeops_formulas.log_sum_exp(
                        self.g,
                        self.f,
                        self.X_s,
                        self.Y_t,
                        self.epsilon.view(-1, 1),
                        self.β_t,
                        self.cost_const,
                    )
                ) / 2
        else:
            if dim == 0:  # g update
                self.g += -self.epsilon.view(-1, 1) * self.pykeops_formulas.log_sum_exp(
                    self.f,
                    self.g,
                    self.Y_t,
                    self.X_s,
                    self.epsilon.view(-1, 1),
                    self.α_s,
                    self.cost_const,
                )
            elif dim == 1:  # f update
                self.f += -self.epsilon.view(-1, 1) * self.pykeops_formulas.log_sum_exp(
                    self.g,
                    self.f,
                    self.X_s,
                    self.Y_t,
                    self.epsilon.view(-1, 1),
                    self.β_t,
                    self.cost_const,
                )

    def _lse_tensorised(self, dim):
        """
        Tensorised log-sum-exp calculation for regular meshed problems

        Parameters
        ----------
        dim : int
            If 0 then update g, if 1 then update f
        """
        # Balanced debias available only
        if self.debias:
            if dim == 1:  # f update
                self.f -= self.epsilon * torch.log(
                    self.tensorise_f(
                        torch.exp(-self.cost_1 / self.epsilon),
                        torch.exp(-self.cost_2 / self.epsilon),
                        torch.exp(self.g / self.epsilon + torch.log(self.β_t)),
                    )
                )
                self.f *= 0.5
            elif dim == 0:  # g update
                self.g -= self.epsilon * torch.log(
                    self.tensorise_f(
                        torch.exp(-self.cost_1 / self.epsilon),
                        torch.exp(-self.cost_2 / self.epsilon),
                        torch.exp(self.f / self.epsilon + torch.log(self.α_s)),
                    )
                )
                self.g *= 0.5
        else:
            if dim == 1:  # f update
                torch.log(
                    self.tensorise_f(
                        torch.exp(-self.cost_1 / self.epsilon),
                        torch.exp(-self.cost_2 / self.epsilon),
                        torch.exp(self.g / self.epsilon + torch.log(self.β_t)),
                    ),
                    out=self.f,
                )
                self.f *= -self.epsilon

            elif dim == 0:  # g update
                torch.log(
                    self.tensorise_f(
                        torch.exp(-self.cost_1 / self.epsilon),
                        torch.exp(-self.cost_2 / self.epsilon),
                        torch.exp(self.f / self.epsilon + torch.log(self.α_s)),
                    ),
                    out=self.g,
                )
                self.g *= -self.epsilon

    def _aprox_sinkhorn_loop(self):
        """
        Loop running the algorithm and checking for nans/infs

        ToDo: Look into running the updates simultaneously/in parallel

        Raises
        ------
        RuntimeWarning
            "sinkhorn loop nans ; over/underflow"
        RuntimeWarning
            sinkhorn loop infs ; over/underflow"
        """
        if self.debias:
            # The potentials are the same hence only one update needed
            self.left_div.sinkhorn_iterate(self.f)
        else:
            self.left_div.sinkhorn_iterate(self.f)
            self.right_div.sinkhorn_iterate(self.g)
                    # Run the parallel computations
            # with ThreadPoolExecutor() as executor:
            #     future_f = executor.submit(self.left_div.sinkhorn_iterate, self.f)
            #     future_g = executor.submit(self.right_div.sinkhorn_iterate, self.g)

            #     future_f.result()
            #     future_g.result()

        # check for nans
        if torch.isnan(self.f).any() or torch.isnan(self.g).any():
            raise RuntimeWarning("sinkhorn loop nans ; over/underflow")
        if torch.isinf(self.f).any() or torch.isinf(self.g).any():
            raise RuntimeWarning("sinkhorn loop infs ; over/underflow")

    # ----------------- Looping call ---------------------------------
    def sinkhorn_algorithm(
        self,
        sinkhorn_steps=None,
        reinitialise=True,
        tol=1e-15,
        f0=None,
        g0=None,
        verbose=False,
        left_divergence=None,
        right_divergence=None,
        aprox=None,
        convergence_checks=None,
        convergence_data=False,
        convergence_data_title="",
        f0_const=0,
        g0_const=0,
        convergence_repeats=1,
        epsilon_annealing=False,
        epsilon_annealing_const=0.99,
        convergence_or_fail=False
    ):
        """Note we overwrite f0, g0"""

        # Convergence checking;
        if sinkhorn_steps is None:
            sinkhorn_steps = max(
                [100, int(-1.1 / self.epsilon * torch.log(self.epsilon))]
            )
        convergence_checks = self.convergence_intialisation(convergence_checks, sinkhorn_steps, convergence_data)

        # To Do: Refactor this so you can iterate the sinkhorn_aglorithm without re-intialising every time
        if reinitialise:
            self.sinkhorn_algorithm_initialisation(
                left_divergence,
                right_divergence,
                aprox,
                f0,
                f0_const,
                g0,
                g0_const
            )

        i_super = 0

        # parameters will nee be parsing, assuming a unit box atm
        if epsilon_annealing:
            if not all(self.tensorise) and self.pykeops:
                self.f = self.pykeops_formulas.starting_potentials(self.Y_t, self.X_s, self.β_t)
                self.g = self.pykeops_formulas.starting_potentials(self.X_s, self.Y_t, self.α_s)

            scale_list = exponential_sequence(torch.sqrt(torch.Tensor([2])), epsilon_annealing_const, torch.sqrt(self.epsilon).cpu())
            store_original_epsilon = self.epsilon.clone()
            for s in scale_list:
                err = torch.inf
                i = 0
                self.epsilon = self._torch_numpy_process(s**2)
                max_its = int(-1 / self.epsilon * torch.log(self.epsilon))
                
                while (i < max_its) and (err > tol) and (max_its >= 1):
                    temp_f, temp_g = self.f.clone(), self.g.clone()

                    # Update loop
                    self._aprox_sinkhorn_loop()
                    
                    # Convergence check in potentials (psuedo?) residual
                    # pylint: disable-next=not-callable
                    f_update = torch.linalg.norm(temp_f - self.f, ord=float("inf"))
                    # pylint: disable-next=not-callable
                    g_update = torch.linalg.norm(temp_g - self.g, ord=float("inf"))
                    err = max(f_update, g_update)

                    i += 1
                    i_super += 1
                
            print(f'Epsilon Annealing, final esp={s**2}, err={err}, its={i}/{max_its}')
            self.epsilon = store_original_epsilon

        # ############ Tru Epsilon loop

        i = 0
        count = 1
        ff = FlipFlop(6, 3)  # look three osciallting numbers?
        while i < sinkhorn_steps:

            # Convergence check
            # Check last to save convergence
            if (i % (sinkhorn_steps // convergence_checks) == 0) or (
                i == sinkhorn_steps - 1
            ):
                if verbose:
                    print(f"-------- Convergence check at {i} -------------")
                temp_f, temp_g = self.f.clone(), self.g.clone()

            # Update loop
            self._aprox_sinkhorn_loop()

            if (i % (sinkhorn_steps // convergence_checks) == 0) or (
                i == sinkhorn_steps - 1
            ):
                # Convergence check in potentials (psuedo?) residual
                # pylint: disable-next=not-callable
                f_update = torch.linalg.norm(temp_f - self.f, ord=float("inf"))
                # pylint: disable-next=not-callable
                g_update = torch.linalg.norm(temp_g - self.g, ord=float("inf"))
                err = max(f_update, g_update)

                if convergence_data:
                    self._update_convergence_dict(f_update, g_update)

                # Add next res and check
                if ff.oscialltion(err):
                    if verbose:
                        print("!!!!! oscillating sinkhorn updates : !!!!!!!!")
                        print(f"updates size = {err}", i)
                    
                    if convergence_or_fail:
                        # ToDo correct these warnings
                        warnings.warn(f"Sinkhorn oscillating and failed to converge in {sinkhorn_steps*(count-1)} iteration| err={err}", ConvergenceWarning)
                        raise RuntimeWarning
                    break

                if verbose:
                    print(f"Iteration {i}, error ", {err})

                if err < tol:
                    if self.debias:
                        print(
                            f"Convergence in debiased f, g updates below {tol} in {i} iterations"
                        )
                    else:
                        print(
                            f"Convergence in f, g updates below {tol} in {i} iterations"
                        )
                    break

                if i == sinkhorn_steps - 1 and err > tol:
                    # restart
                    count += 1
                    if verbose:
                        print(
                            f"!!!!!!!!!!Sinkhorn didnt converge, err={err} | total its: {sinkhorn_steps*count}!!!!!!!!"
                        )
                    i = 0
                    del ff  # Start looking for oscillations
                    ff = FlipFlop(4, 2)  # reduce oscillating looking for

                if count == convergence_repeats + 1:
                    if convergence_or_fail:
                        # ToDo correct these warnings
                        warnings.warn(f"Sinkhorn did not converge in {sinkhorn_steps*(count-1)} iteration| err={err}", ConvergenceWarning)
                        raise RuntimeWarning
                    break

            i += 1
            i_super += 1

        if convergence_data:
            self._dictionary_dump(convergence_data_title)

        return f_update, g_update, i_super

    # ------------------ Running function for the algorithm ----------------------
    def sinkhorn_algorithm_initialisation(
        self,
        left_divergence,
        right_divergence,
        aprox,
        f0,
        f0_const,
        g0,
        g0_const
    ):
        """

        Initialisation function before running the actual sinkhorn loop

        Parameters
        ----------
        left_divergence : class *divergence
            divergence applied to left (source) field of unbalanced ot
        right_divergence : class *divergence
            divergence applied to left (target) field of unbalanced ot
        aprox : str
            Instead of giving the left/right classes, one can instead give
            aprox \in ["kl", "tv", "balanced"]. Which used this on both marginals
        f0 : array
            initial potential f
        f0_const : float
            If f0 is None, initialise array of ones*f0_const
        g0 : array
            initial potential g
        g0_const : float
            If g0 is None, initialise array of ones*g0_const
        convergence_checks : int
            int < sinkhorn_steps, the number of convergence checks to perform as we loop
        sinkhorn_steps : int
            The number of iterations for the sinkhorn algorithm
        convergence_data : bool
            If true, initialise a dictionary tracking convergence information as we loop

        Returns
        -------
        int
            convergence checks

        Raises
        ------
        ValueError
            "Expected aprox one of ['kl', 'balanced', 'tv']"
        ValueError
            "Attempting to debias on a non-symmetric cost matrix, check density inputs"
        """

        # pylint disable=attribute-defined-outside-init
        if left_divergence is None and right_divergence is None:
            if aprox in ["kl", "tv", "balanced"]:
                left_divergence = right_divergence = aprox
            else:
                raise ValueError("Expected aprox one of ['kl', 'balanced', 'tv']")

        if left_divergence in ["kl", "tv", "balanced"]:
            self.left_div = self._process_divergences(left_divergence, side="left")
        else:
            self.left_div = self._process_divergences(left_divergence)

        if right_divergence in ["kl", "tv", "balanced"]:
            self.right_div = self._process_divergences(right_divergence, side="right")
        else:
            self.right_div = self._process_divergences(right_divergence)

        # Initialise potentials as zeros or one times constant or recycling them as
        # typically a good starting guess on given cost matrix
        # ToDO f0/g0_const should have a None and default value
        if hasattr(self, "f") and f0 is None:
            pass
        else:
            self.f = self._process_inputs(f0, *self.α_s.shape, const=f0_const)

        if hasattr(self, "g") and g0 is None:
            pass
        else:
            self.g = self._process_inputs(g0, *self.β_t.shape, const=g0_const)

        if self.debias:
            self.f = self.g
            try:
                if not (self.X_s == self.Y_t).all():
                    raise ValueError(
                        "Attempting to debias on a non-symmetric cost matrix, check density inputs"
                    )
            except AttributeError:  # Error from tuple input
                if not (self.X_s == self.Y_t):
                    # pylint: disable-next=raise-missing-from
                    raise ValueError(
                        "Attempting to debias on a non-symmetric cost matrix, check density inputs"
                    )

        # pylint enable=attribute-defined-outside-init

    def convergence_intialisation(self, convergence_checks, sinkhorn_steps, convergence_data):
        if convergence_checks is None:
            convergence_checks = sinkhorn_steps

        if convergence_data:
            self._initialise_convergence_data_dict(sinkhorn_steps)
        return convergence_checks

    def _process_inputs(self, points, n, m, const=1):
        """
        Processes densities or points or potentials, with default 'None' values as ones*constant. Or convert input to torch type
        """
        if points is None:
            weights = const * torch.ones((n, m)).type(self.dtype) / (n * m)
        else:
            weights = self._clone_process(points, non_blocking=True)
            weights = weights.view(n, m)

        return weights

    def _process_divergences(self, divergence, side=None):
        """
        process divergences, want to eventually have flexibiity to use any
        i.e given a class it'll check for sufficent attributes and run

        Currently implemented ['kl', 'tv', 'balanced']

        Note if you give it a divergence then you need to implement your own LSE type step?
        """

        if side == "left":
            logsumexp = self._process_lse_type()
            update = lambda: logsumexp(dim=1)
        elif side == "right":
            logsumexp = self._process_lse_type()
            update = lambda: logsumexp(dim=0)

        if divergence == "kl":
            return KLDivergence(self.rho, self.epsilon, update)
        elif divergence == "tv":
            return TVDivergence(self.rho, self.epsilon, update)
        elif divergence == "balanced":
            return BalancedDivergence(self.rho, self.epsilon, update)
        elif (
            hasattr(divergence, "primal_cost")
            and hasattr(divergence, "dual_cost")
            and hasattr(divergence, "sinkhorn_iterate")
        ):
            return divergence
        else:
            raise ValueError(
                "Only ['kl', 'tv', 'balanced'] implemented, o/w give own class"
            )

    def _process_lse_type(self):
        """
        Distribution to correct LSE function, depending on regular mesh or pykeops

        Returns
        -------
        function
            log-sum-exp
        """

        if self.tensorise[0] and self.tensorise[1]:
            logsumexp = self._lse_tensorised
        elif self.pykeops:
            logsumexp = self._lse_pykeops
        else:
            logsumexp = self._lse

        return logsumexp

    def _initialise_convergence_data_dict(self, sinkhorn_steps):
        """
        Initialise the convergence data dictionary of analysis

        Parameters
        ----------
        sinkhorn_steps : _type_
            _description_
        """
        self.denp = DetachNumpy()
        # pylint: disable-next=attribute-defined-outside-init
        self.convergence_dict = {}
        self.convergence_dict["info"] = (
            "N_M_eps_Nsteps"
            + str(len(self.f))
            + "_"
            + str(len(self.g))
            + "_"
            + str(self.epsilon)[3:]
            + "_"
            + str(sinkhorn_steps)
        )
        self.convergence_dict["divergence"] = str(self.left_div) + str(self.right_div)
        self.convergence_dict["primal_energy"] = []
        self.convergence_dict["dual_energy"] = []
        self.convergence_dict["f_update"] = []
        self.convergence_dict["g_update"] = []
        self.convergence_dict["epsilon"] = self.epsilon.item()
        self.convergence_dict["N"] = len(self.f)
        self.convergence_dict["M"] = len(self.g)

    def _update_convergence_dict(self, f_update, g_update):
        """Add latest update to convergence dict, or the checkpoints of convergence
        Parameters
        ----------
        f_update : float
            f_i+1 - f_i
        g_update : float
            g_i+1-g_i
        """
        self.convergence_dict["f_update"].append(f_update.item())
        self.convergence_dict["g_update"].append(g_update.item())

    def _dictionary_dump(self, convergence_data_title):
        """Save convergence data dictionary

        Parameters
        ----------
        convergence_data_title : str
            title under which to save the convergence data
        """
        # Save convergence data with title
        if not os.path.exists("convergence_data"):
            os.mkdir("convergence_data")

        f = open(
            "convergence_data/convergence_dict_N_M_eps_"
            + str(len(self.f))
            + "_"
            + str(len(self.g))
            + "_"
            + str(self.epsilon.item())[3:]
            + convergence_data_title
            + "_.pkl",
            "wb",
        )

        dump(self.convergence_dict, f)
        f.close()


# pylint: enable=no-member

if __name__ == "__main__":
    pass
