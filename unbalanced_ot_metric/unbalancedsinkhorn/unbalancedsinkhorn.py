"""
Main code for Sinkhorn algorithm implementation, given any *divergence class.

Solves entropic unbalanced OT between a source alpha (on points X_s) and a target
beta (on points Y_t):

    min_pi  <c, pi> + eps KL(pi | alpha x beta) + rho D(pi_0 | alpha) + rho D(pi_1 | beta)

where pi_0, pi_1 are the marginals of the plan and D is KL, TV or "balanced" (see the
*divergence.py files). The solver works with the dual potentials f (on the source,
shape like alpha) and g (on the target, shape like beta), and the plan is

    pi_ij = alpha_i beta_j exp((f_i + g_j - c_ij) / eps).

Three back-ends compute the log-sum-exp steps (chosen in `_process_lse_type`):
    1. tensorised : regular grids; separable 1D kernels (`_lse_tensorised`)
    2. KeOps      : any point cloud; cost computed on the fly (`_lse_pykeops`)
    3. dense      : any point cloud; full (N, M) cost matrix (`_lse`)

Results (energies, marginals, barycentres) are in the child class `UnbalancedOT`.
"""

import torch
from .pykeops_formula import PyKeOpsFormulas
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
# NOTE: this changes torch's print settings for the whole Python session as soon
# as the package is imported, not only for this module.
torch.set_printoptions(precision=15)

# NOTE: PyKeOpsFormulas, pbcost_cdist, Tensorisation, DetachNumpy (only stored,
# never called), numpy and ThreadPoolExecutor are imported above but not used in
# this file (stage 2 clean-up).

# Sinkhorn Algorithm: Sink((αi)i,(xi)i,(βj)j,(yj)j)
# (Below, "aprox" is the divergence's aprox step: see KLDivergence.aprox_function.
#  The code adds the aprox step after each LSE half-step, as in lines 4 and 6.)
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

    Typical use (through the child class UnbalancedOT or DebiasedUOT):

        uot = UnbalancedOT(pykeops=True)
        uot.parameters(epsilon=0.01, rho=1.0)
        uot.densities(source_points, target_points, source_density, target_density)
        uot.sinkhorn_algorithm(aprox="kl", tol=1e-12)
        total_cost = sum(uot.primal_cost())

    Main attributes
    ---------------
    epsilon, rho, cost_const : torch.Tensor
        Set by `parameters`.
    α_s, β_t : torch.Tensor
        Source and target densities, shape (n1, n2) / (m1, m2) for grids, or
        (N, 1) / (M, 1) for flat point clouds. Set by `densities`.
    f, g : torch.Tensor
        Dual potentials, same shapes as α_s and β_t. With debias=True, f IS g
        (the same tensor). Set by `sinkhorn_algorithm`.
    left_div, right_div : divergence objects
        Marginal penalties for the source and target sides.
    """

    def __init__(self, set_fail=False, cuda_device=None, pykeops=False, debias=False):
        """
        __init__

        Parameters
        ----------
        set_fail : bool, optional
            Passed to TorchNumpyProcess for device configuration, by default False. If True,
            device=='cpu'. else detect what is available.
        cuda_device : int, str or None, optional
            Which GPU to use, e.g. 0 or "cuda:0", by default None. See
            `TorchNumpyProcessing.__init__`.
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

        # NOTE: the ImportError branch can never run: costs.py, pykeops_formula.py
        # and unbalancedot.py import pykeops at module level, so without pykeops
        # the package already fails to import.
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
            (The cost actually used is cost_const / 2 * |x - y|^2; see costs.py.)

        All three become float64 tensors on the chosen device; cost_const is
        reshaped to (1, 1) so KeOps can use it as a parameter. Call this BEFORE
        `densities`, which uses cost_const to build the cost.
        """

        # Processes to correct device, torch and shape
        # pylint: disable=attribute-defined-outside-init
        self.epsilon = self._torch_numpy_process(epsilon)
        self.rho = self._torch_numpy_process(rho)
        self.cost_const = self._torch_numpy_process(cost_const).view(-1, 1)
        # pylint: enable=attribute-defined-outside-init

    def densities(
        self,
        source_points=None,
        target_points=None,
        source_density=None,
        target_density=None,
        observation_points=None,
        forecast_points=None,
        observation_density=None,
        forecast_density=None,
        **kwargs,
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
        source_points (OBSERVATION) : (N, 2), (n1, n2, 2), ((n1), (n2))
            source 2D points, this is the mesh. Not weights.
        target_points (FORECAST) : (M, 2), (m1, m2, 2), ((m1), (m2))
            target 2D points, this is the mesh. Not weights.
        source_density : (N) (n1, n2), optional
            weights associated to the 2D source points, by default None
        target_density : (N) (m1, m2), optional
            weights associated to the 2D target points, by default None
            (Uniform weights 1 / (number of points) when None.)
        observation_points, forecast_points, observation_density, forecast_density : optional
            Aliases for source_points, target_points, source_density and
            target_density, using verification language (observation = source,
            forecast = target). Used only when the source/target name is not given.
        kwargs;
            cost_type: str
                'rigid', 'periodic'  ToDo: [ 'beta', 'beta_periodic']. Need to pass kwarg
            L: int
                 for periodic case

        Raises
        ------
        ValueError
            If either density sums to zero.
        """
        # NOTE: `global` makes n1, n2, m1, m2 module-level variables, shared by
        # every solver object (and every thread). Nothing else reads them; plain
        # local variables would do.
        global n1, n2, m1, m2

        # Resolve aliases: prefer explicit source/target if both are given
        if observation_points is not None and source_points is None:
            source_points = observation_points
        if forecast_points is not None and target_points is None:
            target_points = forecast_points

        if observation_density is not None and source_density is None:
            source_density = observation_density
        if forecast_density is not None and target_density is None:
            target_density = forecast_density

        # Process arguments depending on input, determine whether to tensorise.
        # Note n2, m2 will be one if given no-regular points.
        # (_process_points must be called source first, then target: it fills
        # self.tensorise in that order.)
        n1, n2 = self._process_points(source_points)
        m1, m2 = self._process_points(target_points)

        self.process_initial_point_clouds(
            source_points, target_points, self.cost_const, n1, n2, m1, m2, **kwargs
        )

        # Process associated densities - could be a utils or in torch_numpy... class
        # Shapes: (n1, n2) / (m1, m2) for grids, (N, 1) / (M, 1) for flat points.
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

        Dense back-end, with the full (N, M) matrix `self.cost`. The g update is

            g_j <- g_j - eps * LSE_i[ (f_i + g_j - c_ij) / eps + log alpha_i ]

        Because g_j is constant in the sum over i, this equals the textbook
        softmin  -eps * LSE_i[ (f_i - c_ij) / eps + log alpha_i ].  Writing it as
        an increment keeps the LSE argument near 0 once the potentials are close
        to converged, which is the usual "stabilised" Sinkhorn form. The f update
        is the same with the roles swapped (sum over j, weights beta).

        With debias=True (f is g, symmetric problem) half the increment is added,
        i.e. the new value is the average of the old value and the softmin.

        Parameters
        ----------
        dim : int
            If 0 then update g, if 1 then update f
            (dim is also the axis of the (N, M) matrix that is summed over.)
        """
        # NOTE (efficiency, stage 5): every call recomputes (f + g.T - cost) / eps
        # and log(density) over the whole (N, M) matrix. Precomputing
        # -cost / eps + log(density) halved the time per half-step in a test
        # (N = 1600: 25.6 ms -> 13.2 ms, same result to 1e-16).
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
                    (self.f + self.g.T - self.cost) / self.epsilon
                    + torch.log(self.α_s),
                    dim,
                ).view(-1, 1)
            elif dim == 1:  # f update
                self.f += -self.epsilon * torch.logsumexp(
                    (self.f + self.g.T - self.cost) / self.epsilon
                    + torch.log(self.β_t.T),
                    dim,
                ).view(-1, 1)

    def _lse_pykeops(self, dim):
        """
        Pykeops log-sum-exp calculation on the fly

        Same update as `_lse`, but KeOps computes the cost for each pair as it
        goes, so no (N, M) matrix is stored. `PyKeOpsFormulas.log_sum_exp` always
        sums over its j side, so the arguments are swapped to choose the side:

        - g update: i side = target (Y_t, g), j side = source (X_s, f, alpha)
        - f update: i side = source (X_s, f), j side = target (Y_t, g, beta)

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

        Regular-grid back-end. The Gibbs kernel exp(-C / eps) splits into
        exp(-cost_1 / eps) and exp(-cost_2 / eps), applied with `tensorise_f`:

            f <- -eps * log( K  (beta  * exp(g / eps)) )
            g <- -eps * log( K^T (alpha * exp(f / eps)) )

        Unlike `_lse` this is NOT done in the log domain: it takes exp, multiplies,
        then takes log. That is faster, but can underflow for small eps.
        NOTE: checked on a 20 x 20 grid: eps = 2e-4 raised the NaN error here,
        while the dense back-end stayed finite.

        Results are written into self.f / self.g in place (out=...), because the
        divergence's sinkhorn_iterate then modifies that same tensor.

        Parameters
        ----------
        dim : int
            If 0 then update g, if 1 then update f
        """
        # NOTE (efficiency, stage 5): exp(-cost_1 / eps) and exp(-cost_2 / eps) are
        # recomputed on every call although they only change with eps. Caching
        # them saved 25-60% per iteration in tests (100x100 and 400x400 grids).
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

        One full Sinkhorn iteration: update f (left / source side) then g
        (right / target side). Each `sinkhorn_iterate` runs the LSE half-step
        and then the divergence's aprox step, in place. With debias=True, f and g
        are the same tensor, so one update is enough.

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
        # NOTE: RuntimeWarning is a Warning class but is RAISED here, so it stops
        # the solver like an error. Callers catch it with `except RuntimeWarning`.
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
        convergence_or_fail=False,
    ):
        """Note we overwrite f0, g0

        Run Sinkhorn iterations until the potentials stop changing.

        Parameters
        ----------
        sinkhorn_steps : int, optional
            Iterations per run, by default max(100, -1.1 / eps * log(eps)).
        reinitialise : bool, optional
            Rebuild the divergences and starting potentials first, by default True.
        tol : float, optional
            Stop when max(|f_new - f|_inf, |g_new - g|_inf) < tol, by default 1e-15.
        f0, g0 : array, optional
            Starting potentials. If None, the previous f / g are reused when they
            exist (a warm start), otherwise constants are used (see f0_const).
        verbose : bool, optional
            Print progress at every convergence check, by default False.
        left_divergence, right_divergence : str or divergence object, optional
            Marginal penalty for the source / target side: "kl", "tv",
            "balanced", or an object with primal_cost, dual_cost and
            sinkhorn_iterate.
        aprox : str, optional
            Shortcut that sets both sides to "kl", "tv" or "balanced".
        convergence_checks : int, optional
            How many convergence checks per run, by default sinkhorn_steps (i.e. a
            check on every iteration). A check happens every
            sinkhorn_steps // convergence_checks iterations, and on the last one.
        convergence_data : bool, optional
            Record the size of each update (and energies in UnbalancedOT) and pickle
            them to ./convergence_data/, by default False.
        convergence_data_title : str, optional
            Added to the pickle file name.
        f0_const, g0_const : float, optional
            Starting value when f0 / g0 is None and there is nothing to reuse,
            by default 0.
        convergence_repeats : int, optional
            Number of runs of sinkhorn_steps iterations before giving up, by
            default 1. A restart keeps the current potentials.
        epsilon_annealing : bool, optional
            First solve with a decreasing sequence of eps values (from 2 down to
            the target eps, scaling sqrt(eps) by epsilon_annealing_const each step),
            as a warm start. By default False.
        epsilon_annealing_const : float, optional
            Ratio between successive sqrt(eps) values, by default 0.99. Must be < 1.
        convergence_or_fail : bool, optional
            If True, raise RuntimeWarning (after a ConvergenceWarning) when the run
            oscillates or never reaches tol, instead of just stopping. By default False.

        Returns
        -------
        f_update, g_update : torch.Tensor
            Size of the last change in f and g (inf-norm).
        i_super : int
            Total number of iterations, including annealing.
        """

        # Convergence checking;
        if sinkhorn_steps is None:
            sinkhorn_steps = max(
                [100, int(-1.1 / self.epsilon * torch.log(self.epsilon))]
            )
        convergence_checks = self.convergence_intialisation(
            convergence_checks, sinkhorn_steps, convergence_data
        )

        # To Do: Refactor this so you can iterate the sinkhorn_aglorithm without re-intialising every time
        if reinitialise:
            self.sinkhorn_algorithm_initialisation(
                left_divergence, right_divergence, aprox, f0, f0_const, g0, g0_const
            )

        i_super = 0

        # ---- Optional epsilon annealing: warm start with larger eps first ----
        # parameters will nee be parsing, assuming a unit box atm
        # (The starting value eps = 2, i.e. sqrt(eps) = sqrt(2), assumes the points
        # lie in a box of size about 1.)
        # NOTE: during annealing only self.epsilon changes. The divergence objects
        # keep the target eps they were built with, so the KL aprox step uses the
        # target eps throughout. Only a warm start, but not true annealing for KL.
        if epsilon_annealing:
            # KeOps path: start from the expected cost instead of the given f0, g0.
            # BUG(stage 4): with debias=True this replaces self.f and self.g with
            # two separate tensors, so f is no longer g and the symmetric update
            # only changes f (checked: f and g ended up 0.475 apart).
            # NOTE: cost_const is not passed, so it is ignored for this guess.
            if not all(self.tensorise) and self.pykeops:
                self.f = self.pykeops_formulas.starting_potentials(
                    self.Y_t, self.X_s, self.β_t
                )
                self.g = self.pykeops_formulas.starting_potentials(
                    self.X_s, self.Y_t, self.α_s
                )

            # sqrt(eps) values from sqrt(2) down to sqrt(target eps).
            # BUG(stage 4): epsilon_annealing_const >= 1 makes this loop forever.
            # NOTE: torch.Tensor([2]) is float32, so these eps values carry float32
            # rounding.
            scale_list = exponential_sequence(
                torch.sqrt(torch.Tensor([2])),
                epsilon_annealing_const,
                torch.sqrt(self.epsilon).cpu(),
            )
            store_original_epsilon = self.epsilon.clone()
            for s in scale_list:
                err = torch.inf
                i = 0
                self.epsilon = self._torch_numpy_process(s**2)
                # Iteration budget for this eps; it is negative for eps > 1, which
                # skips the loop for those values.
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

            print(f"Epsilon Annealing, final esp={s**2}, err={err}, its={i}/{max_its}")
            self.epsilon = store_original_epsilon

        # ############ Tru Epsilon loop
        # Main loop at the target eps. Each pass of the while loop is one full
        # iteration (f then g). On "check" iterations the potentials are copied
        # first, so the size of the update can be measured afterwards.
        #
        # Ways out of the loop:
        #   - err < tol                         -> converged
        #   - FlipFlop sees err repeating       -> stalled (oscillating)
        #   - convergence_repeats runs used up  -> not converged
        # (With convergence_or_fail=True the last two raise RuntimeWarning.)
        #
        # NOTE: the default (a check on every iteration) clones f and g every
        # time. Checking less often was 10-20% faster in tests.
        # BUG(stage 4): if sinkhorn_steps == 0 the loop never runs and the
        # `return f_update, ...` at the end raises UnboundLocalError.

        i = 0
        count = 1  # which run of sinkhorn_steps iterations this is (1-based)
        # NOTE: FlipFlop(6, 3) cannot detect an err that alternates between two
        # values (see flipflop.py); FlipFlop(4, 2), used after a restart, can.
        ff = FlipFlop(6, 3)  # look three osciallting numbers?
        while i < sinkhorn_steps:

            # Convergence check
            # Check last to save convergence
            # BUG(stage 4): if convergence_checks > sinkhorn_steps, the integer
            # division is 0 and `i % 0` raises ZeroDivisionError (checked).
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
                        warnings.warn(
                            f"Sinkhorn oscillating and failed to converge in {sinkhorn_steps*(count-1)} iteration| err={err}",
                            ConvergenceWarning,
                        )
                        raise RuntimeWarning
                    break

                if verbose:
                    print(f"Iteration {i}, error ", {err})

                # NOTE: these "Convergence ..." messages print even when
                # verbose=False.
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
                    # Start another run of sinkhorn_steps iterations from the
                    # current potentials, with a fresh, more sensitive FlipFlop.
                    # BUG(stage 4): i is set to 0 here but the `i += 1` at the
                    # bottom of the loop then makes it 1, so each restart runs one
                    # iteration short (checked: 27 instead of 30 iterations for
                    # sinkhorn_steps=10, convergence_repeats=3).
                    count += 1
                    if verbose:
                        print(
                            f"!!!!!!!!!!Sinkhorn didnt converge, err={err} | total its: {sinkhorn_steps*count}!!!!!!!!"
                        )
                    i = 0
                    del ff  # Start looking for oscillations
                    ff = FlipFlop(4, 2)  # reduce oscillating looking for

                # All convergence_repeats runs used without reaching tol.
                if count == convergence_repeats + 1:
                    if convergence_or_fail:
                        # ToDo correct these warnings
                        warnings.warn(
                            f"Sinkhorn did not converge in {sinkhorn_steps*(count-1)} iteration| err={err}",
                            ConvergenceWarning,
                        )
                        raise RuntimeWarning
                    break

            i += 1
            i_super += 1

        if convergence_data:
            self._dictionary_dump(convergence_data_title)

        return f_update, g_update, i_super

    # ------------------ Running function for the algorithm ----------------------
    def sinkhorn_algorithm_initialisation(
        self, left_divergence, right_divergence, aprox, f0, f0_const, g0, g0_const
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

            NOTE: these last three are NOT parameters of this method (they belong to
            `convergence_intialisation`), and it returns nothing, despite the
            Returns section below. Also, "\in" above is an invalid escape in a
            normal string; newer Python versions warn about it.

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
        # BUG(stage 4): the old f / g are reused even after `densities` was called
        # with data of a different size, which then fails with a shape error
        # (checked). Potentials left as NaN by a failed run are also reused.
        if hasattr(self, "f") and f0 is None:
            pass
        else:
            self.f = self._process_inputs(f0, *self.α_s.shape, const=f0_const)

        if hasattr(self, "g") and g0 is None:
            pass
        else:
            self.g = self._process_inputs(g0, *self.β_t.shape, const=g0_const)

        # Symmetric problem (source = target): use ONE tensor for both potentials,
        # so every update to f is also an update to g.
        if self.debias:
            self.f = self.g
            # BUG(stage 4): with tuple input, `X_s == Y_t` compares the tuples'
            # tensors with `==`, which raises "Boolean value of Tensor ... is
            # ambiguous" unless both tuples hold the very same tensor objects
            # (checked).
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

    def convergence_intialisation(
        self, convergence_checks, sinkhorn_steps, convergence_data
    ):
        """
        Default the number of convergence checks, and start the convergence record.

        ("intialisation" is a typo for "initialisation"; kept for now.)

        Parameters
        ----------
        convergence_checks : int or None
            None means a check on every iteration (= sinkhorn_steps).
        sinkhorn_steps : int
            Iterations per run.
        convergence_data : bool
            If True, create `self.convergence_dict`.

        Returns
        -------
        int
            The number of convergence checks.
        """
        # BUG(stage 4): this runs BEFORE sinkhorn_algorithm_initialisation, but
        # _initialise_convergence_data_dict reads self.f and self.left_div. So on
        # a first call convergence_data=True raises AttributeError (checked).
        if convergence_checks is None:
            convergence_checks = sinkhorn_steps

        if convergence_data:
            self._initialise_convergence_data_dict(sinkhorn_steps)
        return convergence_checks

    def _process_inputs(self, points, n, m, const=1):
        """
        Processes densities or points or potentials, with default 'None' values as ones*constant. Or convert input to torch type

        Parameters
        ----------
        points : array or None
            Values to use (copied and reshaped to (n, m)), or None for a constant.
        n, m : int
            Output shape, e.g. (n1, n2) for a grid or (N, 1) for flat points.
        const : float, optional
            Total of the constant array when points is None, by default 1. Each
            entry is const / (n * m), so the default gives a uniform density
            with total mass 1.

        Returns
        -------
        torch.Tensor
            (n, m) float64 tensor on self.device.
        """
        # BUG(stage 4): the same rule is used for the starting potentials, so
        # f0_const=3 gives f = 3 / (n * m) everywhere instead of 3 (checked:
        # 0.12 for 25 points). The docstrings promise "ones * f0_const".
        if points is None:
            weights = const * torch.ones((n, m)).type(self.dtype) / (n * m)
            weights = weights.to(self.device)
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
        (Yes: a custom object is returned as it is, so its sinkhorn_iterate must
        do the whole half-step itself; no update_function is attached.)

        Parameters
        ----------
        divergence : str or object
            "kl", "tv", "balanced", or an object with primal_cost, dual_cost and
            sinkhorn_iterate methods.
        side : {"left", "right", None}
            "left" = source side (its half-step updates f, LSE over dim=1),
            "right" = target side (updates g, LSE over dim=0).

        Returns
        -------
        divergence object
        """

        # The divergence gets a zero-argument closure that runs the chosen LSE
        # back-end for its side. The back-end is chosen once, here.
        # NOTE: if a string is given with side=None, `update` is never defined and
        # the return below raises NameError. The caller always passes a side for
        # strings, so this cannot happen in practice.
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

        Priority: tensorised (both sides regular grids) > KeOps (pykeops=True)
        > dense.

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

        Creates `self.convergence_dict` with empty lists for f_update, g_update,
        primal_energy and dual_energy, plus some problem info.

        Parameters
        ----------
        sinkhorn_steps : int
            Iterations per run; only stored in the "info" string.
        """
        # NOTE: self.denp is stored but never used. len(self.f) is the number of
        # rows (n1 for a grid), not the number of points. str(self.left_div) is the
        # object's default repr (it includes a memory address), and
        # str(self.epsilon)[3:] cuts "tensor(" to "sor(".
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

        Writes a pickle to ./convergence_data/ in the CURRENT working directory,
        creating the folder if needed. The file name holds len(f), len(g) and
        str(epsilon)[3:] (e.g. eps = 0.01 gives "1": the leading "0.0" is cut).

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
