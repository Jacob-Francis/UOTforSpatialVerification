"""
Child class of `UnbalancedSinkhorn` which calculates all costs, dual and other for our metric. This
includes barycenter calculations.

Everything here is computed AFTER `sinkhorn_algorithm` has found the potentials f, g.
The plan is

    pi_ij = alpha_i beta_j exp((f_i + g_j - c_ij) / eps)

and most methods have three versions, picked by `force_type`:

    None        : tensorised if both inputs are regular grids, else KeOps if
                  pykeops=True, else dense (the same priority as the Sinkhorn loop)
    "tensorise" : force the regular-grid version
    "pykeops"   : force the KeOps version (needs pykeops=True)
    anything else (e.g. "None") : the dense version (needs the full cost matrix)

NOTE (memory, stage 5): the tensorised versions here build the FULL 4D plan
pi[k, p, r, s] of size n1*n2*m1*m2, which undoes the point of tensorising
(about 0.66 TiB for a 600 x 500 grid). Measured: a 90 x 90 marginal took 1.4 s
this way and 0.4 ms with `tensorise_f`. Passing force_type="pykeops" avoids it.
"""
import torch
from .utils import _kl
from .unbalancedsinkhorn import UnbalancedSinkhorn
from pykeops.torch import LazyTensor

# For any print statements
# (Same global print setting as in unbalancedsinkhorn.py.)
torch.set_printoptions(precision=15)

# pylint: disable=no-member


class UnbalancedOT(UnbalancedSinkhorn):
    """Process in order: 'self.parameters' --> 'self.densities' --> 'self.sinkhorn_algorithm' ..."
    Order of importance:
    - Tensorisation
    - pykeops
    - neither [storing full cost matrix]
    Child class of `UnbalancedSinkhorn` which calculates all costs, dual and other for our metric. This
    includes barycenter calculations.
    """

    def marginals(self, dim, force_type=None):
        """
        Calculate the marginals of our plan
        dim = 0 : marginal_i/source marginal/pi_0/geoverse
        dim = 1 : marginal_j/target marginal/pi_1/universe

        pi_0[i] = sum_j pi_ij (mass leaving each source point),
        pi_1[j] = sum_i pi_ij (mass arriving at each target point).
        In unbalanced OT these differ from alpha and beta: mass can be created
        or destroyed.

        Parameters
        ----------
        dim : {0, 1}
            0 for the source marginal, 1 for the target marginal.
        force_type : str, optional
            See the module docstring.

        Returns
        -------
        torch.Tensor
            Same shape as α_s (dim=0) or β_t (dim=1). (The tensorised and dense
            versions return (n1, n2) and (N,); the KeOps version reshapes to the
            potential's shape.)
        """
        # NOTE: force_type="pykeops" without pykeops=True raises AttributeError
        # (no self.pykeops_formulas), checked.
        if dim == 0:
            if (
                all(self.tensorise) and force_type is None
            ) or force_type == "tensorise":
                return self._pi(tensorised=True).sum(dim=(2, 3))
            elif (self.pykeops and force_type is None) or force_type == "pykeops":
                return self.pykeops_formulas.marginal_i(
                    self.g.view(-1, 1),
                    self.f.view(-1, 1),
                    self._process_tuple_points_for_pykeops(self.X_s),
                    self._process_tuple_points_for_pykeops(self.Y_t),
                    self.epsilon.view(-1, 1),
                    self.α_s.view(-1, 1),
                    self.β_t.view(-1, 1),
                    self.cost_const,
                ).reshape(self.f.shape)
            else:
                return self._pi(tensorised=False).sum(dim=(1))
        elif dim == 1:
            if (
                all(self.tensorise) and force_type is None
            ) or force_type == "tensorise":
                return self._pi(tensorised=True).sum(dim=(0, 1))
            elif (self.pykeops and force_type is None) or force_type == "pykeops":
                return self.pykeops_formulas.marginal_j(
                    self.g.view(-1, 1),
                    self.f.view(-1, 1),
                    self._process_tuple_points_for_pykeops(self.X_s),
                    self._process_tuple_points_for_pykeops(self.Y_t),
                    self.epsilon.view(-1, 1),
                    self.α_s.view(-1, 1),
                    self.β_t.view(-1, 1),
                    self.cost_const,
                ).reshape(self.g.shape)
            else:
                return self._pi(tensorised=False).sum(dim=(0))
        else:
            raise ValueError("dim need ot be one of 0, 1")

    def marginals_error(self, force_type=None):
        """
        Calculate the (L2) difference between input and output marginals.
        Note that for Unbalanced OT, this is NOT a convergence check but an indication of a loss of mass.

        Returns
        -------
        float, float
            L2 error between marginal distributions. Left/self.pykeops_formulas.marginal_i/source... Right/self.pykeops_formulas.marginal_j/target...
        """
        return torch.norm(
            self.marginals(0, force_type).view(-1, 1) - self.α_s.view(-1, 1)
        ), torch.norm(self.marginals(1, force_type).view(-1, 1) - self.β_t.view(-1, 1))

    def dual_cost(self, force_type=None):
        """
        Calculate the three dual cost terms;

        -D^*(-f|alpha) + -D^*(-g|beta) + eps . <pi -1, alpha x beta>

        ToDo: Check the above corresponds corrceetly

        (As coded, the terms are
            term 1:  -<alpha, D*(-f)>
            term 2:  -<beta,  D*(-g)>
            term 3:  -eps * sum_ij (pi_ij - alpha_i beta_j)
        with D* from the divergence's dual_cost. Note the minus sign on term 3:
        it is -eps <pi/(alpha x beta) - 1, alpha x beta>.)

        At the optimum, sum(primal_cost()) == sum(dual_cost()); see `duality_gap`.

        force_type: str
            One of ['tensorise', 'pykeops', 'None']. Will perform this type of reduction. If None,
            then the oder is  tensorised, pykeops, normal.

        Returns
        -------
        (float, float, float)
            dual cost terms
        """

        if (all(self.tensorise) and force_type is None) or force_type == "tensorise":
            term3 = -self.epsilon * torch.sum(
                (self.α_s[:, :, None, None] * self.β_t[None, None, :, :])
                * (
                    torch.exp(
                        (self.f[:, :, None, None] + self.g[None, None, :, :])
                        / self.epsilon
                    )
                    * torch.exp(-self.cost_1 / self.epsilon)[:, None, :, None]
                    * torch.exp(-self.cost_2 / self.epsilon)[None, :, None, :]
                    - 1
                )
            )
        elif (self.pykeops and force_type is None) or force_type == "pykeops":
            term3 = -self.epsilon * self.pykeops_formulas.dual_energy_kl_transform(
                self.g.view(-1, 1),
                self.f.view(-1, 1),
                self._process_tuple_points_for_pykeops(self.X_s),
                self._process_tuple_points_for_pykeops(self.Y_t),
                self.epsilon.view(-1, 1),
                self.α_s.view(-1, 1),
                self.β_t.view(-1, 1),
                self.cost_const,
            )
        else:  # Long way : (
            term3 = -self.epsilon * torch.sum(
                (self.α_s * self.β_t.T)
                * (torch.exp((self.f + self.g.T - self.cost) / self.epsilon) - 1)
            )
        return (
            -self._dot(self.α_s.squeeze(), self.left_div.dual_cost(-self.f).squeeze()),
            -self._dot(self.β_t.squeeze(), self.right_div.dual_cost(-self.g).squeeze()),
            term3,
        )

    def primal_cost(self, force_type=None):
        """
        Calculate the four primal cost terms;

         <c, pi> + eps KL(pi | alpha x beta) + D(pi_0|alpha) + D(pi_1|beta)

        ToDo: Check the above corresponds correctly

        force_type: str
            One of ['tensorise', 'pykeops', 'None']. Will perform this type of reduction. If None,
            then the oder is  tensorised, pykeops, normal.

        Returns
        -------
        (float, float, float)
            dual cost terms

            (Actually FOUR terms, as listed above: <c, pi>, eps * KL,
            rho * D(pi_0 | alpha), rho * D(pi_1 | beta).)

        """
        # NOTE (efficiency, stage 5): the plan pi is rebuilt for each term that
        # needs it (twice here, plus once in each marginals call): four times per
        # primal_cost call on the tensorised and dense paths.

        if (all(self.tensorise) and force_type is None) or force_type == "tensorise":
            # BUG(stage 4): log(exp(cost_1) * exp(cost_2)) is just cost_1 + cost_2,
            # but exp overflows once the cost passes about 709, giving inf
            # (checked: inf instead of 0.666 on a 60 x 60 domain).
            term1 = torch.sum(
                torch.log(
                    torch.exp(self.cost_1)[:, None, :, None]
                    * torch.exp(self.cost_2)[None, :, None, :]
                )
                * self._pi(tensorised=True)
            )
            term2 = self.epsilon * _kl(
                self._pi(tensorised=True),
                self.α_s[:, :, None, None] * self.β_t[None, None, :, :],
            )
        elif (self.pykeops and force_type is None) or force_type == "pykeops":
            term1 = self.pykeops_formulas.cost_pi(
                self.g.view(-1, 1),
                self.f.view(-1, 1),
                self._process_tuple_points_for_pykeops(self.X_s),
                self._process_tuple_points_for_pykeops(self.Y_t),
                self.epsilon.view(-1, 1),
                self.α_s.view(-1, 1),
                self.β_t.view(-1, 1),
                self.cost_const,
            )
            term2 = self.epsilon * self.pykeops_formulas.primal_energy_kl_term(
                self.g.view(-1, 1),
                self.f.view(-1, 1),
                self._process_tuple_points_for_pykeops(self.X_s),
                self._process_tuple_points_for_pykeops(self.Y_t),
                self.epsilon.view(-1, 1),
                self.α_s.view(-1, 1),
                self.β_t.view(-1, 1),
                self.cost_const,
            )
        else:
            term1 = torch.sum(self.cost * self._pi(tensorised=False))
            term2 = self.epsilon * _kl(
                self._pi(tensorised=False), self.α_s * self.β_t.T
            )

        return (
            term1,
            term2,
            self.rho
            * self.left_div.primal_cost(self.marginals(0, force_type), self.α_s),
            self.rho
            * self.right_div.primal_cost(self.marginals(1, force_type), self.β_t),
        )

    def duality_gap(self, force_type=None):
        """Calculate the duality gap, which does give a notion of convergence in the unbalanced setting!

        Should approach 0 as Sinkhorn converges. Unlike `marginals_error`, it is a
        real convergence measure for unbalanced OT.

        Parameters
        ----------
        force_type : str, optional
            See the module docstring.

        Returns
        -------
        float
            primal - dual cost
        """
        return sum(self.primal_cost(force_type)) - sum(self.dual_cost(force_type))

    # def _cost(self):
    # Presume some four dimensional cost matrix from what I can see
    #     if self.tensorise[0] and self.tensorise[1]:
    #         return self.cost_1[:, None, :, None] + self.cost_2[None, :, None, :]
    #     else:
    #         return self.cost

    def _process_tuple_points_for_pykeops(self, XY):
        """
        Process point sin atuple so that we may still use keops in this case. This is a bit laboured really.

        KeOps needs a flat list of points. A tuple (x-axis, y-axis) becomes all
        (x, y) pairs via cartesian_prod, in the same (row-major, "ij") order as the
        (n1, n2) densities. Arrays are flattened to (N, space_dim).

        Parameters
        ----------
        XY : array or tuple of arrays
            Array being checked and that will then be ran through pykeops.

        Returns
        -------
        array
            Expanded points form tuple or return array
        """

        if isinstance(XY, tuple):
            return torch.cartesian_prod(
                *[self._torch_numpy_process(t) for t in XY]
            ).type_as(self.f)
        else:
            return XY.view(-1, self.pykeops_formulas.space_dim)

    def _dot(self, a, b):
        """Distribute depending on if tensorised of not dot product

        sum of a * b over all entries: tensordot for 2D grid arrays, dot for 1D.

        Parameters
        ----------
        a : array
            (n1, n2) on grids, (N,) otherwise.
        b : array
            Same shape as a.

        Returns
        -------
        torch.Tensor
            Scalar.
        """
        # NOTE: callers pass .squeeze()d tensors, so a single point (N = 1)
        # becomes 0-d and a.dot(b) fails.
        if all(self.tensorise):
            return torch.tensordot(a, b)
        else:
            return a.dot(b)

    def _pi(self, tensorised):
        """
        Form pi calculation, either tensorised with two cost matrices or brute force with full cost matrix.

        pi_ij = alpha_i beta_j exp((f_i + g_j - c_ij) / eps)

        Parameters
        ----------
        tensorised : bool
            True: build the 4D plan pi[k, p, r, s], shape (n1, n2, m1, m2), from
            cost_1 and cost_2 (see the memory NOTE in the module docstring).
            False: the (N, M) plan from the full cost matrix.

        Returns
        -------
        torch.Tensor
            The plan.

        Raises
        ------
        AttributeError
            With tensorised=False when there is no full cost matrix (KeOps or
            grid inputs).
        """
        if tensorised:
            return (
                self.α_s[:, :, None, None]
                * self.β_t[None, None, :, :]
                * torch.exp(
                    (self.f[:, :, None, None] + self.g[None, None, :, :]) / self.epsilon
                )
                * torch.exp(-self.cost_1 / self.epsilon)[:, None, :, None]
                * torch.exp(-self.cost_2 / self.epsilon)[None, :, None, :]
            )
        else:
            try:
                return (
                    self.α_s
                    * self.β_t.T
                    * torch.exp((self.f + self.g.T - self.cost) / self.epsilon)
                )
            except AttributeError:
                raise AttributeError(
                    "cost attribute not created if pykeops was originally used."
                )

    # ------------ Barycentre mappings -----------------------------

    def barycentre_map_of_points(self, which="source"):
        """
        Calculate the barycentre of the input points and weights, aligning with the derivative of points
        to give an approximate mapping based on the probabilistic plan.

        Returns transport (displacement) vectors from the plan:

        - which="source": one vector per TARGET point,
              y_j - (pi-weighted average of the source points sent to y_j)
          i.e. the average direction mass travelled to arrive at y_j.
        - which="target": one vector per SOURCE point,
              x_i - (pi-weighted average of the target points x_i is sent to)
          i.e. it points BACK from the destination to x_i.

        Checked with a known shift (target = source + (0.3, 0)): "source" gave
        mean (0.3, 0) and "target" gave (-0.3, 0), on both the dense and KeOps
        paths.

        With a periodic cost, the x part of the vector is corrected for mass that
        crossed the periodic boundary (KeOps only; see the BUG below).

        Parameters
        ----------
        which : str, optional
            ["source", "target"], by default "source"

        Returns
        -------
        array
            Barycentre array
            (shape (M, 2) for "source", (N, 2) for "target")
        """
        # BUG(stage 4), checked: without pykeops this crashes for
        #   - periodic cost: `periodic_cost_operations` is only defined inside the
        #     `if self.pykeops` branch -> UnboundLocalError
        #   - tuple and (n1, n2, 2) grid inputs: flat (N, 2) points reach
        #     `_barycentres_torch`, whose grid branch expects (n1, n2) arrays ->
        #     RuntimeError (contracted dimensions need to match)

        # Determine barycentre function and handle periodic cost generation
        if self.pykeops:
            bary_function = self._barycentres_pykeops
            if self.cost_kwargs['cost_type'] == 'periodic':
                self._generate_periodic_ind()

                # Common operations for periodic cost type and pykeops
                    # ToDo move intto pykeops formulas

                def periodic_cost_operations(X_s, Y_t, L):
                    """
                    Build the plan pi as a symbolic KeOps LazyTensor (N, M) with the
                    periodic cost, and set self.pbcost_ind (which shift was used).
                    Nothing is computed until a .sum() is taken.

                    NOTE: this repeats the work of `_generate_periodic_ind`, which is
                    called just before.
                    """
                    X_s_2, Y_t_2 = LazyTensor(X_s[:, None, :].contiguous()), LazyTensor(Y_t[None, :, :].contiguous())
                    X_s_lt, Y_t_lt = LazyTensor(X_s[:, 0][:, None, None].contiguous()), LazyTensor(Y_t[:, 0][None, :, None].contiguous())

                    cost_lt = LazyTensor.concatenate((
                        -2 * L * (X_s_lt - Y_t_lt) + L**2,
                        (X_s_lt - Y_t_lt - (X_s_lt - Y_t_lt)),  # hack to make it zero
                        2 * L * (X_s_lt - Y_t_lt) + L**2
                    ))

                    self.pbcost_ind = cost_lt.argmin(dim=2)
                    cost = 0.5 * self.cost_const.item() * (cost_lt.min(dim=2) + (X_s_2 - Y_t_2).square().sum(dim=2))

                    epsilon = LazyTensor(self.epsilon.view(-1,1)[:, None].contiguous())
                    alpha_s, beta_t = LazyTensor(self.α_s[:, None, :].contiguous()), LazyTensor(self.β_t[None, :, :].contiguous())
                    f, g = LazyTensor(self.f[:, None, :].contiguous()), LazyTensor(self.g[None, :, :].contiguous())

                    pi = alpha_s * beta_t * ((f + g - cost) / epsilon).exp()
                    return pi
        else:
            bary_function = self._barycentres_torch

        # Handle tuple input points
        if isinstance(self.X_s, tuple) and isinstance(self.Y_t, tuple):
            if which == "source":
                return self._process_tuple_points_for_pykeops(self.Y_t) - bary_function(
                    self._process_tuple_points_for_pykeops(self.X_s), which
                )
            else:  # which == "target"
                return self._process_tuple_points_for_pykeops(self.X_s) - bary_function(
                    self._process_tuple_points_for_pykeops(self.Y_t), which
                )

        # Handle tensorized input points
        if all(self.tensorise):
            X_s, Y_t = self.X_s.view(-1, 2), self.Y_t.view(-1, 2)
            if which == "source":
                return Y_t - bary_function(X_s, which)
            else:  # which == "target"
                return X_s - bary_function(Y_t, which)

        # General case (flat array inputs) handling based on cost type
        cost_type = self.cost_kwargs['cost_type']
        if cost_type == 'periodic':
            L = self.cost_kwargs['L']
            X_s, Y_t = self.X_s.contiguous(), self.Y_t.contiguous()
            pi = periodic_cost_operations(X_s, Y_t, L)
            
            # ToDo move intto pykeops formulas, make sure the source and target formulations are correct

            # Periodic correction: L * (pbcost_ind - 1) is the x-shift (-L, 0 or +L)
            # used for each pair. Its pi-weighted average, divided by the marginal,
            # is added to the x part of the vector, so mass that wrapped round the
            # boundary gets a short vector instead of one across the whole domain.
            if which == "source":
                temp = (pi * (L * (self.pbcost_ind - 1))).sum(dim=0) / self.marginals(1)
                return Y_t - bary_function(X_s, which) + torch.stack((temp.view(-1,), torch.zeros_like(Y_t[:, 1])), dim=1)
            else:  # which == "target"
                # NOTE (efficiency): the next four lines compute full KeOps
                # reductions and throw the results away (leftover debugging). The
                # "target" call took 11.4 ms against 5.9 ms for "source" (checked).
                (pi * (L * (self.pbcost_ind - 1))).sum(dim=1).contiguous()
                self.marginals(0)
                (pi).sum(dim=1)
                ((L * (self.pbcost_ind - 1))).sum(dim=1)
                temp = (pi * (L * (self.pbcost_ind - 1))).sum(dim=1).contiguous() / self.marginals(0)
                return X_s - bary_function(Y_t, which) + torch.stack((temp.view(-1,), torch.zeros_like(X_s[:, 1])), dim=1)

        elif cost_type == 'rigid':
            if which == "source":
                return self.Y_t - bary_function(self.X_s, which)
            else:  # which == "target"
                return self.X_s - bary_function(self.Y_t, which)

        elif cost_type == 'beta_plane':
            # BUG(stage 4): unfinished. self.f_constant is never set (AttributeError),
            # and the (M, N) difference arrays below do not match the (N, M) plan
            # unless N == M.
            β_scaled =self.cost_kwargs['beta'] * self.Y_t[:, 1].view(1, -1) / self.f_constant + 1
            grad = torch.zeros_like(self.Y_t) if which == "source" else torch.zeros_like(self.X_s)

            grad[:, 0] = (self._pi(tensorised=False) * β_scaled * (self.Y_t[:, 0].view(-1, 1) - self.X_s[:, 0].view(1, -1))).sum(dim=(0 if which == "source" else 1))
            grad[:, 1] = (self._pi(tensorised=False) * β_scaled * (self.Y_t[:, 1].view(-1, 1) - self.X_s[:, 1].view(1, -1))).sum(dim=(0 if which == "source" else 1))

            return grad / (self.β_t if which == "source" else self.α_s)
    
    # def pbc_barycentre_term(self, which):
    #     return (self._pi(tensorised=False) @ self.cost_kwargs['L']*(self.pbcost_ind-1)) / self.marginals(
    #                 0
    #             ).view(-1, 1)
        # 


    def barycentre(self, p, sum_over="source"):
        """
        Barycentric projection of a quantity p through the plan.

        - sum_over="source": p lives on the source points; returns, for each
          target point j, sum_i pi_ij p_i / sum_i pi_ij.
        - sum_over="target": p lives on the target points; returns, for each
          source point i, sum_j pi_ij p_j / sum_j pi_ij.

        With p = the points themselves this is "where the mass came from / goes
        to, on average" (used by `barycentre_map_of_points`).

        Parameters
        ----------
        p : torch.Tensor
            (N, d) or (M, d) values, matching the side summed over.
        sum_over : {"source", "target"}, optional
            By default "source".

        Returns
        -------
        torch.Tensor
            (M, d) for "source", (N, d) for "target".
        """

        # To Do: Add force_type parameter here

        if self.pykeops:
            # Priorities pykeops over tensorisation here, due to memory
            bary_function = self._barycentres_pykeops
        else:
            bary_function = self._barycentres_torch

        return bary_function(p, sum_over)

    def _barycentres_pykeops(self, p, sum_over="source"):
        """Partial Barycentre mapping, the gibbs kernal term NOT full map
         - barycentre assuming G and P match in the first dimension

        We want this to take grided points and return a grid
        Or take flat and return flat

        Parameters
        ----------
        p : torch.Tensor (N, M, D) / (N,D)
            quantity to pass through approximate barycentre mapping
        sum_over : str, optional
            ["source", "target"], by default "source"

        Returns
        -------
        torch.Tensor
            pi (p) barycentre approximation

            (Normalised: divided by the marginal, see `barycentre`.) Always flat:
            (M, D) for "source", (N, D) for "target", even for grid inputs.

        Raises
        ------
        ValueError
            If sum_over is not "source" or "target".
        """
        # The KeOps formula always sums over its j side, so the source/target
        # roles are swapped to choose which side is summed over:
        #   "target": i = source (output per source point), j = target
        #   "source": i = target (output per target point), j = source
        D = p.shape[-1]
        if sum_over == "target":
            return self.pykeops_formulas.barycentres(
                self.g.view(-1, 1),
                self.f.view(-1, 1),
                self._process_tuple_points_for_pykeops(self.X_s),
                self._process_tuple_points_for_pykeops(self.Y_t),
                self.epsilon.view(-1, 1),
                self.α_s.view(-1, 1),
                self.β_t.view(-1, 1),
                p.view(-1, D),  # /should this ever be view(-1, 2?)
                self.cost_const,
            )  # .view(*self.f.shape, D)
        elif sum_over == "source":
            return self.pykeops_formulas.barycentres(
                self.f.view(-1, 1),
                self.g.view(-1, 1),
                self._process_tuple_points_for_pykeops(self.Y_t),
                self._process_tuple_points_for_pykeops(self.X_s),
                self.epsilon.view(-1, 1),
                self.β_t.view(-1, 1),
                self.α_s.view(-1, 1),
                p.view(-1, D),  # /should this ever be view(-1, 2?)
                self.cost_const,
            )  # .view(*self.g.shape, D)
        else:
            raise ValueError("sum_over = 'target' or 'source' required inputs")
    
    def _pi_transport_pykeops(self, p, sum_over="source"):
        """Partial Barycentre mapping, the gibbs kernal term NOT full map
         - barycentre assuming G and P match in the first dimension

        We want this to take grided points and return a grid
        Or take flat and return flat

        Parameters
        ----------
        p : torch.Tensor (N, M, D) / (N,D)
            quantity to pass through approximate barycentre mapping
        sum_over : str, optional
            ["source", "target"], by default "source"

        Returns
        -------
        torch.Tensor
            pi (p) barycentre approximation

            NOT normalised: sum over the chosen side of pi_ij p, without dividing
            by the marginal (unlike `_barycentres_pykeops`). Flat (M, D) for
            "source", (N, D) for "target".

        Raises
        ------
        ValueError
            If sum_over is not "source" or "target".
        """
        # Same argument layout as _barycentres_pykeops; only the formula differs.
        D = p.shape[-1]
        if sum_over == "target":
            return self.pykeops_formulas.pi_transport(
                self.g.view(-1, 1),
                self.f.view(-1, 1),
                self._process_tuple_points_for_pykeops(self.X_s),
                self._process_tuple_points_for_pykeops(self.Y_t),
                self.epsilon.view(-1, 1),
                self.α_s.view(-1, 1),
                self.β_t.view(-1, 1),
                p.view(-1, D),  # /should this ever be view(-1, 2?)
                self.cost_const,
            )  # .view(*self.f.shape, D)
        elif sum_over == "source":
            return self.pykeops_formulas.pi_transport(
                self.f.view(-1, 1),
                self.g.view(-1, 1),
                self._process_tuple_points_for_pykeops(self.Y_t),
                self._process_tuple_points_for_pykeops(self.X_s),
                self.epsilon.view(-1, 1),
                self.β_t.view(-1, 1),
                self.α_s.view(-1, 1),
                p.view(-1, D),  # /should this ever be view(-1, 2?)
                self.cost_const,
            )  # .view(*self.g.shape, D)
        else:
            raise ValueError("sum_over = 'target' or 'source' required inputs")

    def _barycentres_torch(self, p, sum_over="source"):
        """
        Partial Barycentre mapping, the gibbs kernal term NOT full map
         - barycentre assuming G and P match in the first dimension

        We want this to take grided points and return a grid
        Or take flat and return flat

        Torch version of `barycentre`, using the full plan from `_pi`.

        Parameters
        ----------
        p : torch.Tensor
            Flat inputs: (N, d) / (M, d). Grid inputs: an (n1, n2) / (m1, m2)
            SCALAR field only.
        sum_over : {"source", "target"}

        Returns
        -------
        torch.Tensor
            Flat: (M, d) / (N, d). Grid: (m1, m2) / (n1, n2).
        """
        # BUG(stage 4): on grids, a vector field p of shape (n1, n2, d) fails with
        # a broadcasting error (checked), and so do the flat points that
        # barycentre_map_of_points passes in. The grid branch also builds the
        # full 4D plan (see the module memory NOTE).

        if len(p.shape) == 1:
            d = 1
        else:
            d = p.shape[-1]

        if sum_over == "source":
            if all(self.tensorise):
                return torch.tensordot(
                    self._pi(tensorised=True),
                    p[:, :, None, None],
                    dims=([0, 1], [0, 1]),
                ).squeeze() / self.marginals(1) # ToDo: Check this broadcasting for tensorisation
            else:
                return (self._pi(tensorised=False).T @ p).view(-1, d) / self.marginals(
                    1
                ).view(-1, 1)
        elif sum_over == "target":
            if all(self.tensorise):
                return torch.tensordot(
                    self._pi(tensorised=True),
                    p[None, None, :, :],
                    dims=([2, 3], [2, 3]),
                ).squeeze() / self.marginals(0)
            else:
                return (self._pi(tensorised=False) @ p).view(-1, d) / self.marginals(
                    0
                ).view(-1, 1)
        else:
            raise ValueError(
                "Required input to have shape matching one of the potentials shape; (N, b) - dim=0 or (M, b) dim=1"
            )

    # ------- Overwrite convergence data ---------
    def _update_convergence_dict(self, f_update, g_update):
        """
        Like the parent version, but also record the total primal and dual
        energies at every convergence check.

        NOTE: computing both energies is far more expensive than a Sinkhorn
        iteration (on the dense and grid paths it rebuilds the full plan several
        times), and by default there is a check on every iteration.
        """
        self.convergence_dict["f_update"].append(f_update.item())
        self.convergence_dict["g_update"].append(g_update.item())
        self.convergence_dict["primal_energy"].append(sum(self.primal_cost()).item())
        self.convergence_dict["dual_energy"].append(sum(self.dual_cost()).item())


# pylint: enable=no-member

if __name__ == "__main__":
    pass
