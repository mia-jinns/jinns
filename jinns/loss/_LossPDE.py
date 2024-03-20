"""
Main module to implement a PDE loss in jinns
"""

import warnings
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jinns.loss._Losses import (
    dynamic_loss_apply,
    boundary_condition_apply,
    normalization_loss_apply,
    observations_loss_apply,
    sobolev_reg_apply,
    initial_condition_apply,
    constraints_system_loss_apply,
)
from jinns.data._DataGenerators import PDEStatioBatch, PDENonStatioBatch
from jinns.utils._utils import (
    _get_vmap_in_axes_params,
    _set_derivatives,
    _update_eq_params_dict,
)
from jinns.utils._pinn import PINN
from jinns.utils._spinn import SPINN
from jinns.loss._operators import _sobolev

_IMPLEMENTED_BOUNDARY_CONDITIONS = [
    "dirichlet",
    "von neumann",
    "vonneumann",
]


@register_pytree_node_class
class LossPDEAbstract:
    """
    Super class for the actual Pinn loss classes. This class should not be
    used. It serves for common attributes between LossPDEStatio and
    LossPDENonStatio


    **Note:** LossPDEAbstract is jittable. Hence it implements the tree_flatten() and
    tree_unflatten methods.
    """

    def __init__(
        self,
        u,
        loss_weights,
        derivative_keys=None,
        norm_key=None,
        norm_borders=None,
        norm_samples=None,
    ):
        """
        Parameters
        ----------
        u
            the PINN object
        loss_weights
            a dictionary with values used to ponderate each term in the loss
            function. Valid keys are `dyn_loss`, `norm_loss`, `boundary_loss`
            and `observations`
            Note that we can have jnp.arrays with the same dimension of
            `u` which then ponderates each output of `u`
        derivative_keys
            A dict of lists of strings. In the dict, the key must correspond to
            the loss term keywords. Then each of the values must correspond to keys in the parameter
            dictionary (*at top level only of the parameter dictionary*).
            It enables selecting the set of parameters
            with respect to which the gradients of the dynamic
            loss are computed. If nothing is provided, we set ["nn_params"] for all loss term
            keywords, this is what is typically
            done in solving forward problems, when we only estimate the
            equation solution with a PINN. If some loss terms keywords are
            missing we set their value to ["nn_params"] by default for the
            same reason
        norm_key
            Jax random key to draw samples in for the Monte Carlo computation
            of the normalization constant. Default is None
        norm_borders
            tuple of (min, max) of the boundaray values of the space over which
            to integrate in the computation of the normalization constant.
            A list of tuple for higher dimensional problems. Default None.
        norm_samples
            Fixed sample point in the space over which to compute the
            normalization constant. Default is None

        Raises
        ------
        RuntimeError
            When provided an invalid combination of `norm_key`, `norm_borders`
            and `norm_samples`. See note below.

        **Note:** If `norm_key` and `norm_borders` and `norm_samples` are `None`
        then no normalization loss in enforced.
        If `norm_borders` and `norm_samples` are given while
        `norm_samples` is `None` then samples are drawn at each loss evaluation.
        Otherwise, if `norm_samples` is given, those samples are used.
        """

        self.u = u
        if derivative_keys is None:
            # be default we only take gradient wrt nn_params
            derivative_keys = {
                k: ["nn_params"]
                for k in [
                    "dyn_loss",
                    "boundary_loss",
                    "norm_loss",
                    "initial_condition",
                    "observations",
                    "sobolev",
                ]
            }
        if isinstance(derivative_keys, list):
            # if the user only provided a list, this defines the gradient taken
            # for all the loss entries
            derivative_keys = {
                k: derivative_keys
                for k in [
                    "dyn_loss",
                    "boundary_loss",
                    "norm_loss",
                    "initial_condition",
                    "observations",
                    "sobolev",
                ]
            }

        self.derivative_keys = derivative_keys
        self.loss_weights = loss_weights
        self.norm_borders = norm_borders
        self.norm_key = norm_key
        self.norm_samples = norm_samples

        if norm_key is None and norm_borders is None and norm_samples is None:
            # if there is None of the 3 above, that means we don't consider
            # normalization loss
            self.normalization_loss = None
        elif (
            norm_key is not None and norm_borders is not None and norm_samples is None
        ):  # this ordering so that by default priority is to given mc_samples
            self.norm_sample_method = "generate"
            if not isinstance(self.norm_borders[0], tuple):
                self.norm_borders = (self.norm_borders,)
            self.norm_xmin, self.norm_xmax = [], []
            for i, _ in enumerate(self.norm_borders):
                self.norm_xmin.append(self.norm_borders[i][0])
                self.norm_xmax.append(self.norm_borders[i][1])
            self.int_length = jnp.prod(
                jnp.array(
                    [
                        self.norm_xmax[i] - self.norm_xmin[i]
                        for i in range(len(self.norm_borders))
                    ]
                )
            )
            self.normalization_loss = True
        elif norm_samples is None:
            raise RuntimeError(
                "norm_borders should always provided then either"
                " norm_samples (fixed norm_samples) or norm_key (random norm_samples)"
                " is required."
            )
        else:
            # ok, we are sure we have norm_samples given by the user
            self.norm_sample_method = "user"
            if not isinstance(self.norm_borders[0], tuple):
                self.norm_borders = (self.norm_borders,)
            self.norm_xmin, self.norm_xmax = [], []
            for i, _ in enumerate(self.norm_borders):
                self.norm_xmin.append(self.norm_borders[i][0])
                self.norm_xmax.append(self.norm_borders[i][1])
            self.int_length = jnp.prod(
                jnp.array(
                    [
                        self.norm_xmax[i] - self.norm_xmin[i]
                        for i in range(len(self.norm_borders))
                    ]
                )
            )
            self.normalization_loss = True

    def get_norm_samples(self):
        """
        Returns a batch of points in the domain for integration when the
        normalization constraint is enforced. The batch of points is either
        fixed (provided by the user) or regenerated at each iteration.
        """
        if self.norm_sample_method == "user":
            return self.norm_samples
        if self.norm_sample_method == "generate":
            ## NOTE TODO CHECK the performances of this for loop
            norm_samples = []
            for d in range(len(self.norm_borders)):
                self.norm_key, subkey = jax.random.split(self.norm_key)
                norm_samples.append(
                    jax.random.uniform(
                        subkey,
                        shape=(1000, 1),
                        minval=self.norm_xmin[d],
                        maxval=self.norm_xmax[d],
                    )
                )
            self.norm_samples = jnp.concatenate(norm_samples, axis=-1)
            return self.norm_samples
        raise RuntimeError("Problem with the value of self.norm_sample_method")

    def tree_flatten(self):
        children = (self.norm_key, self.norm_samples, self.loss_weights)
        aux_data = {
            "norm_borders": self.norm_borders,
            "derivative_keys": self.derivative_keys,
            "u": self.u,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(self, aux_data, children):
        (norm_key, norm_samples, loss_weights) = children
        pls = self(
            aux_data["u"],
            loss_weights,
            aux_data["derivative_keys"],
            norm_key,
            aux_data["norm_borders"],
            norm_samples,
        )
        return pls


@register_pytree_node_class
class LossPDEStatio(LossPDEAbstract):
    r"""Loss object for a stationary partial differential equation

    .. math::
        \mathcal{N}[u](x) = 0, \forall x  \in \Omega

    where :math:`\mathcal{N}[\cdot]` is a differential operator and the
    boundary condition is :math:`u(x)=u_b(x)` The additional condition of
    integrating to 1 can be included, i.e. :math:`\int u(x)\mathrm{d}x=1`.


    **Note:** LossPDEStatio is jittable. Hence it implements the tree_flatten() and
    tree_unflatten methods.
    """

    def __init__(
        self,
        u,
        loss_weights,
        dynamic_loss,
        derivative_keys=None,
        omega_boundary_fun=None,
        omega_boundary_condition=None,
        omega_boundary_dim=None,
        norm_key=None,
        norm_borders=None,
        norm_samples=None,
        sobolev_m=None,
        obs_slice=None,
    ):
        r"""
        Parameters
        ----------
        u
            the PINN object
        loss_weights
            a dictionary with values used to ponderate each term in the loss
            function. Valid keys are `dyn_loss`, `norm_loss`, `boundary_loss`
            and `observations`.
            Note that we can have jnp.arrays with the same dimension of
            `u` which then ponderates each output of `u`
        dynamic_loss
            the stationary PDE dynamic part of the loss, basically the differential
            operator :math:` \mathcal{N}[u](t)`. Should implement a method
            `dynamic_loss.evaluate(t, u, params)`.
            Can be None in order to access only some part of the evaluate call
            results.
        derivative_keys
            A dict of lists of strings. In the dict, the key must correspond to
            the loss term keywords. Then each of the values must correspond to keys in the parameter
            dictionary (*at top level only of the parameter dictionary*).
            It enables selecting the set of parameters
            with respect to which the gradients of the dynamic
            loss are computed. If nothing is provided, we set ["nn_params"] for all loss term
            keywords, this is what is typically
            done in solving forward problems, when we only estimate the
            equation solution with a PINN. If some loss terms keywords are
            missing we set their value to ["nn_params"] by default for the same
            reason
        omega_boundary_fun
            The function to be matched in the border condition (can be None)
            or a dictionary of such function. In this case, the keys are the
            facets and the values are the functions. The keys must be in the
            following order: 1D -> ["xmin", "xmax"], 2D -> ["xmin", "xmax",
            "ymin", "ymax"]. Note that high order boundaries are currently not
            implemented. A value in the dict can be None, this means we do not
            enforce a particular boundary condition on this facet.
            The facet called "xmin", resp. "xmax" etc., in 2D,
            refers to the set of 2D points with fixed "xmin", resp. "xmax", etc.
        omega_boundary_condition
            Either None (no condition), or a string defining the boundary
            condition e.g. Dirichlet or Von Neumann, or a dictionary of such
            strings. In this case, the keys are the
            facets and the values are the strings. The keys must be in the
            following order: 1D -> ["xmin", "xmax"], 2D -> ["xmin", "xmax",
            "ymin", "ymax"]. Note that high order boundaries are currently not
            implemented. A value in the dict can be None, this means we do not
            enforce a particular boundary condition on this facet.
            The facet called "xmin", resp. "xmax" etc., in 2D,
            refers to the set of 2D points with fixed "xmin", resp. "xmax", etc.
        omega_boundary_dim
            Either None, or a jnp.s\_ or a dict of jnp.s\_ with keys following
            the logic of omega_boundary_fun. It indicates which dimension(s) of
            the PINN will be forced to match the boundary condition
            Note that it must be a slice and not an integer (a preprocessing of the
            user provided argument takes care of it)
        norm_key
            Jax random key to draw samples in for the Monte Carlo computation
            of the normalization constant. Default is None
        norm_borders
            tuple of (min, max) of the boundaray values of the space over which
            to integrate in the computation of the normalization constant.
            A list of tuple for higher dimensional problems. Default None.
        norm_samples
            Fixed sample point in the space over which to compute the
            normalization constant. Default is None
        sobolev_m
            An integer. Default is None.
            It corresponds to the Sobolev regularization order as proposed in
            *Convergence and error analysis of PINNs*,
            Doumeche et al., 2023, https://arxiv.org/pdf/2305.01240.pdf
        obs_slice
            slice object specifying the begininning/ending
            slice of u output(s) that is observed (this is then useful for
            multidim PINN). Default is None.


        Raises
        ------
        ValueError
            If conditions on omega_boundary_condition and omega_boundary_fun
            are not respected
        """

        super().__init__(
            u, loss_weights, derivative_keys, norm_key, norm_borders, norm_samples
        )

        if omega_boundary_condition is None or omega_boundary_fun is None:
            warnings.warn(
                "Missing boundary function or no boundary condition."
                "Boundary function is thus ignored."
            )
        else:
            if isinstance(omega_boundary_condition, dict):
                for _, v in omega_boundary_condition.items():
                    if v is not None and not any(
                        v.lower() in s for s in _IMPLEMENTED_BOUNDARY_CONDITIONS
                    ):
                        raise NotImplementedError(
                            f"The boundary condition {omega_boundary_condition} is not"
                            f"implemented yet. Try one of :"
                            f"{_IMPLEMENTED_BOUNDARY_CONDITIONS}."
                        )
            else:
                if not any(
                    omega_boundary_condition.lower() in s
                    for s in _IMPLEMENTED_BOUNDARY_CONDITIONS
                ):
                    raise NotImplementedError(
                        f"The boundary condition {omega_boundary_condition} is not"
                        f"implemented yet. Try one of :"
                        f"{_IMPLEMENTED_BOUNDARY_CONDITIONS}."
                    )
                if isinstance(omega_boundary_fun, dict) and isinstance(
                    omega_boundary_condition, dict
                ):
                    if (
                        not (
                            list(omega_boundary_fun.keys()) == ["xmin", "xmax"]
                            and list(omega_boundary_condition.keys())
                            == ["xmin", "xmax"]
                        )
                    ) or (
                        not (
                            list(omega_boundary_fun.keys())
                            == ["xmin", "xmax", "ymin", "ymax"]
                            and list(omega_boundary_condition.keys())
                            == ["xmin", "xmax", "ymin", "ymax"]
                        )
                    ):
                        raise ValueError(
                            "The key order (facet order) in the "
                            "boundary condition dictionaries is incorrect"
                        )

        self.omega_boundary_fun = omega_boundary_fun
        self.omega_boundary_condition = omega_boundary_condition

        self.omega_boundary_dim = omega_boundary_dim
        if isinstance(self.omega_boundary_fun, dict):
            if self.omega_boundary_dim is None:
                self.omega_boundary_dim = {
                    k: jnp.s_[::] for k in self.omega_boundary_fun.keys()
                }
            if list(self.omega_boundary_dim.keys()) != list(
                self.omega_boundary_fun.keys()
            ):
                raise ValueError(
                    "If omega_boundary_fun is a dict,"
                    " omega_boundary_dim should be a dict with the same keys"
                )
            for k, v in self.omega_boundary_dim.items():
                if isinstance(v, int):
                    # rewrite it as a slice to ensure that axis does not disappear when
                    # indexing
                    self.omega_boundary_dim[k] = jnp.s_[v : v + 1]

        else:
            if self.omega_boundary_dim is None:
                self.omega_boundary_dim = jnp.s_[::]
            if isinstance(self.omega_boundary_dim, int):
                # rewrite it as a slice to ensure that axis does not disappear when
                # indexing
                self.omega_boundary_dim = jnp.s_[
                    self.omega_boundary_dim : self.omega_boundary_dim + 1
                ]
            if not isinstance(self.omega_boundary_dim, slice):
                raise ValueError("self.omega_boundary_dim must be a jnp.s_" " object")

        self.dynamic_loss = dynamic_loss

        self.sobolev_m = sobolev_m
        if self.sobolev_m is not None:
            self.sobolev_reg = _sobolev(
                self.u, self.sobolev_m
            )  # we return a function, that way
            # the order of sobolev_m is static and the conditional in the recursive
            # function is properly set
            self.sobolev_m = self.sobolev_m
        else:
            self.sobolev_reg = None

        if self.normalization_loss is None:
            self.loss_weights["norm_loss"] = 0

        if self.omega_boundary_fun is None:
            self.loss_weights["boundary_loss"] = 0

        if self.sobolev_reg is None:
            self.loss_weights["sobolev"] = 0

        if (
            isinstance(self.omega_boundary_fun, dict)
            and not isinstance(self.omega_boundary_condition, dict)
        ) or (
            not isinstance(self.omega_boundary_fun, dict)
            and isinstance(self.omega_boundary_condition, dict)
        ):
            raise ValueError(
                "if one of self.omega_boundary_fun or "
                "self.omega_boundary_condition is dict, the other should be too."
            )

        self.obs_slice = obs_slice
        if self.obs_slice is None:
            self.obs_slice = jnp.s_[...]

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, params, batch):
        """
        Evaluate the loss function at a batch of points for given parameters.


        Parameters
        ---------
        params
            The dictionary of parameters of the model.
            Typically, it is a dictionary of
            dictionaries: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter
        batch
            A PDEStatioBatch object.
            Such a named tuple is composed of a batch of points in the
            domain, a batch of points in the domain
            border and an optional additional batch of parameters (eg. for
            metamodeling) and an optional additional batch of observed
            inputs/outputs/parameters
        """
        omega_batch, _ = batch.inside_batch, batch.border_batch

        vmap_in_axes_x = (0,)

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            # update eq_params with the batches of generated params
            params = _update_eq_params_dict(params, batch.param_batch_dict)

        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)

        # dynamic part
        params_ = _set_derivatives(params, "dyn_loss", self.derivative_keys)
        if self.dynamic_loss is not None:
            mse_dyn_loss = dynamic_loss_apply(
                self.dynamic_loss.evaluate,
                self.u,
                (omega_batch,),
                params_,
                vmap_in_axes_x + vmap_in_axes_params,
                self.loss_weights["dyn_loss"],
            )
        else:
            mse_dyn_loss = jnp.array(0.0)

        # normalization part
        params_ = _set_derivatives(params, "norm_loss", self.derivative_keys)
        if self.normalization_loss is not None:
            mse_norm_loss = normalization_loss_apply(
                self.u,
                (self.get_norm_samples(),),
                params_,
                vmap_in_axes_x + vmap_in_axes_params,
                self.int_length,
                self.loss_weights["norm_loss"],
            )
        else:
            mse_norm_loss = jnp.array(0.0)
            self.loss_weights["norm_loss"] = 0

        # boundary part
        params_ = _set_derivatives(params, "boundary_loss", self.derivative_keys)
        if self.omega_boundary_condition is not None:
            mse_boundary_loss = boundary_condition_apply(
                self.u,
                batch,
                params_,
                self.omega_boundary_fun,
                self.omega_boundary_condition,
                self.omega_boundary_dim,
                self.loss_weights["boundary_loss"],
            )
        else:
            mse_boundary_loss = jnp.array(0.0)

        # Observation mse
        if batch.obs_batch_dict is not None:
            # update params with the batches of observed params
            params = _update_eq_params_dict(params, batch.obs_batch_dict["eq_params"])

            params_ = _set_derivatives(params, "observations", self.derivative_keys)
            mse_observation_loss = observations_loss_apply(
                self.u,
                (batch.obs_batch_dict["pinn_in"],),
                params_,
                vmap_in_axes_x + vmap_in_axes_params,
                batch.obs_batch_dict["val"],
                self.loss_weights["observations"],
                self.obs_slice,
            )
        else:
            mse_observation_loss = jnp.array(0.0)
            self.loss_weights["observations"] = 0

        # Sobolev regularization
        params_ = _set_derivatives(params, "sobolev", self.derivative_keys)
        if self.sobolev_reg is not None:
            mse_sobolev_loss = sobolev_reg_apply(
                self.u,
                (omega_batch,),
                params_,
                vmap_in_axes_x + vmap_in_axes_params,
                self.sobolev_reg,
                self.loss_weights["sobolev"],
            )
        else:
            mse_sobolev_loss = jnp.array(0.0)
            self.loss_weights["sobolev"] = 0

        # total loss
        total_loss = (
            mse_dyn_loss
            + mse_norm_loss
            + mse_boundary_loss
            + mse_observation_loss
            + mse_sobolev_loss
        )
        return total_loss, (
            {
                "dyn_loss": mse_dyn_loss,
                "norm_loss": mse_norm_loss,
                "boundary_loss": mse_boundary_loss,
                "observations": mse_observation_loss,
                "sobolev": mse_sobolev_loss,
                "initial_condition": jnp.array(0.0),  # for compatibility in the
                # tree_map of SystemLoss
            }
        )

    def tree_flatten(self):
        children = (self.norm_key, self.norm_samples, self.loss_weights)
        aux_data = {
            "u": self.u,
            "dynamic_loss": self.dynamic_loss,
            "derivative_keys": self.derivative_keys,
            "omega_boundary_fun": self.omega_boundary_fun,
            "omega_boundary_condition": self.omega_boundary_condition,
            "omega_boundary_dim": self.omega_boundary_dim,
            "norm_borders": self.norm_borders,
            "sobolev_m": self.sobolev_m,
            "obs_slice": self.obs_slice,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (norm_key, norm_samples, loss_weights) = children
        pls = cls(
            aux_data["u"],
            loss_weights,
            aux_data["dynamic_loss"],
            aux_data["derivative_keys"],
            aux_data["omega_boundary_fun"],
            aux_data["omega_boundary_condition"],
            aux_data["omega_boundary_dim"],
            norm_key,
            aux_data["norm_borders"],
            norm_samples,
            aux_data["sobolev_m"],
            aux_data["obs_slice"],
        )
        return pls


@register_pytree_node_class
class LossPDENonStatio(LossPDEStatio):
    r"""Loss object for a stationary partial differential equation

    .. math::
        \mathcal{N}[u](t, x) = 0, \forall t \in I, \forall x \in \Omega

    where :math:`\mathcal{N}[\cdot]` is a differential operator.
    The boundary condition is :math:`u(t, x)=u_b(t, x),\forall
    x\in\delta\Omega, \forall t`.
    The initial condition is :math:`u(0, x)=u_0(x), \forall x\in\Omega`
    The additional condition of
    integrating to 1 can be included, i.e., :math:`\int u(t, x)\mathrm{d}x=1`.


    **Note:** LossPDENonStatio is jittable. Hence it implements the tree_flatten() and
    tree_unflatten methods.
    """

    def __init__(
        self,
        u,
        loss_weights,
        dynamic_loss,
        derivative_keys=None,
        omega_boundary_fun=None,
        omega_boundary_condition=None,
        omega_boundary_dim=None,
        initial_condition_fun=None,
        norm_key=None,
        norm_borders=None,
        norm_samples=None,
        sobolev_m=None,
        obs_slice=None,
    ):
        r"""
        Parameters
        ----------
        u
            the PINN object
        loss_weights
            dictionary of values for loss term ponderation
            Note that we can have jnp.arrays with the same dimension of
            `u` which then ponderates each output of `u`
        dynamic_loss
            A Dynamic loss object whose evaluate method corresponds to the
            dynamic term in the loss
            Can be None in order to access only some part of the evaluate call
            results.
        derivative_keys
            A dict of lists of strings. In the dict, the key must correspond to
            the loss term keywords. Then each of the values must correspond to keys in the parameter
            dictionary (*at top level only of the parameter dictionary*).
            It enables selecting the set of parameters
            with respect to which the gradients of the dynamic
            loss are computed. If nothing is provided, we set ["nn_params"] for all loss term
            keywords, this is what is typically
            done in solving forward problems, when we only estimate the
            equation solution with a PINN. If some loss terms keywords are
            missing we set their value to ["nn_params"] by default for the same
            reason
        omega_boundary_fun
            The function to be matched in the border condition (can be None)
            or a dictionary of such function. In this case, the keys are the
            facets and the values are the functions. The keys must be in the
            following order: 1D -> ["xmin", "xmax"], 2D -> ["xmin", "xmax",
            "ymin", "ymax"]. Note that high order boundaries are currently not
            implemented. A value in the dict can be None, this means we do not
            enforce a particular boundary condition on this facet.
            The facet called "xmin", resp. "xmax" etc., in 2D,
            refers to the set of 2D points with fixed "xmin", resp. "xmax", etc.
        omega_boundary_condition
            Either None (no condition), or a string defining the boundary
            condition e.g. Dirichlet or Von Neumann, or a dictionary of such
            strings. In this case, the keys are the
            facets and the values are the strings. The keys must be in the
            following order: 1D -> ["xmin", "xmax"], 2D -> ["xmin", "xmax",
            "ymin", "ymax"]. Note that high order boundaries are currently not
            implemented. A value in the dict can be None, this means we do not
            enforce a particular boundary condition on this facet.
            The facet called "xmin", resp. "xmax" etc., in 2D,
            refers to the set of 2D points with fixed "xmin", resp. "xmax", etc.
        omega_boundary_dim
            Either None, or a jnp.s\_ or a dict of jnp.s\_ with keys following
            the logic of omega_boundary_fun. It indicates which dimension(s) of
            the PINN will be forced to match the boundary condition
            Note that it must be a slice and not an integer (a preprocessing of the
            user provided argument takes care of it)
        initial_condition_fun
            A function representing the temporal initial condition. If None
            (default) then no initial condition is applied
        norm_key
            Jax random key to draw samples in for the Monte Carlo computation
            of the normalization constant. Default is None
        norm_borders
            tuple of (min, max) of the boundaray values of the space over which
            to integrate in the computation of the normalization constant.
            A list of tuple for higher dimensional problems. Default None.
        norm_samples
            Fixed sample point in the space over which to compute the
            normalization constant. Default is None
        sobolev_m
            An integer. Default is None.
            It corresponds to the Sobolev regularization order as proposed in
            *Convergence and error analysis of PINNs*,
            Doumeche et al., 2023, https://arxiv.org/pdf/2305.01240.pdf
        obs_slice
            slice object specifying the begininning/ending
            slice of u output(s) that is observed (this is then useful for
            multidim PINN). Default is None.


        """

        super().__init__(
            u,
            loss_weights,
            dynamic_loss,
            derivative_keys,
            omega_boundary_fun,
            omega_boundary_condition,
            omega_boundary_dim,
            norm_key,
            norm_borders,
            norm_samples,
            sobolev_m=sobolev_m,
            obs_slice=obs_slice,
        )
        if initial_condition_fun is None:
            warnings.warn(
                "Initial condition wasn't provided. Be sure to cover for that"
                "case (e.g by. hardcoding it into the PINN output)."
            )
        self.initial_condition_fun = initial_condition_fun

        self.sobolev_m = sobolev_m
        if self.sobolev_m is not None:
            # This overwrite the wrongly initialized self.sobolev_reg with
            # statio=True in the LossPDEStatio init
            self.sobolev_reg = _sobolev(self.u, self.sobolev_m, statio=False)
            # we return a function, that way
            # the order of sobolev_m is static and the conditional in the recursive
            # function is properly set
        else:
            self.sobolev_reg = None

        if self.sobolev_reg is None:
            self.loss_weights["sobolev"] = 0

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(
        self,
        params,
        batch,
    ):
        """
        Evaluate the loss function at a batch of points for given parameters.


        Parameters
        ---------
        params
            The dictionary of parameters of the model.
            Typically, it is a dictionary of
            dictionaries: `eq_params` and `nn_params`, respectively the
            differential equation parameters and the neural network parameter
        batch
            A PDENonStatioBatch object.
            Such a named tuple is composed of a batch of points in
            the domain, a batch of points in the domain
            border, a batch of time points and an optional additional batch
            of parameters (eg. for metamodeling) and an optional additional batch of observed
            inputs/outputs/parameters
        """

        omega_batch, omega_border_batch, times_batch = (
            batch.inside_batch,
            batch.border_batch,
            batch.temporal_batch,
        )
        n = omega_batch.shape[0]
        nt = times_batch.shape[0]
        times_batch = times_batch.reshape(nt, 1)

        def rep_times(k):
            return jnp.repeat(times_batch, k, axis=0)

        vmap_in_axes_x_t = (0, 0)

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            eq_params_batch_dict = batch.param_batch_dict

            # feed the eq_params with the batch
            for k in eq_params_batch_dict.keys():
                params["eq_params"][k] = eq_params_batch_dict[k]

        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)

        if isinstance(self.u, PINN):
            omega_batch = jnp.tile(omega_batch, reps=(nt, 1))  # it is tiled
            times_batch = rep_times(n)  # it is repeated

        # dynamic part
        params_ = _set_derivatives(params, "dyn_loss", self.derivative_keys)
        if self.dynamic_loss is not None:
            mse_dyn_loss = dynamic_loss_apply(
                self.dynamic_loss.evaluate,
                self.u,
                (times_batch, omega_batch),
                params_,
                vmap_in_axes_x_t + vmap_in_axes_params,
                self.loss_weights["dyn_loss"],
            )
        else:
            mse_dyn_loss = jnp.array(0.0)

        # normalization part
        params_ = _set_derivatives(params, "norm_loss", self.derivative_keys)
        if self.normalization_loss is not None:
            mse_norm_loss = normalization_loss_apply(
                self.u,
                (times_batch, self.get_norm_samples()),
                params_,
                vmap_in_axes_x_t + vmap_in_axes_params,
                self.int_length,
                self.loss_weights["norm_loss"],
            )
        else:
            mse_norm_loss = jnp.array(0.0)

        # boundary part
        params_ = _set_derivatives(params, "boundary_loss", self.derivative_keys)
        if self.omega_boundary_fun is not None:
            mse_boundary_loss = boundary_condition_apply(
                self.u,
                batch,
                params_,
                self.omega_boundary_fun,
                self.omega_boundary_condition,
                self.omega_boundary_dim,
                self.loss_weights["boundary_loss"],
            )
        else:
            mse_boundary_loss = jnp.array(0.0)

        # initial condition
        params_ = _set_derivatives(params, "initial_condition", self.derivative_keys)
        if self.initial_condition_fun is not None:
            mse_initial_condition = initial_condition_apply(
                self.u,
                omega_batch,
                params_,
                (0,) + vmap_in_axes_params,
                self.initial_condition_fun,
                n,
                self.loss_weights["initial_condition"],
            )
        else:
            mse_initial_condition = jnp.array(0.0)

        # Observation mse
        if batch.obs_batch_dict is not None:
            # update params with the batches of observed params
            params = _update_eq_params_dict(params, batch.obs_batch_dict["eq_params"])

            params_ = _set_derivatives(params, "observations", self.derivative_keys)
            mse_observation_loss = observations_loss_apply(
                self.u,
                (
                    batch.obs_batch_dict["pinn_in"][:, 0:1],
                    batch.obs_batch_dict["pinn_in"][:, 1:],
                ),
                params_,
                vmap_in_axes_x_t + vmap_in_axes_params,
                batch.obs_batch_dict["val"],
                self.loss_weights["observations"],
                self.obs_slice,
            )
        else:
            mse_observation_loss = jnp.array(0.0)
            self.loss_weights["observations"] = 0

        # Sobolev regularization
        params_ = _set_derivatives(params, "sobolev", self.derivative_keys)
        if self.sobolev_reg is not None:
            mse_sobolev_loss = sobolev_reg_apply(
                self.u,
                (omega_batch, times_batch),
                params_,
                vmap_in_axes_x_t + vmap_in_axes_params,
                self.sobolev_reg,
                self.loss_weights["sobolev"],
            )
        else:
            mse_sobolev_loss = jnp.array(0.0)
            self.loss_weights["sobolev"] = 0.0

        # total loss
        total_loss = (
            mse_dyn_loss
            + mse_norm_loss
            + mse_boundary_loss
            + mse_initial_condition
            + mse_observation_loss
            + mse_sobolev_loss
        )

        return total_loss, (
            {
                "dyn_loss": mse_dyn_loss,
                "norm_loss": mse_norm_loss,
                "boundary_loss": mse_boundary_loss,
                "initial_condition": mse_initial_condition,
                "observations": mse_observation_loss,
                "sobolev": mse_sobolev_loss,
            }
        )

    def tree_flatten(self):
        children = (self.norm_key, self.norm_samples, self.loss_weights)
        aux_data = {
            "u": self.u,
            "dynamic_loss": self.dynamic_loss,
            "derivative_keys": self.derivative_keys,
            "omega_boundary_fun": self.omega_boundary_fun,
            "omega_boundary_condition": self.omega_boundary_condition,
            "omega_boundary_dim": self.omega_boundary_dim,
            "initial_condition_fun": self.initial_condition_fun,
            "norm_borders": self.norm_borders,
            "sobolev_m": self.sobolev_m,
            "obs_slice": self.obs_slice,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (norm_key, norm_samples, loss_weights) = children
        pls = cls(
            aux_data["u"],
            loss_weights,
            aux_data["dynamic_loss"],
            aux_data["derivative_keys"],
            aux_data["omega_boundary_fun"],
            aux_data["omega_boundary_condition"],
            aux_data["omega_boundary_dim"],
            aux_data["initial_condition_fun"],
            norm_key,
            aux_data["norm_borders"],
            norm_samples,
            aux_data["sobolev_m"],
            aux_data["obs_slice"],
        )
        return pls


@register_pytree_node_class
class SystemLossPDE:
    """
    Class to implement a system of PDEs.
    The goal is to give maximum freedom to the user. The class is created with
    a dict of dynamic loss, and dictionaries of all the objects that are used
    in LossPDENonStatio and LossPDEStatio. When then iterate
    over the dynamic losses that compose the system. All the PINNs with all the
    parameter dictionaries are passed as arguments to each dynamic loss
    evaluate functions; it is inside the dynamic loss that specification are
    performed.

    **Note:** All the dictionaries (except `dynamic_loss_dict`) must have the same keys.
    Indeed, these dictionaries (except `dynamic_loss_dict`) are tied to one
    solution.

    **Note:** SystemLossPDE is jittable. Hence it implements the tree_flatten() and
    tree_unflatten methods.
    """

    def __init__(
        self,
        u_dict,
        loss_weights,
        dynamic_loss_dict,
        nn_type_dict,
        derivative_keys_dict=None,
        omega_boundary_fun_dict=None,
        omega_boundary_condition_dict=None,
        omega_boundary_dim_dict=None,
        initial_condition_fun_dict=None,
        norm_key_dict=None,
        norm_borders_dict=None,
        norm_samples_dict=None,
        sobolev_m_dict=None,
        obs_slice_dict=None,
    ):
        r"""
        Parameters
        ----------
        u_dict
            A dict of PINNs
        loss_weights
            A dictionary of dictionaries with values used to
            ponderate each term in the loss
            function. The keys of the nested
            dictionaries must share the keys of `u_dict`. Note that the values
            at the leaf level can have jnp.arrays with the same dimension of
            `u` which then ponderates each output of `u`
        dynamic_loss_dict
            A dict of dynamic part of the loss, basically the differential
            operator :math:`\mathcal{N}[u](t)`.
        nn_type_dict
            A dict whose keys are that of u_dict whose value is either
            `nn_statio` or `nn_nonstatio` which signifies either the PINN has a
            time component in input or not.
        derivative_keys_dict
            A dict of derivative keys as defined in LossODE. The key of this
            dict must be that of `dynamic_loss_dict` at least and specify how
            to compute gradient for the `dyn_loss` loss term at least (see the
            check at the beginning of the present `__init__` function.
            Other keys of this dict might be that of `u_dict` to specify how to
            compute gradients for all the different constraints. If those keys
            are not specified then the default behaviour for `derivative_keys`
            of LossODE is used
        omega_boundary_fun_dict
            A dict of dict of functions (see doc for `omega_boundary_fun` in
            LossPDEStatio or LossPDENonStatio). Default is None.
            Must share the keys of `u_dict`.
        omega_boundary_condition_dict
            A dict of dict of strings (see doc for
            `omega_boundary_condition_dict` in
            LossPDEStatio or LossPDENonStatio). Default is None.
            Must share the keys of `u_dict`
        omega_boundary_dim_dict
            A dict of dict of slices (see doc for `omega_boundary_dim` in
            LossPDEStatio or LossPDENonStatio). Default is None.
            Must share the keys of `u_dict`
        initial_condition_fun_dict
            A dict of functions representing the temporal initial condition. If None
            (default) then no temporal boundary condition is applied
            Must share the keys of `u_dict`
        norm_key_dict
            A dict of Jax random keys to draw samples in for the Monte Carlo computation
            of the normalization constant. Default is None
            Must share the keys of `u_dict`
        norm_borders_dict
            A dict of tuples of (min, max) of the boundaray values of the space over which
            to integrate in the computation of the normalization constant.
            A list of tuple for higher dimensional problems. Default None.
            Must share the keys of `u_dict`
        norm_samples_dict
            A dict of fixed sample point in the space over which to compute the
            normalization constant. Default is None
            Must share the keys of `u_dict`
        sobolev_m
            Default is None. A dictionary of integers, one per key which must
            match `u_dict`.
            It corresponds to the Sobolev regularization order as proposed in
            *Convergence and error analysis of PINNs*,
            Doumeche et al., 2023, https://arxiv.org/pdf/2305.01240.pdf
        obs_slice_dict
            dict of obs_slice, with keys from `u_dict` to designate the
            output(s) channels that are forced to observed values, for each
            PINNs. Default is None. But if a value is given, all the entries of
            `u_dict` must be represented here with default value `jnp.s_[...]`
            if no particular slice is to be given


        Raises
        ------
        ValueError
            if initial condition is not a dict of tuple
        ValueError
            if the dictionaries that should share the keys of u_dict do not
        """
        # a dictionary that will be useful at different places
        self.u_dict_with_none = {k: None for k in u_dict.keys()}
        # First, for all the optional dict,
        # if the user did not provide at all this optional argument,
        # we make sure there is a null ponderating loss_weight and we
        # create a dummy dict with the required keys and all the values to
        # None
        if omega_boundary_fun_dict is None:
            self.omega_boundary_fun_dict = self.u_dict_with_none
        else:
            self.omega_boundary_fun_dict = omega_boundary_fun_dict
        if omega_boundary_condition_dict is None:
            self.omega_boundary_condition_dict = self.u_dict_with_none
        else:
            self.omega_boundary_condition_dict = omega_boundary_condition_dict
        if omega_boundary_dim_dict is None:
            self.omega_boundary_dim_dict = self.u_dict_with_none
        else:
            self.omega_boundary_dim_dict = omega_boundary_dim_dict
        if initial_condition_fun_dict is None:
            self.initial_condition_fun_dict = self.u_dict_with_none
        else:
            self.initial_condition_fun_dict = initial_condition_fun_dict
        if norm_key_dict is None:
            self.norm_key_dict = self.u_dict_with_none
        else:
            self.norm_key_dict = norm_key_dict
        if norm_borders_dict is None:
            self.norm_borders_dict = self.u_dict_with_none
        else:
            self.norm_borders_dict = norm_borders_dict
        if norm_samples_dict is None:
            self.norm_samples_dict = self.u_dict_with_none
        else:
            self.norm_samples_dict = norm_samples_dict
        if sobolev_m_dict is None:
            self.sobolev_m_dict = self.u_dict_with_none
        else:
            self.sobolev_m_dict = sobolev_m_dict
        if obs_slice_dict is None:
            self.obs_slice_dict = {k: jnp.s_[...] for k in u_dict.keys()}
        else:
            self.obs_slice_dict = obs_slice_dict
            if u_dict.keys() != obs_slice_dict.keys():
                raise ValueError("obs_slice_dict should have same keys as u_dict")
        if derivative_keys_dict is None:
            self.derivative_keys_dict = {
                k: None
                for k in set(list(dynamic_loss_dict.keys()) + list(u_dict.keys()))
            }
            # set() because we can have duplicate entries and in this case we
            # say it corresponds to the same derivative_keys_dict entry
        else:
            self.derivative_keys_dict = derivative_keys_dict
        # but then if the user did not provide anything, we must at least have
        # a default value for the dynamic_loss_dict keys entries in
        # self.derivative_keys_dict since the computation of dynamic losses is
        # made without create a lossODE object that would provide the
        # default values
        for k in dynamic_loss_dict.keys():
            if self.derivative_keys_dict[k] is None:
                self.derivative_keys_dict[k] = {"dyn_loss": ["nn_params"]}

        # Second we make sure that all the dicts (except dynamic_loss_dict) have the same keys
        if (
            u_dict.keys() != nn_type_dict.keys()
            or u_dict.keys() != self.omega_boundary_fun_dict.keys()
            or u_dict.keys() != self.omega_boundary_condition_dict.keys()
            or u_dict.keys() != self.omega_boundary_dim_dict.keys()
            or u_dict.keys() != self.initial_condition_fun_dict.keys()
            or u_dict.keys() != self.norm_key_dict.keys()
            or u_dict.keys() != self.norm_borders_dict.keys()
            or u_dict.keys() != self.norm_samples_dict.keys()
            or u_dict.keys() != self.sobolev_m_dict.keys()
        ):
            raise ValueError("All the dicts concerning the PINNs should have same keys")

        self.dynamic_loss_dict = dynamic_loss_dict
        self.u_dict = u_dict
        # TODO nn_type should become a class attribute now that we have PINN
        # class and SPINNs class
        self.nn_type_dict = nn_type_dict

        self.loss_weights = loss_weights  # This calls the setter

        # Third, in order not to benefit from LossPDEStatio and
        # LossPDENonStatio and in order to factorize code, we create internally
        # some losses object to implement the constraints on the solutions.
        # We will not use the dynamic loss term
        self.u_constraints_dict = {}
        for i in self.u_dict.keys():
            if self.nn_type_dict[i] == "nn_statio":
                self.u_constraints_dict[i] = LossPDEStatio(
                    u=u_dict[i],
                    loss_weights={
                        "dyn_loss": 0.0,
                        "norm_loss": 1.0,
                        "boundary_loss": 1.0,
                        "observations": 1.0,
                        "sobolev": 1.0,
                    },
                    dynamic_loss=None,
                    derivative_keys=self.derivative_keys_dict[i],
                    omega_boundary_fun=self.omega_boundary_fun_dict[i],
                    omega_boundary_condition=self.omega_boundary_condition_dict[i],
                    omega_boundary_dim=self.omega_boundary_dim_dict[i],
                    norm_key=self.norm_key_dict[i],
                    norm_borders=self.norm_borders_dict[i],
                    norm_samples=self.norm_samples_dict[i],
                    sobolev_m=self.sobolev_m_dict[i],
                    obs_slice=self.obs_slice_dict[i],
                )
            elif self.nn_type_dict[i] == "nn_nonstatio":
                self.u_constraints_dict[i] = LossPDENonStatio(
                    u=u_dict[i],
                    loss_weights={
                        "dyn_loss": 0.0,
                        "norm_loss": 1.0,
                        "boundary_loss": 1.0,
                        "observations": 1.0,
                        "initial_condition": 1.0,
                        "sobolev": 1.0,
                    },
                    dynamic_loss=None,
                    derivative_keys=self.derivative_keys_dict[i],
                    omega_boundary_fun=self.omega_boundary_fun_dict[i],
                    omega_boundary_condition=self.omega_boundary_condition_dict[i],
                    omega_boundary_dim=self.omega_boundary_dim_dict[i],
                    initial_condition_fun=self.initial_condition_fun_dict[i],
                    norm_key=self.norm_key_dict[i],
                    norm_borders=self.norm_borders_dict[i],
                    norm_samples=self.norm_samples_dict[i],
                    sobolev_m=self.sobolev_m_dict[i],
                )
            else:
                raise ValueError(
                    f"Wrong value for nn_type_dict[i], got {nn_type_dict[i]}"
                )

        # for convenience in the tree_map of evaluate,
        # we separate the two derivative keys dict
        self.derivative_keys_dyn_loss_dict = {
            k: self.derivative_keys_dict[k]
            for k in self.dynamic_loss_dict.keys() & self.derivative_keys_dict.keys()
        }
        self.derivative_keys_u_dict = {
            k: self.derivative_keys_dict[k]
            for k in self.u_dict.keys() & self.derivative_keys_dict.keys()
        }

        # also make sure we only have PINNs or SPINNs
        if not (
            all(isinstance(value, PINN) for value in u_dict.values())
            or all(isinstance(value, SPINN) for value in u_dict.values())
        ):
            raise ValueError(
                "We only accept dictionary of PINNs or dictionary of SPINNs"
            )

    @property
    def loss_weights(self):
        return self._loss_weights

    @loss_weights.setter
    def loss_weights(self, value):
        self._loss_weights = {}
        for k, v in value.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if not isinstance(vv, (int, float)) and not (
                        isinstance(vv, jnp.ndarray)
                        and ((vv.shape == (1,) or len(vv.shape) == 0))
                    ):
                        # TODO improve that
                        raise ValueError(
                            f"loss values cannot be vectorial here, got {vv}"
                        )
                if k == "dyn_loss":
                    if v.keys() == self.dynamic_loss_dict.keys():
                        self._loss_weights[k] = v
                    else:
                        raise ValueError(
                            "Keys in nested dictionary of loss_weights"
                            " do not match dynamic_loss_dict keys"
                        )
                else:
                    if v.keys() == self.u_dict.keys():
                        self._loss_weights[k] = v
                    else:
                        raise ValueError(
                            "Keys in nested dictionary of loss_weights"
                            " do not match u_dict keys"
                        )
            else:
                if not isinstance(v, (int, float)) and not (
                    isinstance(v, jnp.ndarray)
                    and ((v.shape == (1,) or len(v.shape) == 0))
                ):
                    # TODO improve that
                    raise ValueError(f"loss values cannot be vectorial here, got {v}")
                if k == "dyn_loss":
                    self._loss_weights[k] = {
                        kk: v for kk in self.dynamic_loss_dict.keys()
                    }
                else:
                    self._loss_weights[k] = {kk: v for kk in self.u_dict.keys()}
        # Some special checks below
        if all(v is None for k, v in self.sobolev_m_dict.items()):
            self._loss_weights["sobolev"] = {k: 0 for k in self.u_dict.keys()}
        if "observations" not in value.keys():
            self._loss_weights["observations"] = {k: 0 for k in self.u_dict.keys()}
        if all(v is None for k, v in self.omega_boundary_fun_dict.items()) or all(
            v is None for k, v in self.omega_boundary_condition_dict.items()
        ):
            self._loss_weights["boundary_loss"] = {k: 0 for k in self.u_dict.keys()}
        if (
            all(v is None for k, v in self.norm_key_dict.items())
            or all(v is None for k, v in self.norm_borders_dict.items())
            or all(v is None for k, v in self.norm_samples_dict.items())
        ):
            self._loss_weights["norm_loss"] = {k: 0 for k in self.u_dict.keys()}
        if all(v is None for k, v in self.initial_condition_fun_dict.items()):
            self._loss_weights["initial_condition"] = {k: 0 for k in self.u_dict.keys()}

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(
        self,
        params_dict,
        batch,
    ):
        """
        Evaluate the loss function at a batch of points for given parameters.


        Parameters
        ---------
        params_dict
            A dictionary of dictionaries of parameters of the model.
            Typically, it is a dictionary of dictionaries of
            dictionaries: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter
        batch
            A PDEStatioBatch or PDENonStatioBatch object.
            Such named tuples are composed of  batch of points in the
            domain, a batch of points in the domain
            border, (a batch of time points a for PDENonStatioBatch) and an
            optional additional batch of parameters (eg. for metamodeling)
            and an optional additional batch of observed
            inputs/outputs/parameters
        """
        if self.u_dict.keys() != params_dict["nn_params"].keys():
            raise ValueError("u_dict and params_dict[nn_params] should have same keys ")

        if isinstance(batch, PDEStatioBatch):
            omega_batch, _ = batch.inside_batch, batch.border_batch
            n = omega_batch.shape[0]
            vmap_in_axes_x_or_x_t = (0,)

            batches = (omega_batch,)
        elif isinstance(batch, PDENonStatioBatch):
            omega_batch, _, times_batch = (
                batch.inside_batch,
                batch.border_batch,
                batch.temporal_batch,
            )
            n = omega_batch.shape[0]
            nt = times_batch.shape[0]
            times_batch = times_batch.reshape(nt, 1)

            def rep_times(k):
                return jnp.repeat(times_batch, k, axis=0)

            # Moreover...
            if isinstance(list(self.u_dict.values())[0], PINN):
                omega_batch = jnp.tile(omega_batch, reps=(nt, 1))  # it is tiled
                times_batch = rep_times(n)  # it is repeated

            batches = (omega_batch, times_batch)
            vmap_in_axes_x_or_x_t = (0, 0)
        else:
            raise ValueError("Wrong type of batch")

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            eq_params_batch_dict = batch.param_batch_dict

            # feed the eq_params with the batch
            for k in eq_params_batch_dict.keys():
                params_dict["eq_params"][k] = eq_params_batch_dict[k]

        vmap_in_axes_params = _get_vmap_in_axes_params(
            batch.param_batch_dict, params_dict
        )

        def dyn_loss_for_one_key(dyn_loss, derivative_key, loss_weight):
            """The function used in tree_map"""
            params_dict_ = _set_derivatives(params_dict, "dyn_loss", derivative_key)
            return dynamic_loss_apply(
                dyn_loss.evaluate,
                self.u_dict,
                batches,
                params_dict_,
                vmap_in_axes_x_or_x_t + vmap_in_axes_params,
                loss_weight,
                u_type=type(list(self.u_dict.values())[0]),
            )

        dyn_loss_mse_dict = jax.tree_util.tree_map(
            dyn_loss_for_one_key,
            self.dynamic_loss_dict,
            self.derivative_keys_dyn_loss_dict,
            self._loss_weights["dyn_loss"],
        )
        mse_dyn_loss = jax.tree_util.tree_reduce(
            lambda x, y: x + y, jax.tree_util.tree_leaves(dyn_loss_mse_dict)
        )

        # boundary conditions, normalization conditions, observation_loss,
        # initial condition... loss this is done via the internal
        # LossPDEStatio and NonStatio
        loss_weight_struct = {
            "dyn_loss": "*",
            "norm_loss": "*",
            "boundary_loss": "*",
            "observations": "*",
            "initial_condition": "*",
            "sobolev": "*",
        }
        # we need to do the following for the tree_mapping to work
        if batch.obs_batch_dict is None:
            batch = batch._replace(obs_batch_dict=self.u_dict_with_none)
        total_loss, res_dict = constraints_system_loss_apply(
            self.u_constraints_dict,
            batch,
            params_dict,
            self._loss_weights,
            loss_weight_struct,
        )

        # Add the mse_dyn_loss from the previous computations
        total_loss += mse_dyn_loss
        res_dict["dyn_loss"] += mse_dyn_loss
        return total_loss, res_dict

    def tree_flatten(self):
        children = (
            self.norm_key_dict,
            self.norm_samples_dict,
            self.initial_condition_fun_dict,
            self._loss_weights,
        )
        aux_data = {
            "u_dict": self.u_dict,
            "dynamic_loss_dict": self.dynamic_loss_dict,
            "norm_borders_dict": self.norm_borders_dict,
            "omega_boundary_fun_dict": self.omega_boundary_fun_dict,
            "omega_boundary_condition_dict": self.omega_boundary_condition_dict,
            "nn_type_dict": self.nn_type_dict,
            "sobolev_m_dict": self.sobolev_m_dict,
            "derivative_keys_dict": self.derivative_keys_dict,
            "obs_slice_dict": self.obs_slice_dict,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            norm_key_dict,
            norm_samples_dict,
            initial_condition_fun_dict,
            loss_weights,
        ) = children
        loss_ode = cls(
            loss_weights=loss_weights,
            norm_key_dict=norm_key_dict,
            norm_samples_dict=norm_samples_dict,
            initial_condition_fun_dict=initial_condition_fun_dict,
            **aux_data,
        )

        return loss_ode
