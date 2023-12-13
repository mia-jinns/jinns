import jax
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import register_pytree_node_class
from functools import partial
import warnings
from jinns.loss._boundary_conditions import (
    _compute_boundary_loss_statio,
    _compute_boundary_loss_nonstatio,
)
from jinns.loss._DynamicLoss import ODE, PDEStatio, PDENonStatio
from jinns.data._DataGenerators import PDEStatioBatch, PDENonStatioBatch
from jinns.utils._utils import (
    _get_vmap_in_axes_params,
    _get_grid,
    _check_user_func_return,
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
            for i in range(len(self.norm_borders)):
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
            for i in range(len(self.norm_borders)):
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
        elif self.norm_sample_method == "generate":
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
        else:
            raise RuntimeError("Problem with the value of self.norm_sample_method")

    def tree_flatten(self):
        children = (self.norm_key, self.norm_samples, self.loss_weights)
        aux_data = {
            "norm_borders": self.norm_borders,
            "u": self.u,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(self, aux_data, children):
        (norm_key, norm_samples, loss_weights) = children
        pls = self(
            aux_data["u"],
            loss_weights,
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
        omega_boundary_fun=None,
        omega_boundary_condition=None,
        norm_key=None,
        norm_borders=None,
        norm_samples=None,
        obs_batch=None,
        sobolev_m=None,
    ):
        """
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
            operator :math:`\mathcal{N}[u](t)`. Should implement a method
            `dynamic_loss.evaluate(t, u, params)`.
            Can be None in order to access only some part of the evaluate call
            results.
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
        obs_batch:
            A list containing 2 jnp.array of the same size (on their first
            axis) encoding for observations [:math:`x_i, u(x_i)`] for
            computing a MSE. Default is None.
        sobolev_m
            An integer. Default is None.
            It corresponds to the Sobolev regularization order as proposed in
            *Convergence and error analysis of PINNs*,
            Doumeche et al., 2023, https://arxiv.org/pdf/2305.01240.pdf


        Raises
        ------
        ValueError
            If conditions on obs_batch are not respected
        ValueError
            If conditions on omega_boundary_condition and omega_boundary_fun
            are not respected
        """

        if obs_batch is not None:
            if len(obs_batch) != 2:
                raise ValueError(
                    f"obs_batch must be a list of size 2. You gave {len(obs_batch)}"
                )
            elif any([isinstance(b, jnp.array) for b in obs_batch]):
                raise ValueError("Every element of obs_batch should be a jnp.array.")
            n_obs = obs_batch[0].shape[0]
            if any([b.shape[0] != n_obs for b in obs_batch]):
                raise ValueError(
                    "Every jnp array should have the same size of the first axis (number of observations)."
                )
            if obs_batch[1].ndim == 1:
                # if omega domain is unidimensional make sure that the x
                # obs_batch is (nb_obs, 1)
                obs_batch[1] = obs_batch[1][:, None]

        super().__init__(u, loss_weights, norm_key, norm_borders, norm_samples)

        if omega_boundary_condition is None or omega_boundary_fun is None:
            warnings.warn(
                "Missing boundary function or no boundary condition."
                "Boundary function is thus ignored."
            )
        else:
            if type(omega_boundary_condition) is dict:
                for k, v in omega_boundary_condition.items():
                    if v is not None and not any(
                        [v.lower() in s for s in _IMPLEMENTED_BOUNDARY_CONDITIONS]
                    ):
                        raise NotImplementedError(
                            f"The boundary condition {omega_boundary_condition} is not"
                            f"implemented yet. Try one of :"
                            f"{_IMPLEMENTED_BOUNDARY_CONDITIONS}."
                        )
            else:
                if not any(
                    [
                        omega_boundary_condition.lower() in s
                        for s in _IMPLEMENTED_BOUNDARY_CONDITIONS
                    ]
                ):
                    raise NotImplementedError(
                        f"The boundary condition {omega_boundary_condition} is not"
                        f"implemented yet. Try one of :"
                        f"{_IMPLEMENTED_BOUNDARY_CONDITIONS}."
                    )
                if (
                    type(omega_boundary_fun) is dict
                    and type(omega_boundary_condition) is dict
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
        self.dynamic_loss = dynamic_loss
        self.obs_batch = obs_batch

        if sobolev_m is not None:
            self.sobolev_reg = _sobolev(
                self.u, sobolev_m
            )  # we return a function, that way
            # the order of sobolev_m is static and the conditional in the recursive
            # function is properly set
        else:
            self.sobolev_reg = None

        if self.normalization_loss is None:
            self.loss_weights["norm_loss"] = 0

        if self.omega_boundary_fun is None:
            self.loss_weights["boundary_loss"] = 0

        if self.sobolev_reg is None:
            self.loss_weights["sobolev"] = 0

        if (
            type(self.omega_boundary_fun) is dict
            and not (type(self.omega_boundary_condition) is dict)
        ) or (
            not (type(self.omega_boundary_fun) is dict)
            and type(self.omega_boundary_condition) is dict
        ):
            raise ValueError(
                "if one of self.omega_boundary_fun or "
                "self.omega_boundary_condition is dict, the other should be too."
            )

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
            A tuple.
            A batch of points in the domain and a batch of points in the domain
            border
        """
        omega_batch, omega_border_batch = batch.inside_batch, batch.border_batch
        n = omega_batch.shape[0]

        vmap_in_axes_x = (0,)

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            eq_params_batch_dict = batch.param_batch_dict

            # feed the eq_params with the batch
            for k in eq_params_batch_dict.keys():
                params["eq_params"][k] = eq_params_batch_dict[k]

        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)

        # dynamic part
        if self.dynamic_loss is not None:
            if isinstance(self.u, PINN):
                v_dyn_loss = vmap(
                    lambda x, params: self.dynamic_loss.evaluate(
                        x,
                        self.u,
                        params,
                    ),
                    vmap_in_axes_x + vmap_in_axes_params,
                    0,
                )
                mse_dyn_loss = jnp.mean(
                    self.loss_weights["dyn_loss"]
                    * jnp.sum(v_dyn_loss(omega_batch, params) ** 2, axis=-1)
                )
            elif isinstance(self.u, SPINN):
                residuals = self.dynamic_loss.evaluate(omega_batch, self.u, params)
                mse_dyn_loss = jnp.mean(
                    self.loss_weights["dyn_loss"] * jnp.sum(residuals**2, axis=-1)
                )

        else:
            mse_dyn_loss = 0

        # normalization part
        if self.normalization_loss is not None:
            if isinstance(self.u, PINN):
                v_u = vmap(
                    partial(
                        self.u,
                        u_params=params["nn_params"],
                        eq_params=jax.lax.stop_gradient(params["eq_params"]),
                    ),
                    (0),
                    0,
                )
                mse_norm_loss = self.loss_weights["norm_loss"] * jnp.mean(
                    jnp.abs(
                        jnp.mean(v_u(self.get_norm_samples()), axis=-1)
                        * self.int_length
                        - 1
                    )
                    ** 2
                )
            elif isinstance(self.u, SPINN):
                norm_samples = self.get_norm_samples()
                res = self.u(
                    norm_samples,
                    params["nn_params"],
                    jax.lax.stop_gradient(params["eq_params"]),
                )
                mse_norm_loss = self.loss_weights["norm_loss"] * (
                    jnp.abs(
                        jnp.mean(
                            jnp.mean(res, axis=-1),
                            axis=tuple(range(res.ndim - 1)),
                        )
                        * self.int_length
                        - 1
                    )
                    ** 2
                )
        else:
            mse_norm_loss = 0
            self.loss_weights["norm_loss"] = 0

        # boundary part
        if self.omega_boundary_condition is not None:
            if type(self.omega_boundary_fun) is dict:
                # means self.omega_boundary_condition is dict too because of
                # check in init
                mse_boundary_loss = 0
                for idx, facet in enumerate(self.omega_boundary_fun.keys()):
                    if self.omega_boundary_condition[facet] is not None:
                        mse_boundary_loss += jnp.mean(
                            self.loss_weights["boundary_loss"]
                            * _compute_boundary_loss_statio(
                                self.omega_boundary_condition[facet],
                                self.omega_boundary_fun[facet],
                                omega_border_batch[..., idx],
                                self.u,
                                params,
                                idx,
                            )
                        )
            else:
                mse_boundary_loss = 0
                for facet in range(omega_border_batch.shape[-1]):
                    mse_boundary_loss += jnp.mean(
                        self.loss_weights["boundary_loss"]
                        * _compute_boundary_loss_statio(
                            self.omega_boundary_condition,
                            self.omega_boundary_fun,
                            omega_border_batch[..., facet],
                            self.u,
                            params,
                            facet,
                        )
                    )
        else:
            mse_boundary_loss = 0

        # Observation MSE (if obs_batch provided)
        # NOTE that it does not use jax.lax.stop_gradient on "eq_params" here
        # since we may wish to optimize on it.
        if self.obs_batch is not None:
            # TODO implement for SPINN
            if isinstance(self.u, PINN):
                v_u = vmap(
                    lambda x: self.u(x, params["nn_params"], params["eq_params"]),
                    0,
                    0,
                )
                mse_observation_loss = jnp.mean(
                    self.loss_weights["observations"]
                    * jnp.sum(
                        (v_u(self.obs_batch[0][:, None]) - self.obs_batch[1]) ** 2,
                        axis=-1,
                    )
                )
            elif isinstance(self.u, SPINN):
                raise RuntimeError(
                    "observation loss term not yet implemented for SPINNs"
                )
        else:
            mse_observation_loss = 0
            self.loss_weights["observations"] = 0

        # Sobolev regularization
        if self.sobolev_reg is not None:
            # TODO implement for SPINN
            if isinstance(self.u, PINN):
                v_sob_reg = vmap(
                    lambda x: self.sobolev_reg(
                        x,
                        params["nn_params"],
                        jax.lax.stop_gradient(params["eq_params"]),
                    ),
                    (0, 0),
                    0,
                )
                mse_sobolev_loss = self.loss_weights["sobolev"] * jnp.mean(
                    v_sob_reg(omega_batch)
                )
            elif isinstance(self.u, SPINN):
                raise RuntimeError("Sobolev loss term not yet implemented for SPINNs")
        else:
            mse_sobolev_loss = 0
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
            }
        )

    def tree_flatten(self):
        children = (self.norm_key, self.norm_samples, self.obs_batch, self.loss_weights)
        aux_data = {
            "u": self.u,
            "dynamic_loss": self.dynamic_loss,
            "omega_boundary_fun": self.omega_boundary_fun,
            "omega_boundary_condition": self.omega_boundary_condition,
            "norm_borders": self.norm_borders,
            "sobolev_m": self.sobolev_m,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (norm_key, norm_samples, obs_batch, loss_weights) = children
        pls = cls(
            aux_data["u"],
            loss_weights,
            aux_data["dynamic_loss"],
            aux_data["omega_boundary_fun"],
            aux_data["omega_boundary_condition"],
            norm_key,
            aux_data["norm_borders"],
            norm_samples,
            obs_batch,
            aux_data["sobolev_m"],
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
        omega_boundary_fun=None,
        omega_boundary_condition=None,
        initial_condition_fun=None,
        norm_key=None,
        norm_borders=None,
        norm_samples=None,
        obs_batch=None,
        sobolev_m=None,
    ):
        """
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
        obs_batch
            A list of 3 jnp.array of the same size (on their 1st axis) encoding
            for observations [:math:`t_i, x_i, u(t_i, x_i)`] for computing a
            MSE. Default is None: no MSE term in the global loss.
        sobolev_m
            An integer. Default is None.
            It corresponds to the Sobolev regularization order as proposed in
            *Convergence and error analysis of PINNs*,
            Doumeche et al., 2023, https://arxiv.org/pdf/2305.01240.pdf


        Raises
        ------
        ValueError
            If conditions on obs_batch are not respected
        """

        # If given, test that obs_batch is in the correct format
        if obs_batch is not None:
            if len(obs_batch) != 3:
                raise ValueError(
                    f"obs_batch must be a list of size 3. You gave {len(obs_batch)}"
                )
            elif not all([isinstance(b, jnp.ndarray) for b in obs_batch]):
                raise ValueError("Every element of obs_batch should be a jnp.array.")
            n_obs = obs_batch[0].shape[0]
            if any([b.shape[0] != n_obs for b in obs_batch]):
                raise ValueError(
                    "Every jnp array should have the same size of the first axis (number of observations)."
                )
            if obs_batch[1].ndim == 1:
                # if omega domain is unidimensional make sure that the x
                # obs_batch is (nb_obs, 1)
                obs_batch[1] = obs_batch[1][:, None]

        super().__init__(
            u,
            loss_weights,
            dynamic_loss,
            omega_boundary_fun,
            omega_boundary_condition,
            norm_key,
            norm_borders,
            norm_samples,
            obs_batch=None,
            # must be set to None because the set up performed
            # in the super class does not match
            sobolev_m=sobolev_m,
        )
        self.initial_condition_fun = initial_condition_fun
        self.obs_batch = obs_batch  # Set after call to super()

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
            A tuple.
            A batch of points in the domain, a batch of points in the domain
            border and a batch of time points
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
            omega_batch_ = jnp.tile(omega_batch, reps=(nt, 1))  # it is tiled
            times_batch_ = rep_times(n)  # it is repeated

        # dynamic part
        if self.dynamic_loss is not None:
            if isinstance(self.u, PINN):
                v_dyn_loss = vmap(
                    lambda t, x, params: self.dynamic_loss.evaluate(
                        t, x, self.u, params
                    ),
                    vmap_in_axes_x_t + vmap_in_axes_params,
                    0,
                )
                residuals = v_dyn_loss(times_batch_, omega_batch_, params)
                mse_dyn_loss = jnp.mean(
                    self.loss_weights["dyn_loss"] * jnp.sum(residuals**2, axis=-1)
                )
            # TODO implement Causality is all you need (not yet implemented)
            #  epsilon = 0.01
            #  times_batch_ = jnp.sort(times_batch_)
            #  val_dyn_loss = v_dyn_loss(times_batch_, omega_batch_, params)
            #  causality_is_all_you_need = jax.lax.stop_gradient(jnp.roll(jnp.exp(-epsilon *
            #      jnp.cumsum(val_dyn_loss)), shift=1,
            #      axis=0)) * val_dyn_loss
            #  mse_dyn_loss = jnp.mean(causality_is_all_you_need ** 2)
            elif isinstance(self.u, SPINN):
                residuals = self.dynamic_loss.evaluate(
                    times_batch, omega_batch, self.u, params
                )
                mse_dyn_loss = jnp.mean(
                    self.loss_weights["dyn_loss"] * jnp.sum(residuals**2, axis=-1)
                )
            # TODO implement Causality is all you need (not yet implemented)
            #    epsilon = 0.01
            #    times_batch = jnp.sort(times_batch)
            #    residuals = jax.lax.stop_gradient(jnp.roll(jnp.exp(-epsilon *
            #        jnp.cumsum(jnp.cumsum(residuals, axis=-2), axis=-1)
            #        ),
            #        shift=1, axis=0)) * residuals
        else:
            mse_dyn_loss = 0

        # normalization part
        if self.normalization_loss is not None:
            if isinstance(self.u, PINN):
                v_u = vmap(
                    vmap(
                        lambda t, x: self.u(
                            t,
                            x,
                            params["nn_params"],
                            jax.lax.stop_gradient(params["eq_params"]),
                        ),
                        in_axes=(None, 0),
                    ),
                    in_axes=(0, None),
                )
                res = v_u(times_batch, self.get_norm_samples())
                # the outer mean() below is for the times stamps
                mse_norm_loss = self.loss_weights["norm_loss"] * jnp.mean(
                    jnp.abs(jnp.mean(res, axis=(-2, -1)) * self.int_length - 1) ** 2
                )
            elif isinstance(self.u, SPINN):
                norm_samples = self.get_norm_samples()
                assert norm_samples.shape[0] % times_batch.shape[0] == 0
                rep_t = norm_samples.shape[0] // times_batch.shape[0]
                res = self.u(
                    jnp.repeat(times_batch, rep_t, axis=0),
                    norm_samples,
                    params["nn_params"],
                    jax.lax.stop_gradient(params["eq_params"]),
                )
                # the outer mean() below is for the times stamps
                mse_norm_loss = self.loss_weights["norm_loss"] * jnp.mean(
                    jnp.abs(
                        jnp.mean(
                            jnp.mean(res, axis=-1),
                            axis=(d + 1 for d in range(res.ndim - 2)),
                        )
                        * self.int_length
                        - 1
                    )
                    ** 2
                )

        else:
            mse_norm_loss = 0

        # boundary part
        if self.omega_boundary_fun is not None:
            if type(self.omega_boundary_fun) is dict:
                # means self.omega_boundary_condition is dict too because of
                # check in init
                mse_boundary_loss = 0
                for idx, facet in enumerate(self.omega_boundary_fun.keys()):
                    if self.omega_boundary_condition[facet] is not None:
                        mse_boundary_loss += jnp.mean(
                            self.loss_weights["boundary_loss"]
                            * _compute_boundary_loss_nonstatio(
                                self.omega_boundary_condition[facet],
                                self.omega_boundary_fun[facet],
                                times_batch,
                                omega_border_batch[..., idx],
                                self.u,
                                params,
                                idx,
                            )
                        )
            else:
                mse_boundary_loss = 0
                for facet in range(omega_border_batch.shape[-1]):
                    mse_boundary_loss += jnp.mean(
                        self.loss_weights["boundary_loss"]
                        * _compute_boundary_loss_nonstatio(
                            self.omega_boundary_condition,
                            self.omega_boundary_fun,
                            times_batch,
                            omega_border_batch[..., facet],
                            self.u,
                            params,
                            facet,
                        )
                    )
        else:
            mse_boundary_loss = 0

        # initial condition
        if self.initial_condition_fun is not None:
            if isinstance(self.u, PINN):
                v_u_t0 = vmap(
                    lambda x: self.initial_condition_fun(x)
                    - self.u(
                        t=jnp.zeros((1,)),
                        x=x,
                        u_params=params["nn_params"],
                        eq_params=jax.lax.stop_gradient(params["eq_params"]),
                    ),
                    (0),
                    0,
                )
                res = v_u_t0(omega_batch)
                mse_initial_condition = jnp.mean(
                    self.loss_weights["initial_condition"] * jnp.sum(res**2, axis=-1)
                )
            elif isinstance(self.u, SPINN):
                values = lambda x: self.u(
                    jnp.repeat(jnp.zeros((1, 1)), omega_batch.shape[0], axis=0),
                    x,
                    params["nn_params"],
                    jax.lax.stop_gradient(params["eq_params"]),
                )[0]
                omega_batch_grid = _get_grid(omega_batch)
                v_ini = values(omega_batch)
                ini = _check_user_func_return(
                    self.initial_condition_fun(omega_batch_grid), v_ini.shape
                )
                res = ini - v_ini
                mse_initial_condition = jnp.mean(
                    self.loss_weights["initial_condition"] * jnp.sum(res**2, axis=-1)
                )
        else:
            mse_initial_condition = 0

        # Observation MSE (if obs_batch provided)
        # NOTE that it does not use jax.lax.stop_gradient on "eq_params" here
        # since we may wish to optimize on it.
        if self.obs_batch is not None:
            # TODO implement for SPINN
            if isinstance(self.u, PINN):
                v_u = vmap(
                    lambda t, x: self.u(t, x, params["nn_params"], params["eq_params"]),
                    (0, 0),
                    0,
                )
                mse_observation_loss = jnp.mean(
                    self.loss_weights["observations"]
                    * jnp.mean(
                        (
                            v_u(self.obs_batch[0][:, None], self.obs_batch[1])
                            - self.obs_batch[2]
                        )
                        ** 2,
                        axis=-1,
                    )
                )
            elif isinstance(self.u, SPINN):
                raise RuntimeError(
                    "observation loss term not yet implemented for SPINNs"
                )
        else:
            mse_observation_loss = 0
            self.loss_weights["observations"] = 0

        # Sobolev regularization
        if self.sobolev_reg is not None:
            # TODO implement for SPINN
            if isinstance(self.u, PINN):
                v_sob_reg = vmap(
                    lambda t, x: self.sobolev_reg(
                        t,
                        x,
                        params["nn_params"],
                        jax.lax.stop_gradient(params["eq_params"]),
                    ),
                    (0, 0),
                    0,
                )
                mse_sobolev_loss = self.loss_weights["sobolev"] * jnp.mean(
                    v_sob_reg(omega_batch_, times_batch_)
                )
            elif isinstance(self.u, SPINN):
                raise RuntimeError("Sobolev loss term not yet implemented for SPINNs")
        else:
            mse_sobolev_loss = 0
            self.loss_weights["sobolev"] = 0

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
        children = (self.norm_key, self.norm_samples, self.obs_batch, self.loss_weights)
        aux_data = {
            "u": self.u,
            "dynamic_loss": self.dynamic_loss,
            "omega_boundary_fun": self.omega_boundary_fun,
            "omega_boundary_condition": self.omega_boundary_condition,
            "initial_condition_fun": self.initial_condition_fun,
            "norm_borders": self.norm_borders,
            "sobolev_m": self.sobolev_m,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (norm_key, norm_samples, obs_batch, loss_weights) = children
        pls = cls(
            aux_data["u"],
            loss_weights,
            aux_data["dynamic_loss"],
            aux_data["omega_boundary_fun"],
            aux_data["omega_boundary_condition"],
            aux_data["initial_condition_fun"],
            norm_key,
            aux_data["norm_borders"],
            norm_samples,
            obs_batch,
            aux_data["sobolev_m"],
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
        omega_boundary_fun_dict=None,
        omega_boundary_condition_dict=None,
        initial_condition_fun_dict=None,
        norm_key_dict=None,
        norm_borders_dict=None,
        norm_samples_dict=None,
        obs_batch_dict=None,
        sobolev_m_dict=None,
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
        omega_boundary_fun_dict
            A dict of functions to be matched in the border condition, or a
            dict of dict of functions (see doc for `omega_boundary_fun` in
            LossPDEStatio or LossPDENonStatio).
            Must share the keys of `u_dict`
        omega_boundary_condition_dict
            A dict of either None (no condition), or a string defining the boundary
            condition e.g. Dirichlet or Von Neumann, or a dict of dict of
            strings (see doc for `omega_boundary_fun` in
            LossPDEStatio or LossPDENonStatio).
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
        obs_batch_dict
            Default is None.
            A dictionary, a key per element of `u_dict`, with value being a tuple
            containing 2 jnp.array of the same size (on their first
            axis) encoding the timestep and the observations [:math:`x_i, u(x_i)`].
            A particular key can be None.
            Must share the keys of `u_dict`
        sobolev_m
            Default is None. A dictionary of integers, one per key which must
            match `u_dict`.
            It corresponds to the Sobolev regularization order as proposed in
            *Convergence and error analysis of PINNs*,
            Doumeche et al., 2023, https://arxiv.org/pdf/2305.01240.pdf


        Raises
        ------
        ValueError
            if initial condition is not a dict of tuple
        ValueError
            if the dictionaries that should share the keys of u_dict do not
        """
        # First, for all the optional dict,
        # if the user did not provide at all this optional argument,
        # we make sure there is a null ponderating loss_weight and we
        # create a dummy dict with the required keys and all the values to
        # None
        if obs_batch_dict is None:
            self.obs_batch_dict = {k: None for k in u_dict.keys()}
        else:
            self.obs_batch_dict = obs_batch_dict
        if omega_boundary_fun_dict is None:
            self.omega_boundary_fun_dict = {k: None for k in u_dict.keys()}
        else:
            self.omega_boundary_fun_dict = omega_boundary_fun_dict
        if omega_boundary_condition_dict is None:
            self.omega_boundary_condition_dict = {k: None for k in u_dict.keys()}
        else:
            self.omega_boundary_condition_dict = omega_boundary_condition_dict
        if initial_condition_fun_dict is None:
            self.initial_condition_fun_dict = {k: None for k in u_dict.keys()}
        else:
            self.initial_condition_fun_dict = initial_condition_fun_dict
        if norm_key_dict is None:
            self.norm_key_dict = {k: None for k in u_dict.keys()}
        else:
            self.norm_key_dict = norm_key_dict
        if norm_borders_dict is None:
            self.norm_borders_dict = {k: None for k in u_dict.keys()}
        else:
            self.norm_borders_dict = norm_borders_dict
        if norm_samples_dict is None:
            self.norm_samples_dict = {k: None for k in u_dict.keys()}
        else:
            self.norm_samples_dict = norm_samples_dict
        if sobolev_m_dict is None:
            self.sobolev_m_dict = {k: None for k in u_dict.keys()}
        else:
            self.sobolev_m_dict = sobolev_m_dict

        # Second we make sure that all the dicts (except dynamic_loss_dict) have the same keys
        if (
            u_dict.keys() != nn_type_dict.keys()
            or u_dict.keys() != self.obs_batch_dict.keys()
            or u_dict.keys() != self.omega_boundary_fun_dict.keys()
            or u_dict.keys() != self.omega_boundary_condition_dict.keys()
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
                    omega_boundary_fun=self.omega_boundary_fun_dict[i],
                    omega_boundary_condition=self.omega_boundary_condition_dict[i],
                    norm_key=self.norm_key_dict[i],
                    norm_borders=self.norm_borders_dict[i],
                    norm_samples=self.norm_samples_dict[i],
                    obs_batch=self.obs_batch_dict[i],
                    sobolev_m=self.sobolev_m_dict[i],
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
                    omega_boundary_fun=self.omega_boundary_fun_dict[i],
                    omega_boundary_condition=self.omega_boundary_condition_dict[i],
                    initial_condition_fun=self.initial_condition_fun[i],
                    norm_key=self.norm_key_dict[i],
                    norm_borders=self.norm_borders_dict[i],
                    norm_samples=self.norm_samples_dict[i],
                    obs_batch=self.obs_batch_dict[i],
                    sobolev_m=self.sobolev_m_dict[i],
                )
            else:
                raise ValueError(
                    f"Wrong value for nn_type_dict[i], got " "{nn_type_dict[i]}"
                )

        # also make sure we only have PINNs or SPINNs
        if not (
            all(type(value) == PINN for value in u_dict.values())
            or all(type(value) == SPINN for value in u_dict.values())
        ):
            raise ValueError(
                "We only accept dictionary of PINNs or dictionary" " of SPINNs"
            )

    @property
    def loss_weights(self):
        return self._loss_weights

    @loss_weights.setter
    def loss_weights(self, value):
        self._loss_weights = {}
        for k, v in value.items():
            if isinstance(v, dict):
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
                if k == "dyn_loss":
                    self._loss_weights[k] = {
                        kk: v for kk in self.dynamic_loss_dict.keys()
                    }
                else:
                    self._loss_weights[k] = {kk: v for kk in self.u_dict.keys()}
        # Some special checks below
        if all(v is None for k, v in self.sobolev_m_dict.items()):
            self._loss_weights["sobolev"] = {k: 0 for k in self.u_dict.keys()}
        if all(v is None for k, v in self.obs_batch_dict.items()):
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
            A batch of time points at which to evaluate the loss
        """
        if self.u_dict.keys() != params_dict["nn_params"].keys():
            raise ValueError("u_dict and params_dict[nn_params] should have same keys ")

        if isinstance(batch, PDEStatioBatch):
            omega_batch, omega_border_batch = batch.inside_batch, batch.border_batch
            n = omega_batch.shape[0]
        elif isinstance(batch, PDENonStatioBatch):
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

        else:
            raise ValueError("Wrong type of batch")

        vmap_in_axes_x = (0,)
        vmap_in_axes_x_t = (0, 0)

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

        mse_dyn_loss = 0
        mse_boundary_loss = 0
        mse_norm_loss = 0
        mse_initial_condition = 0
        mse_observation_loss = 0
        mse_sobolev_loss = 0

        for i in self.dynamic_loss_dict.keys():
            # dynamic part
            if isinstance(self.dynamic_loss_dict[i], PDEStatio):
                # Below we just look at the first element because we suppose we
                # must only have SPINNs or only PINNs
                if isinstance(list(self.u_dict.values())[0], PINN):
                    v_dyn_loss = vmap(
                        lambda x, params_dict: self.dynamic_loss_dict[i].evaluate(
                            x,
                            self.u_dict,
                            params_dict,
                        ),
                        vmap_in_axes_x + vmap_in_axes_params,
                        0,
                    )
                    mse_dyn_loss += jnp.mean(
                        self._loss_weights["dyn_loss"][i]
                        * jnp.sum(v_dyn_loss(omega_batch, params_dict) ** 2, axis=-1)
                    )
                elif isinstance(list(self.u_dict.values())[0], SPINN):
                    residuals = self.dynamic_loss_dict[i].evaluate(
                        omega_batch, self.u_dict, params_dict
                    )
                    mse_dyn_loss += jnp.mean(
                        self._loss_weights["dyn_loss"][i]
                        * jnp.sum(
                            residuals**2,
                            axis=-1,
                        )
                    )
            else:
                if isinstance(list(self.u_dict.values())[0], PINN):
                    v_dyn_loss = vmap(
                        lambda t, x, params_dict: self.dynamic_loss_dict[i].evaluate(
                            t, x, self.u_dict, params_dict
                        ),
                        vmap_in_axes_x_t + vmap_in_axes_params,
                        0,
                    )

                    tile_omega_batch = jnp.tile(omega_batch, reps=(nt, 1))

                    omega_batch_ = jnp.tile(omega_batch, reps=(nt, 1))  # it is tiled
                    times_batch_ = rep_times(n)  # it is repeated

                    mse_dyn_loss += jnp.mean(
                        self._loss_weights["dyn_loss"][i]
                        * jnp.sum(
                            v_dyn_loss(times_batch_, omega_batch_, params_dict) ** 2,
                            axis=-1,
                        )
                    )
                elif isinstance(list(self.u_dict.values())[0], SPINN):
                    residuals = self.dynamic_loss_dict[i].evaluate(
                        times_batch, omega_batch, self.u_dict, params_dict
                    )
                    mse_dyn_loss += jnp.mean(
                        self._loss_weights["dyn_loss"][i]
                        * jnp.sum(
                            residuals**2,
                            axis=-1,
                        )
                    )

        # boundary conditions, normalization conditions, observation_loss,
        # initial condition... loss this is done via the internal
        # LossPDEStatio and NonStatio
        for i in self.u_dict.keys():
            _, res_dict = self.u_constraints_dict[i].evaluate(
                {
                    "nn_params": (
                        params_dict["nn_params"][i]
                        if isinstance(params_dict["nn_params"], dict)
                        else params_dict["nn_params"]
                    ),
                    "eq_params": params_dict["eq_params"],
                },
                batch,
            )
            # note that the results have already been scaled internally in the
            # call to evaluate
            mse_boundary_loss += (
                self._loss_weights["boundary_loss"][i] * res_dict["boundary_loss"]
            )
            mse_norm_loss += self._loss_weights["norm_loss"][i] * res_dict["norm_loss"]
            mse_observation_loss += (
                self._loss_weights["observations"][i] * res_dict["observations"]
            )
            mse_sobolev_loss += self._loss_weights["sobolev"][i] * res_dict["sobolev"]
            if self.nn_type_dict[i] == "nn_nonstatio":
                mse_initial_condition += (
                    self._loss_weights["initial_condition"][i]
                    * res_dict["initial_condition"]
                )

        # total loss
        total_loss = (
            mse_dyn_loss
            + mse_norm_loss
            + mse_boundary_loss
            + mse_observation_loss
            + mse_sobolev_loss
        )
        return_dict = {
            "dyn_loss": mse_dyn_loss,
            "norm_loss": mse_norm_loss,
            "boundary_loss": mse_boundary_loss,
            "observations": mse_observation_loss,
            "sobolev": mse_sobolev_loss,
        }

        if isinstance(batch, PDENonStatioBatch):
            total_loss += mse_initial_condition
            return_dict["initial_condition"] = mse_initial_condition

        return total_loss, return_dict

    def tree_flatten(self):
        children = (
            self.obs_batch_dict,
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
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            obs_batch_dict,
            norm_key_dict,
            norm_samples_dict,
            initial_condition_fun_dict,
            loss_weights,
        ) = children
        loss_ode = cls(
            loss_weights=loss_weights,
            obs_batch_dict=obs_batch_dict,
            norm_key_dict=norm_key_dict,
            norm_samples_dict=norm_samples_dict,
            initial_condition_fun_dict=initial_condition_fun_dict,
            **aux_data,
        )

        return loss_ode
