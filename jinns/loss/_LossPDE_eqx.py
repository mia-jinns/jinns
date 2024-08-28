# pylint: disable=unsubscriptable-object, no-member
"""
Main module to implement a PDE loss in jinns
"""

import abc
from typing import Union, Dict, Callable
import warnings
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import equinox as eqx
from jaxtyping import PyTree, Float, Array, Key
from jinns.loss._Losses import (
    dynamic_loss_apply,
    boundary_condition_apply,
    normalization_loss_apply,
    observations_loss_apply,
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
from jinns.loss._DynamicLossAbstract_eqx import PDEStatio, PDENonStatio

_IMPLEMENTED_BOUNDARY_CONDITIONS = [
    "dirichlet",
    "von neumann",
    "vonneumann",
]

_LOSS_WEIGHT_KEYS_PDESTATIO = [
    "observations",
    "norm_loss",
    "boundary_loss",
    "dyn_loss",
]

_LOSS_WEIGHT_KEYS_PDENONSTATIO = _LOSS_WEIGHT_KEYS_PDESTATIO + ["initial_condition"]

_MSE_TERMS_PDESTATIO = _LOSS_WEIGHT_KEYS_PDESTATIO
_MSE_TERMS_PDENONSTATIO = _LOSS_WEIGHT_KEYS_PDENONSTATIO


class _LossPDEAbstract_eqx(eqx.Module):
    """
    Parameters
    ----------

    loss_weights
        a dictionary with values used to ponderate each term in the loss
        function. Valid keys are in _LOSS_WEIGHT_KEYS_PDENONSTATIO
        Note that we can have jnp.arrays with the same dimension of
        `u` which then ponderates each output of `u`
    derivative_keys
        A dict of lists of strings. In the dict, the key must correspond to
        the loss term keywords (valid keys are in _MSE_TERMS_PDENONSTATIO). Then each of the values must correspond to keys in the parameter
        dictionary (*at top level only of the parameter dictionary*).
        It enables selecting the set of parameters
        with respect to which the gradients of the dynamic
        loss are computed. If nothing is provided, we set ["nn_params"] for all loss term
        keywords, this is what is typically
        done in solving forward problems, when we only estimate the
        equation solution with a PINN. If some loss terms keywords are
        missing we set their value to ["nn_params"] by default for the
        same reason
    norm_samples
        Fixed sample point in the space over which to compute the
        normalization constant. Default is None. Note that contrary to
        LossPDEAbstract defined in custom Python classes, we perform no check
        on this argument!
    obs_slice
        slice object specifying the begininning/ending
        slice of u output(s) that is observed (this is then useful for
        multidim PINN). Default is None.
    """

    # kw_only in base class is motivated here: https://stackoverflow.com/a/69822584
    derivative_keys: Union[list, None] = eqx.field(
        kw_only=True, default=None, static=True
    )
    loss_weights: Union[Dict[str, Union[Float, Array]], None] = eqx.field(
        kw_only=True, default=None, static=True
    )
    omega_boundary_fun: Union[Callable, Dict[str, Callable], None] = eqx.field(
        kw_only=True, default=None, static=True
    )
    omega_boundary_condition: Union[str, Dict[str, str], None] = eqx.field(
        kw_only=True, default=None, static=True
    )
    omega_boundary_dim: Union[slice, Dict[str, slice], None] = eqx.field(
        kw_only=True, default=None, static=True
    )
    norm_samples: Union[Array, None] = eqx.field(kw_only=True, default=None)
    obs_slice: Union[slice, None] = eqx.field(kw_only=True, default=None, static=True)

    def __post_init__(self):
        """
        Note that neither __init__ or __post_init__ are called when udating a
        Module with eqx.tree_at
        """
        if self.derivative_keys is None:
            self.derivative_keys = {k: ["nn_params"] for k in _MSE_TERMS_PDENONSTATIO}
        if isinstance(self.derivative_keys, list):
            self.derivative_keys = {
                k: self.derivative_keys for k in _MSE_TERMS_PDENONSTATIO
            }

        for k in _LOSS_WEIGHT_KEYS_PDENONSTATIO:
            if k not in self.loss_weights.keys():
                self.loss_weights[k] = 0

        if self.obs_slice is None:
            self.obs_slice = jnp.s_[...]

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

        if self.omega_boundary_condition is None or self.omega_boundary_fun is None:
            warnings.warn(
                "Missing boundary function or no boundary condition."
                "Boundary function is thus ignored."
            )
        else:
            if isinstance(self.omega_boundary_condition, dict):
                for _, v in self.omega_boundary_condition.items():
                    if v is not None and not any(
                        v.lower() in s for s in _IMPLEMENTED_BOUNDARY_CONDITIONS
                    ):
                        raise NotImplementedError(
                            f"The boundary condition {self.omega_boundary_condition} is not"
                            f"implemented yet. Try one of :"
                            f"{_IMPLEMENTED_BOUNDARY_CONDITIONS}."
                        )
            else:
                if not any(
                    self.omega_boundary_condition.lower() in s
                    for s in _IMPLEMENTED_BOUNDARY_CONDITIONS
                ):
                    raise NotImplementedError(
                        f"The boundary condition {self.omega_boundary_condition} is not"
                        f"implemented yet. Try one of :"
                        f"{_IMPLEMENTED_BOUNDARY_CONDITIONS}."
                    )
                if isinstance(self.omega_boundary_fun, dict) and isinstance(
                    self.omega_boundary_condition, dict
                ):
                    if (
                        not (
                            list(self.omega_boundary_fun.keys()) == ["xmin", "xmax"]
                            and list(self.omega_boundary_condition.keys())
                            == ["xmin", "xmax"]
                        )
                    ) or (
                        not (
                            list(self.omega_boundary_fun.keys())
                            == ["xmin", "xmax", "ymin", "ymax"]
                            and list(self.omega_boundary_condition.keys())
                            == ["xmin", "xmax", "ymin", "ymax"]
                        )
                    ):
                        raise ValueError(
                            "The key order (facet order) in the "
                            "boundary condition dictionaries is incorrect"
                        )

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

    @abc.abstractmethod
    def evaluate(
        self: eqx.Module,
        params: PyTree,
        batch: Union[PDEStatioBatch, PDENonStatioBatch],
    ) -> tuple[Float, dict]:
        raise NotImplementedError


class LossPDEStatio_eqx(_LossPDEAbstract_eqx):
    r"""Loss object for a stationary partial differential equation

    .. math::
        \mathcal{N}[u](x) = 0, \forall x  \in \Omega

    where :math:`\mathcal{N}[\cdot]` is a differential operator and the
    boundary condition is :math:`u(x)=u_b(x)` The additional condition of
    integrating to 1 can be included, i.e. :math:`\int u(x)\mathrm{d}x=1`.

    Parameters
    ----------
    u
        the PINN object
    dynamic_loss
        the stationary PDE dynamic part of the loss, basically the differential
        operator :math:` \mathcal{N}[u](t)`. Should implement a method
        `dynamic_loss.evaluate(t, u, params)`.
        Can be None in order to access only some part of the evaluate call
        results.
    key
        A JAX PRNG Key for the loss class treated as an attribute. Default is
        None. This field is provided for future developments and additional
        losses that might need some randomness. Note that special care must be
        taken when splitting the key because in-place updates are forbidden in
        eqx.Modules.
    loss_weights
        a dictionary with values used to ponderate each term in the loss
        function. Valid keys are `dyn_loss`, `norm_loss`, `boundary_loss`,
        `observations`.
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
    norm_samples
        Fixed sample point in the space over which to compute the
        normalization constant. Default is None. Note that contrary to
        LossPDEAbstract defined in custom Python classes, we perform no check
        on this argument!
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

    u: eqx.Module
    dynamic_loss: Union[eqx.Module, None]
    key: Union[Key, None] = eqx.field(kw_only=True, default=None)

    def __post_init__(self):
        """
        Note that neither __init__ or __post_init__ are called when udating a
        Module with eqx.tree_at!
        """
        super().__post_init__()  # because __init__ or __post_init__ of Base
        # class is not automatically called

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
        if self.norm_samples is not None:
            mse_norm_loss = normalization_loss_apply(
                self.u,
                (self.norm_samples,),
                params_,
                vmap_in_axes_x + vmap_in_axes_params,
                self.int_length,
                self.loss_weights["norm_loss"],
            )
        else:
            mse_norm_loss = jnp.array(0.0)

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

        # total loss
        total_loss = (
            mse_dyn_loss + mse_norm_loss + mse_boundary_loss + mse_observation_loss
        )
        return total_loss, (
            {
                "dyn_loss": mse_dyn_loss,
                "norm_loss": mse_norm_loss,
                "boundary_loss": mse_boundary_loss,
                "observations": mse_observation_loss,
                "initial_condition": jnp.array(0.0),  # for compatibility in the
                # tree_map of SystemLoss
            }
        )


class LossPDENonStatio_eqx(LossPDEStatio_eqx):
    r"""Loss object for a stationary partial differential equation

    .. math::
        \mathcal{N}[u](t, x) = 0, \forall t \in I, \forall x \in \Omega

    where :math:`\mathcal{N}[\cdot]` is a differential operator.
    The boundary condition is :math:`u(t, x)=u_b(t, x),\forall
    x\in\delta\Omega, \forall t`.
    The initial condition is :math:`u(0, x)=u_0(x), \forall x\in\Omega`
    The additional condition of
    integrating to 1 can be included, i.e., :math:`\int u(t, x)\mathrm{d}x=1`.

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
    obs_slice
        slice object specifying the begininning/ending
        slice of u output(s) that is observed (this is then useful for
        multidim PINN). Default is None.

    """

    initial_condition_fun: Union[Callable, None] = eqx.field(
        kw_only=True, default=None, static=True
    )

    def __post_init__(self):
        """
        Note that neither __init__ or __post_init__ are called when udating a
        Module with eqx.tree_at!
        """
        super().__post_init__()  # because __init__ or __post_init__ of Base
        # class is not automatically called

        if self.initial_condition_fun is None:
            warnings.warn(
                "Initial condition wasn't provided. Be sure to cover for that"
                "case (e.g by. hardcoding it into the PINN output)."
            )

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

        times_batch = batch.times_x_inside_batch[:, 0:1]
        omega_batch = batch.times_x_inside_batch[:, 1:]
        n = omega_batch.shape[0]

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
        if self.norm_samples is not None:
            mse_norm_loss = normalization_loss_apply(
                self.u,
                (times_batch, self.norm_samples),
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

        # total loss
        total_loss = (
            mse_dyn_loss
            + mse_norm_loss
            + mse_boundary_loss
            + mse_initial_condition
            + mse_observation_loss
        )

        return total_loss, (
            {
                "dyn_loss": mse_dyn_loss,
                "norm_loss": mse_norm_loss,
                "boundary_loss": mse_boundary_loss,
                "initial_condition": mse_initial_condition,
                "observations": mse_observation_loss,
            }
        )


class SystemLossPDE_eqx(eqx.Module):
    r"""
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
    key_dict
        A dictionary of JAX PRNG keys. The dictionary keys of key_dict must
        match that of u_dict. See LossPDEStatio_eqx or LossPDENonStatio_eqx for
        more details.
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
    norm_samples_dict
        A dict of fixed sample point in the space over which to compute the
        normalization constant. Default is None
        Must share the keys of `u_dict`
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

    # Contrary to the losses above, we need to declare u_dict and
    # dynamic_loss_dict as static because of the str typed keys which are not
    # valid JAX type (and not because of the ODE or eqx.Module)
    # TODO do not use a dictionary here!
    u_dict: Dict[str, eqx.Module] = eqx.field(static=True)
    dynamic_loss_dict: Dict[str, Union[PDEStatio, PDENonStatio]] = eqx.field(
        static=True
    )
    key_dict: Union[Dict[Union[Key, None], None]] = eqx.field(
        kw_only=True, default=None
    )
    derivative_keys_dict: Union[Dict[str, Union[list, None]], None] = eqx.field(
        kw_only=True, default=None, static=True
    )
    loss_weights: Union[Dict[str, Dict[str, Union[Float, Array]]], None] = eqx.field(
        kw_only=True, default=None, static=True
    )
    omega_boundary_fun_dict: Union[
        Dict[str, Union[Callable, Dict[str, Callable]]], None
    ] = eqx.field(kw_only=True, default=None, static=True)
    omega_boundary_condition_dict: Union[
        Dict[str, Union[str, Dict[str, str]]], None
    ] = eqx.field(kw_only=True, default=None, static=True)
    omega_boundary_dim_dict: Union[Dict[str, Union[slice, Dict[str, slice]]], None] = (
        eqx.field(kw_only=True, default=None, static=True)
    )
    initial_condition_fun_dict: Union[Dict[str, Union[Callable, None]], None] = (
        eqx.field(kw_only=True, default=None, static=True)
    )
    norm_samples_dict: Union[Dict[str, Union[Array, None]], None] = eqx.field(
        kw_only=True, default=None
    )
    obs_slice_dict: Union[Dict[str, Union[slice, None]], None] = eqx.field(
        kw_only=True, default=None, static=True
    )

    # following have init=False and are set in the __post_init__
    u_constraints_dict: Dict[str, list] = eqx.field(init=False, static=True)
    derivative_keys_u_dict: Dict[str, list] = eqx.field(init=False, static=True)
    derivative_keys_dyn_loss_dict: Dict[str, list] = eqx.field(init=False, static=True)
    u_dict_with_none: Dict[str, None] = eqx.field(init=False, static=True)

    def __post_init__(self):
        # a dictionary that will be useful at different places
        self.u_dict_with_none = {k: None for k in self.u_dict.keys()}
        # First, for all the optional dict,
        # if the user did not provide at all this optional argument,
        # we make sure there is a null ponderating loss_weight and we
        # create a dummy dict with the required keys and all the values to
        # None
        if self.key_dict is None:
            self.key_dict = self.u_dict_with_none
        if self.omega_boundary_fun_dict is None:
            self.omega_boundary_fun_dict = self.u_dict_with_none
        if self.omega_boundary_condition_dict is None:
            self.omega_boundary_condition_dict = self.u_dict_with_none
        if self.omega_boundary_dim_dict is None:
            self.omega_boundary_dim_dict = self.u_dict_with_none
        if self.initial_condition_fun_dict is None:
            self.initial_condition_fun_dict = self.u_dict_with_none
        if self.norm_samples_dict is None:
            self.norm_samples_dict = self.u_dict_with_none
        if self.obs_slice_dict is None:
            self.obs_slice_dict = {k: jnp.s_[...] for k in self.u_dict.keys()}
            if self.u_dict.keys() != self.obs_slice_dict.keys():
                raise ValueError("obs_slice_dict should have same keys as u_dict")
        if self.derivative_keys_dict is None:
            self.derivative_keys_dict = {
                k: None
                for k in set(
                    list(self.dynamic_loss_dict.keys()) + list(self.u_dict.keys())
                )
            }
            # set() because we can have duplicate entries and in this case we
            # say it corresponds to the same derivative_keys_dict entry
        # but then if the user did not provide anything, we must at least have
        # a default value for the dynamic_loss_dict keys entries in
        # self.derivative_keys_dict since the computation of dynamic losses is
        # made without create a lossODE object that would provide the
        # default values
        for k in self.dynamic_loss_dict.keys():
            if self.derivative_keys_dict[k] is None:
                self.derivative_keys_dict[k] = {"dyn_loss": ["nn_params"]}

        # Second we make sure that all the dicts (except dynamic_loss_dict) have the same keys
        if (
            self.u_dict.keys() != self.key_dict.keys()
            or self.u_dict.keys() != self.omega_boundary_fun_dict.keys()
            or self.u_dict.keys() != self.omega_boundary_condition_dict.keys()
            or self.u_dict.keys() != self.omega_boundary_dim_dict.keys()
            or self.u_dict.keys() != self.initial_condition_fun_dict.keys()
            or self.u_dict.keys() != self.norm_samples_dict.keys()
        ):
            raise ValueError("All the dicts concerning the PINNs should have same keys")

        self.loss_weights = self.set_loss_weights(self.loss_weights)

        # Third, in order not to benefit from LossPDEStatio and
        # LossPDENonStatio and in order to factorize code, we create internally
        # some losses object to implement the constraints on the solutions.
        # We will not use the dynamic loss term
        self.u_constraints_dict = {}
        for i in self.u_dict.keys():
            if self.u_dict[i].eq_type == "statio_PDE":
                self.u_constraints_dict[i] = LossPDEStatio_eqx(
                    u=self.u_dict[i],
                    loss_weights={
                        "dyn_loss": 0.0,
                        "norm_loss": 1.0,
                        "boundary_loss": 1.0,
                        "observations": 1.0,
                    },
                    dynamic_loss=None,
                    key=self.key_dict[i],
                    derivative_keys=self.derivative_keys_dict[i],
                    omega_boundary_fun=self.omega_boundary_fun_dict[i],
                    omega_boundary_condition=self.omega_boundary_condition_dict[i],
                    omega_boundary_dim=self.omega_boundary_dim_dict[i],
                    norm_samples=self.norm_samples_dict[i],
                    obs_slice=self.obs_slice_dict[i],
                )
            elif self.u_dict[i].eq_type == "nonstatio_PDE":
                self.u_constraints_dict[i] = LossPDENonStatio_eqx(
                    u=self.u_dict[i],
                    loss_weights={
                        "dyn_loss": 0.0,
                        "norm_loss": 1.0,
                        "boundary_loss": 1.0,
                        "observations": 1.0,
                        "initial_condition": 1.0,
                    },
                    dynamic_loss=None,
                    key=self.key_dict[i],
                    derivative_keys=self.derivative_keys_dict[i],
                    omega_boundary_fun=self.omega_boundary_fun_dict[i],
                    omega_boundary_condition=self.omega_boundary_condition_dict[i],
                    omega_boundary_dim=self.omega_boundary_dim_dict[i],
                    initial_condition_fun=self.initial_condition_fun_dict[i],
                    norm_key=self.norm_key_dict[i],
                    norm_borders=self.norm_borders_dict[i],
                    norm_samples=self.norm_samples_dict[i],
                )
            else:
                raise ValueError(
                    "Wrong value for self.u_dict[i].eq_type[i], "
                    f"got {self.u_dict[i].eq_type[i]}"
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
            all(isinstance(value, PINN) for value in self.u_dict.values())
            or all(isinstance(value, SPINN) for value in self.u_dict.values())
        ):
            raise ValueError(
                "We only accept dictionary of PINNs or dictionary of SPINNs"
            )

    def set_loss_weights(self, value):
        _loss_weights = {}
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
                        _loss_weights[k] = v
                    else:
                        raise ValueError(
                            "Keys in nested dictionary of loss_weights"
                            " do not match dynamic_loss_dict keys"
                        )
                else:
                    if v.keys() == self.u_dict.keys():
                        _loss_weights[k] = v
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
                    _loss_weights[k] = {kk: v for kk in self.dynamic_loss_dict.keys()}
                else:
                    _loss_weights[k] = {kk: v for kk in self.u_dict.keys()}
        # Some special checks below
        if "observations" not in value.keys():
            _loss_weights["observations"] = {k: 0 for k in self.u_dict.keys()}
        if all(v is None for k, v in self.omega_boundary_fun_dict.items()) or all(
            v is None for k, v in self.omega_boundary_condition_dict.items()
        ):
            _loss_weights["boundary_loss"] = {k: 0 for k in self.u_dict.keys()}
        if all(v is None for k, v in self.norm_samples_dict.items()):
            _loss_weights["norm_loss"] = {k: 0 for k in self.u_dict.keys()}
        if all(v is None for k, v in self.initial_condition_fun_dict.items()):
            _loss_weights["initial_condition"] = {k: 0 for k in self.u_dict.keys()}
        return _loss_weights

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
            vmap_in_axes_x_or_x_t = (0,)

            batches = (omega_batch,)
        elif isinstance(batch, PDENonStatioBatch):
            times_batch = batch.times_x_inside_batch[:, 0:1]
            omega_batch = batch.times_x_inside_batch[:, 1:]

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
            self.loss_weights["dyn_loss"],
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
        }
        # we need to do the following for the tree_mapping to work
        if batch.obs_batch_dict is None:
            batch = batch._replace(obs_batch_dict=self.u_dict_with_none)
        total_loss, res_dict = constraints_system_loss_apply(
            self.u_constraints_dict,
            batch,
            params_dict,
            self.loss_weights,
            loss_weight_struct,
        )

        # Add the mse_dyn_loss from the previous computations
        total_loss += mse_dyn_loss
        res_dict["dyn_loss"] += mse_dyn_loss
        return total_loss, res_dict
