# pylint: disable=unsubscriptable-object, no-member
"""
Main module to implement a PDE loss in jinns
"""
from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

import abc
from dataclasses import InitVar, fields
from typing import TYPE_CHECKING, Dict, Callable
import warnings
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, Key, Int
from jinns.loss._loss_utils import (
    dynamic_loss_apply,
    boundary_condition_apply,
    normalization_loss_apply,
    observations_loss_apply,
    initial_condition_apply,
    constraints_system_loss_apply,
)
from jinns.data._DataGenerators import append_obs_batch
from jinns.parameters._params import (
    _get_vmap_in_axes_params,
    _update_eq_params_dict,
)
from jinns.parameters._derivative_keys import (
    _set_derivatives,
    DerivativeKeysPDEStatio,
    DerivativeKeysPDENonStatio,
)
from jinns.loss._loss_weights import (
    LossWeightsPDEStatio,
    LossWeightsPDENonStatio,
    LossWeightsPDEDict,
)
from jinns.loss._DynamicLossAbstract import PDEStatio, PDENonStatio
from jinns.nn._pinn import PINN
from jinns.nn._spinn import SPINN
from jinns.data._Batchs import PDEStatioBatch, PDENonStatioBatch


if TYPE_CHECKING:
    from jinns.utils._types import *

_IMPLEMENTED_BOUNDARY_CONDITIONS = [
    "dirichlet",
    "von neumann",
    "vonneumann",
]


class _LossPDEAbstract(eqx.Module):
    """
    Parameters
    ----------

    loss_weights : LossWeightsPDEStatio | LossWeightsPDENonStatio, default=None
        The loss weights for the differents term : dynamic loss,
        initial condition (if LossWeightsPDENonStatio), boundary conditions if
        any, normalization loss if any and observations if any.
        All fields are set to 1.0 by default.
    derivative_keys : DerivativeKeysPDEStatio | DerivativeKeysPDENonStatio, default=None
        Specify which field of `params` should be differentiated for each
        composant of the total loss. Particularily useful for inverse problems.
        Fields can be "nn_params", "eq_params" or "both". Those that should not
        be updated will have a `jax.lax.stop_gradient` called on them. Default
        is `"nn_params"` for each composant of the loss.
    omega_boundary_fun : Callable | Dict[str, Callable], default=None
         The function to be matched in the border condition (can be None) or a
         dictionary of such functions as values and keys as described
         in `omega_boundary_condition`.
    omega_boundary_condition : str | Dict[str, str], default=None
        Either None (no condition, by default), or a string defining
        the boundary condition (Dirichlet or Von Neumann),
        or a dictionary with such strings as values. In this case,
        the keys are the facets and must be in the following order:
        1D -> [“xmin”, “xmax”], 2D -> [“xmin”, “xmax”, “ymin”, “ymax”].
        Note that high order boundaries are currently not implemented.
        A value in the dict can be None, this means we do not enforce
        a particular boundary condition on this facet.
        The facet called “xmin”, resp. “xmax” etc., in 2D,
        refers to the set of 2D points with fixed “xmin”, resp. “xmax”, etc.
    omega_boundary_dim : slice | Dict[str, slice], default=None
        Either None, or a slice object or a dictionary of slice objects as
        values and keys as described in `omega_boundary_condition`.
        `omega_boundary_dim` indicates which dimension(s) of the PINN
        will be forced to match the boundary condition.
        Note that it must be a slice and not an integer
        (but a preprocessing of the user provided argument takes care of it)
    norm_samples : Float[Array, "nb_norm_samples dimension"], default=None
        Monte-Carlo sample points for computing the
        normalization constant. Default is None.
    norm_weights : Float[Array, "nb_norm_samples"] | float | int, default=None
        The importance sampling weights for Monte-Carlo integration of the
        normalization constant. Must be provided if `norm_samples` is provided.
        `norm_weights` should have the same leading dimension as
        `norm_samples`.
        Alternatively, the user can pass a float or an integer.
        These corresponds to the weights $w_k = \frac{1}{q(x_k)}$ where
        $q(\cdot)$ is the proposal p.d.f. and $x_k$ are the Monte-Carlo samples.
    obs_slice : slice, default=None
        slice object specifying the begininning/ending of the PINN output
        that is observed (this is then useful for multidim PINN). Default is None.
    params : InitVar[Params], default=None
        The main Params object of the problem needed to instanciate the
        DerivativeKeysODE if the latter is not specified.
    """

    # NOTE static=True only for leaf attributes that are not valid JAX types
    # (ie. jax.Array cannot be static) and that we do not expect to change
    # kw_only in base class is motivated here: https://stackoverflow.com/a/69822584
    derivative_keys: DerivativeKeysPDEStatio | DerivativeKeysPDENonStatio | None = (
        eqx.field(kw_only=True, default=None)
    )
    loss_weights: LossWeightsPDEStatio | LossWeightsPDENonStatio | None = eqx.field(
        kw_only=True, default=None
    )
    omega_boundary_fun: Callable | Dict[str, Callable] | None = eqx.field(
        kw_only=True, default=None, static=True
    )
    omega_boundary_condition: str | Dict[str, str] | None = eqx.field(
        kw_only=True, default=None, static=True
    )
    omega_boundary_dim: slice | Dict[str, slice] | None = eqx.field(
        kw_only=True, default=None, static=True
    )
    norm_samples: Float[Array, "nb_norm_samples dimension"] | None = eqx.field(
        kw_only=True, default=None
    )
    norm_weights: Float[Array, "nb_norm_samples"] | float | int | None = eqx.field(
        kw_only=True, default=None
    )
    obs_slice: slice | None = eqx.field(kw_only=True, default=None, static=True)

    params: InitVar[Params] = eqx.field(kw_only=True, default=None)

    def __post_init__(self, params=None):
        """
        Note that neither __init__ or __post_init__ are called when udating a
        Module with eqx.tree_at
        """
        if self.derivative_keys is None:
            # be default we only take gradient wrt nn_params
            try:
                self.derivative_keys = (
                    DerivativeKeysPDENonStatio(params=params)
                    if isinstance(self, LossPDENonStatio)
                    else DerivativeKeysPDEStatio(params=params)
                )
            except ValueError as exc:
                raise ValueError(
                    "Problem at self.derivative_keys initialization "
                    f"received {self.derivative_keys=} and {params=}"
                ) from exc

        if self.loss_weights is None:
            self.loss_weights = (
                LossWeightsPDENonStatio()
                if isinstance(self, LossPDENonStatio)
                else LossWeightsPDEStatio()
            )

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
                raise ValueError("self.omega_boundary_dim must be a jnp.s_ object")

        if self.norm_samples is not None:
            if self.norm_weights is None:
                raise ValueError(
                    "`norm_weights` must be provided when `norm_samples` is used!"
                )
            try:
                assert self.norm_weights.shape[0] == self.norm_samples.shape[0]
            except (AssertionError, AttributeError):
                if isinstance(self.norm_weights, (int, float)):
                    self.norm_weights = jnp.array(
                        [self.norm_weights], dtype=jax.dtypes.canonicalize_dtype(float)
                    )
                else:
                    raise ValueError(
                        "`norm_weights` should have the same leading dimension"
                        " as `norm_samples`,"
                        f" got shape {self.norm_weights.shape} and"
                        f" shape {self.norm_samples.shape}."
                    )

    @abc.abstractmethod
    def evaluate(
        self: eqx.Module,
        params: Params,
        batch: PDEStatioBatch | PDENonStatioBatch,
    ) -> tuple[Float, dict]:
        raise NotImplementedError


class LossPDEStatio(_LossPDEAbstract):
    r"""Loss object for a stationary partial differential equation

    $$
        \mathcal{N}[u](x) = 0, \forall x  \in \Omega
    $$

    where $\mathcal{N}[\cdot]$ is a differential operator and the
    boundary condition is $u(x)=u_b(x)$ The additional condition of
    integrating to 1 can be included, i.e. $\int u(x)\mathrm{d}x=1$.

    Parameters
    ----------
    u : eqx.Module
        the PINN
    dynamic_loss : DynamicLoss
        the stationary PDE dynamic part of the loss, basically the differential
        operator $\mathcal{N}[u](x)$. Should implement a method
        `dynamic_loss.evaluate(x, u, params)`.
        Can be None in order to access only some part of the evaluate call
        results.
    key : Key
        A JAX PRNG Key for the loss class treated as an attribute. Default is
        None. This field is provided for future developments and additional
        losses that might need some randomness. Note that special care must be
        taken when splitting the key because in-place updates are forbidden in
        eqx.Modules.
    loss_weights : LossWeightsPDEStatio, default=None
        The loss weights for the differents term : dynamic loss,
        boundary conditions if any, normalization loss if any and
        observations if any.
        All fields are set to 1.0 by default.
    derivative_keys : DerivativeKeysPDEStatio, default=None
        Specify which field of `params` should be differentiated for each
        composant of the total loss. Particularily useful for inverse problems.
        Fields can be "nn_params", "eq_params" or "both". Those that should not
        be updated will have a `jax.lax.stop_gradient` called on them. Default
        is `"nn_params"` for each composant of the loss.
    omega_boundary_fun : Callable | Dict[str, Callable], default=None
         The function to be matched in the border condition (can be None) or a
         dictionary of such functions as values and keys as described
         in `omega_boundary_condition`.
    omega_boundary_condition : str | Dict[str, str], default=None
        Either None (no condition, by default), or a string defining
        the boundary condition (Dirichlet or Von Neumann),
        or a dictionary with such strings as values. In this case,
        the keys are the facets and must be in the following order:
        1D -> [“xmin”, “xmax”], 2D -> [“xmin”, “xmax”, “ymin”, “ymax”].
        Note that high order boundaries are currently not implemented.
        A value in the dict can be None, this means we do not enforce
        a particular boundary condition on this facet.
        The facet called “xmin”, resp. “xmax” etc., in 2D,
        refers to the set of 2D points with fixed “xmin”, resp. “xmax”, etc.
    omega_boundary_dim : slice | Dict[str, slice], default=None
        Either None, or a slice object or a dictionary of slice objects as
        values and keys as described in `omega_boundary_condition`.
        `omega_boundary_dim` indicates which dimension(s) of the PINN
        will be forced to match the boundary condition.
        Note that it must be a slice and not an integer
        (but a preprocessing of the user provided argument takes care of it)
    norm_samples : Float[Array, "nb_norm_samples dimension"], default=None
        Monte-Carlo sample points for computing the
        normalization constant. Default is None.
    norm_weights : Float[Array, "nb_norm_samples"] | float | int, default=None
        The importance sampling weights for Monte-Carlo integration of the
        normalization constant. Must be provided if `norm_samples` is provided.
        `norm_weights` should have the same leading dimension as
        `norm_samples`.
        Alternatively, the user can pass a float or an integer.
        These corresponds to the weights $w_k = \frac{1}{q(x_k)}$ where
        $q(\cdot)$ is the proposal p.d.f. and $x_k$ are the Monte-Carlo samples.
    obs_slice : slice, default=None
        slice object specifying the begininning/ending of the PINN output
        that is observed (this is then useful for multidim PINN). Default is None.
    params : InitVar[Params], default=None
        The main Params object of the problem needed to instanciate the
        DerivativeKeysODE if the latter is not specified.


    Raises
    ------
    ValueError
        If conditions on omega_boundary_condition and omega_boundary_fun
        are not respected
    """

    # NOTE static=True only for leaf attributes that are not valid JAX types
    # (ie. jax.Array cannot be static) and that we do not expect to change

    u: eqx.Module
    dynamic_loss: DynamicLoss | None
    key: Key | None = eqx.field(kw_only=True, default=None)

    vmap_in_axes: tuple[Int] = eqx.field(init=False, static=True)

    def __post_init__(self, params=None):
        """
        Note that neither __init__ or __post_init__ are called when udating a
        Module with eqx.tree_at!
        """
        super().__post_init__(
            params=params
        )  # because __init__ or __post_init__ of Base
        # class is not automatically called

        self.vmap_in_axes = (0,)  # for x only here

    def _get_dynamic_loss_batch(
        self, batch: PDEStatioBatch
    ) -> Float[Array, "batch_size dimension"]:
        return batch.domain_batch

    def _get_normalization_loss_batch(
        self, _
    ) -> Float[Array, "nb_norm_samples dimension"]:
        return (self.norm_samples,)

    def _get_observations_loss_batch(
        self, batch: PDEStatioBatch
    ) -> Float[Array, "batch_size obs_dim"]:
        return (batch.obs_batch_dict["pinn_in"],)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(
        self, params: Params, batch: PDEStatioBatch
    ) -> tuple[Float[Array, "1"], dict[str, float]]:
        """
        Evaluate the loss function at a batch of points for given parameters.


        Parameters
        ---------
        params
            Parameters at which the loss is evaluated
        batch
            Composed of a batch of points in the
            domain, a batch of points in the domain
            border and an optional additional batch of parameters (eg. for
            metamodeling) and an optional additional batch of observed
            inputs/outputs/parameters
        """
        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            # update eq_params with the batches of generated params
            params = _update_eq_params_dict(params, batch.param_batch_dict)

        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)

        # dynamic part
        if self.dynamic_loss is not None:
            mse_dyn_loss = dynamic_loss_apply(
                self.dynamic_loss.evaluate,
                self.u,
                self._get_dynamic_loss_batch(batch),
                _set_derivatives(params, self.derivative_keys.dyn_loss),
                self.vmap_in_axes + vmap_in_axes_params,
                self.loss_weights.dyn_loss,
            )
        else:
            mse_dyn_loss = jnp.array(0.0)

        # normalization part
        if self.norm_samples is not None:
            mse_norm_loss = normalization_loss_apply(
                self.u,
                self._get_normalization_loss_batch(batch),
                _set_derivatives(params, self.derivative_keys.norm_loss),
                vmap_in_axes_params,
                self.norm_weights,
                self.loss_weights.norm_loss,
            )
        else:
            mse_norm_loss = jnp.array(0.0)

        # boundary part
        if self.omega_boundary_condition is not None:
            mse_boundary_loss = boundary_condition_apply(
                self.u,
                batch,
                _set_derivatives(params, self.derivative_keys.boundary_loss),
                self.omega_boundary_fun,
                self.omega_boundary_condition,
                self.omega_boundary_dim,
                self.loss_weights.boundary_loss,
            )
        else:
            mse_boundary_loss = jnp.array(0.0)

        # Observation mse
        if batch.obs_batch_dict is not None:
            # update params with the batches of observed params
            params = _update_eq_params_dict(params, batch.obs_batch_dict["eq_params"])

            mse_observation_loss = observations_loss_apply(
                self.u,
                self._get_observations_loss_batch(batch),
                _set_derivatives(params, self.derivative_keys.observations),
                self.vmap_in_axes + vmap_in_axes_params,
                batch.obs_batch_dict["val"],
                self.loss_weights.observations,
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


class LossPDENonStatio(LossPDEStatio):
    r"""Loss object for a stationary partial differential equation

    $$
        \mathcal{N}[u](t, x) = 0, \forall t \in I, \forall x \in \Omega
    $$

    where $\mathcal{N}[\cdot]$ is a differential operator.
    The boundary condition is $u(t, x)=u_b(t, x),\forall
    x\in\delta\Omega, \forall t$.
    The initial condition is $u(0, x)=u_0(x), \forall x\in\Omega$
    The additional condition of
    integrating to 1 can be included, i.e., $\int u(t, x)\mathrm{d}x=1$.

    Parameters
    ----------
    u : eqx.Module
        the PINN
    dynamic_loss : DynamicLoss
        the non stationary PDE dynamic part of the loss, basically the differential
        operator $\mathcal{N}[u](t, x)$. Should implement a method
        `dynamic_loss.evaluate(t, x, u, params)`.
        Can be None in order to access only some part of the evaluate call
        results.
    key : Key
        A JAX PRNG Key for the loss class treated as an attribute. Default is
        None. This field is provided for future developments and additional
        losses that might need some randomness. Note that special care must be
        taken when splitting the key because in-place updates are forbidden in
        eqx.Modules.
        reason
    loss_weights : LossWeightsPDENonStatio, default=None
        The loss weights for the differents term : dynamic loss,
        boundary conditions if any, initial condition, normalization loss if any and
        observations if any.
        All fields are set to 1.0 by default.
    derivative_keys : DerivativeKeysPDENonStatio, default=None
        Specify which field of `params` should be differentiated for each
        composant of the total loss. Particularily useful for inverse problems.
        Fields can be "nn_params", "eq_params" or "both". Those that should not
        be updated will have a `jax.lax.stop_gradient` called on them. Default
        is `"nn_params"` for each composant of the loss.
    omega_boundary_fun : Callable | Dict[str, Callable], default=None
         The function to be matched in the border condition (can be None) or a
         dictionary of such functions as values and keys as described
         in `omega_boundary_condition`.
    omega_boundary_condition : str | Dict[str, str], default=None
        Either None (no condition, by default), or a string defining
        the boundary condition (Dirichlet or Von Neumann),
        or a dictionary with such strings as values. In this case,
        the keys are the facets and must be in the following order:
        1D -> [“xmin”, “xmax”], 2D -> [“xmin”, “xmax”, “ymin”, “ymax”].
        Note that high order boundaries are currently not implemented.
        A value in the dict can be None, this means we do not enforce
        a particular boundary condition on this facet.
        The facet called “xmin”, resp. “xmax” etc., in 2D,
        refers to the set of 2D points with fixed “xmin”, resp. “xmax”, etc.
    omega_boundary_dim : slice | Dict[str, slice], default=None
        Either None, or a slice object or a dictionary of slice objects as
        values and keys as described in `omega_boundary_condition`.
        `omega_boundary_dim` indicates which dimension(s) of the PINN
        will be forced to match the boundary condition.
        Note that it must be a slice and not an integer
        (but a preprocessing of the user provided argument takes care of it)
    norm_samples : Float[Array, "nb_norm_samples dimension"], default=None
        Monte-Carlo sample points for computing the
        normalization constant. Default is None.
    norm_weights : Float[Array, "nb_norm_samples"] | float | int, default=None
        The importance sampling weights for Monte-Carlo integration of the
        normalization constant. Must be provided if `norm_samples` is provided.
        `norm_weights` should have the same leading dimension as
        `norm_samples`.
        Alternatively, the user can pass a float or an integer.
        These corresponds to the weights $w_k = \frac{1}{q(x_k)}$ where
        $q(\cdot)$ is the proposal p.d.f. and $x_k$ are the Monte-Carlo samples.
    obs_slice : slice, default=None
        slice object specifying the begininning/ending of the PINN output
        that is observed (this is then useful for multidim PINN). Default is None.
    initial_condition_fun : Callable, default=None
        A function representing the temporal initial condition. If None
        (default) then no initial condition is applied
    params : InitVar[Params], default=None
        The main Params object of the problem needed to instanciate the
        DerivativeKeysODE if the latter is not specified.

    """

    # NOTE static=True only for leaf attributes that are not valid JAX types
    # (ie. jax.Array cannot be static) and that we do not expect to change
    initial_condition_fun: Callable | None = eqx.field(
        kw_only=True, default=None, static=True
    )

    _max_norm_samples_omega: Int = eqx.field(init=False, static=True)
    _max_norm_time_slices: Int = eqx.field(init=False, static=True)

    def __post_init__(self, params=None):
        """
        Note that neither __init__ or __post_init__ are called when udating a
        Module with eqx.tree_at!
        """
        super().__post_init__(
            params=params
        )  # because __init__ or __post_init__ of Base
        # class is not automatically called

        self.vmap_in_axes = (0,)  # for t_x

        if self.initial_condition_fun is None:
            warnings.warn(
                "Initial condition wasn't provided. Be sure to cover for that"
                "case (e.g by. hardcoding it into the PINN output)."
            )

        # witht the variables below we avoid memory overflow since a cartesian
        # product is taken
        self._max_norm_time_slices = 100
        self._max_norm_samples_omega = 1000

    def _get_dynamic_loss_batch(
        self, batch: PDENonStatioBatch
    ) -> Float[Array, "batch_size 1+dimension"]:
        return batch.domain_batch

    def _get_normalization_loss_batch(
        self, batch: PDENonStatioBatch
    ) -> Float[Array, "nb_norm_time_slices nb_norm_samples dimension"]:
        return (
            batch.domain_batch[: self._max_norm_time_slices, 0:1],
            self.norm_samples[: self._max_norm_samples_omega],
        )

    def _get_observations_loss_batch(
        self, batch: PDENonStatioBatch
    ) -> tuple[Float[Array, "batch_size 1"], Float[Array, "batch_size dimension"]]:
        return (batch.obs_batch_dict["pinn_in"],)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(
        self, params: Params, batch: PDENonStatioBatch
    ) -> tuple[Float[Array, "1"], dict[str, float]]:
        """
        Evaluate the loss function at a batch of points for given parameters.


        Parameters
        ---------
        params
            Parameters at which the loss is evaluated
        batch
            Composed of a batch of points in
            the domain, a batch of points in the domain
            border, a batch of time points and an optional additional batch
            of parameters (eg. for metamodeling) and an optional additional batch of observed
            inputs/outputs/parameters
        """
        omega_batch = batch.initial_batch

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            # update eq_params with the batches of generated params
            params = _update_eq_params_dict(params, batch.param_batch_dict)

        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)

        # For mse_dyn_loss, mse_norm_loss, mse_boundary_loss,
        # mse_observation_loss we use the evaluate from parent class
        partial_mse, partial_mse_terms = super().evaluate(params, batch)

        # initial condition
        if self.initial_condition_fun is not None:
            mse_initial_condition = initial_condition_apply(
                self.u,
                omega_batch,
                _set_derivatives(params, self.derivative_keys.initial_condition),
                (0,) + vmap_in_axes_params,
                self.initial_condition_fun,
                self.loss_weights.initial_condition,
            )
        else:
            mse_initial_condition = jnp.array(0.0)

        # total loss
        total_loss = partial_mse + mse_initial_condition

        return total_loss, {
            **partial_mse_terms,
            "initial_condition": mse_initial_condition,
        }


class SystemLossPDE(eqx.Module):
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
    u_dict : Dict[str, eqx.Module]
        dict of PINNs
    loss_weights : LossWeightsPDEDict
        A dictionary of LossWeightsODE
    derivative_keys_dict : Dict[str, DerivativeKeysPDEStatio | DerivativeKeysPDENonStatio], default=None
        A dictionnary of DerivativeKeysPDEStatio or DerivativeKeysPDENonStatio
        specifying what field of `params`
        should be used during gradient computations for each of the terms of
        the total loss, for each of the loss in the system. Default is
        `"nn_params`" everywhere.
    dynamic_loss_dict : Dict[str, PDEStatio | PDENonStatio]
        A dict of dynamic part of the loss, basically the differential
        operator $\mathcal{N}[u](t, x)$ or $\mathcal{N}[u](x)$.
    key_dict : Dict[str, Key], default=None
        A dictionary of JAX PRNG keys. The dictionary keys of key_dict must
        match that of u_dict. See LossPDEStatio or LossPDENonStatio for
        more details.
    omega_boundary_fun_dict : Dict[str, Callable | Dict[str, Callable] | None], default=None
        A dict of of function or of dict of functions or of None
        (see doc for `omega_boundary_fun` in
        LossPDEStatio or LossPDENonStatio). Default is None.
        Must share the keys of `u_dict`.
    omega_boundary_condition_dict : Dict[str, str | Dict[str, str] | None], default=None
        A dict of strings or of dict of strings or of None
        (see doc for `omega_boundary_condition_dict` in
        LossPDEStatio or LossPDENonStatio). Default is None.
        Must share the keys of `u_dict`
    omega_boundary_dim_dict : Dict[str, slice | Dict[str, slice] | None], default=None
        A dict of slices or of dict of slices or of None
        (see doc for `omega_boundary_dim` in
        LossPDEStatio or LossPDENonStatio). Default is None.
        Must share the keys of `u_dict`
    initial_condition_fun_dict : Dict[str, Callable | None], default=None
        A dict of functions representing the temporal initial condition (None
        value is possible). If None
        (default) then no temporal boundary condition is applied
        Must share the keys of `u_dict`
    norm_samples_dict : Dict[str, Float[Array, "nb_norm_samples dimension"] | None, default=None
        A dict of Monte-Carlo sample points for computing the
        normalization constant. Default is None.
        Must share the keys of `u_dict`
    norm_weights_dict : Dict[str, Array[Float, "nb_norm_samples"] | float | int | None] | None, default=None
        A dict of jnp.array with the same keys as `u_dict`. The importance
        sampling weights for Monte-Carlo integration of the
        normalization constant for each element of u_dict. Must be provided if
        `norm_samples_dict` is provided.
        `norm_weights_dict[key]` should have the same leading dimension as
        `norm_samples_dict[key]` for each `key`.
        Alternatively, the user can pass a float or an integer.
        For each key, an array of similar shape to `norm_samples_dict[key]`
        or shape `(1,)` is expected. These corresponds to the weights $w_k =
        \frac{1}{q(x_k)}$ where $q(\cdot)$ is the proposal p.d.f. and $x_k$ are
        the Monte-Carlo samples. Default is None
    obs_slice_dict : Dict[str, slice | None] | None, default=None
        dict of obs_slice, with keys from `u_dict` to designate the
        output(s) channels that are forced to observed values, for each
        PINNs. Default is None. But if a value is given, all the entries of
        `u_dict` must be represented here with default value `jnp.s_[...]`
        if no particular slice is to be given
    params : InitVar[ParamsDict], default=None
        The main Params object of the problem needed to instanciate the
        DerivativeKeysODE if the latter is not specified.

    """

    # NOTE static=True only for leaf attributes that are not valid JAX types
    # (ie. jax.Array cannot be static) and that we do not expect to change
    u_dict: Dict[str, eqx.Module]
    dynamic_loss_dict: Dict[str, PDEStatio | PDENonStatio]
    key_dict: Dict[str, Key] | None = eqx.field(kw_only=True, default=None)
    derivative_keys_dict: Dict[
        str, DerivativeKeysPDEStatio | DerivativeKeysPDENonStatio | None
    ] = eqx.field(kw_only=True, default=None)
    omega_boundary_fun_dict: Dict[str, Callable | Dict[str, Callable] | None] | None = (
        eqx.field(kw_only=True, default=None, static=True)
    )
    omega_boundary_condition_dict: Dict[str, str | Dict[str, str] | None] | None = (
        eqx.field(kw_only=True, default=None, static=True)
    )
    omega_boundary_dim_dict: Dict[str, slice | Dict[str, slice] | None] | None = (
        eqx.field(kw_only=True, default=None, static=True)
    )
    initial_condition_fun_dict: Dict[str, Callable | None] | None = eqx.field(
        kw_only=True, default=None, static=True
    )
    norm_samples_dict: Dict[str, Float[Array, "nb_norm_samples dimension"]] | None = (
        eqx.field(kw_only=True, default=None)
    )
    norm_weights_dict: (
        Dict[str, Float[Array, "nb_norm_samples dimension"] | float | int | None] | None
    ) = eqx.field(kw_only=True, default=None)
    obs_slice_dict: Dict[str, slice | None] | None = eqx.field(
        kw_only=True, default=None, static=True
    )

    # For the user loss_weights are passed as a LossWeightsPDEDict (with internal
    # dictionary having keys in u_dict and / or dynamic_loss_dict)
    loss_weights: InitVar[LossWeightsPDEDict | None] = eqx.field(
        kw_only=True, default=None
    )
    params_dict: InitVar[ParamsDict] = eqx.field(kw_only=True, default=None)

    # following have init=False and are set in the __post_init__
    u_constraints_dict: Dict[str, LossPDEStatio | LossPDENonStatio] = eqx.field(
        init=False
    )
    derivative_keys_dyn_loss: DerivativeKeysPDEStatio | DerivativeKeysPDENonStatio = (
        eqx.field(init=False)
    )
    u_dict_with_none: Dict[str, None] = eqx.field(init=False)
    # internally the loss weights are handled with a dictionary
    _loss_weights: Dict[str, dict] = eqx.field(init=False)

    def __post_init__(self, loss_weights=None, params_dict=None):
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
        if self.norm_weights_dict is None:
            self.norm_weights_dict = self.u_dict_with_none
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
            # we need both because the constraints (all but dyn_loss) will be
            # done by iterating on u_dict while the dyn_loss will be by
            # iterating on dynamic_loss_dict. So each time we will require dome
            # derivative_keys_dict

        # derivative keys for the u_constraints. Note that we create missing
        # DerivativeKeysODE around a Params object and not ParamsDict
        # this works because u_dict.keys == params_dict.nn_params.keys()
        for k in self.u_dict.keys():
            if self.derivative_keys_dict[k] is None:
                if self.u_dict[k].eq_type == "statio_PDE":
                    self.derivative_keys_dict[k] = DerivativeKeysPDEStatio(
                        params=params_dict.extract_params(k)
                    )
                else:
                    self.derivative_keys_dict[k] = DerivativeKeysPDENonStatio(
                        params=params_dict.extract_params(k)
                    )

        # Second we make sure that all the dicts (except dynamic_loss_dict) have the same keys
        if (
            self.u_dict.keys() != self.key_dict.keys()
            or self.u_dict.keys() != self.omega_boundary_fun_dict.keys()
            or self.u_dict.keys() != self.omega_boundary_condition_dict.keys()
            or self.u_dict.keys() != self.omega_boundary_dim_dict.keys()
            or self.u_dict.keys() != self.initial_condition_fun_dict.keys()
            or self.u_dict.keys() != self.norm_samples_dict.keys()
            or self.u_dict.keys() != self.norm_weights_dict.keys()
        ):
            raise ValueError("All the dicts concerning the PINNs should have same keys")

        self._loss_weights = self.set_loss_weights(loss_weights)

        # Third, in order not to benefit from LossPDEStatio and
        # LossPDENonStatio and in order to factorize code, we create internally
        # some losses object to implement the constraints on the solutions.
        # We will not use the dynamic loss term
        self.u_constraints_dict = {}
        for i in self.u_dict.keys():
            if self.u_dict[i].eq_type == "statio_PDE":
                self.u_constraints_dict[i] = LossPDEStatio(
                    u=self.u_dict[i],
                    loss_weights=LossWeightsPDENonStatio(
                        dyn_loss=0.0,
                        norm_loss=1.0,
                        boundary_loss=1.0,
                        observations=1.0,
                        initial_condition=1.0,
                    ),
                    dynamic_loss=None,
                    key=self.key_dict[i],
                    derivative_keys=self.derivative_keys_dict[i],
                    omega_boundary_fun=self.omega_boundary_fun_dict[i],
                    omega_boundary_condition=self.omega_boundary_condition_dict[i],
                    omega_boundary_dim=self.omega_boundary_dim_dict[i],
                    norm_samples=self.norm_samples_dict[i],
                    norm_weights=self.norm_weights_dict[i],
                    obs_slice=self.obs_slice_dict[i],
                )
            elif self.u_dict[i].eq_type == "nonstatio_PDE":
                self.u_constraints_dict[i] = LossPDENonStatio(
                    u=self.u_dict[i],
                    loss_weights=LossWeightsPDENonStatio(
                        dyn_loss=0.0,
                        norm_loss=1.0,
                        boundary_loss=1.0,
                        observations=1.0,
                        initial_condition=1.0,
                    ),
                    dynamic_loss=None,
                    key=self.key_dict[i],
                    derivative_keys=self.derivative_keys_dict[i],
                    omega_boundary_fun=self.omega_boundary_fun_dict[i],
                    omega_boundary_condition=self.omega_boundary_condition_dict[i],
                    omega_boundary_dim=self.omega_boundary_dim_dict[i],
                    initial_condition_fun=self.initial_condition_fun_dict[i],
                    norm_samples=self.norm_samples_dict[i],
                    norm_weights=self.norm_weights_dict[i],
                    obs_slice=self.obs_slice_dict[i],
                )
            else:
                raise ValueError(
                    "Wrong value for self.u_dict[i].eq_type[i], "
                    f"got {self.u_dict[i].eq_type[i]}"
                )

        # derivative keys for the dynamic loss. Note that we create a
        # DerivativeKeysODE around a ParamsDict object because a whole
        # params_dict is feed to DynamicLoss.evaluate functions (extract_params
        # happen inside it)
        self.derivative_keys_dyn_loss = DerivativeKeysPDENonStatio(params=params_dict)

        # also make sure we only have PINNs or SPINNs
        if not (
            all(isinstance(value, PINN) for value in self.u_dict.values())
            or all(isinstance(value, SPINN) for value in self.u_dict.values())
        ):
            raise ValueError(
                "We only accept dictionary of PINNs or dictionary of SPINNs"
            )

    def set_loss_weights(
        self, loss_weights_init: LossWeightsPDEDict
    ) -> dict[str, dict]:
        """
        This rather complex function enables the user to specify a simple
        loss_weights=LossWeightsPDEDict(dyn_loss=1., initial_condition=Tmax)
        for ponderating values being applied to all the equations of the
        system... So all the transformations are handled here
        """
        _loss_weights = {}
        for k in fields(loss_weights_init):
            v = getattr(loss_weights_init, k.name)
            if isinstance(v, dict):
                for vv in v.keys():
                    if not isinstance(vv, (int, float)) and not (
                        isinstance(vv, Array)
                        and ((vv.shape == (1,) or len(vv.shape) == 0))
                    ):
                        # TODO improve that
                        raise ValueError(
                            f"loss values cannot be vectorial here, got {vv}"
                        )
                if k.name == "dyn_loss":
                    if v.keys() == self.dynamic_loss_dict.keys():
                        _loss_weights[k.name] = v
                    else:
                        raise ValueError(
                            "Keys in nested dictionary of loss_weights"
                            " do not match dynamic_loss_dict keys"
                        )
                else:
                    if v.keys() == self.u_dict.keys():
                        _loss_weights[k.name] = v
                    else:
                        raise ValueError(
                            "Keys in nested dictionary of loss_weights"
                            " do not match u_dict keys"
                        )
            if v is None:
                _loss_weights[k.name] = {kk: 0 for kk in self.u_dict.keys()}
            else:
                if not isinstance(v, (int, float)) and not (
                    isinstance(v, Array) and ((v.shape == (1,) or len(v.shape) == 0))
                ):
                    # TODO improve that
                    raise ValueError(f"loss values cannot be vectorial here, got {v}")
                if k.name == "dyn_loss":
                    _loss_weights[k.name] = {
                        kk: v for kk in self.dynamic_loss_dict.keys()
                    }
                else:
                    _loss_weights[k.name] = {kk: v for kk in self.u_dict.keys()}
        return _loss_weights

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(
        self,
        params_dict: ParamsDict,
        batch: PDEStatioBatch | PDENonStatioBatch,
    ) -> tuple[Float[Array, "1"], dict[str, float]]:
        """
        Evaluate the loss function at a batch of points for given parameters.


        Parameters
        ---------
        params_dict
            Parameters at which the losses of the system are evaluated
        batch
            Such named tuples are composed of  batch of points in the
            domain, a batch of points in the domain
            border, (a batch of time points a for PDENonStatioBatch) and an
            optional additional batch of parameters (eg. for metamodeling)
            and an optional additional batch of observed
            inputs/outputs/parameters
        """
        if self.u_dict.keys() != params_dict.nn_params.keys():
            raise ValueError("u_dict and params_dict[nn_params] should have same keys ")

        vmap_in_axes = (0,)

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            eq_params_batch_dict = batch.param_batch_dict

            # feed the eq_params with the batch
            for k in eq_params_batch_dict.keys():
                params_dict.eq_params[k] = eq_params_batch_dict[k]

        vmap_in_axes_params = _get_vmap_in_axes_params(
            batch.param_batch_dict, params_dict
        )

        def dyn_loss_for_one_key(dyn_loss, loss_weight):
            """The function used in tree_map"""
            return dynamic_loss_apply(
                dyn_loss.evaluate,
                self.u_dict,
                (
                    batch.domain_batch
                    if isinstance(batch, PDEStatioBatch)
                    else batch.domain_batch
                ),
                _set_derivatives(params_dict, self.derivative_keys_dyn_loss.dyn_loss),
                vmap_in_axes + vmap_in_axes_params,
                loss_weight,
                u_type=list(self.u_dict.values())[0].__class__.__base__,
            )

        dyn_loss_mse_dict = jax.tree_util.tree_map(
            dyn_loss_for_one_key,
            self.dynamic_loss_dict,
            self._loss_weights["dyn_loss"],
            is_leaf=lambda x: isinstance(
                x, (PDEStatio, PDENonStatio)
            ),  # before when dynamic losses
            # where plain (unregister pytree) node classes, we could not traverse
            # this level. Now that dynamic losses are eqx.Module they can be
            # traversed by tree map recursion. Hence we need to specify to that
            # we want to stop at this level
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
            batch = append_obs_batch(batch, self.u_dict_with_none)
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
