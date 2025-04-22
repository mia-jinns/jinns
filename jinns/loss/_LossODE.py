# pylint: disable=unsubscriptable-object, no-member
"""
Main module to implement a ODE loss in jinns
"""
from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from dataclasses import InitVar
from typing import TYPE_CHECKING, Any
from types import EllipsisType
import abc
import warnings
import jax
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
from jaxtyping import Float, Array
from jinns.loss._loss_utils import (
    dynamic_loss_apply,
    observations_loss_apply,
)
from jinns.parameters._params import (
    _get_vmap_in_axes_params,
    _update_eq_params_dict,
)
from jinns.parameters._derivative_keys import _set_derivatives, DerivativeKeysODE
from jinns.loss._loss_weights import LossWeightsODE

if TYPE_CHECKING:
    from jinns.utils._types import *


class _LossODEAbstract(eqx.Module):
    """
    Parameters
    ----------

    loss_weights : LossWeightsODE, default=None
        The loss weights for the differents term : dynamic loss,
        initial condition and eventually observations if any. All fields are
        set to 1.0 by default.
    derivative_keys : DerivativeKeysODE, default=None
        Specify which field of `params` should be differentiated for each
        composant of the total loss. Particularily useful for inverse problems.
        Fields can be "nn_params", "eq_params" or "both". Those that should not
        be updated will have a `jax.lax.stop_gradient` called on them. Default
        is `"nn_params"` for each composant of the loss.
    initial_condition : tuple[float | Float[Array, "1"], Float[Array, "dim"]], default=None
        tuple of length 2 with initial condition $(t_0, u_0)$.
    obs_slice : EllipsisType | slice | None, default=None
        Slice object specifying the begininning/ending
        slice of u output(s) that is observed. This is useful for
        multidimensional PINN, with partially observed outputs.
        Default is None (whole output is observed).
    params : InitVar[Params], default=None
        The main Params object of the problem needed to instanciate the
        DerivativeKeysODE if the latter is not specified.
    """

    # NOTE static=True only for leaf attributes that are not valid JAX types
    # (ie. jax.Array cannot be static) and that we do not expect to change
    # kw_only in base class is motivated here: https://stackoverflow.com/a/69822584
    derivative_keys: DerivativeKeysODE | None = eqx.field(kw_only=True, default=None)
    loss_weights: LossWeightsODE | None = eqx.field(kw_only=True, default=None)
    initial_condition: tuple[float | Float[Array, "1"], Float[Array, "dim"]] | None = (
        eqx.field(kw_only=True, default=None)
    )
    obs_slice: EllipsisType | slice | None = eqx.field(
        kw_only=True, default=None, static=True
    )

    params: InitVar[Params[Any]] = eqx.field(default=None, kw_only=True)

    def __post_init__(self, params: Params[Any] | None = None):
        if self.loss_weights is None:
            self.loss_weights = LossWeightsODE()

        if self.derivative_keys is None:
            # by default we only take gradient wrt nn_params
            if params is None:
                raise ValueError(
                    "Problem at self.derivative_keys initialization "
                    f"received {self.derivative_keys=} and {params=}"
                )
            self.derivative_keys = DerivativeKeysODE(params=params)
        if self.initial_condition is None:
            warnings.warn(
                "Initial condition wasn't provided. Be sure to cover for that"
                "case (e.g by. hardcoding it into the PINN output)."
            )
        else:
            if (
                not isinstance(self.initial_condition, tuple)
                or len(self.initial_condition) != 2
            ):
                raise ValueError(
                    "Initial condition should be a tuple of len 2 with (t0, u0), "
                    f"{self.initial_condition} was passed."
                )
            # some checks/reshaping for t0
            t0, u0 = self.initial_condition
            if isinstance(t0, Array):
                if not t0.shape:  # e.g. user input: jnp.array(0.)
                    t0 = jnp.array([t0])
                elif t0.shape != (1,):
                    raise ValueError(
                        f"Wrong t0 input (self.initial_condition[0]) It should be"
                        f"a float or an array of shape (1,). Got shape: {t0.shape}"
                    )
            if isinstance(t0, float):  # e.g. user input: 0
                t0 = jnp.array([t0])
            self.initial_condition = (t0, u0)

        if self.obs_slice is None:
            self.obs_slice = jnp.s_[...]

        if self.loss_weights is None:
            self.loss_weights = LossWeightsODE()

    @abc.abstractmethod
    def evaluate(
        self: eqx.Module, params: Params[Array | int], batch: ODEBatch
    ) -> tuple[Float, dict]:
        raise NotImplementedError


class LossODE(_LossODEAbstract):
    r"""Loss object for an ordinary differential equation

    $$
    \mathcal{N}[u](t) = 0, \forall t \in I
    $$

    where $\mathcal{N}[\cdot]$ is a differential operator and the
    initial condition is $u(t_0)=u_0$.


    Parameters
    ----------
    loss_weights : LossWeightsODE, default=None
        The loss weights for the differents term : dynamic loss,
        initial condition and eventually observations if any. All fields are
        set to 1.0 by default.
    derivative_keys : DerivativeKeysODE, default=None
        Specify which field of `params` should be differentiated for each
        composant of the total loss. Particularily useful for inverse problems.
        Fields can be "nn_params", "eq_params" or "both". Those that should not
        be updated will have a `jax.lax.stop_gradient` called on them. Default
        is `"nn_params"` for each composant of the loss.
    initial_condition : tuple[float | Float[Array, "1"]], default=None
        tuple of length 2 with initial condition $(t_0, u_0)$.
    obs_slice : EllipsisType | slice | None, default=None
        Slice object specifying the begininning/ending
        slice of u output(s) that is observed. This is useful for
        multidimensional PINN, with partially observed outputs.
        Default is None (whole output is observed).
    params : InitVar[Params[Any]], default=None
        The main Params object of the problem needed to instanciate the
        DerivativeKeysODE if the latter is not specified.
    u : eqx.Module
        the PINN
    dynamic_loss : DynamicLoss
        the ODE dynamic part of the loss, basically the differential
        operator $\mathcal{N}[u](t)$. Should implement a method
        `dynamic_loss.evaluate(t, u, params)`.
        Can be None in order to access only some part of the evaluate call.

    Raises
    ------
    ValueError
        if initial condition is not a tuple.
    """

    # NOTE static=True only for leaf attributes that are not valid JAX types
    # (ie. jax.Array cannot be static) and that we do not expect to change
    u: AbstractPINN
    dynamic_loss: DynamicLoss | None

    vmap_in_axes: tuple[int] = eqx.field(init=False, static=True)

    def __post_init__(self, params: Params[Any] | None = None):
        super().__post_init__(
            params=params
        )  # because __init__ or __post_init__ of Base
        # class is not automatically called

        self.vmap_in_axes = (0,)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(
        self, params: Params[Array | int], batch: ODEBatch
    ) -> tuple[Float[Array, "1"], dict[str, Array]]:
        """
        Evaluate the loss function at a batch of points for given parameters.


        Parameters
        ---------
        params
            Parameters at which the loss is evaluated
        batch
            Composed of a batch of time points
            at which to evaluate the differential operator. An optional additional batch of parameters (eg. for metamodeling) and an optional additional batch of observed inputs/outputs/parameters can
            be supplied.
        """
        temporal_batch = batch.temporal_batch

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            # update params with the batches of generated params
            params = _update_eq_params_dict(params, batch.param_batch_dict)

        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)

        ## dynamic part
        if self.dynamic_loss is not None:
            mse_dyn_loss = dynamic_loss_apply(
                self.dynamic_loss.evaluate,
                self.u,
                temporal_batch,
                _set_derivatives(params, self.derivative_keys.dyn_loss),  # type: ignore
                self.vmap_in_axes + vmap_in_axes_params,
                self.loss_weights.dyn_loss,  # type: ignore
            )
        else:
            mse_dyn_loss = jnp.array(0.0)

        # initial condition
        if self.initial_condition is not None:
            vmap_in_axes = (None,) + vmap_in_axes_params
            if not jax.tree_util.tree_leaves(vmap_in_axes):
                # test if only None in vmap_in_axes to avoid the value error:
                # `vmap must have at least one non-None value in in_axes`
                v_u = self.u
            else:
                v_u = vmap(self.u, (None,) + vmap_in_axes_params)
            t0, u0 = self.initial_condition  # pylint: disable=unpacking-non-sequence
            u0 = jnp.array(u0)
            mse_initial_condition = jnp.mean(
                self.loss_weights.initial_condition  # type: ignore
                * jnp.sum(
                    (
                        v_u(
                            t0,
                            _set_derivatives(
                                params, self.derivative_keys.initial_condition  # type: ignore
                            ),
                        )
                        - u0
                    )
                    ** 2,
                    axis=-1,
                )
            )
        else:
            mse_initial_condition = jnp.array(0.0)

        if batch.obs_batch_dict is not None:
            # update params with the batches of observed params
            params = _update_eq_params_dict(params, batch.obs_batch_dict["eq_params"])

            # MSE loss wrt to an observed batch
            mse_observation_loss = observations_loss_apply(
                self.u,
                batch.obs_batch_dict["pinn_in"],
                _set_derivatives(params, self.derivative_keys.observations),  # type: ignore
                self.vmap_in_axes + vmap_in_axes_params,
                batch.obs_batch_dict["val"],
                self.loss_weights.observations,  # type: ignore
                self.obs_slice,
            )
        else:
            mse_observation_loss = jnp.array(0.0)

        # total loss
        total_loss = mse_dyn_loss + mse_initial_condition + mse_observation_loss
        return total_loss, (
            {
                "dyn_loss": mse_dyn_loss,
                "initial_condition": mse_initial_condition,
                "observations": mse_observation_loss,
            }
        )
