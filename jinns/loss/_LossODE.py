# pylint: disable=unsubscriptable-object, no-member
"""
Main module to implement a ODE loss in jinns
"""
from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from dataclasses import InitVar, fields
from typing import TYPE_CHECKING, Dict
import abc
import warnings
import jax
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
from jaxtyping import Float, Array, Int
from jinns.data._DataGenerators import append_obs_batch
from jinns.loss._loss_utils import (
    dynamic_loss_apply,
    constraints_system_loss_apply,
    observations_loss_apply,
)
from jinns.parameters._params import (
    _get_vmap_in_axes_params,
    _update_eq_params_dict,
)
from jinns.parameters._derivative_keys import _set_derivatives, DerivativeKeysODE
from jinns.loss._loss_weights import LossWeightsODE, LossWeightsODEDict
from jinns.loss._DynamicLossAbstract import ODE
from jinns.nn._pinn import PINN

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
    initial_condition : tuple, default=None
        tuple of length 2 with initial condition $(t_0, u_0)$.
    obs_slice : Slice, default=None
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
    initial_condition: tuple | None = eqx.field(kw_only=True, default=None)
    obs_slice: slice | None = eqx.field(kw_only=True, default=None, static=True)

    params: InitVar[Params] = eqx.field(default=None, kw_only=True)

    def __post_init__(self, params=None):
        if self.loss_weights is None:
            self.loss_weights = LossWeightsODE()

        if self.derivative_keys is None:
            try:
                # be default we only take gradient wrt nn_params
                self.derivative_keys = DerivativeKeysODE(params=params)
            except ValueError as exc:
                raise ValueError(
                    "Problem at self.derivative_keys initialization "
                    f"received {self.derivative_keys=} and {params=}"
                ) from exc
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

        if self.obs_slice is None:
            self.obs_slice = jnp.s_[...]

        if self.loss_weights is None:
            self.loss_weights = LossWeightsODE()

    @abc.abstractmethod
    def evaluate(
        self: eqx.Module, params: Params, batch: ODEBatch
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
    initial_condition : tuple, default=None
        tuple of length 2 with initial condition $(t_0, u_0)$.
    obs_slice Slice, default=None
        Slice object specifying the begininning/ending
        slice of u output(s) that is observed. This is useful for
        multidimensional PINN, with partially observed outputs.
        Default is None (whole output is observed).
    params : InitVar[Params], default=None
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
    u: eqx.Module
    dynamic_loss: DynamicLoss | None

    vmap_in_axes: tuple[Int] = eqx.field(init=False, static=True)

    def __post_init__(self, params=None):
        super().__post_init__(
            params=params
        )  # because __init__ or __post_init__ of Base
        # class is not automatically called

        self.vmap_in_axes = (0,)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(
        self, params: Params, batch: ODEBatch
    ) -> tuple[Float[Array, "1"], dict[str, float]]:
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
                _set_derivatives(params, self.derivative_keys.dyn_loss),
                self.vmap_in_axes + vmap_in_axes_params,
                self.loss_weights.dyn_loss,
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
            t0 = jnp.array([t0])
            u0 = jnp.array(u0)
            mse_initial_condition = jnp.mean(
                self.loss_weights.initial_condition
                * jnp.sum(
                    (
                        v_u(
                            t0,
                            _set_derivatives(
                                params, self.derivative_keys.initial_condition
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
                (batch.obs_batch_dict["pinn_in"],),
                _set_derivatives(params, self.derivative_keys.observations),
                self.vmap_in_axes + vmap_in_axes_params,
                batch.obs_batch_dict["val"],
                self.loss_weights.observations,
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


class SystemLossODE(eqx.Module):
    r"""
    Class to implement a system of ODEs.
    The goal is to give maximum freedom to the user. The class is created with
    a dict of dynamic loss and a dict of initial conditions. Then, it iterates
    over the dynamic losses that compose the system. All PINNs are passed as
    arguments to each dynamic loss evaluate functions, along with all the
    parameter dictionaries. All specification is left to the responsability
    of the user, inside the dynamic loss.

    **Note:** All the dictionaries (except `dynamic_loss_dict`) must have the same keys.
    Indeed, these dictionaries (except `dynamic_loss_dict`) are tied to one
    solution.

    Parameters
    ----------
    u_dict : Dict[str, eqx.Module]
        dict of PINNs
    loss_weights : LossWeightsODEDict
        A dictionary of LossWeightsODE
    derivative_keys_dict : Dict[str, DerivativeKeysODE], default=None
        A dictionnary of DerivativeKeysODE specifying what field of `params`
        should be used during gradient computations for each of the terms of
        the total loss, for each of the loss in the system. Default is
        `"nn_params`" everywhere.
    initial_condition_dict : Dict[str, tuple], default=None
        dict of tuple of length 2 with initial condition $(t_0, u_0)$
        Must share the keys of `u_dict`. Default is None. No initial
        condition is permitted when the initial condition is hardcoded in
        the PINN architecture for example
    dynamic_loss_dict : Dict[str, ODE]
        dict of dynamic part of the loss, basically the differential
        operator $\mathcal{N}[u](t)$. Should implement a method
        `dynamic_loss.evaluate(t, u, params)`
    obs_slice_dict : Dict[str, Slice]
        dict of obs_slice, with keys from `u_dict` to designate the
        output(s) channels that are observed, for each
        PINNs. Default is None. But if a value is given, all the entries of
        `u_dict` must be represented here with default value `jnp.s_[...]`
        if no particular slice is to be given.
    params_dict : InitVar[ParamsDict], default=None
        The main Params object of the problem needed to instanciate the
        DerivativeKeysODE if the latter is not specified.

    Raises
    ------
    ValueError
        if initial condition is not a dict of tuple.
    ValueError
        if the dictionaries that should share the keys of u_dict do not.
    """

    # NOTE static=True only for leaf attributes that are not valid JAX types
    # (ie. jax.Array cannot be static) and that we do not expect to change
    u_dict: Dict[str, eqx.Module]
    dynamic_loss_dict: Dict[str, ODE]
    derivative_keys_dict: Dict[str, DerivativeKeysODE | None] | None = eqx.field(
        kw_only=True, default=None
    )
    initial_condition_dict: Dict[str, tuple] | None = eqx.field(
        kw_only=True, default=None
    )

    obs_slice_dict: Dict[str, slice | None] | None = eqx.field(
        kw_only=True, default=None, static=True
    )  # We are at an "leaf" attribute here (slice, not valid JAX type). Since
    # we do not expect it to change with put a static=True here. But note that
    # this is the only static for all the SystemLossODE attribute, since all
    # other are composed of more complex structures ("non-leaf")

    # For the user loss_weights are passed as a LossWeightsODEDict (with internal
    # dictionary having keys in u_dict and / or dynamic_loss_dict)
    loss_weights: InitVar[LossWeightsODEDict | None] = eqx.field(
        kw_only=True, default=None
    )
    params_dict: InitVar[ParamsDict] = eqx.field(kw_only=True, default=None)

    u_constraints_dict: Dict[str, LossODE] = eqx.field(init=False)
    derivative_keys_dyn_loss: DerivativeKeysODE = eqx.field(init=False)

    u_dict_with_none: Dict[str, None] = eqx.field(init=False)
    # internally the loss weights are handled with a dictionary
    _loss_weights: Dict[str, dict] = eqx.field(init=False)

    def __post_init__(self, loss_weights=None, params_dict=None):
        # a dictionary that will be useful at different places
        self.u_dict_with_none = {k: None for k in self.u_dict.keys()}
        if self.initial_condition_dict is None:
            self.initial_condition_dict = self.u_dict_with_none
        else:
            if self.u_dict.keys() != self.initial_condition_dict.keys():
                raise ValueError(
                    "initial_condition_dict should have same keys as u_dict"
                )
        if self.obs_slice_dict is None:
            self.obs_slice_dict = {k: jnp.s_[...] for k in self.u_dict.keys()}
        else:
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
                self.derivative_keys_dict[k] = DerivativeKeysODE(
                    params=params_dict.extract_params(k)
                )

        self._loss_weights = self.set_loss_weights(loss_weights)

        # The constaints on the solutions will be implemented by reusing a
        # LossODE class without dynamic loss term
        self.u_constraints_dict = {}
        for i in self.u_dict.keys():
            self.u_constraints_dict[i] = LossODE(
                u=self.u_dict[i],
                loss_weights=LossWeightsODE(
                    dyn_loss=0.0,
                    initial_condition=1.0,
                    observations=1.0,
                ),
                dynamic_loss=None,
                derivative_keys=self.derivative_keys_dict[i],
                initial_condition=self.initial_condition_dict[i],
                obs_slice=self.obs_slice_dict[i],
            )

        # derivative keys for the dynamic loss. Note that we create a
        # DerivativeKeysODE around a ParamsDict object because a whole
        # params_dict is feed to DynamicLoss.evaluate functions (extract_params
        # happen inside it)
        self.derivative_keys_dyn_loss = DerivativeKeysODE(params=params_dict)

    def set_loss_weights(self, loss_weights_init):
        """
        This rather complex function enables the user to specify a simple
        loss_weights=LossWeightsODEDict(dyn_loss=1., initial_condition=Tmax)
        for ponderating values being applied to all the equations of the
        system... So all the transformations are handled here
        """
        _loss_weights = {}
        for k in fields(loss_weights_init):
            v = getattr(loss_weights_init, k.name)
            if isinstance(v, dict):
                for vv in v.values():
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
            elif v is None:
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

    def evaluate(self, params_dict: ParamsDict, batch: ODEBatch) -> Float[Array, "1"]:
        """
        Evaluate the loss function at a batch of points for given parameters.


        Parameters
        ---------
        params
            A ParamsDict object
        batch
            A ODEBatch object.
            Such a named tuple is composed of a batch of time points
            at which to evaluate an optional additional batch of parameters (eg. for
            metamodeling) and an optional additional batch of observed
            inputs/outputs/parameters
        """
        if (
            isinstance(params_dict.nn_params, dict)
            and self.u_dict.keys() != params_dict.nn_params.keys()
        ):
            raise ValueError("u_dict and params_dict.nn_params should have same keys ")

        temporal_batch = batch.temporal_batch

        vmap_in_axes_t = (0,)

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            # update params with the batches of generated params
            params = _update_eq_params_dict(params, batch.param_batch_dict)

        vmap_in_axes_params = _get_vmap_in_axes_params(
            batch.param_batch_dict, params_dict
        )

        def dyn_loss_for_one_key(dyn_loss, loss_weight):
            """This function is used in tree_map"""
            return dynamic_loss_apply(
                dyn_loss.evaluate,
                self.u_dict,
                temporal_batch,
                _set_derivatives(params_dict, self.derivative_keys_dyn_loss.dyn_loss),
                vmap_in_axes_t + vmap_in_axes_params,
                loss_weight,
                u_type=PINN,
            )

        dyn_loss_mse_dict = jax.tree_util.tree_map(
            dyn_loss_for_one_key,
            self.dynamic_loss_dict,
            self._loss_weights["dyn_loss"],
            is_leaf=lambda x: isinstance(x, ODE),  # before when dynamic losses
            # where plain (unregister pytree) node classes, we could not traverse
            # this level. Now that dynamic losses are eqx.Module they can be
            # traversed by tree map recursion. Hence we need to specify to that
            # we want to stop at this level
        )
        mse_dyn_loss = jax.tree_util.tree_reduce(
            lambda x, y: x + y, jax.tree_util.tree_leaves(dyn_loss_mse_dict)
        )

        # initial conditions and observation_loss via the internal LossODE
        loss_weight_struct = {
            "dyn_loss": "*",
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
