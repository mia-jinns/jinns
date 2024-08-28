# pylint: disable=unsubscriptable-object, no-member
"""
Main module to implement a ODE loss in jinns
"""

from typing import Union, Dict
import abc
import warnings
import jax
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
from jaxtyping import PyTree, Float, Array
from jinns.data._DataGenerators import ODEBatch
from jinns.utils._utils import (
    _get_vmap_in_axes_params,
    _set_derivatives,
    _update_eq_params_dict,
)
from jinns.loss._Losses import (
    dynamic_loss_apply,
    constraints_system_loss_apply,
    observations_loss_apply,
)
from jinns.utils._pinn import PINN
from jinns.loss._DynamicLossAbstract_eqx import ODE

_MSE_TERMS_ODE = ["observations", "dyn_loss", "initial_condition"]


class _LossODEAbstract_eqx(eqx.Module):
    """
    Parameters
    ----------

    loss_weights
        a dictionary with values used to ponderate each term in the loss
        function. Valid keys are that of _MSE_TERMS_ODE
        Note that we can have jnp.arrays with the same dimension of
        `u` which then ponderates each output of `u`
    derivative_keys
        A dict of lists of strings. In the dict, the key must correspond to
        the loss term keywords (valid keys are in _MSE_TERMS_ODE). Then each of the values must correspond to keys in the parameter
        dictionary (*at top level only of the parameter dictionary*).
        It enables selecting the set of parameters
        with respect to which the gradients of the dynamic
        loss are computed. If nothing is provided, we set ["nn_params"] for all loss term
        keywords, this is what is typically
        done in solving forward problems, when we only estimate the
        equation solution with a PINN. If some loss terms keywords are
        missing we set their value to ["nn_params"] by default for the
        same reason
    initial_condition
        tuple of length 2 with initial condition :math:`(t0, u0)`.
        Can be None in order to
        access only some part of the evaluate call results.
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
    initial_condition: Union[tuple, None] = eqx.field(
        kw_only=True, default=None, static=True
    )
    obs_slice: Union[slice, None] = eqx.field(kw_only=True, default=None, static=True)

    def __post_init__(self):
        if self.derivative_keys is None:
            # be default we only take gradient wrt nn_params
            self.derivative_keys = {k: ["nn_params"] for k in _MSE_TERMS_ODE}
        if isinstance(self.derivative_keys, list):
            # if the user only provided a list, this defines the gradient taken
            # for all the loss entries
            self.derivative_keys = {k: self.derivative_keys for k in _MSE_TERMS_ODE}
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

        for k in _MSE_TERMS_ODE:
            if k not in self.loss_weights.keys():
                self.loss_weights[k] = 0

    @abc.abstractmethod
    def evaluate(
        self: eqx.Module, params: PyTree, batch: ODEBatch
    ) -> tuple[Float, dict]:
        raise NotImplementedError


class LossODE_eqx(_LossODEAbstract_eqx):
    r"""Loss object for an ordinary differential equation

    .. math::
        \mathcal{N}[u](t) = 0, \forall t \in I

    where :math:`\mathcal{N}[\cdot]` is a differential operator and the
    initial condition is :math:`u(t_0)=u_0`.


    Parameters
    ----------
    u
        the PINN
    dynamic_loss
        the ODE dynamic part of the loss, basically the differential
        operator :math:`\mathcal{N}[u](t)`. Should implement a method
        `dynamic_loss.evaluate(t, u, params)`.
        Can be None in order to
        access only some part of the evaluate call results.
    loss_weights
        a dictionary with values used to ponderate each term in the loss
        function. Valid keys are `dyn_loss`, `initial_condition` and `observations`
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
    initial_condition :
        tuple of length 2 with initial condition :math:`(t0, u0)`.
        Can be None in order to
        access only some part of the evaluate call results.
    obs_slice:
        slice object specifying the begininning/ending
        slice of u output(s) that is observed (this is then useful for
        multidim PINN). Default is None.

    Raises
    ------
    ValueError
        if initial condition is not a tuple.
    """

    u: eqx.Module
    dynamic_loss: Union[eqx.Module, None]

    def __post_init__(self):
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
            A ODEBatch object.
            Such a named tuple is composed of a batch of time points
            at which to evaluate an optional additional batch of parameters (eg. for
            metamodeling) and an optional additional batch of observed
            inputs/outputs/parameters
        """
        temporal_batch = batch.temporal_batch

        vmap_in_axes_t = (0,)

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            # update params with the batches of generated params
            params = _update_eq_params_dict(params, batch.param_batch_dict)

        vmap_in_axes_params = _get_vmap_in_axes_params(batch.param_batch_dict, params)

        ## dynamic part
        params_ = _set_derivatives(params, "dyn_loss", self.derivative_keys)
        if self.dynamic_loss is not None:
            mse_dyn_loss = dynamic_loss_apply(
                self.dynamic_loss.evaluate,
                self.u,
                (temporal_batch,),
                params_,
                vmap_in_axes_t + vmap_in_axes_params,
                self.loss_weights["dyn_loss"],
            )
        else:
            mse_dyn_loss = jnp.array(0.0)

        # initial condition
        params_ = _set_derivatives(params, "initial_condition", self.derivative_keys)
        if self.initial_condition is not None:
            vmap_in_axes = (None,) + vmap_in_axes_params
            if not jax.tree_util.tree_leaves(vmap_in_axes):
                # test if only None in vmap_in_axes to avoid the value error:
                # `vmap must have at least one non-None value in in_axes`
                v_u = self.u
            else:
                v_u = vmap(self.u, (None,) + vmap_in_axes_params)
            t0, u0 = self.initial_condition  # pylint: disable=unpacking-non-sequence
            t0 = jnp.array(t0)
            u0 = jnp.array(u0)
            mse_initial_condition = jnp.mean(
                self.loss_weights["initial_condition"]
                * jnp.sum((v_u(t0, params_) - u0) ** 2, axis=-1)
            )
        else:
            mse_initial_condition = jnp.array(0.0)

        if batch.obs_batch_dict is not None:
            # update params with the batches of observed params
            params = _update_eq_params_dict(params, batch.obs_batch_dict["eq_params"])

            # MSE loss wrt to an observed batch
            params_ = _set_derivatives(params, "observations", self.derivative_keys)
            mse_observation_loss = observations_loss_apply(
                self.u,
                (batch.obs_batch_dict["pinn_in"],),
                params_,
                vmap_in_axes_t + vmap_in_axes_params,
                batch.obs_batch_dict["val"],
                self.loss_weights["observations"],
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


class SystemLossODE_eqx(eqx.Module):
    r"""
    Class to implement a system of ODEs.
    The goal is to give maximum freedom to the user. The class is created with
    a dict of dynamic loss and a dict of initial conditions. When then iterate
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
        dict of PINNs
    loss_weights
        A dictionary of dictionaries with values used to
        ponderate each term in the loss
        function. Valid keys in the first dictionary are `dyn_loss`,
        `initial_condition` and `observations`. The keys of the nested
        dictionaries must share the keys of `u_dict`. Note that the values
        at the leaf level can have jnp.arrays with the same dimension of
        `u` which then ponderates each output of `u`
    derivative_keys_dict
        A dict of derivative keys as defined in LossODE. The key of this
        dict must be that of `dynamic_loss_dict` at least and specify how
        to compute gradient for the `dyn_loss` loss term at least (see the
        check at the beginning of the present `__init__` function.
        Other keys of this dict might be that of `u_dict` to specify how to
        compute gradients for all the different constraints. If those keys
        are not specified then the default behaviour for `derivative_keys`
        of LossODE is used
    initial_condition_dict
        dict of tuple of length 2 with initial condition :math:`(t_0, u_0)`
        Must share the keys of `u_dict`. Default is None. No initial
        condition is permitted when the initial condition is hardcoded in
        the PINN architecture for example
    dynamic_loss_dict
        dict of dynamic part of the loss, basically the differential
        operator :math:`\mathcal{N}[u](t)`. Should implement a method
        `dynamic_loss.evaluate(t, u, params)`
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
    dynamic_loss_dict: Dict[str, ODE] = eqx.field(static=True)
    derivative_keys_dict: Union[Dict[str, Union[list, None]], None] = eqx.field(
        kw_only=True, default=None, static=True
    )
    loss_weights: Union[Dict[str, Dict[str, Union[Float, Array]]], None] = eqx.field(
        kw_only=True, default=None, static=True
    )
    initial_condition_dict: Union[Dict[str, tuple], None] = eqx.field(
        kw_only=True, default=None, static=True
    )
    obs_slice_dict: Union[Dict[str, Union[slice, None]], None] = eqx.field(
        kw_only=True, default=None, static=True
    )

    u_constraints_dict: Dict[str, list] = eqx.field(init=False, static=True)
    derivative_keys_u_dict: Dict[str, list] = eqx.field(init=False, static=True)
    derivative_keys_dyn_loss_dict: Dict[str, list] = eqx.field(init=False, static=True)
    u_dict_with_none: Dict[str, None] = eqx.field(init=False, static=True)

    def __post_init__(self):
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
        else:
            self.derivative_keys_dict = self.derivative_keys_dict

        # but then if the user did not provide anything, we must at least have
        # a default value for the dynamic_loss_dict keys entries in
        # self.derivative_keys_dict since the computation of dynamic losses is
        # made without create a lossODE object that would provide the
        # default values
        for k in self.dynamic_loss_dict.keys():
            if self.derivative_keys_dict[k] is None:
                self.derivative_keys_dict[k] = {"dyn_loss": ["nn_params"]}

        self.loss_weights = self.set_loss_weights(self.loss_weights)

        # The constaints on the solutions will be implemented by reusing a
        # LossODE class without dynamic loss term
        self.u_constraints_dict = {}
        for i in self.u_dict.keys():
            self.u_constraints_dict[i] = LossODE_eqx(
                u=self.u_dict[i],
                loss_weights={
                    "dyn_loss": 0.0,
                    "initial_condition": 1.0,
                    "observations": 1.0,
                },
                dynamic_loss=None,
                derivative_keys=self.derivative_keys_dict[i],
                initial_condition=self.initial_condition_dict[i],
                obs_slice=self.obs_slice_dict[i],
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
        if all(v is None for k, v in self.initial_condition_dict.items()):
            _loss_weights["initial_condition"] = {k: 0 for k in self.u_dict.keys()}
        if "observations" not in value.keys():
            _loss_weights["observations"] = {k: 0 for k in self.u_dict.keys()}

        return _loss_weights

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, params_dict, batch):
        """
        Evaluate the loss function at a batch of points for given parameters.


        Parameters
        ---------
        params
            A dictionary of dictionaries of parameters of the model.
            Typically, it is a dictionary of dictionaries of
            dictionaries: `eq_params` and `nn_params``, respectively the
            differential equation parameters and the neural network parameter.
            Note that params_dict["nn_params"] need not be a dictionary anymore
            but can directly be the parameters. It is useful when working with
            neural networks sharing the same parameters
        batch
            A ODEBatch object.
            Such a named tuple is composed of a batch of time points
            at which to evaluate an optional additional batch of parameters (eg. for
            metamodeling) and an optional additional batch of observed
            inputs/outputs/parameters
        """
        if (
            isinstance(params_dict["nn_params"], dict)
            and self.u_dict.keys() != params_dict["nn_params"].keys()
        ):
            raise ValueError("u_dict and params_dict[nn_params] should have same keys ")

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

        def dyn_loss_for_one_key(dyn_loss, derivative_key, loss_weight):
            """This function is used in tree_map"""
            params_dict_ = _set_derivatives(params_dict, "dyn_loss", derivative_key)
            return dynamic_loss_apply(
                dyn_loss.evaluate,
                self.u_dict,
                (temporal_batch,),
                params_dict_,
                vmap_in_axes_t + vmap_in_axes_params,
                loss_weight,
                u_type=PINN,
            )

        dyn_loss_mse_dict = jax.tree_util.tree_map(
            dyn_loss_for_one_key,
            self.dynamic_loss_dict,
            self.derivative_keys_dyn_loss_dict,
            self.loss_weights["dyn_loss"],
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
