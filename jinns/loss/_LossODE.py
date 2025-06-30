"""
Main module to implement a ODE loss in jinns
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from dataclasses import InitVar
from typing import TYPE_CHECKING, TypedDict, Callable
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
    initial_condition_check,
)
from jinns.parameters._params import (
    _get_vmap_in_axes_params,
    _update_eq_params_dict,
)
from jinns.parameters._derivative_keys import _set_derivatives, DerivativeKeysODE
from jinns.loss._loss_weights import LossWeightsODE
from jinns.loss._abstract_loss import AbstractLoss
from jinns.loss._loss_components import ODEComponents
from jinns.parameters._params import Params

if TYPE_CHECKING:
    # imports only used in type hints
    from jinns.data._Batchs import ODEBatch
    from jinns.nn._abstract_pinn import AbstractPINN
    from jinns.loss import ODE

    class LossDictODE(TypedDict):
        dyn_loss: Float[Array, " "]
        initial_condition: Float[Array, " "]
        observations: Float[Array, " "]


class _LossODEAbstract(AbstractLoss):
    r"""
    Parameters
    ----------

    loss_weights : LossWeightsODE, default=None
        The loss weights for the differents term : dynamic loss,
        initial condition and eventually observations if any.
        Can be updated according to a specific algorithm. See
        `update_weight_method`
    update_weight_method : Literal['soft_adapt', 'lr_annealing', 'ReLoBRaLo'], default=None
        Default is None meaning no update for loss weights. Otherwise a string
    derivative_keys : DerivativeKeysODE, default=None
        Specify which field of `params` should be differentiated for each
        composant of the total loss. Particularily useful for inverse problems.
        Fields can be "nn_params", "eq_params" or "both". Those that should not
        be updated will have a `jax.lax.stop_gradient` called on them. Default
        is `"nn_params"` for each composant of the loss.
    initial_condition : tuple[
            Float[Array, "n_cond "],
            Float[Array, "n_cond dim"]
        ] |
        tuple[int | float | Float[Array, " "],
              int | float | Float[Array, " dim"]
        ] | None, default=None
        Most of the time, a tuple of length 2 with initial condition $(t_0, u_0)$.
        From jinns v1.5.1 we accept tuples of jnp arrays with shape (n_cond, 1) for t0 and (n_cond, dim) for u0. This is useful to include observed conditions at different time points, such as *e.g* final conditions. It was designed to implement $\mathcal{L}^{aux}$ from _Systems biology informed deep learning for inferring parameters and hidden dynamics_, Alireza Yazdani et al., 2020
    obs_slice : EllipsisType | slice | None, default=None
        Slice object specifying the begininning/ending
        slice of u output(s) that is observed. This is useful for
        multidimensional PINN, with partially observed outputs.
        Default is None (whole output is observed).
    params : InitVar[Params[Array]], default=None
        The main Params object of the problem needed to instanciate the
        DerivativeKeysODE if the latter is not specified.
    """

    # NOTE static=True only for leaf attributes that are not valid JAX types
    # (ie. jax.Array cannot be static) and that we do not expect to change
    # kw_only in base class is motivated here: https://stackoverflow.com/a/69822584
    derivative_keys: DerivativeKeysODE | None = eqx.field(kw_only=True, default=None)
    loss_weights: LossWeightsODE | None = eqx.field(kw_only=True, default=None)
    initial_condition: (
        tuple[Float[Array, " n_cond 1"], Float[Array, " n_cond dim"]]
        | tuple[int | float | Float[Array, " "], int | float | Float[Array, " dim"]]
        | None
    ) = eqx.field(kw_only=True, default=None)
    obs_slice: EllipsisType | slice | None = eqx.field(
        kw_only=True, default=None, static=True
    )

    params: InitVar[Params[Array]] = eqx.field(default=None, kw_only=True)

    def __post_init__(self, params: Params[Array] | None = None):
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
            # some checks/reshaping for t0 and u0
            t0, u0 = self.initial_condition
            if isinstance(t0, Array):
                # at the end we want to end up with t0 of shape (:, 1) to account for
                # possibly several data points
                if t0.ndim <= 1:
                    # in this case we assume t0 belongs one (initial)
                    # condition
                    t0 = initial_condition_check(t0, dim_size=1)[
                        None, :
                    ]  # make a (1, 1) here
                if t0.ndim > 2:
                    raise ValueError(
                        "It t0 is an Array, it represents n_cond"
                        " imposed conditions and must be of shape (n_cond, 1)"
                    )
            else:
                # in this case t0 clearly represents one (initial) condition
                t0 = initial_condition_check(t0, dim_size=1)[
                    None, :
                ]  # make a (1, 1) here
            if isinstance(u0, Array):
                # at the end we want to end up with u0 of shape (:, dim) to account for
                # possibly several data points
                if not u0.shape:
                    # in this case we assume u0 belongs to one (initial)
                    # condition
                    u0 = initial_condition_check(u0, dim_size=1)[
                        None, :
                    ]  # make a (1, 1) here
                elif u0.ndim == 1:
                    # in this case we assume u0 belongs to one (initial)
                    # condition
                    u0 = initial_condition_check(u0, dim_size=u0.shape[0])[
                        None, :
                    ]  # make a (1, dim) here
                if u0.ndim > 2:
                    raise ValueError(
                        "It u0 is an Array, it represents n_cond "
                        "imposed conditions and must be of shape (n_cond, dim)"
                    )
            else:
                # at the end we want to end up with u0 of shape (:, dim) to account for
                # possibly several data points
                u0 = initial_condition_check(u0, dim_size=None)[
                    None, :
                ]  # make a (1, 1) here

            if t0.shape[0] != u0.shape[0] or t0.ndim != u0.ndim:
                raise ValueError(
                    "t0 and u0 must represent a same number of initial"
                    " conditial conditions"
                )

            self.initial_condition = (t0, u0)

        if self.obs_slice is None:
            self.obs_slice = jnp.s_[...]

        if self.loss_weights is None:
            self.loss_weights = LossWeightsODE()

    @abc.abstractmethod
    def __call__(self, *_, **__):
        pass

    @abc.abstractmethod
    def evaluate(
        self: eqx.Module, params: Params[Array], batch: ODEBatch
    ) -> tuple[Float[Array, " "], LossDictODE]:
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
        initial condition and eventually observations if any.
        Can be updated according to a specific algorithm. See
        `update_weight_method`
    update_weight_method : Literal['soft_adapt', 'lr_annealing', 'ReLoBRaLo'], default=None
        Default is None meaning no update for loss weights. Otherwise a string
    derivative_keys : DerivativeKeysODE, default=None
        Specify which field of `params` should be differentiated for each
        composant of the total loss. Particularily useful for inverse problems.
        Fields can be "nn_params", "eq_params" or "both". Those that should not
        be updated will have a `jax.lax.stop_gradient` called on them. Default
        is `"nn_params"` for each composant of the loss.
    initial_condition : tuple[float | Float[Array, " 1"]], default=None
        tuple of length 2 with initial condition $(t_0, u_0)$.
    obs_slice : EllipsisType | slice | None, default=None
        Slice object specifying the begininning/ending
        slice of u output(s) that is observed. This is useful for
        multidimensional PINN, with partially observed outputs.
        Default is None (whole output is observed).
    params : InitVar[Params[Array]], default=None
        The main Params object of the problem needed to instanciate the
        DerivativeKeysODE if the latter is not specified.
    u : eqx.Module
        the PINN
    dynamic_loss : ODE
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
    dynamic_loss: ODE | None

    vmap_in_axes: tuple[int] = eqx.field(init=False, static=True)

    def __post_init__(self, params: Params[Array] | None = None):
        super().__post_init__(
            params=params
        )  # because __init__ or __post_init__ of Base
        # class is not automatically called

        self.vmap_in_axes = (0,)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate_by_terms(
        self, params: Params[Array], batch: ODEBatch
    ) -> tuple[
        ODEComponents[Float[Array, " "] | None], ODEComponents[Float[Array, " "] | None]
    ]:
        """
        Evaluate the loss function at a batch of points for given parameters.

        We retrieve two PyTrees with loss values and gradients for each term

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
            dyn_loss_fun = lambda p: dynamic_loss_apply(
                self.dynamic_loss.evaluate,  # type: ignore
                self.u,
                temporal_batch,
                _set_derivatives(p, self.derivative_keys.dyn_loss),  # type: ignore
                self.vmap_in_axes + vmap_in_axes_params,
            )
        else:
            dyn_loss_fun = None

        # initial condition
        if self.initial_condition is not None:
            t0, u0 = self.initial_condition
            u0 = jnp.array(u0)

            # first construct the plain init loss no vmaping
            initial_condition_fun__ = lambda t, u, p: jnp.sum(
                (
                    self.u(
                        t,
                        _set_derivatives(
                            p,
                            self.derivative_keys.initial_condition,  # type: ignore
                        ),
                    )
                    - u
                )
                ** 2,
                axis=0,
            )
            # now vmap over the number of conditions (first dim of t0 and u0)
            # and take the mean
            initial_condition_fun_ = lambda p: jnp.mean(
                vmap(initial_condition_fun__, (0, 0, None))(t0, u0, p)
            )
            # now vmap over the the possible batch of parameters and take the
            # average. Note that we then finally have a cartesian product
            # between the batch of parameters (if any) and the number of
            # conditions (if any)
            if not jax.tree_util.tree_leaves(vmap_in_axes_params):
                # if there is no parameter batch to vmap over we cannot call
                # vmap because calling vmap must be done with at least one non
                # None in_axes or out_axes
                initial_condition_fun = initial_condition_fun_
            else:
                initial_condition_fun = lambda p: jnp.mean(
                    vmap(initial_condition_fun_, vmap_in_axes_params)(p)
                )
        else:
            initial_condition_fun = None

        if batch.obs_batch_dict is not None:
            # update params with the batches of observed params
            params_obs = _update_eq_params_dict(
                params, batch.obs_batch_dict["eq_params"]
            )

            # MSE loss wrt to an observed batch
            obs_loss_fun = lambda po: observations_loss_apply(
                self.u,
                batch.obs_batch_dict["pinn_in"],
                _set_derivatives(po, self.derivative_keys.observations),  # type: ignore
                self.vmap_in_axes + vmap_in_axes_params,
                batch.obs_batch_dict["val"],
                self.obs_slice,
            )
        else:
            params_obs = None
            obs_loss_fun = None

        # get the unweighted mses for each loss term as well as the gradients
        all_funs: ODEComponents[Callable[[Params[Array]], Array] | None] = (
            ODEComponents(dyn_loss_fun, initial_condition_fun, obs_loss_fun)
        )
        all_params: ODEComponents[Params[Array] | None] = ODEComponents(
            params, params, params_obs
        )
        mses_grads = jax.tree.map(
            lambda fun, params: self.get_gradients(fun, params),
            all_funs,
            all_params,
            is_leaf=lambda x: x is None,
        )

        mses = jax.tree.map(
            lambda leaf: leaf[0], mses_grads, is_leaf=lambda x: isinstance(x, tuple)
        )
        grads = jax.tree.map(
            lambda leaf: leaf[1], mses_grads, is_leaf=lambda x: isinstance(x, tuple)
        )

        return mses, grads

    def evaluate(
        self, params: Params[Array], batch: ODEBatch
    ) -> tuple[Float[Array, " "], ODEComponents[Float[Array, " "] | None]]:
        """
        Evaluate the loss function at a batch of points for given parameters.

        We retrieve the total value itself and a PyTree with loss values for each term

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
        loss_terms, _ = self.evaluate_by_terms(params, batch)

        loss_val = self.ponderate_and_sum_loss(loss_terms)

        return loss_val, loss_terms
