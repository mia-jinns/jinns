"""
Main module to implement a ODE loss in jinns
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from dataclasses import InitVar
from typing import TYPE_CHECKING, Callable, Any, cast
from types import EllipsisType
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
    update_eq_params,
)
from jinns.parameters._derivative_keys import _set_derivatives, DerivativeKeysODE
from jinns.loss._loss_weights import LossWeightsODE
from jinns.loss._abstract_loss import AbstractLoss
from jinns.loss._loss_components import ODEComponents
from jinns.parameters._params import Params
from jinns.data._Batchs import ODEBatch

if TYPE_CHECKING:
    # imports only used in type hints
    from jinns.nn._abstract_pinn import AbstractPINN
    from jinns.loss import ODE

    InitialConditionUser = (
        tuple[Float[Array, " n_cond "], Float[Array, " n_cond dim"]]
        | tuple[int | float | Float[Array, " "], int | float | Float[Array, " dim"]]
    )

    InitialCondition = (
        tuple[Float[Array, " n_cond "], Float[Array, " n_cond dim"]]
        | tuple[Float[Array, " "], Float[Array, " dim"]]
    )


class LossODE(
    AbstractLoss[
        LossWeightsODE, ODEBatch, ODEComponents[Array | None], DerivativeKeysODE
    ]
):
    r"""Loss object for an ordinary differential equation

    $$
    \mathcal{N}[u](t) = 0, \forall t \in I
    $$

    where $\mathcal{N}[\cdot]$ is a differential operator and the
    initial condition is $u(t_0)=u_0$.

    Parameters
    ----------
    u : eqx.Module
        the PINN
    dynamic_loss : ODE
        the ODE dynamic part of the loss, basically the differential
        operator $\mathcal{N}[u](t)$. Should implement a method
        `dynamic_loss.evaluate(t, u, params)`.
        Can be None in order to access only some part of the evaluate call.
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
        ], default=None
        Most of the time, a tuple of length 2 with initial condition $(t_0, u_0)$.
        From jinns v1.5.1 we accept tuples of jnp arrays with shape (n_cond, 1) for t0 and (n_cond, dim) for u0. This is useful to include observed conditions at different time points, such as *e.g* final conditions. It was designed to implement $\mathcal{L}^{aux}$ from _Systems biology informed deep learning for inferring parameters and hidden dynamics_, Alireza Yazdani et al., 2020
    obs_slice : EllipsisType | slice, default=None
        Slice object specifying the begininning/ending
        slice of u output(s) that is observed. This is useful for
        multidimensional PINN, with partially observed outputs.
        Default is None (whole output is observed).
    params : InitVar[Params[Array]], default=None
        The main Params object of the problem needed to instanciate the
        DerivativeKeysODE if the latter is not specified.
    Raises
    ------
    ValueError
        if initial condition is not a tuple.
    """

    # NOTE static=True only for leaf attributes that are not valid JAX types
    # (ie. jax.Array cannot be static) and that we do not expect to change
    u: AbstractPINN
    dynamic_loss: ODE | None
    derivative_keys: DerivativeKeysODE
    loss_weights: LossWeightsODE
    initial_condition: InitialCondition | None
    obs_slice: EllipsisType | slice = eqx.field(static=True)
    params: InitVar[Params[Array] | None]

    def __init__(
        self,
        *,
        u: AbstractPINN,
        dynamic_loss: ODE | None,
        loss_weights: LossWeightsODE | None = None,
        derivative_keys: DerivativeKeysODE | None = None,
        initial_condition: InitialConditionUser | None = None,
        obs_slice: EllipsisType | slice | None = None,
        params: Params[Array] | None = None,
        **kwargs: Any,  # this is for arguments for super()
    ):
        if loss_weights is None:
            self.loss_weights = LossWeightsODE()
        else:
            self.loss_weights = loss_weights

        if derivative_keys is None:
            # by default we only take gradient wrt nn_params
            if params is None:
                raise ValueError(
                    "Problem at derivative_keys initialization "
                    f"received {derivative_keys=} and {params=}"
                )
            derivative_keys = DerivativeKeysODE(params=params)
        else:
            derivative_keys = derivative_keys

        super().__init__(
            loss_weights=self.loss_weights,
            derivative_keys=derivative_keys,
            vmap_in_axes=(0,),
            **kwargs,
        )
        self.u = u
        self.dynamic_loss = dynamic_loss
        if self.update_weight_method is not None and jnp.any(
            jnp.array(jax.tree.leaves(self.loss_weights)) == 0
        ):
            warnings.warn(
                "self.update_weight_method is activated while some loss "
                "weights are zero. The update weight method will likely "
                "update the zero weight to some non-zero value. Check that "
                "this is the desired behaviour."
            )

        if initial_condition is None:
            warnings.warn(
                "Initial condition wasn't provided. Be sure to cover for that"
                "case (e.g by. hardcoding it into the PINN output)."
            )
            self.initial_condition = initial_condition
        else:
            if len(initial_condition) != 2:
                raise ValueError(
                    "Initial condition should be a tuple of len 2 with (t0, u0), "
                    f"{initial_condition} was passed."
                )
            # some checks/reshaping for t0 and u0
            t0, u0 = initial_condition
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

        if obs_slice is None:
            self.obs_slice = jnp.s_[...]
        else:
            self.obs_slice = obs_slice

    def evaluate_by_terms(
        self,
        opt_params: Params[Array],
        batch: ODEBatch,
        *,
        non_opt_params: Params[Array] | None = None,
    ) -> tuple[
        ODEComponents[Float[Array, " "] | None], ODEComponents[Float[Array, " "] | None]
    ]:
        """
        Evaluate the loss function at a batch of points for given parameters.

        We retrieve two PyTrees with loss values and gradients for each term

        Parameters
        ---------
        opt_params
            Parameters, which are optimized, at which the loss is evaluated
        batch
            Composed of a batch of points in the
            domain, a batch of points in the domain
            border and an optional additional batch of parameters (eg. for
            metamodeling) and an optional additional batch of observed
            inputs/outputs/parameters
        non_opt_params
            Parameters, which are not optimized, at which the loss is evaluated
        """
        if non_opt_params is not None:
            params = eqx.combine(opt_params, non_opt_params)
        else:
            params = opt_params

        temporal_batch = batch.temporal_batch

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            # update params with the batches of generated params
            params = update_eq_params(params, batch.param_batch_dict)

        vmap_in_axes_params = _get_vmap_in_axes_params(
            cast(eqx.Module, batch.param_batch_dict), params
        )

        ## dynamic part
        if self.dynamic_loss is not None:
            dyn_loss_eval = self.dynamic_loss.evaluate
            dyn_loss_fun: Callable[[Params[Array]], Array] | None = (
                lambda p: dynamic_loss_apply(
                    dyn_loss_eval,
                    self.u,
                    temporal_batch,
                    _set_derivatives(p, self.derivative_keys.dyn_loss),
                    self.vmap_in_axes + vmap_in_axes_params,
                )
            )
        else:
            dyn_loss_fun = None

        if self.initial_condition is not None:
            # initial condition
            t0, u0 = self.initial_condition

            # first construct the plain init loss no vmaping
            initial_condition_fun__: Callable[[Array, Array, Params[Array]], Array] = (
                lambda t, u, p: jnp.sum(
                    (
                        self.u(
                            t,
                            _set_derivatives(
                                p,
                                self.derivative_keys.initial_condition,
                            ),
                        )
                        - u
                    )
                    ** 2,
                    axis=0,
                )
            )
            # now vmap over the number of conditions (first dim of t0 and u0)
            # and take the mean
            initial_condition_fun_: Callable[[Params[Array]], Array] = (
                lambda p: jnp.mean(
                    vmap(initial_condition_fun__, (0, 0, None))(t0, u0, p)
                )
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
                initial_condition_fun: Callable[[Params[Array]], Array] | None = (
                    lambda p: jnp.mean(
                        vmap(initial_condition_fun_, vmap_in_axes_params)(p)
                    )
                )
        else:
            initial_condition_fun = None

        if batch.obs_batch_dict is not None:
            # update params with the batches of observed params
            params_obs = update_eq_params(params, batch.obs_batch_dict["eq_params"])

            pinn_in, val = (
                batch.obs_batch_dict["pinn_in"],
                batch.obs_batch_dict["val"],
            )  # the reason for this intruction is https://github.com/microsoft/pyright/discussions/8340

            # MSE loss wrt to an observed batch
            obs_loss_fun: Callable[[Params[Array]], Array] | None = (
                lambda po: observations_loss_apply(
                    self.u,
                    pinn_in,
                    _set_derivatives(po, self.derivative_keys.observations),
                    self.vmap_in_axes + vmap_in_axes_params,
                    val,
                    self.obs_slice,
                )
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

        # Note that the lambda functions below are with type: ignore just
        # because the lambda are not type annotated, but there is no proper way
        # to do this and we should assign the lambda to a type hinted variable
        # before hand: this is not practical, let us not get mad at this
        mses_grads = jax.tree.map(
            self.get_gradients,
            all_funs,
            all_params,
            is_leaf=lambda x: x is None,
        )

        mses = jax.tree.map(
            lambda leaf: leaf[0],  # type: ignore
            mses_grads,
            is_leaf=lambda x: isinstance(x, tuple),
        )
        grads = jax.tree.map(
            lambda leaf: leaf[1],  # type: ignore
            mses_grads,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        return mses, grads
