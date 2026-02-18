"""
Main module to implement a ODE loss in jinns
"""

from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from dataclasses import InitVar
from typing import TYPE_CHECKING, Callable, Any, ClassVar
import warnings
import jax
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
from jaxtyping import Float, Array
from jinns.loss._loss_utils import initial_condition_check, mean_sum_reduction
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
    params: InitVar[Params[Array] | None]
    reduction_functions: ClassVar[ODEComponents[Callable]] = eqx.field(
        static=True,
        default=ODEComponents(
            dyn_loss=mean_sum_reduction,
            initial_condition=mean_sum_reduction,
            observations=mean_sum_reduction,
        ),
    )

    def __init__(
        self,
        *,
        u: AbstractPINN,
        dynamic_loss: ODE | None,
        loss_weights: LossWeightsODE | None = None,
        derivative_keys: DerivativeKeysODE | None = None,
        initial_condition: InitialConditionUser | None = None,
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
            self.derivative_keys = DerivativeKeysODE(params=params)
        else:
            self.derivative_keys = derivative_keys
        self.u = u
        self.dynamic_loss = dynamic_loss

        # NOTE unclear why the AbstractVar of super need to be passed
        # explicitely
        super().__init__(
            dynamic_loss=self.dynamic_loss,
            loss_weights=self.loss_weights,
            derivative_keys=self.derivative_keys,
            u=self.u,
            **kwargs,
        )
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

    def evaluate_by_terms(
        self,
        batch: ODEBatch,
        params: Params[Array],
    ) -> ODEComponents[
        tuple[
            Callable | None,
            tuple[Array | None, ...] | Array | None,
            Params[Array] | None,
            tuple[tuple[int, ...]] | tuple[int, ...] | None,
        ]
    ]:
        """
        Evaluate the loss function at a batch object for given
        parameters (ie, the vmap is done on the returned functions of
        evaluate_by_terms)

        We retrieve two PyTrees with loss values and gradients for each term

        Parameters
        ---------
        params
            Parameters
        batch
            Composed of a batch of points in the
            domain, a batch of points in the domain
            border and an optional additional batch of parameters (eg. for
            metamodeling) and an optional additional batch of observed
            inputs/outputs/parameters
        """

        # dynamic part
        temporal_batch = batch.temporal_batch
        dyn_loss_fun = self._get_dyn_loss_fun()

        if self.initial_condition is not None:
            # initial condition
            t0, u0 = self.initial_condition

            # first construct the plain init loss no vmaping
            initial_condition_fun__: Callable[[Array, Array, Params[Array]], Array] = (
                lambda t, u, p: self.u(
                    t,
                    _set_derivatives(
                        p,
                        self.derivative_keys.initial_condition,
                    ),
                )
                - u
            )
            # now vmap over the number of conditions (first dim of t0 and u0)
            # and take the mean
            initial_condition_fun: Callable[[Params[Array]], Array] = (
                lambda p: jnp.mean(
                    vmap(initial_condition_fun__, (0, 0, None))(t0, u0, p),
                    axis=0,
                )
            )

            ## NOTE NOTE not clear if below can be totally suppressed
            ## NOTE that this is the vmap below and its reduction which
            # disappear since now we vmap over the possible batch of paramter
            # from outside this function
            # now vmap over the the possible batch of parameters and take the
            # average. Note that we then finally have a cartesian product
            # between the batch of parameters (if any) and the number of
            # conditions (if any)
            # if not jax.tree_util.tree_leaves(vmap_in_axes_params):
            #    # if there is no parameter batch to vmap over we cannot call
            #    # vmap because calling vmap must be done with at least one non
            #    # None in_axes or out_axes
            #    initial_condition_fun = initial_condition_fun_
            # else:
            #    initial_condition_fun: Callable[[Params[Array]], Array] | None = (
            #        lambda p: jnp.mean(
            #            vmap(initial_condition_fun_, vmap_in_axes_params)(p)
            #        )
            #    )
            # initial_condition_fun: Callable[[Params[Array]], Array] | None = (
            #    lambda p: jnp.mean(initial_condition_fun_(p)) # this get vmap
            #    # over p thanks to vmap_axes_params at outer level
            # )
        else:
            initial_condition_fun = None

        if batch.obs_batch_dict is not None:
            obs_batch, obs_params, obs_loss_fun = (
                self._get_obs_batch_params_and_loss_fun(params, batch.obs_batch_dict)
            )
        else:
            obs_params = None
            obs_loss_fun = None
            obs_batch = None, None

        all_funs_and_params: ODEComponents[
            tuple[
                Callable | None,
                tuple[Array | None, ...] | Array | None,
                Params[Array] | None,
                tuple[tuple[int, ...]] | tuple[int, ...] | None,
            ]
        ] = ODEComponents(
            dyn_loss=(dyn_loss_fun, temporal_batch, params, (0,)),
            initial_condition=(initial_condition_fun, None, params, None),
            observations=(
                obs_loss_fun,
                obs_batch,
                obs_params,
                ((0, 0),),  # nested tuple for generic parametrization of the
                # vmap when batch is also a tuple
            ),
        )
        return all_funs_and_params
