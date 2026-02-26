from __future__ import annotations

import abc
from typing import (
    TYPE_CHECKING,
    Self,
    Literal,
    Callable,
    TypeVar,
    Generic,
    Any,
    cast,
    get_args,
)
from dataclasses import InitVar
from types import EllipsisType
import warnings
from jaxtyping import PRNGKeyArray, Array, PyTree, Float
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jinns.loss._loss_utils import dynamic_loss_apply, observations_loss_apply
from jinns.parameters._params import Params, _get_vmap_in_axes_params, update_eq_params
from jinns.loss._loss_weight_updates import (
    soft_adapt,
    lr_annealing,
    ReLoBRaLo,
    prior_loss,
)
from jinns.loss._DynamicLossAbstract import DynamicLoss
from jinns.utils._types import (
    AnyLossComponents,
    AnyBatch,
    AnyLossWeights,
    AnyDerivativeKeys,
)
from jinns.parameters._derivative_keys import _set_derivatives
from jinns.nn._pinn import PINN
from jinns.nn._spinn import SPINN
from jinns.nn._hyperpinn import HyperPINN

if TYPE_CHECKING:
    from jinns.loss._DynamicLossAbstract import DynamicLoss
    from jinns.nn._abstract_pinn import AbstractPINN

L = TypeVar(
    "L", bound=AnyLossWeights
)  # we want to be able to use one of the element of AnyLossWeights
# that is https://stackoverflow.com/a/79534258 via `bound`

B = TypeVar(
    "B", bound=AnyBatch
)  # The above comment also works with Unions (https://docs.python.org/3/library/typing.html#typing.TypeVar)
# We then do the same TypeVar to be able to use one of the element of AnyBatch
# in the evaluate_by_terms methods of child classes.
C = TypeVar(
    "C", bound=AnyLossComponents[Array | None]
)  # The above comment also works with Unions (https://docs.python.org/3/library/typing.html#typing.TypeVar)

DK = TypeVar("DK", bound=AnyDerivativeKeys)

# In the cases above, without the bound, we could not have covariance on
# the type because it would break LSP. Note that covariance on the return type
# is authorized in LSP hence we do not need the same TypeVar instruction for
# the return types of evaluate_by_terms for example!


AvailableUpdateWeightMethods = Literal[
    "softadapt", "soft_adapt", "prior_loss", "lr_annealing", "ReLoBRaLo"
]


class AbstractLoss(eqx.Module, Generic[L, B, C, DK]):
    """
    About the call:
    https://github.com/patrick-kidger/equinox/issues/1002 + https://docs.kidger.site/equinox/pattern/
    """

    dynamic_loss: eqx.AbstractVar[tuple[DynamicLoss, ...] | None]
    derivative_keys: eqx.AbstractVar[DK]
    loss_weights: eqx.AbstractVar[L]
    u: eqx.AbstractVar[AbstractPINN]
    reduction_functions: eqx.AbstractClassVar[C] = eqx.field(init=False)
    vmap_loss_fun: eqx.AbstractClassVar[C] = eqx.field(init=False)
    obs_slice: tuple[EllipsisType | slice, ...] = eqx.field(static=True, kw_only=True)
    loss_weight_scales: L = eqx.field(init=False)
    update_weight_method: AvailableUpdateWeightMethods | None = eqx.field(
        kw_only=True, default=None, static=True
    )
    keep_initial_loss_weight_scales: InitVar[bool] = eqx.field(
        default=True, kw_only=True
    )

    def __init__(
        self,
        *,
        dynamic_loss,
        u,
        loss_weights,
        derivative_keys,
        obs_slice=None,
        update_weight_method=None,
        keep_initial_loss_weight_scales=False,
    ):
        if self.update_weight_method is not None and jnp.any(
            jnp.array(jax.tree.leaves(self.loss_weights)) == 0
        ):
            warnings.warn(
                "self.update_weight_method is activated while some loss "
                "weights are zero. The update weight method will likely "
                "update the zero weight to some non-zero value. Check that "
                "this is the desired behaviour."
            )
        if update_weight_method is not None and update_weight_method not in get_args(
            AvailableUpdateWeightMethods
        ):
            raise ValueError(f"{update_weight_method=} is not a valid method")
        self.update_weight_method = update_weight_method
        self.loss_weights = loss_weights
        self.derivative_keys = derivative_keys
        self.dynamic_loss = dynamic_loss
        self.u = u
        if keep_initial_loss_weight_scales:
            self.loss_weight_scales = self.loss_weights
            if self.update_weight_method is not None:
                warnings.warn(
                    "Loss weights out from update_weight_method will still be"
                    " multiplied by the initial input loss_weights"
                )
        else:
            self.loss_weight_scales = optax.tree_utils.tree_ones_like(self.loss_weights)
            # self.loss_weight_scales will contain None where self.loss_weights
            # has None

        if obs_slice is None:
            self.obs_slice = (jnp.s_[...],)
        elif not isinstance(obs_slice, tuple):
            self.obs_slice = (obs_slice,)
        else:
            self.obs_slice = obs_slice

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.evaluate(*args, **kwargs)

    @abc.abstractmethod
    def evaluate_by_terms(
        self,
        batch: B,
        params: Params[Array],
    ) -> C:
        pass

    def _get_dyn_loss_fun(
        self,
    ) -> Callable[[Array, Params[Array]], tuple[Array, ...]] | None:
        """
        Common for all XDELoss
        """
        if self.dynamic_loss != (None,):
            # Note, for the record, multiple dynamic losses
            # have been introduced in MR 92
            # As opposed to obs_loss_fun, here the batch is the same for all
            # dyn loss
            dyn_loss_fun: Callable[[Array, Params[Array]], tuple[Array, ...]] | None = (
                lambda b, p: jax.tree.map(
                    lambda d: dynamic_loss_apply(
                        d,
                        self.u,
                        b,
                        _set_derivatives(p, self.derivative_keys.dyn_loss),
                    ),
                    self.dynamic_loss,
                    is_leaf=lambda x: isinstance(x, DynamicLoss),  # do not traverse
                    # further than first level
                )
            )
        else:
            dyn_loss_fun = None

        return dyn_loss_fun

    def _get_obs_loss_fun(
        self,
    ) -> Callable[[tuple[Array, Array], Params[Array], Array, EllipsisType], Array]:
        """
        Common for all XDELoss
        """
        # Note, for the record, multiple DGObs
        # (leading to batch.obs_batch_dict being tuple | None)
        # have been introduced in MR 92
        # As opposed to dyn_loss_fun, here the batch is different for each
        # obs_loss_fun
        # See more explanation in vmap_loss_fun_observations
        obs_loss_fun: Callable[
            [tuple[Array, Array], Params[Array], Array, EllipsisType], Array
        ] = lambda b, po, obs_eq_params, slice_: observations_loss_apply(
            self.u,
            b,
            _set_derivatives(
                update_eq_params(  # NOTE update_eq_params is here
                    po, obs_eq_params
                ),
                self.derivative_keys.observations,
            ),
            slice_,
        )

        return obs_loss_fun

    def evaluate(
        self,
        opt_params: Params[Array],
        batch: B,
        *,
        non_opt_params: Params[Array] | None = None,
        ret_std_grad_terms: bool = False,
        # ret_std_grad_terms
    ) -> tuple[Float[Array, " "], C]:
        """
        Evaluate the loss function at a batch of points for given parameters.

        We retrieve the total value itself and a PyTree with loss values for each term

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
            Parameters, which are non optimized, at which the loss is evaluated
        """
        if non_opt_params is not None:
            params = eqx.combine(opt_params, non_opt_params)
        else:
            params = opt_params

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            # update params with the batches of generated params
            params = update_eq_params(params, batch.param_batch_dict)

        if isinstance(self.u, (PINN, HyperPINN)):
            # NOTE each loss function is vmapped generically here
            # before reduction

            vmap_in_axes_params = _get_vmap_in_axes_params(
                cast(eqx.Module, batch.param_batch_dict), params
            )

            # create a PyTree of vmapped functions (loss terms)
            # we could technically vmap evaluate by terms with a PyTree in axes
            # of type XDEBatch but this would for identical batch length...

            # We vmap each function returned by evaluate by terms via the
            # following tree map. `vmap_loss_fun` is a function which does this
            # vmap, with some subtleties depending on the function term
            # NOTE: we keep it as a function of batch and params (b and p)
            # until then it to be able to call `jax.jacrev`
            evaluate_by_terms = lambda b, p: jax.tree.map(
                lambda vlf, args: vlf(*args, vmap_in_axes_params=vmap_in_axes_params),
                self.vmap_loss_fun,
                self.evaluate_by_terms(b, p),
                is_leaf=lambda x: (
                    isinstance(x, tuple) and (callable(x[0]) or x[0] is None)
                ),  # only traverse first layer
            )

            # next we reduce the output of each loss term function
            # NOTE: we keep it as a function of batch and params (b and p)
            # until then it to be able to call `jax.jacrev`
            evaluate_by_terms_reduced = lambda b, p: jax.tree.map(
                lambda red_fun, v_eval: red_fun(v_eval),
                self.reduction_functions,
                evaluate_by_terms(b, p),
            )

            # if we simply call the previous function we get the value of all
            # loss terms
            loss_terms = evaluate_by_terms_reduced(batch, params)
        elif isinstance(self.u, SPINN):
            # NOTE there is no vmap here on each loss functin
            # as the SPINN expects the full batch
            def loss_fun(fun, _b, _p, *args):
                """
                *args because for PINN and their vmap we needed to pass more
                arguments
                """
                if fun is None:
                    return None
                return fun(_b, _p)

            evaluate_by_terms_reduced = lambda b, p: jax.tree.map(
                lambda red_fun, f_b_p_in_axes: red_fun(loss_fun(*f_b_p_in_axes)),
                self.reduction_functions,
                self.evaluate_by_terms(b, p),
                is_leaf=lambda x: (
                    isinstance(x, tuple) and (callable(x[0]) or x[0] is None)
                ),  # only traverse first layer
            )
            loss_terms = evaluate_by_terms_reduced(batch, params)
        else:
            raise ValueError(
                f"Bad type for self.u. Got {type(self.u)}, expected PINN or SPINN"
            )

        if ret_std_grad_terms:
            # jacrev instead of grad to differentiate through the XDEComponents
            # Pytree
            grad_terms = jax.jacrev(evaluate_by_terms_reduced, argnums=1)(batch, params)
            return loss_terms, grad_terms

        loss_val = self.ponderate_and_sum_loss(loss_terms)
        # NOTE: @hgangloff: I don't understand why we return a 2-tuple in both
        # cases but with different arguments ?
        # I think it would be more readable to be consistant with a
        # return loss_val, loss_terms, grad_terms if ret_std_grad_terms
        # else: return loss_val, loss_terms, None
        return loss_val, loss_terms

    def evaluate_natural_gradient(
        self,
        opt_params: Params[Array],
        batch: B,
        *,
        non_opt_params: Params[Array] | None = None,
        ret_nat_grad_terms: bool = False,
        # ret_std_grad_terms
    ) -> tuple[Float[Array, " "], C]:
        """ """
        if non_opt_params is not None:
            params = eqx.combine(opt_params, non_opt_params)
        else:
            params = opt_params

        # Retrieve the optional eq_params_batch
        # and update eq_params with the latter
        # and update vmap_in_axes
        if batch.param_batch_dict is not None:
            # update params with the batches of generated params
            params = update_eq_params(params, batch.param_batch_dict)

        if not isinstance(self.u, (PINN, HyperPINN)):
            raise ValueError("Not implemented for SPINNs")

        vmap_in_axes_params = _get_vmap_in_axes_params(
            cast(eqx.Module, batch.param_batch_dict), params
        )

        # create a PyTree of vmapped functions (loss terms)
        # we could technically vmap evaluate by terms with a PyTree in axes
        # of type XDEBatch but this would for identical batch length...
        # We vmap each function returned by evaluate by terms via the
        # following tree map. `vmap_loss_fun` is a function which does this
        # vmap, with some subtleties depending on the function term
        # NOTE: we keep it as a function of batch and params (b and p)
        # until then it to be able to call `jax.jacrev`
        evaluate_by_terms = lambda b, p: jax.tree.map(
            lambda vlf, args: vlf(*args, vmap_in_axes_params=vmap_in_axes_params),
            self.vmap_loss_fun,
            self.evaluate_by_terms(b, p),
            is_leaf=lambda x: (
                isinstance(x, tuple) and (callable(x[0]) or x[0] is None)
            ),  # only traverse first layer
        )
        loss_terms = evaluate_by_terms(batch, params)

        if ret_nat_grad_terms:
            jacrev_evaluate_by_terms = lambda b, p: jax.tree.map(
                lambda vlf, args: vlf(
                    *args,
                    vmap_in_axes_params=vmap_in_axes_params,
                    jacrev=True,
                ),
                self.vmap_loss_fun,
                self.evaluate_by_terms(b, p),
                is_leaf=lambda x: (
                    isinstance(x, tuple) and (callable(x[0]) or x[0] is None)
                ),  # only traverse first layer
            )

            # Return the unreduced gradients and loss terms for each sample for
            # each loss
            grad_terms = jacrev_evaluate_by_terms(batch, params)
            return loss_terms, grad_terms

        # NOTE: (related to my comment at the end of `evaluate`) should we rather keep the same signature and return loss_terms, None ?
        return loss_terms

    def ponderate_and_sum_loss(self, terms: C) -> Array:
        """
        Get total loss from individual loss terms and weights

        tree.leaves is needed to get rid of None from non used loss terms
        """
        weights = jax.tree.leaves(
            self.loss_weights,
            is_leaf=lambda x: eqx.is_inexact_array(x) and x is not None,
        )
        terms_list = jax.tree.leaves(
            terms, is_leaf=lambda x: eqx.is_inexact_array(x) and x is not None
        )
        if len(weights) == len(terms_list):
            return jnp.sum(jnp.array(weights) * jnp.array(terms_list))
        raise ValueError(
            "The numbers of declared loss weights and "
            "declared loss terms do not concord "
            f" got {len(weights)} and {len(terms_list)}. "
            "If you passed tuple of dyn_loss, make sure to pass "
            "tuple of loss weights at LossWeights.dyn_loss."
            "If you passed tuple of obs datasets, make sure to pass "
            "tuple of loss weights at LossWeights.observations."
        )

    def ponderate_and_sum_gradient(self, terms: C) -> Params[Array | None]:
        """
        Get total gradients from individual loss gradients and weights
        for each parameter

        tree.leaves is needed to get rid of None from non used loss terms
        """
        weights = jax.tree.leaves(
            self.loss_weights,
            is_leaf=lambda x: eqx.is_inexact_array(x) and x is not None,
        )
        grads = jax.tree.leaves(terms, is_leaf=lambda x: isinstance(x, Params))
        # gradient terms for each individual loss for each parameter (several
        # Params structures)
        weights_pytree = jax.tree.map(
            lambda w: optax.tree_utils.tree_full_like(grads[0], w), weights
        )  # We need several Params structures full of the weight scalar
        weighted_grads = jax.tree.map(
            lambda w, p: w * p, weights_pytree, grads, is_leaf=eqx.is_inexact_array
        )  # Now we can multiply
        return jax.tree.map(
            lambda *grads: jnp.sum(jnp.array(grads), axis=0),
            *weighted_grads,
            is_leaf=eqx.is_inexact_array,
        )

    def update_weights(
        self: Self,
        iteration_nb: int,
        loss_terms: PyTree,
        stored_loss_terms: PyTree,
        grad_terms: PyTree,
        key: PRNGKeyArray,
    ) -> Self:
        """
        Update the loss weights according to a predefined scheme
        """
        if self.update_weight_method == "soft_adapt":
            new_weights = soft_adapt(
                self.loss_weights, iteration_nb, loss_terms, stored_loss_terms
            )
        elif self.update_weight_method == "prior_loss":
            new_weights = prior_loss(self.loss_weights, iteration_nb, stored_loss_terms)
        elif self.update_weight_method == "lr_annealing":
            new_weights = lr_annealing(self.loss_weights, grad_terms)
        elif self.update_weight_method == "ReLoBRaLo":
            new_weights = ReLoBRaLo(
                self.loss_weights, iteration_nb, loss_terms, stored_loss_terms, key
            )
        else:
            raise ValueError("update_weight_method for loss weights not implemented")

        # Below we update the non None entry in the PyTree self.loss_weights
        # we directly get the non None entries because None is not treated as a
        # leaf

        new_weights = jax.lax.cond(
            iteration_nb == 0,
            lambda nw: nw,
            lambda nw: jnp.array(jax.tree.leaves(self.loss_weight_scales)) * nw,
            new_weights,
        )
        return eqx.tree_at(
            lambda pt: jax.tree.leaves(pt.loss_weights), self, new_weights
        )
