from __future__ import annotations

import abc
from typing import Self, Literal, Callable, TypeVar, Generic, Any
from jaxtyping import PRNGKeyArray, Array, PyTree, Float
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jinns.parameters._params import Params
from jinns.loss._loss_weight_updates import soft_adapt, lr_annealing, ReLoBRaLo
from jinns.utils._types import (
    AnyLossComponents,
    AnyBatch,
    AnyLossWeights,
    AnyDerivativeKeys,
)

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


class AbstractLoss(eqx.Module, Generic[L, B, C, DK]):
    """
    About the call:
    https://github.com/patrick-kidger/equinox/issues/1002 + https://docs.kidger.site/equinox/pattern/
    """

    derivative_keys: eqx.AbstractVar[DK]
    loss_weights: eqx.AbstractVar[L]
    update_weight_method: Literal["soft_adapt", "lr_annealing", "ReLoBRaLo"] | None = (
        eqx.field(kw_only=True, default=None, static=True)
    )
    vmap_in_axes: tuple[int] = eqx.field(static=True)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.evaluate(*args, **kwargs)

    @abc.abstractmethod
    def evaluate_by_terms(
        self,
        opt_params: Params[Array],
        batch: B,
        *,
        non_opt_params: Params[Array] | None = None,
    ) -> tuple[C, C]:
        pass

    def evaluate(
        self,
        opt_params: Params[Array],
        batch: B,
        *,
        non_opt_params: Params[Array] | None = None,
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
        loss_terms, _ = self.evaluate_by_terms(
            opt_params, batch, non_opt_params=non_opt_params
        )

        loss_val = self.ponderate_and_sum_loss(loss_terms)

        return loss_val, loss_terms

    def get_gradients(
        self, fun: Callable[[Params[Array]], Array], params: Params[Array]
    ) -> tuple[Array, Array]:
        """
        params already filtered with derivative keys here
        """
        if fun is None:
            return None, None
        value_grad_loss = jax.value_and_grad(fun)
        loss_val, grads = value_grad_loss(params)
        return loss_val, grads

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
            f" got {len(weights)} and {len(terms_list)}"
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
        return eqx.tree_at(
            lambda pt: jax.tree.leaves(pt.loss_weights), self, new_weights
        )
