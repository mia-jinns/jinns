from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Self, Literal, Callable
from jaxtyping import Array, PyTree, Key
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jinns.loss._loss_weights import AbstractLossWeights
from jinns.parameters._params import Params
from jinns.loss._loss_weight_updates import soft_adapt, lr_annealing, ReLoBRaLo

if TYPE_CHECKING:
    from jinns.utils._types import AnyLossComponents, AnyBatch


class AbstractLoss(eqx.Module):
    """
    About the call:
    https://github.com/patrick-kidger/equinox/issues/1002 + https://docs.kidger.site/equinox/pattern/
    """

    loss_weights: AbstractLossWeights
    update_weight_method: Literal["soft_adapt", "lr_annealing", "ReLoBRaLo"] | None = (
        eqx.field(kw_only=True, default=None, static=True)
    )

    @abc.abstractmethod
    def __call__(self, *_, **__) -> Array:
        pass

    @abc.abstractmethod
    def evaluate_by_terms(
        self, params: Params[Array], batch: AnyBatch
    ) -> tuple[AnyLossComponents, AnyLossComponents]:
        pass

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

    def ponderate_and_sum_loss(self, terms):
        """
        Get total loss from individual loss terms and weights

        tree.leaves is needed to get rid of None from non used loss terms
        """
        weights = jax.tree.leaves(
            self.loss_weights,
            is_leaf=lambda x: eqx.is_inexact_array(x) and x is not None,
        )
        terms = jax.tree.leaves(
            terms, is_leaf=lambda x: eqx.is_inexact_array(x) and x is not None
        )
        if len(weights) == len(terms):
            return jnp.sum(jnp.array(weights) * jnp.array(terms))
        else:
            raise ValueError(
                "The numbers of declared loss weights and "
                "declared loss terms do not concord "
                f" got {len(weights)} and {len(terms)}"
            )

    def ponderate_and_sum_gradient(self, terms):
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
        key: Key,
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
