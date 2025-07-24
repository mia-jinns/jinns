from __future__ import annotations

import abc
from dataclasses import InitVar
from typing import TYPE_CHECKING, Self, Literal, Callable, get_args
from jaxtyping import Array, PyTree, Key, Float
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jinns.loss._loss_weights import AbstractLossWeights
from jinns.parameters._params import Params
from jinns.loss._loss_weight_updates import (
    soft_adapt,
    prior_loss,
    lr_annealing,
    ReLoBRaLo,
)

if TYPE_CHECKING:
    from jinns.utils._types import AnyLossComponents, AnyBatch


AvailableUpdateWeightMethods = Literal[
    "soft_adapt", "prior_loss", "lr_annealing", "ReLoBRaLo"
]


class AbstractLoss(eqx.Module):
    """
    About the call:
    https://github.com/patrick-kidger/equinox/issues/1002 + https://docs.kidger.site/equinox/pattern/
    """

    loss_weights: AbstractLossWeights
    loss_weight_scales: AbstractLossWeights = eqx.field(init=False)
    update_weight_method: AvailableUpdateWeightMethods | None = eqx.field(
        kw_only=True, default=None, static=True
    )
    keep_initial_loss_weight_scales: InitVar[bool] = eqx.field(
        default=True, kw_only=True
    )

    def __post_init__(self, keep_initial_loss_weight_scales: bool = True):
        if (
            self.update_weight_method is not None
            and self.update_weight_method not in get_args(AvailableUpdateWeightMethods)
        ):
            raise ValueError("update_weight_method is not a valid method")
        if keep_initial_loss_weight_scales:
            self.loss_weight_scales = self.loss_weights
        else:
            self.loss_weight_scales = optax.tree_utils.tree_ones_like(self.loss_weights)
            # self.loss_weight_scales will contain None where self.loss_weights
            # has None

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
                f" got {len(weights)} and {len(terms)}. "
                "If you passed tuple of dyn_loss, make sure to pass "
                "tuple of loss weights at LossWeights.dyn_loss."
                "If you passed tuple of obs datasets, make sure to pass "
                "tuple of loss weights at LossWeights.observations."
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

    def evaluate(
        self, params: Params[Array], batch: AnyBatch
    ) -> tuple[Float[Array, " "], AnyLossComponents]:
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
