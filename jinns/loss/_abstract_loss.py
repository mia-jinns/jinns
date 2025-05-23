from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from operator import add
from jaxtyping import Array
import equinox as eqx
import jax
import jax.numpy as jnp
from jinns.loss._loss_weights import AbstractLossWeights

if TYPE_CHECKING:
    from jinns.parameters._params import Params
    from jinns.utils._types import AnyLossComponents, AnyBatch


class AbstractLoss(eqx.Module):
    """
    About the call:
    https://github.com/patrick-kidger/equinox/issues/1002 + https://docs.kidger.site/equinox/pattern/
    """

    loss_weights = eqx.AbstractVar[AbstractLossWeights]

    @abc.abstractmethod
    def __call__(self, *_, **__) -> Array:
        pass

    @abc.abstractmethod
    def evaluate_by_terms(
        self, params: Params[Array], batch: AnyBatch
    ) -> tuple[AnyLossComponents, AnyLossComponents]:
        pass

    def get_gradients(self, fun, params):
        """
        params already filtered with derivative keys here
        """
        value_grad_loss = jax.value_and_grad(fun)
        loss_val, grads = value_grad_loss(params)
        return loss_val, grads

    def ponderate_and_sum(self, terms):
        """
        Get total loss from individual terms (loss or gradients for example) and weights
        """
        return jax.tree.reduce(
            add,
            jax.tree.map(lambda w, l: w * l, self.loss_weights, terms),
            initializer=jax.tree.map(jnp.zeros_like, terms),
        )
