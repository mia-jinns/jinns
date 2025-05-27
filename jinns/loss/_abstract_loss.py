import abc
from typing import TypeVar, Generic, TYPE_CHECKING
from operator import add
from jaxtyping import Array
import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jinns.parameters._params import Params
    from jinns.loss._loss_weights import AbstractLossWeights
    from jinns.utils._types import AnyLossComponents, AnyBatch

T = TypeVar("T")


class ODEComponents(eqx.Module, Generic[T]):
    """
    Provides a template for ODE components with generic types.
    One can inherit to specialize and add methods and attributes
    We do not enforce keyword only to avoid being to verbose (this then can
    work like a tuple)
    """

    dyn_loss: T
    initial_condition: T
    observations: T


class PDEStatioComponents(eqx.Module, Generic[T]):
    dyn_loss: T
    norm_loss: T
    boundary_loss: T
    observations: T


class PDENonStatioComponents(PDEStatioComponents[T]):
    initial_condition: T


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
