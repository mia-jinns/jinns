"""
Formalize the loss weights data structure
"""

from __future__ import annotations
from typing import Callable, TypeVar, Generic

from jaxtyping import Array
import equinox as eqx
import jax
import jax.numpy as jnp

from jinns.loss._loss_components import (
    ODEComponents,
    PDEStatioComponents,
    PDENonStatioComponents,
)

T = TypeVar("T")


def soft_adapt(
    loss_weights: AbstractLossWeights[T], loss_terms: T, stored_loss_terms: T, _: T
) -> AbstractLossWeights[T]:
    r"""
    Implement the simple strategy given in
    https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/theory/advanced_schemes.html#softadapt

    $$
    w_j(i)= \frac{\exp(\frac{L_j(i)}{L_j(i-1)+\epsilon}-\mu(i))}
    {\sum_{k=1}^{n_{loss}}\exp(\frac{L_k(i)}{L_k(i-1)+\epsilon}-\mu(i)}
    $$
    """
    # is_leaf test for None to avoid non used XDEComponents
    ratio_pytree = jax.tree.map(
        lambda lt, slt: lt / (slt[-1] + 1e-6) - jnp.max(lt / (slt[-1] + 1e-6)),
        loss_terms,
        stored_loss_terms,
        is_leaf=lambda x: eqx.is_inexact_array(x),
    )
    ratio_leaves = jax.tree.leaves(
        ratio_pytree, is_leaf=lambda x: eqx.is_inexact_array(x) and x is not None
    )
    jax.debug.print("{x}", x=jax.nn.softmax(jnp.array(ratio_leaves)))
    return eqx.tree_at(
        lambda pt: jax.tree.leaves(
            pt, is_leaf=lambda x: eqx.is_inexact_array(x) and x is not None
        ),
        loss_weights,
        jax.nn.softmax(jnp.array(ratio_leaves)),
    )


class AbstractLossWeights(eqx.Module, Generic[T]):
    """
    A Protocol, mainly so that we are able to give a type to
    the abstract loss_weights attribute in AbstractLoss
    """

    update_fun: Callable[[AbstractLossWeights, T, T, T], AbstractLossWeights] | None = (
        eqx.field(kw_only=True, default=None, static=True)
    )


class LossWeightsODE(AbstractLossWeights[ODEComponents[Array | None]]):
    dyn_loss: Array = eqx.field(kw_only=True, default=None)
    initial_condition: Array = eqx.field(kw_only=True, default=None)
    observations: Array = eqx.field(kw_only=True, default=None)


class LossWeightsPDEStatio(AbstractLossWeights[PDEStatioComponents[Array | None]]):
    dyn_loss: Array = eqx.field(kw_only=True, default=None)
    norm_loss: Array = eqx.field(kw_only=True, default=None)
    boundary_loss: Array = eqx.field(kw_only=True, default=None)
    observations: Array = eqx.field(kw_only=True, default=None)


class LossWeightsPDENonStatio(
    AbstractLossWeights[PDENonStatioComponents[Array | None]],
):
    dyn_loss: Array = eqx.field(kw_only=True, default=None)
    norm_loss: Array = eqx.field(kw_only=True, default=None)
    boundary_loss: Array = eqx.field(kw_only=True, default=None)
    observations: Array = eqx.field(kw_only=True, default=None)
    initial_condition: Array = eqx.field(kw_only=True, default=None)
