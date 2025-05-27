"""
Formalize the loss weights data structure
"""

from __future__ import annotations
from typing import TypeVar, Generic, Self, Literal

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
    loss_weights: AbstractLossWeights[T],
    loss_terms: T,
    stored_loss_terms: T,
) -> Array:
    r"""
    Implement the simple strategy given in
    https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/theory/advanced_schemes.html#softadapt

    $$
    w_j(i)= \frac{\exp(\frac{L_j(i)}{L_j(i-1)+\epsilon}-\mu(i))}
    {\sum_{k=1}^{n_{loss}}\exp(\frac{L_k(i)}{L_k(i-1)+\epsilon}-\mu(i)}
    $$

    Note all the commented is_leaf arguments:
    (*) no need since None are not treated as leaves by default and
    the only other leaves are the is_inexact_array we want
    """
    # is_leaf test for None to avoid non used XDEComponents
    ratio_pytree = jax.tree.map(
        lambda lt, slt: lt / (slt[-1] + 1e-6) - jnp.max(lt / (slt[-1] + 1e-6)),
        loss_terms,
        stored_loss_terms,
        # is_leaf=lambda x: eqx.is_inexact_array(x), (*)
    )
    ratio_leaves = jax.tree.leaves(
        ratio_pytree  # , is_leaf=eqx.is_inexact_array and x is not None (*)
    )
    return jax.nn.softmax(jnp.array(ratio_leaves))


class AbstractLossWeights(eqx.Module, Generic[T]):
    """

    (*) need eqx.is_inexact_array to avoid the update_method
    """

    update_method: Literal["soft_adapt"] | None = eqx.field(
        static=True, kw_only=True, default=None
    )

    def update(self: Self, loss_terms: T, stored_loss_terms: T, grad_terms: T) -> Self:
        if self.update_method == "soft_adapt":
            new_weights = soft_adapt(self, loss_terms, stored_loss_terms)
        else:
            raise ValueError("Update method for loss weights not implemented")
        return eqx.tree_at(
            lambda pt: jax.tree.leaves(
                pt,
                is_leaf=lambda x: eqx.is_inexact_array(x),  # and x is not None (*)
            ),
            self,
            new_weights,
        )


class LossWeightsODE(AbstractLossWeights[ODEComponents[Array | float | None]]):
    dyn_loss: Array | float | None = eqx.field(kw_only=True, default=None)
    initial_condition: Array | float | None = eqx.field(kw_only=True, default=None)
    observations: Array | float | None = eqx.field(kw_only=True, default=None)


class LossWeightsPDEStatio(
    AbstractLossWeights[PDEStatioComponents[Array | float | None]]
):
    dyn_loss: Array | float | None = eqx.field(kw_only=True, default=None)
    norm_loss: Array | float | None = eqx.field(kw_only=True, default=None)
    boundary_loss: Array | float | None = eqx.field(kw_only=True, default=None)
    observations: Array | float | None = eqx.field(kw_only=True, default=None)


class LossWeightsPDENonStatio(
    AbstractLossWeights[PDENonStatioComponents[Array | float | None]],
):
    dyn_loss: Array | float | None = eqx.field(kw_only=True, default=None)
    norm_loss: Array | float | None = eqx.field(kw_only=True, default=None)
    boundary_loss: Array | float | None = eqx.field(kw_only=True, default=None)
    observations: Array | float | None = eqx.field(kw_only=True, default=None)
    initial_condition: Array | float | None = eqx.field(kw_only=True, default=None)
