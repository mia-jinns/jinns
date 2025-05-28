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


def lr_annealing(
    loss_weights: AbstractLossWeights[T],
    grad_terms: T,
    decay_factor: float = 0.9,  # 0.9 is the recommended value from the article
) -> Array:
    r"""
    Implementation of the Learning rate annealing
    Algorithm 1 in the paper UNDERSTANDING AND MITIGATING GRADIENT PATHOLOGIES IN PHYSICS-INFORMED NEURAL NETWORKS

    (a) Compute $\hat{\lambda}_i$ by
    $$
        \hat{\lambda}_i = \frac{\max_{\theta}\{|\nabla_\theta \mathcal{L}_r (\theta_n)|\}}{mean(|\nabla_\theta \mathcal{L}_i (\theta_n)|)}, \quad i=1,\dots, M,
    $$

    (b) Update the weights $\lambda_i$ using a moving average of the form
    $$
        \lambda_i = (1-\alpha) \lambda_{i-1} + \alpha \hat{\lambda}_i, \quad i=1, \dots, M.
    $$

    """
    assert hasattr(grad_terms, "dyn_loss")
    dyn_loss_grads = getattr(grad_terms, "dyn_loss")
    data_fit_grads = [
        getattr(grad_terms, att) if hasattr(grad_terms, att) else None
        for att in ["norm_loss", "boundary_loss", "observations", "initial_condition"]
    ]

    dyn_loss_grads_leaves = jax.tree.leaves(
        dyn_loss_grads,
        is_leaf=eqx.is_inexact_array,
    )
    max_dyn_loss_grads = jnp.max(jnp.absolute(jnp.array(dyn_loss_grads_leaves)))

    mean_gradients = jax.tree.map(
        lambda t: jnp.mean(jnp.absolute(jnp.array(jax.tree.leaves(t)))),
        data_fit_grads,
        is_leaf=eqx.is_inexact_array,
    )

    lambda_hat = max_dyn_loss_grads / jnp.array(jax.tree.leaves(mean_gradients))
    old_weights = jnp.array(
        jax.tree.leaves(
            loss_weights,
            is_leaf=eqx.is_inexact_array,
        )
    )

    new_weigths = (1 - decay_factor) * old_weights[1:] + decay_factor * lambda_hat
    return jnp.hstack([old_weights[0], new_weigths])


class AbstractLossWeights(eqx.Module, Generic[T]):
    """

    (*) need eqx.is_inexact_array to avoid the update_method
    """

    update_method: Literal["soft_adapt"] | Literal["lr_annealing"] | None = eqx.field(
        static=True, kw_only=True, default=None
    )

    def update(self: Self, loss_terms: T, stored_loss_terms: T, grad_terms: T) -> Self:
        if self.update_method == "soft_adapt":
            new_weights = soft_adapt(self, loss_terms, stored_loss_terms)
        elif self.update_method == "lr_annealing":
            new_weights = lr_annealing(self, grad_terms)
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
