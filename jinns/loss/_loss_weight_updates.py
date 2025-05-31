"""
Formalize the loss weights data structure
"""

from __future__ import annotations
from typing import TypeVar

from jaxtyping import Array
import equinox as eqx
import jax
import jax.numpy as jnp

from jinns.loss._loss_weights import AbstractLossWeights

T = TypeVar("T")


def soft_adapt(
    loss_weights: AbstractLossWeights[T],
    iteration_nb: int,
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
    the only other leaves are the is_inexact_array we want. Otherwise
    is_leaf test for None to avoid non used XDEComponents
    """

    def do_nothing(loss_weights, _, __):
        return jnp.array(
            jax.tree.leaves(loss_weights, is_leaf=eqx.is_inexact_array), dtype=float
        )

    def soft_adapt_(_, loss_terms, stored_loss_terms):
        ratio_pytree = jax.tree.map(
            lambda lt, slt: lt / (slt[iteration_nb - 1] + 1e-6),
            loss_terms,
            stored_loss_terms,
            # is_leaf=lambda x: eqx.is_inexact_array(x), (*)
        )
        mu = jax.tree.reduce(jnp.maximum, ratio_pytree, initializer=jnp.array(-jnp.inf))
        ratio_pytree = jax.tree.map(lambda r: r - mu, ratio_pytree)
        ratio_leaves = jax.tree.leaves(
            ratio_pytree  # , is_leaf=eqx.is_inexact_array and x is not None (*)
        )
        return jax.nn.softmax(jnp.array(ratio_leaves))

    return jax.lax.cond(
        iteration_nb == 0,
        lambda op: do_nothing(*op),
        lambda op: soft_adapt_(*op),
        (loss_weights, loss_terms, stored_loss_terms),
    )


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
