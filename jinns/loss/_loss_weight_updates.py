"""
A collection of specific weight update schemes in jinns
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from jaxtyping import Array, Key
import jax.numpy as jnp
import jax
import equinox as eqx

if TYPE_CHECKING:
    from jinns.loss._loss_weights import AbstractLossWeights
    from jinns.utils._types import AnyLossComponents


def soft_adapt(
    loss_weights: AbstractLossWeights,
    iteration_nb: int,
    loss_terms: AnyLossComponents,
    stored_loss_terms: AnyLossComponents,
) -> Array:
    r"""
    Implement the simple strategy given in
    https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/theory/advanced_schemes.html#softadapt

    $$
    w_j(i)= \frac{\exp(\frac{L_j(i)}{L_j(i-1)+\epsilon}-\mu(i))}
    {\sum_{k=1}^{n_{loss}}\exp(\frac{L_k(i)}{L_k(i-1)+\epsilon}-\mu(i)}
    $$

    Note that since None is not treated as a leaf by jax tree.util functions,
    we naturally avoid None components from loss_terms, stored_loss_terms etc.!
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
        )
        mu = jax.tree.reduce(jnp.maximum, ratio_pytree, initializer=jnp.array(-jnp.inf))
        ratio_pytree = jax.tree.map(lambda r: r - mu, ratio_pytree)
        ratio_leaves = jax.tree.leaves(ratio_pytree)
        return jax.nn.softmax(jnp.array(ratio_leaves))

    return jax.lax.cond(
        iteration_nb == 0,
        lambda op: do_nothing(*op),
        lambda op: soft_adapt_(*op),
        (loss_weights, loss_terms, stored_loss_terms),
    )


def ReLoBRaLo(
    loss_weights: AbstractLossWeights,
    iteration_nb: int,
    loss_terms: AnyLossComponents,
    stored_loss_terms: AnyLossComponents,
    key: Key,
    decay_factor: float = 0.9,
    tau: float = 1,  ## referred to as temperature in the article
    p: float = 0.9,
):
    r"""
    Implementing the extension of softadapt: Relative Loss Balancing with random LookBack
    """
    n_loss = len(jax.tree.leaves(loss_terms))  # number of loss terms
    epsilon = 1e-6

    def do_nothing(loss_weights, _):
        return jnp.array(
            jax.tree.leaves(loss_weights, is_leaf=eqx.is_inexact_array), dtype=float
        )

    def compute_softmax_weights(current, reference):
        ratio_pytree = jax.tree.map(
            lambda lt, ref: lt / (ref + epsilon),
            current,
            reference,
        )
        mu = jax.tree.reduce(jnp.maximum, ratio_pytree, initializer=-jnp.inf)
        ratio_pytree = jax.tree.map(lambda r: r - mu, ratio_pytree)
        ratio_leaves = jax.tree.leaves(ratio_pytree)
        return jax.nn.softmax(jnp.array(ratio_leaves))

    def soft_adapt_prev(stored_loss_terms):
        # ω_j(i-1)
        prev_terms = jax.tree.map(lambda slt: slt[iteration_nb - 1], stored_loss_terms)
        prev_prev_terms = jax.tree.map(
            lambda slt: slt[iteration_nb - 2], stored_loss_terms
        )
        return compute_softmax_weights(prev_terms, prev_prev_terms)

    def look_back(loss_terms, stored_loss_terms):
        # ω̂_j^(i,0)
        initial_terms = jax.tree.map(lambda slt: tau * slt[0], stored_loss_terms)
        weights = compute_softmax_weights(loss_terms, initial_terms)
        return n_loss * weights

    def soft_adapt_current(loss_terms, stored_loss_terms):
        # ω_j(i)
        prev_terms = jax.tree.map(lambda slt: slt[iteration_nb - 1], stored_loss_terms)
        return compute_softmax_weights(loss_terms, prev_terms)

    # Bernoulli variable for random lookback
    rho = jax.random.bernoulli(key, p).astype(float)

    # Base case for first iteration
    def first_iter_case(_):
        return do_nothing(loss_weights, None)

    # Case for iteration >= 1
    def subsequent_iter_case(_):
        # Compute historical weights
        def hist_weights_case1(_):
            return soft_adapt_current(loss_terms, stored_loss_terms)

        def hist_weights_case2(_):
            return rho * soft_adapt_prev(stored_loss_terms) + (1 - rho) * look_back(
                loss_terms, stored_loss_terms
            )

        loss_weights_hist = jax.lax.cond(
            iteration_nb < 2,
            hist_weights_case1,
            hist_weights_case2,
            None,
        )

        # Compute and return final weights
        adaptive_weights = soft_adapt_current(loss_terms, stored_loss_terms)
        return decay_factor * loss_weights_hist + (1 - decay_factor) * adaptive_weights

    return jax.lax.cond(
        iteration_nb == 0,
        first_iter_case,
        subsequent_iter_case,
        None,
    )


def lr_annealing(
    loss_weights: AbstractLossWeights,
    grad_terms: AnyLossComponents,
    decay_factor: float = 0.9,  # 0.9 is the recommended value from the article
    eps: float = 1e-6,
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

    Note that since None is not treated as a leaf by jax tree.util functions,
    we naturally avoid None components from loss_terms, stored_loss_terms etc.!

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

    max_dyn_loss_grads = jnp.max(
        jnp.stack([jnp.max(jnp.abs(g)) for g in dyn_loss_grads_leaves])
    )

    mean_gradients = [
        jnp.mean(jnp.stack([jnp.abs(jnp.mean(g)) for g in jax.tree.leaves(t)]))
        for t in data_fit_grads
        if t is not None and jax.tree.leaves(t)
    ]

    lambda_hat = max_dyn_loss_grads / (jnp.array(mean_gradients) + eps)
    old_weights = jnp.array(
        jax.tree.leaves(
            loss_weights,
        )
    )

    new_weights = (1 - decay_factor) * old_weights[1:] + decay_factor * lambda_hat
    return jnp.hstack([old_weights[0], new_weights])
