from typing import NamedTuple
from jaxtyping import Array
import jax
import optax
import jax.numpy as jnp
import equinox as eqx


def proj_g(p_eq_params_g):
    """ """
    p_0 = jnp.maximum(0, p_eq_params_g)

    return p_0  # eqx.tree_at(lambda pt: (pt.g), p_eq_params, (p_0))


def proj_i(p_eq_params_i):
    """ """
    p_1 = jnp.fill_diagonal(
        p_eq_params_i, jnp.minimum(0, jnp.diag(p_eq_params_i)), inplace=False
    )

    return p_1  # eqx.tree_at(lambda pt: (pt.i,), p_eq_params, (p_1,))


class EmptyState(NamedTuple):
    pass


def update_and_project(param: str) -> optax._src.base.GradientTransformation:
    """
    An optax GradientTransformation which updates the parameters and project them
    to enforce some constraints on the parameters
    """

    def init_fn(params: optax._src.base.Params) -> EmptyState:
        return EmptyState()

    def update_fn(grads, state, params=None):
        grads_ = grads
        params_ = params

        grads = getattr(grads.eq_params, param)
        params = getattr(params.eq_params, param)

        # new_params = jax.tree.map(lambda g, p: p - learning_rate * g, grads, params)

        # 1) Update
        new_params = jax.tree.map(
            lambda u, v: v + u, params, grads
        )  # grads = previous updates that have not been applied
        # hence the previous lines are equivalent to an optax.apply_updates

        if param == "g":
            next_x = proj_g(new_params)
        elif param == "i":
            next_x = proj_i(new_params)
        else:
            raise ValueError

        # 2) Project
        # We return prox_update - xold to be compatible with
        # with optax.apply_update(xold, additive_update) which is additive
        additive_update_ = jax.tree.map(lambda u, v: v - u, params, next_x)

        additive_update = eqx.tree_at(
            lambda pt: (getattr(pt.eq_params, param),), grads_, (additive_update_,)
        )

        return additive_update, EmptyState()

    return optax._src.base.GradientTransformation(init_fn, update_fn)


class ProxGLVState(NamedTuple):
    t: Array
    y: Array


def proj_i_prox(p_eq_params_i, l1reg, learning_rate):
    """ """
    alpha = l1reg * learning_rate
    save_diag = jnp.diag(p_eq_params_i)
    p_1 = jnp.maximum(p_eq_params_i - alpha, 0) + jnp.minimum(p_eq_params_i + alpha, 0)
    p_1 = jnp.fill_diagonal(p_1, jnp.minimum(0, save_diag), inplace=False)

    return p_1


def soft_thresholding_additive_update(
    learning_rate: optax.ScalarOrSchedule, l1reg: float, param: str
) -> optax._src.base.GradientTransformation:
    """Soft thresholding operator, given input gradients `grads` return an
    update
    u <- - params + prox(params, grads, lr, l1reg)

    For prox_lasso, prox = max(0, params - lr * grads - lr * l1reg)

    Parameters
    ----------
    learning_rate : optax.GradientTransformation
        _description_
    l1reg : float, optional
        _description_, by default 1

    Returns
    -------
    base.GradientTransformation
    """

    def init_fn(params: optax._src.base.Params) -> ProxGLVState:
        return ProxGLVState(t=jnp.array(1.0), y=getattr(params.eq_params, param))

    def update_fn(grads, state, params=None):
        grads_ = grads
        params_ = params
        grads = getattr(grads.eq_params, param)
        params = getattr(params.eq_params, param)

        # new_params = jax.tree.map(lambda g, p: p - learning_rate * g, grads, state.y)
        new_params = jax.tree.map(
            lambda u, v: v + u, params, grads
        )  # grads = previous updates

        if param == "g":
            next_x = proj_g(new_params)
        elif param == "i":
            next_x = proj_i_prox(new_params, l1reg, learning_rate)
        else:
            raise ValueError
        next_t = 0.5 * (1 + jnp.sqrt(1 + 4 * state.t**2))
        diff_x = jax.tree.map(lambda u, v: u - v, next_x, params)
        next_y = jax.tree.map(
            lambda u, v: u + (state.t - 1) / next_t * v, next_x, diff_x
        )

        # (following jaxopt Proximal code we could also add a err parameter
        # to stop updates below a certain tolerance)

        additive_update_ = jax.tree.map(lambda u, v: v - u, params, next_x)
        additive_update = eqx.tree_at(
            lambda pt: (getattr(pt.eq_params, param),), grads_, (additive_update_,)
        )

        state = ProxGLVState(t=next_t, y=next_y)

        return additive_update, state

    return optax._src.base.GradientTransformation(init_fn, update_fn)
