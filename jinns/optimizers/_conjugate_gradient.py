"""Matrix-free conjugate gradient with preconditioner.
From Scimba source code : https://gitlab.com/scimba/scimba/-/blob/main/src/scimba_jax/utils/conjugate_gradient.py
"""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

NDARRAY_TYPE = jnp.ndarray


def _id(a: NDARRAY_TYPE) -> NDARRAY_TYPE:
    return a


# @jax.jit(static_argnames=["g", "pginv"]) # only valid in recent versions of jax
@partial(jax.jit, static_argnames=["g", "pginv"])
def conjugate_gradient(
    g: Callable,
    b: NDARRAY_TYPE,
    rtol: float = 1e-10,
    max_iter: int | None = None,
    pginv: Callable | None = None,
):
    """Matrix-free conjugate gradient with preconditioner for Gx=b.

    Implements algo 5.1 (page 10) in [Frangella and al. 2021].

    Args:
        g: The product matrix-vector function for G.
        b: The right member.
        rtol: the relative error to stop.
        max_iter: the max number of iterations.
        pginv: The product matrix-vector function for the preconditioner,
            which should approximate inverse of G.

    Returns:
        the approximate solution and the numer of iterations.

    [Frangella and al. 2021] RANDOMIZED NYSTROM PRECONDITIONING,
        Z. Frangella, J. A. Tropp, M. Udell
        https://arxiv.org/abs/2110.02820
    """
    n = b.shape[0]
    max_iter = n if max_iter is None else max_iter
    pginv = _id if pginv is None else pginv
    x = jnp.zeros_like(b)
    r = b - g(x)
    nbrtol = jnp.sqrt(jnp.dot(b, b)) * rtol
    z = pginv(r)
    p = z
    rz = jnp.dot(r, z)

    def cond_fn(state):
        _, r, _, _, _, i = state
        return (jnp.sqrt(jnp.dot(r, r)) > nbrtol) & (i <= max_iter)

    def body_fn(state):
        x, r, z, p, rz, i = state
        v = g(p)
        alpha = rz / jnp.dot(p, v)
        x = x + alpha * p
        r = r - alpha * v
        z = pginv(r)
        nrz = jnp.dot(r, z)
        beta = nrz / rz
        p = z + beta * p
        rz = nrz
        return (x, r, z, p, rz, i + 1)

    x, _, _, _, _, n_iter = jax.lax.while_loop(cond_fn, body_fn, (x, r, z, p, rz, 0))
    return x, n_iter
