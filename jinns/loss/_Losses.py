"""
Implement diverse loss functions
"""

import jax
import jax.numpy as jnp
from jax import vmap

from jinns.utils._pinn import PINN
from jinns.utils._spinn import SPINN


def dynamic_loss_apply(dyn_loss, u, batches, params, vmap_axes, u_type=None):
    if u_type == PINN or isinstance(u, PINN):
        v_dyn_loss = vmap(
            lambda *args: dyn_loss(
                *args[:-1], u, args[-1]  # we must place the params at the end
            ),
            vmap_axes,
            0,
        )
        residuals = v_dyn_loss(*batches, params)
        mse_dyn_loss = jnp.mean(jnp.sum(residuals**2, axis=-1))
    elif u_type == SPINN or isinstance(u, SPINN):
        residuals = dyn_loss(*batches, u, params)
        mse_dyn_loss = jnp.mean(jnp.sum(residuals**2, axis=-1))

    return mse_dyn_loss
