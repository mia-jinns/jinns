"""
Formalize the loss weights data structure
"""

from jaxtyping import Array, Float
import jax.numpy as jnp
import equinox as eqx


class LossWeightsODE(eqx.Module):
    dyn_loss: Array | Float = eqx.field(
        kw_only=True, default=0.0, converter=jnp.asarray
    )
    initial_condition: Array | Float = eqx.field(
        kw_only=True, default=0.0, converter=jnp.asarray
    )
    observations: Array | Float = eqx.field(
        kw_only=True, default=0.0, converter=jnp.asarray
    )


class LossWeightsPDEStatio(eqx.Module):
    dyn_loss: Array | Float = eqx.field(
        kw_only=True, default=0.0, converter=jnp.asarray
    )
    norm_loss: Array | Float = eqx.field(
        kw_only=True, default=0.0, converter=jnp.asarray
    )
    boundary_loss: Array | Float = eqx.field(
        kw_only=True, default=0.0, converter=jnp.asarray
    )
    observations: Array | Float = eqx.field(
        kw_only=True, default=0.0, converter=jnp.asarray
    )


class LossWeightsPDENonStatio(eqx.Module):
    dyn_loss: Array | Float = eqx.field(
        kw_only=True, default=0.0, converter=jnp.asarray
    )
    norm_loss: Array | Float = eqx.field(
        kw_only=True, default=0.0, converter=jnp.asarray
    )
    boundary_loss: Array | Float = eqx.field(
        kw_only=True, default=0.0, converter=jnp.asarray
    )
    observations: Array | Float = eqx.field(
        kw_only=True, default=0.0, converter=jnp.asarray
    )
    initial_condition: Array | Float = eqx.field(
        kw_only=True, default=0.0, converter=jnp.asarray
    )
