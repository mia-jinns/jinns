"""
Formalize the loss weights data structure
"""

from __future__ import annotations

from jaxtyping import Array
import jax.numpy as jnp
import equinox as eqx

from jinns.loss._loss_components import (
    ODEComponents,
    PDEStatioComponents,
    PDENonStatioComponents,
)


def lw_converter(x: Array | None) -> Array | None:
    if x is None:
        return x
    else:
        return jnp.asarray(x)


class LossWeightsODE(ODEComponents[Array | None]):
    """
    Value given at initialization is converted to a jnp.array orunmodified if None.
    This means that at initialization, the user can pass a float or int
    """

    dyn_loss: Array | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    initial_condition: Array | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    observations: Array | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )


class LossWeightsPDEStatio(PDEStatioComponents[Array | None]):
    """
    Value given at initialization is converted to a jnp.array orunmodified if None.
    This means that at initialization, the user can pass a float or int
    """

    dyn_loss: Array | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    norm_loss: Array | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    boundary_loss: Array | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    observations: Array | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )


class LossWeightsPDENonStatio(PDENonStatioComponents[Array | None]):
    """
    Value given at initialization is converted to a jnp.array orunmodified if None.
    This means that at initialization, the user can pass a float or int
    """

    dyn_loss: Array | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    norm_loss: Array | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    boundary_loss: Array | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    observations: Array | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    initial_condition: Array | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
