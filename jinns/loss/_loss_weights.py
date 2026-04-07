"""
Formalize the loss weights data structure
"""

from __future__ import annotations
from typing import overload

from jaxtyping import Array
import jax.numpy as jnp
import equinox as eqx

from jinns.loss._loss_components import (
    ODEComponents,
    PDEStatioComponents,
    PDENonStatioComponents,
)


# NOTE that overload is the correct way to annotate function following
# `lw_converter` design
# https://stackoverflow.com/a/52449229
# https://typing.python.org/en/latest/spec/overload.html#overload-definitions
@overload
def lw_converter(x: None) -> None: ...
@overload
def lw_converter(x: Array) -> Array: ...


def lw_converter(
    x: tuple[float | int, ...] | Array | None,
) -> tuple[Array, ...] | Array | None:
    if x is None:
        return x
    elif isinstance(x, tuple):
        # user might input tuple of scalar loss weights to account for cases
        # when dyn loss is also a tuple of (possibly 1D) dyn_loss
        return tuple(jnp.asarray(x_) for x_ in x)
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
