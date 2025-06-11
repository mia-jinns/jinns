"""
Formalize the loss weights data structure
"""

from __future__ import annotations
from typing import TypeVar, Generic

from jaxtyping import Array
import jax.numpy as jnp
import equinox as eqx

from jinns.loss._loss_components import (
    ODEComponents,
    PDEStatioComponents,
    PDENonStatioComponents,
)

T = TypeVar("T")


def lw_converter(x):
    if x is None:
        return x
    else:
        return jnp.asarray(x)


class AbstractLossWeights(eqx.Module, Generic[T]):
    """
    An abstract class, currently only useful for type hints
    """


class LossWeightsODE(AbstractLossWeights[ODEComponents[Array | float | None]]):
    dyn_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    initial_condition: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    observations: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )


class LossWeightsPDEStatio(
    AbstractLossWeights[PDEStatioComponents[Array | float | None]]
):
    dyn_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    norm_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    boundary_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    observations: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )


class LossWeightsPDENonStatio(
    AbstractLossWeights[PDENonStatioComponents[Array | float | None]],
):
    dyn_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    norm_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    boundary_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    observations: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
    initial_condition: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=lw_converter
    )
