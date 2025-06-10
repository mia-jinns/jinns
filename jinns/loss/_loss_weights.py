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


class AbstractLossWeights(eqx.Module, Generic[T]):
    """
    An abstract class, currently only useful for type hints
    """


class LossWeightsODE(AbstractLossWeights[ODEComponents[Array | float | None]]):
    dyn_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=jnp.asarray
    )
    initial_condition: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=jnp.asarray
    )
    observations: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=jnp.asarray
    )


class LossWeightsPDEStatio(
    AbstractLossWeights[PDEStatioComponents[Array | float | None]]
):
    dyn_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=jnp.asarray
    )
    norm_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=jnp.asarray
    )
    boundary_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=jnp.asarray
    )
    observations: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=jnp.asarray
    )


class LossWeightsPDENonStatio(
    AbstractLossWeights[PDENonStatioComponents[Array | float | None]],
):
    dyn_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=jnp.asarray
    )
    norm_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=jnp.asarray
    )
    boundary_loss: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=jnp.asarray
    )
    observations: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=jnp.asarray
    )
    initial_condition: Array | float | None = eqx.field(
        kw_only=True, default=None, converter=jnp.asarray
    )
