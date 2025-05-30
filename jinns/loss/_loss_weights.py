"""
Formalize the loss weights data structure
"""

from __future__ import annotations
from typing import TypeVar, Generic

from jaxtyping import Array
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
    dyn_loss: Array | float | None = eqx.field(kw_only=True, default=None)
    initial_condition: Array | float | None = eqx.field(kw_only=True, default=None)
    observations: Array | float | None = eqx.field(kw_only=True, default=None)


class LossWeightsPDEStatio(
    AbstractLossWeights[PDEStatioComponents[Array | float | None]]
):
    dyn_loss: Array | float | None = eqx.field(kw_only=True, default=None)
    norm_loss: Array | float | None = eqx.field(kw_only=True, default=None)
    boundary_loss: Array | float | None = eqx.field(kw_only=True, default=None)
    observations: Array | float | None = eqx.field(kw_only=True, default=None)


class LossWeightsPDENonStatio(
    AbstractLossWeights[PDENonStatioComponents[Array | float | None]],
):
    dyn_loss: Array | float | None = eqx.field(kw_only=True, default=None)
    norm_loss: Array | float | None = eqx.field(kw_only=True, default=None)
    boundary_loss: Array | float | None = eqx.field(kw_only=True, default=None)
    observations: Array | float | None = eqx.field(kw_only=True, default=None)
    initial_condition: Array | float | None = eqx.field(kw_only=True, default=None)
