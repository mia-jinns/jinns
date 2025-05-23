"""
Formalize the loss weights data structure
"""

from __future__ import annotations
from typing import Callable, Self, TypeVar, Generic

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
    This abstraction so that we factorize the tedious update_fun definition
    (with a generic type also), and also so that we are able to give a type to
    the abstract loss_weights attribute in AbstractLoss
    """

    update_fun: Callable[[Self, T, T, T], Self] | None = eqx.field(
        kw_only=True, default=None
    )


class LossWeightsODE(
    ODEComponents[Array | float], AbstractLossWeights[ODEComponents[Array]]
):
    dyn_loss: Array | float = eqx.field(kw_only=True, default=0.0)
    initial_condition: Array | float = eqx.field(kw_only=True, default=0.0)
    observations: Array | float = eqx.field(kw_only=True, default=0.0)


class LossWeightsPDEStatio(
    PDEStatioComponents[Array | float], AbstractLossWeights[PDEStatioComponents[Array]]
):
    dyn_loss: Array | float = eqx.field(kw_only=True, default=0.0)
    norm_loss: Array | float = eqx.field(kw_only=True, default=0.0)
    boundary_loss: Array | float = eqx.field(kw_only=True, default=0.0)
    observations: Array | float = eqx.field(kw_only=True, default=0.0)


class LossWeightsPDENonStatio(
    PDENonStatioComponents[Array | float],
    AbstractLossWeights[PDENonStatioComponents[Array]],
):
    dyn_loss: Array | float = eqx.field(kw_only=True, default=0.0)
    norm_loss: Array | float = eqx.field(kw_only=True, default=0.0)
    boundary_loss: Array | float = eqx.field(kw_only=True, default=0.0)
    observations: Array | float = eqx.field(kw_only=True, default=0.0)
    initial_condition: Array | float = eqx.field(kw_only=True, default=0.0)
