"""
Formalize the loss weights data structure
"""

from typing import Dict
from jaxtyping import Array, Float
import equinox as eqx


class LossWeightsODE(eqx.Module):

    dyn_loss: Array | Float | None = eqx.field(kw_only=True, default=1.0)
    initial_condition: Array | Float | None = eqx.field(kw_only=True, default=1.0)
    observations: Array | Float | None = eqx.field(kw_only=True, default=1.0)


class LossWeightsODEDict(eqx.Module):

    dyn_loss: Dict[str, Array | Float | None] = eqx.field(kw_only=True, default=None)
    initial_condition: Dict[str, Array | Float | None] = eqx.field(
        kw_only=True, default=None
    )
    observations: Dict[str, Array | Float | None] = eqx.field(
        kw_only=True, default=None
    )


class LossWeightsPDEStatio(eqx.Module):

    dyn_loss: Array | Float | None = eqx.field(kw_only=True, default=1.0)
    norm_loss: Array | Float | None = eqx.field(kw_only=True, default=1.0)
    boundary_loss: Array | Float | None = eqx.field(kw_only=True, default=1.0)
    observations: Array | Float | None = eqx.field(kw_only=True, default=1.0)


class LossWeightsPDENonStatio(eqx.Module):

    dyn_loss: Array | Float | None = eqx.field(kw_only=True, default=1.0)
    norm_loss: Array | Float | None = eqx.field(kw_only=True, default=1.0)
    boundary_loss: Array | Float | None = eqx.field(kw_only=True, default=1.0)
    observations: Array | Float | None = eqx.field(kw_only=True, default=1.0)
    initial_condition: Array | Float | None = eqx.field(kw_only=True, default=1.0)


class LossWeightsPDEDict(eqx.Module):
    """
    Only one type of LossWeights data structure for the SystemLossPDE:
    Include the initial condition always for the code to be more generic
    """

    dyn_loss: Dict[str, Array | Float | None] = eqx.field(kw_only=True, default=1.0)
    norm_loss: Dict[str, Array | Float | None] = eqx.field(kw_only=True, default=1.0)
    boundary_loss: Dict[str, Array | Float | None] = eqx.field(
        kw_only=True, default=1.0
    )
    observations: Dict[str, Array | Float | None] = eqx.field(kw_only=True, default=1.0)
    initial_condition: Dict[str, Array | Float | None] = eqx.field(
        kw_only=True, default=1.0
    )
