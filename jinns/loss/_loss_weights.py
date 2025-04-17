"""
Formalize the loss weights data structure
"""

from jaxtyping import Array, Float
import equinox as eqx


class LossWeightsODE(eqx.Module):

    dyn_loss: Array | Float | None = eqx.field(kw_only=True, default=1.0)
    initial_condition: Array | Float | None = eqx.field(kw_only=True, default=1.0)
    observations: Array | Float | None = eqx.field(kw_only=True, default=1.0)


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
