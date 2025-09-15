from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TypeAlias, TYPE_CHECKING, Callable, TypeVar
from jaxtyping import Float, Array

from jinns.data._Batchs import ODEBatch, PDEStatioBatch, PDENonStatioBatch, ObsBatchDict
from jinns.loss._loss_weights import (
    LossWeightsODE,
    LossWeightsPDEStatio,
    LossWeightsPDENonStatio,
)
from jinns.loss._loss_components import (
    ODEComponents,
    PDEStatioComponents,
    PDENonStatioComponents,
)

AnyBatch: TypeAlias = ODEBatch | PDENonStatioBatch | PDEStatioBatch | ObsBatchDict

AnyLossWeights: TypeAlias = (
    LossWeightsODE | LossWeightsPDEStatio | LossWeightsPDENonStatio
)

# Note that syntax change starting from 3.12
_T = TypeVar("_T")
AnyLossComponents: TypeAlias = (
    ODEComponents[_T] | PDEStatioComponents[_T] | PDENonStatioComponents[_T]
)

if TYPE_CHECKING:
    from jinns.loss._LossODE import LossODE
    from jinns.loss._LossPDE import LossPDEStatio, LossPDENonStatio

    # Here we define types available for the whole package
    BoundaryConditionFun: TypeAlias = Callable[
        [Float[Array, " dim"] | Float[Array, " dim + 1"]], Float[Array, " dim_solution"]
    ]

    AnyLoss: TypeAlias = LossODE | LossPDEStatio | LossPDENonStatio
