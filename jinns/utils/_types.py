from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TypeAlias, TYPE_CHECKING, Callable
from jaxtyping import Float, Array

from jinns.data._Batchs import ODEBatch, PDEStatioBatch, PDENonStatioBatch
from jinns.loss._loss_components import (
    ODEComponents,
    PDEStatioComponents,
    PDENonStatioComponents,
)

AnyBatch: TypeAlias = ODEBatch | PDENonStatioBatch | PDEStatioBatch

# here we would like a type from 3.12
# (https://typing.python.org/en/latest/spec/aliases.html#type-statement) so
# that we could have a generic AnyLossComponents
AnyLossComponents: TypeAlias = (
    ODEComponents | PDEStatioComponents | PDENonStatioComponents
)

if TYPE_CHECKING:
    from jinns.loss._LossODE import LossODE
    from jinns.loss._LossPDE import LossPDEStatio, LossPDENonStatio

    # Here we define types available for the whole package
    BoundaryConditionFun: TypeAlias = Callable[
        [Float[Array, " dim"] | Float[Array, " dim + 1"]], Float[Array, " dim_solution"]
    ]

    AnyLoss: TypeAlias = LossODE | LossPDEStatio | LossPDENonStatio
