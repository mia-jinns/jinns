from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TypeAlias, TYPE_CHECKING, NewType, TypedDict, Callable
from jaxtyping import Float, Array

if TYPE_CHECKING:
    from jinns.data._Batchs import ODEBatch, PDEStatioBatch, PDENonStatioBatch
    from jinns.loss._LossODE import LossODE
    from jinns.loss._LossPDE import LossPDEStatio, LossPDENonStatio

    # Here we define types available for the whole package
    BoundaryConditionFun: TypeAlias = Callable[
        [Float[Array, "dim"] | Float[Array, "dim + 1"]], Float[Array, "dim_solution"]
    ]

    AnyBatch: TypeAlias = ODEBatch | PDENonStatioBatch | PDEStatioBatch
    AnyLoss: TypeAlias = LossODE | LossPDEStatio | LossPDENonStatio
