# pragma: exclude file
from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TypeAlias, TYPE_CHECKING, NewType, TypedDict, Callable
from jaxtyping import Float, Array

if TYPE_CHECKING:
    from jinns.data._DataGenerators import (
        DataGeneratorODE,
        CubicMeshPDEStatio,
        CubicMeshPDENonStatio,
    )
    from jinns.data._Batchs import ODEBatch, PDEStatioBatch, PDENonStatioBatch
    from jinns.loss._LossODE import LossODE
    from jinns.loss._LossPDE import LossPDEStatio, LossPDENonStatio

    # Here we define types available for the whole package
    BoundaryConditionFun: TypeAlias = Callable[
        [Float[Array, "dim"] | Float[Array, "dim + 1"]], Float[Array, "dim_solution"]
    ]

    AnyDataGenerator = DataGeneratorODE | CubicMeshPDEStatio | CubicMeshPDENonStatio

    AnyBatch = ODEBatch | PDENonStatioBatch | PDEStatioBatch

    # rar_operands = NewType("rar_operands", tuple[Any, AnyParams, AnyDataGenerator, Int])

    # main_carry = NewType(
    #    "main_carry",
    #    tuple[
    #        Int,
    #        Any,
    #        OptimizationContainer,
    #        OptimizationExtraContainer,
    #        DataGeneratorContainer,
    #        AbstractValidationModule,
    #        LossContainer,
    #        StoredObjectContainer,
    #        Float[Array, "n_iter"],
    #    ],
    # )
