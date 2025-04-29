# pragma: exclude file
from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TypeAlias, TYPE_CHECKING, NewType, TypedDict, Callable
from jaxtyping import Int

if TYPE_CHECKING:
    from jinns.parameters._params import Params
    from jinns.data._DataGenerators import (
        DataGeneratorODE,
        CubicMeshPDEStatio,
        CubicMeshPDENonStatio,
        DataGeneratorObservations,
        DataGeneratorParameter,
    )

    from jinns.loss import DynamicLoss
    from jinns.data._Batchs import *
    from jinns.nn._abstract_pinn import AbstractPINN
    from jinns.nn._pinn import PINN
    from jinns.nn._hyperpinn import HyperPINN
    from jinns.nn._spinn_mlp import SPINN
    from jinns.utils._containers import *
    from jinns.validation._validation import AbstractValidationModule

    BoundaryConditionFun: TypeAlias = Callable[
        [Float[Array, "dim"] | Float[Array, "dim + 1"]], Float[Array, "dim_solution"]
    ]

    AnyParams: TypeAlias = Params

    AnyDataGenerator: TypeAlias = (
        DataGeneratorODE | CubicMeshPDEStatio | CubicMeshPDENonStatio
    )

    AnyPINN: TypeAlias = PINN | HyperPINN | SPINN

    AnyBatch: TypeAlias = ODEBatch | PDEStatioBatch | PDENonStatioBatch

    rar_operands = NewType("rar_operands", tuple[Any, AnyParams, AnyDataGenerator, Int])

    main_carry = NewType(
        "main_carry",
        tuple[
            Int,
            Any,
            OptimizationContainer,
            OptimizationExtraContainer,
            DataGeneratorContainer,
            AbstractValidationModule,
            LossContainer,
            StoredObjectContainer,
            Float[Array, "n_iter"],
        ],
    )
