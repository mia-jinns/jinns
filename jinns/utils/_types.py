# pragma: exclude file
from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TypeAlias, TYPE_CHECKING, NewType
from jaxtyping import Int

if TYPE_CHECKING:
    from jinns.loss._LossPDE import (
        LossPDEStatio,
        LossPDENonStatio,
        SystemLossPDE,
    )

    from jinns.loss._LossODE import LossODE, SystemLossODE
    from jinns.parameters._params import Params, ParamsDict
    from jinns.data._DataGenerators import (
        DataGeneratorODE,
        CubicMeshPDEStatio,
        CubicMeshPDENonStatio,
        DataGeneratorObservations,
        DataGeneratorParameter,
        DataGeneratorObservationsMultiPINNs,
    )

    from jinns.loss import DynamicLoss
    from jinns.data._Batchs import *
    from jinns.nn._pinn import PINN
    from jinns.nn._hyperpinn import HyperPINN
    from jinns.nn._spinn_mlp import SPINN
    from jinns.utils._containers import *
    from jinns.validation._validation import AbstractValidationModule

    AnyLoss: TypeAlias = (
        LossPDEStatio | LossPDENonStatio | SystemLossPDE | LossODE | SystemLossODE
    )

    AnyParams: TypeAlias = Params | ParamsDict

    AnyDataGenerator: TypeAlias = (
        DataGeneratorODE | CubicMeshPDEStatio | CubicMeshPDENonStatio
    )

    AnyPINN: TypeAlias = PINN | HyperPINN | SPINN

    AnyBatch: TypeAlias = ODEBatch | PDEStatioBatch | PDENonStatioBatch
    rar_operands = NewType(
        "rar_operands", tuple[AnyLoss, AnyParams, AnyDataGenerator, Int]
    )

    main_carry = NewType(
        "main_carry",
        tuple[
            Int,
            AnyLoss,
            OptimizationContainer,
            OptimizationExtraContainer,
            DataGeneratorContainer,
            AbstractValidationModule,
            LossContainer,
            StoredObjectContainer,
            Float[Array, "n_iter"],
        ],
    )
