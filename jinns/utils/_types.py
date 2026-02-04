from __future__ import (
    annotations,
)  # https://docs.python.org/3/library/typing.html#constant

from typing import TypeAlias, TYPE_CHECKING, Callable, TypeVar
from jaxtyping import Float, Array, PRNGKeyArray

from jinns.data._Batchs import ODEBatch, PDEStatioBatch, PDENonStatioBatch, ObsBatchDict
from jinns.loss._loss_weights import (
    LossWeightsODE,
    LossWeightsPDEStatio,
    LossWeightsPDENonStatio,
)
from jinns.parameters._derivative_keys import (
    DerivativeKeysODE,
    DerivativeKeysPDENonStatio,
    DerivativeKeysPDEStatio,
)
from jinns.loss._loss_components import (
    ODEComponents,
    PDEStatioComponents,
    PDENonStatioComponents,
)

AnyBatch: TypeAlias = ODEBatch | PDENonStatioBatch | PDEStatioBatch | ObsBatchDict

AnyDerivativeKeys: TypeAlias = (
    DerivativeKeysODE | DerivativeKeysPDEStatio | DerivativeKeysPDENonStatio
)
AnyLossWeights: TypeAlias = (
    LossWeightsODE | LossWeightsPDEStatio | LossWeightsPDENonStatio
)

# Note that syntax change starting from 3.12
_T = TypeVar("_T")
AnyLossComponents: TypeAlias = (
    ODEComponents[_T] | PDEStatioComponents[_T] | PDENonStatioComponents[_T]
)

if TYPE_CHECKING:
    from jinns.utils._containers import (
        DataGeneratorContainer,
        OptimizationContainer,
        OptimizationExtraContainer,
        LossContainer,
        StoredObjectContainer,
    )
    from jinns.validation._validation import AbstractValidationModule
    from jinns.loss._abstract_loss import AbstractLoss
    from jinns.loss._LossODE import LossODE
    from jinns.loss._LossPDE import LossPDEStatio, LossPDENonStatio
    from jinns.parameters._Params import Params
    from jinns.nn._abstract_pinn import AbstractPINN
    from jinns.loss._BoundaryConditionAbstract import BoundaryCondition

    # Here we define types available for the whole package
    BoundaryConditionFun: TypeAlias = Callable[
        [Float[Array, " dim"] | Float[Array, " dim + 1"]], Float[Array, " dim_solution"]
    ]

    BoundaryEquationU: TypeAlias = Callable[
        [
            BoundaryCondition,
            Float[Array, " InputDim n_facet"],
            AbstractPINN,
            Params[Array],
        ],
        tuple[Float[Array, " InputDim"]],
    ]
    BoundaryEquationF: TypeAlias = Callable[
        [BoundaryCondition, Float[Array, " InputDim n_facet"], Params[Array], bool],
        tuple[Float[Array, " InputDim"]],
    ]
    BoundaryEquationUOnFacet: TypeAlias = Callable[
        [BoundaryCondition, Float[Array, " InputDim"], AbstractPINN, Params[Array]],
        Float[Array, " InputDim"],
    ]
    BoundaryEquationFOnFacet: TypeAlias = Callable[
        [BoundaryCondition, Float[Array, " InputDim"], Params[Array], bool],
        Float[Array, " InputDim"],
    ]

    AnyLoss: TypeAlias = LossODE | LossPDEStatio | LossPDENonStatio

    SolveCarry: TypeAlias = tuple[
        int,
        AbstractLoss,
        OptimizationContainer,
        OptimizationExtraContainer,
        DataGeneratorContainer,
        AbstractValidationModule | None,
        LossContainer,
        StoredObjectContainer,
        Float[Array, " n_iter"] | None,
        PRNGKeyArray | None,
    ]

    SolveAlternateCarry: TypeAlias = tuple[
        int,
        AbstractLoss,
        OptimizationContainer,
        OptimizationExtraContainer,
        DataGeneratorContainer,
        LossContainer,
        StoredObjectContainer,
        PRNGKeyArray | None,
    ]
