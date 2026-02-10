from ._DynamicLossAbstract import DynamicLoss, ODE, PDEStatio, PDENonStatio
from ._LossODE import LossODE
from ._LossPDE import LossPDEStatio, LossPDENonStatio
from ._DynamicLoss import (
    GeneralizedLotkaVolterra,
    BurgersEquation,
    FPENonStatioLoss2D,
    OU_FPENonStatioLoss2D,
    FisherKPP,
    NavierStokesMassConservation2DStatio,
)
from ._BoundaryConditionAbstract import BoundaryConditionAbstract
from ._BoundaryCondition import Dirichlet, Neumann
from ._loss_weights import (
    LossWeightsODE,
    LossWeightsPDENonStatio,
    LossWeightsPDEStatio,
)
from ._loss_weight_updates import soft_adapt, lr_annealing, ReLoBRaLo

from ._operators import (
    divergence_fwd,
    divergence_rev,
    laplacian_fwd,
    laplacian_rev,
    vectorial_laplacian_fwd,
    vectorial_laplacian_rev,
)

from ._loss_utils import equation_on_all_facets_equal

__all__ = [
    "DynamicLoss",
    "ODE",
    "PDEStatio",
    "PDENonStatio",
    "LossODE",
    "LossPDEStatio",
    "LossPDENonStatio",
    "GeneralizedLotkaVolterra",
    "BurgersEquation",
    "FPENonStatioLoss2D",
    "OU_FPENonStatioLoss2D",
    "FisherKPP",
    "NavierStokesMassConservation2DStatio",
    "LossWeightsODE",
    "LossWeightsPDEStatio",
    "LossWeightsPDENonStatio",
    "divergence_fwd",
    "divergence_rev",
    "laplacian_fwd",
    "laplacian_rev",
    "vectorial_laplacian_fwd",
    "vectorial_laplacian_rev",
    "soft_adapt",
    "lr_annealing",
    "ReLoBRaLo",
    "Dirichlet",
    "Neumann",
    "BoundaryConditionAbstract",
    "equation_on_all_facets_equal",
]
