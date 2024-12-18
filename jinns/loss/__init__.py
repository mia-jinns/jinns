from ._DynamicLossAbstract import DynamicLoss, ODE, PDEStatio, PDENonStatio
from ._LossODE import LossODE, SystemLossODE
from ._LossPDE import LossPDEStatio, LossPDENonStatio, SystemLossPDE
from ._DynamicLoss import (
    GeneralizedLotkaVolterra,
    BurgersEquation,
    FPENonStatioLoss2D,
    OU_FPENonStatioLoss2D,
    FisherKPP,
    MassConservation2DStatio,
    NavierStokes2DStatio,
)
from ._loss_weights import (
    LossWeightsODE,
    LossWeightsODEDict,
    LossWeightsPDENonStatio,
    LossWeightsPDEStatio,
    LossWeightsPDEDict,
)

from ._operators import (
    divergence_fwd,
    divergence_rev,
    laplacian_fwd,
    laplacian_rev,
    vectorial_laplacian_fwd,
    vectorial_laplacian_rev,
)
