from ._DynamicLossAbstract import DynamicLoss, ODE, PDEStatio, PDENonStatio
from ._LossODE import LossODE, SystemLossODE
from ._LossPDE import LossPDEStatio, LossPDENonStatio, SystemLossPDE
from ._DynamicLoss import (
    GeneralizedLotkaVolterra,
    BurgerEquation,
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
    _div_fwd,
    _div_rev,
    _laplacian_fwd,
    _laplacian_rev,
    _vectorial_laplacian,
)
