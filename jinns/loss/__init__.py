from ._DynamicLossAbstract import ODE, PDEStatio, PDENonStatio
from ._DynamicLoss import (
    FisherKPP,
    BurgerEquation,
    GeneralizedLotkaVolterra,
    OU_FPENonStatioLoss2D,
    MassConservation2DStatio,
    NavierStokes2DStatio,
)
from ._LossPDE import LossPDENonStatio, LossPDEStatio, SystemLossPDE
from ._LossODE import LossODE, SystemLossODE
