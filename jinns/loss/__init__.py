from ._DynamicLossAbstract import ODE, PDEStatio, PDENonStatio
from ._DynamicLoss import (
    FisherKPP,
    Malthus,
    BurgerEquation,
    GeneralizedLotkaVolterra,
    OU_FPENonStatioLoss2D,
    ConvectionDiffusionNonStatio,
    MassConservation2DStatio,
    NavierStokes2DStatio,
)
from ._LossPDE import LossPDENonStatio, LossPDEStatio, SystemLossPDE
from ._LossODE import LossODE, SystemLossODE
