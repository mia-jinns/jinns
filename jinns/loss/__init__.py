from ._DynamicLossAbstract import ODE, PDEStatio, PDENonStatio
from ._DynamicLoss import (
    FisherKPP,
    Malthus,
    BurgerEquation,
    GeneralizedLotkaVolterra,
    OU_FPEStatioLoss1D,
    CIR_FPEStatioLoss1D,
    OU_FPEStatioLoss2D,
    OU_FPENonStatioLoss1D,
    CIR_FPENonStatioLoss1D,
    Sinus_FPENonStatioLoss1D,
    OU_FPENonStatioLoss2D,
    ConvectionDiffusionNonStatio,
    MassConservation2DStatio,
    NavierStokes2DStatio,
)
from ._LossPDE import LossPDENonStatio, LossPDEStatio, SystemLossPDE
from ._LossODE import LossODE, SystemLossODE
