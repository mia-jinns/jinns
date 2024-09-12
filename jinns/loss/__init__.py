from ._DynamicLossAbstract import ODE, PDEStatio, PDENonStatio
from ._LossODE import LossODE, SystemLossODE
from ._LossPDE import LossPDEStatio, LossPDENonStatio, SystemLossPDE
from ._DynamicLoss import (
    GeneralizedLotkaVolterra,
    BurgerEquation,
    OU_FPENonStatioLoss2D,
    FisherKPP,
    MassConservation2DStatio,
    NavierStokes2DStatio,
)
from ._loss_weights import *
