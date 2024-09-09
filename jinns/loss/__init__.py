from ._DynamicLossAbstract_eqx import ODE, PDEStatio, PDENonStatio
from ._LossODE_eqx import LossODE_eqx, SystemLossODE_eqx
from ._LossPDE_eqx import LossPDEStatio_eqx, LossPDENonStatio_eqx, SystemLossPDE_eqx
from ._DynamicLoss_eqx import (
    GeneralizedLotkaVolterra_eqx,
    BurgerEquation_eqx,
    OU_FPENonStatioLoss2D_eqx,
    FisherKPP_eqx,
    MassConservation2DStatio_eqx,
    NavierStokes2DStatio_eqx,
)
from ._loss_weights import *
