# Your model : dynamic (or physics) loss


## ODE

::: jinns.loss.ODE
    options:
        members:
            - equation

::: jinns.loss.GeneralizedLotkaVolterra


## Stationary PDE

::: jinns.loss.PDEStatio
    options:
        members:
            - equation

::: jinns.loss.MassConservation2DStatio

::: jinns.loss.NavierStokes2DStatio

## Non-stationary PDE

::: jinns.loss.PDENonStatio
    options:
        members:
            - equation

::: jinns.loss.BurgerEquation

::: jinns.loss.FisherKPP

::: jinns.loss.FPENonStatioLoss2D

::: jinns.loss.OU_FPENonStatioLoss2D
