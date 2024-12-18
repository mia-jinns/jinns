# Your model : the differential operator, a.k.a the dynamic loss

## Abstract Class

::: jinns.loss.DynamicLoss
    options:
        members: False
        heading_level: 3

## ODE

::: jinns.loss.ODE
    options:
        members:
            - equation
        heading_level: 3

::: jinns.loss.GeneralizedLotkaVolterra
    options:
      heading_level: 3

## Stationary PDE

::: jinns.loss.PDEStatio
    options:
        members:
            - equation
        heading_level: 3

::: jinns.loss.MassConservation2DStatio
    options:
      heading_level: 3

::: jinns.loss.NavierStokes2DStatio
    options:
      heading_level: 3

## Non-stationary PDE

::: jinns.loss.PDENonStatio
    options:
        members:
            - equation
        heading_level: 3

::: jinns.loss.BurgersEquation
    options:
      heading_level: 3

::: jinns.loss.FisherKPP
    options:
      heading_level: 3

::: jinns.loss.FPENonStatioLoss2D
    options:
      heading_level: 3

::: jinns.loss.OU_FPENonStatioLoss2D
    options:
      heading_level: 3
