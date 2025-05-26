from typing import TypeVar, Generic
import equinox as eqx

T = TypeVar("T")


class ODEComponents(eqx.Module, Generic[T]):
    """
    Provides a template for ODE components with generic types.
    One can inherit to specialize and add methods and attributes
    We do not enforce keyword only to avoid being to verbose (this then can
    work like a tuple)
    """

    dyn_loss: T
    initial_condition: T
    observations: T


class PDEStatioComponents(eqx.Module, Generic[T]):
    dyn_loss: T
    norm_loss: T
    boundary_loss: T
    observations: T


class PDENonStatioComponents(PDEStatioComponents[T]):
    initial_condition: T
