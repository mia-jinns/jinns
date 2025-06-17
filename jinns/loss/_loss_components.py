from typing import TypeVar, Generic
from dataclasses import fields
import equinox as eqx

T = TypeVar("T")


class XDEComponentsAbstract(eqx.Module, Generic[T]):
    """
    Provides a template for ODE components with generic types.
    One can inherit to specialize and add methods and attributes
    We do not enforce keyword only to avoid being to verbose (this then can
    work like a tuple)
    """

    def items(self):
        """
        For the dataclass to be iterated like a dictionary.
        Practical and retrocompatible with old code when loss components were
        dictionaries
        """
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if getattr(self, field.name) is not None
        }.items()


class ODEComponents(XDEComponentsAbstract[T]):
    dyn_loss: T
    initial_condition: T
    observations: T


class PDEStatioComponents(XDEComponentsAbstract[T]):
    dyn_loss: T
    norm_loss: T
    boundary_loss: T
    observations: T


class PDENonStatioComponents(PDEStatioComponents[T]):
    initial_condition: T
