from typing import TypeVar, Generic

from jinns.utils._ItemizableModule import ItemizableModule

T = TypeVar("T")


class ODEComponents(ItemizableModule, Generic[T]):
    dyn_loss: T
    initial_condition: T
    observations: T


class PDEStatioComponents(ItemizableModule, Generic[T]):
    dyn_loss: T
    norm_loss: T
    boundary_loss: T
    observations: T


class PDENonStatioComponents(PDEStatioComponents[T]):
    initial_condition: T
