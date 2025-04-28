import abc
from typing import TypeVar, Generic
from jaxtyping import Array
import equinox as eqx

EqType = TypeVar("EqType")


class AbstractPINN(eqx.Module, Generic[EqType]):
    """
    Basically just a way to add a __call__ to an eqx.Module.
    The way to go for correct type hints apparently
    https://github.com/patrick-kidger/equinox/issues/1002 + https://docs.kidger.site/equinox/pattern/
    """

    eq_type: eqx.AbstractVar[EqType]

    @abc.abstractmethod
    def __call__(self, *_, **__) -> Array:
        pass
