from __future__ import annotations
import abc
from typing import Self, TYPE_CHECKING, TypeVar, Generic
import equinox as eqx

if TYPE_CHECKING:
    from jinns.utils._types import AnyBatch


B = TypeVar(
    "B", bound=AnyBatch
)  # The above comment also works with Unions (https://docs.python.org/3/library/typing.html#typing.TypeVar)
# We then do the same TypeVar to be able to use one of the element of AnyBatch
# in the evaluate_by_terms methods of child classes.


class AbstractDataGenerator(eqx.Module, Generic[B]):
    """
    Basically just a way to add a get_batch() to an eqx.Module.
    The way to go for correct type hints apparently
    https://github.com/patrick-kidger/equinox/issues/1002 + https://docs.kidger.site/equinox/pattern/
    """

    @abc.abstractmethod
    def get_batch(self) -> tuple[Self, B]:  # type: ignore
        pass
