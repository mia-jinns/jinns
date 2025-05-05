from __future__ import annotations
import abc
from typing import Self, TYPE_CHECKING
import equinox as eqx

if TYPE_CHECKING:
    from jinns.utils._types import AnyBatch


class AbstractDataGenerator(eqx.Module):
    """
    Basically just a way to add a get_batch() to an eqx.Module.
    The way to go for correct type hints apparently
    https://github.com/patrick-kidger/equinox/issues/1002 + https://docs.kidger.site/equinox/pattern/
    """

    @abc.abstractmethod
    def get_batch(self) -> tuple[type[Self], AnyBatch]:  # type: ignore
        pass
