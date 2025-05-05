import abc
from jaxtyping import Array
import equinox as eqx


class AbstractLoss(eqx.Module):
    """
    Basically just a way to add a __call__ to an eqx.Module.
    The way to go for correct type hints apparently
    https://github.com/patrick-kidger/equinox/issues/1002 + https://docs.kidger.site/equinox/pattern/
    """

    @abc.abstractmethod
    def __call__(self, *_, **__) -> Array:
        pass
