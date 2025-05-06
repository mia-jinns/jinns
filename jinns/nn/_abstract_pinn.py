import abc
from typing import Literal, Any
from jaxtyping import Array
import equinox as eqx

from jinns.nn._utils import _PyTree_to_Params
from jinns.parameters._params import Params


class AbstractPINN(eqx.Module):
    """
    Basically just a way to add a __call__ to an eqx.Module.
    The way to go for correct type hints apparently
    https://github.com/patrick-kidger/equinox/issues/1002 + https://docs.kidger.site/equinox/pattern/
    """

    eq_type: eqx.AbstractVar[Literal["ODE", "statio_PDE", "nonstatio_PDE"]]

    @abc.abstractmethod
    @_PyTree_to_Params
    def __call__(self, inputs: Any, params: Params[Array], *args, **kwargs) -> Any:
        pass
