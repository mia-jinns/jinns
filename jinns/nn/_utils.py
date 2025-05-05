from typing import Any, ParamSpec, Callable, Concatenate
from jaxtyping import PyTree, Array
from jinns.parameters._params import Params


P = ParamSpec("P")


def _PyTree_to_Params(
    call_fun: Callable[
        Concatenate[Any, Any, PyTree | Params[Array], P],
        Any,
    ],
) -> Callable[
    Concatenate[Any, Any, PyTree | Params[Array], P],
    Any,
]:
    def wrapper(
        self: Any,
        inputs: Any,
        params: PyTree | Params[Array],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        if isinstance(params, PyTree):
            params = Params(nn_params=params, eq_params={})
        return call_fun(self, inputs, params, *args, **kwargs)

    return wrapper
