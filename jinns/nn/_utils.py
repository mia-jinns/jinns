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
    """
    Decorator to be used around __call__ functions of PINNs, SPINNs, etc. It
    authorizes the __call__ with `params` being directly be the
    PyTree (SPINN, PINN_MLP, ...) that we get out of `eqx.combine`

    This generic approach enables to cleanly handle type hints, up to the small
    effort required to understand type hints for decorators (ie ParamSpec).
    """

    def wrapper(
        self: Any,
        inputs: Any,
        params: PyTree | Params[Array],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        if isinstance(params, PyTree) and not isinstance(params, Params):
            params = Params(nn_params=params, eq_params={})
        return call_fun(self, inputs, params, *args, **kwargs)

    return wrapper
